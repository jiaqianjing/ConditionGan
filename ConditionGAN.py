import torch
from torchvision import datasets, transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision.utils import save_image
import os, time

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


# Gnet
class G(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        # noise img: [B, 100] -> [B, 256]
        self.fc1 = nn.Linear(100, 256)
        # self.bn1 = nn.BatchNorm1d(256)

        # label: [B, 10] -> [B, 256]
        self.lfc1 = nn.Linear(10, 256)
        # self.lbn1 = nn.BatchNorm1d(256)

        # fusion(img, label) shape: [B, 512]  -> [B, 512]
        self.fc2 = nn.Linear(512, 512)
        self.bn2 = nn.BatchNorm1d(512)

        # [B, 512] -> [B, 1024]
        self.fc3 = nn.Linear(512, 1024)
        self.bn3 = nn.BatchNorm1d(1024)

        # [B, 1024] -> [B, 28*28]
        self.fc4 = nn.Linear(1024, 28 * 28)

    def forward(self, input, label):
        x = F.leaky_relu(self.fc1(input), negative_slope=0.2)
        y = F.leaky_relu(self.lfc1(label), negative_slope=0.2)
        # fusion img and label
        x = torch.cat([x, y], 1)
        x = F.leaky_relu(self.bn2(self.fc2(x)), negative_slope=0.2)
        x = F.leaky_relu(self.bn3(self.fc3(x)), negative_slope=0.2)
        # normal output -> [-1, 1]
        x = torch.tanh(self.fc4(x))
        return x

    # init weight
    def weight_init(self, mean, std):
        for i in self._modules:
            if isinstance(i, nn.Linear):
                i.weight.data.normal_(mean, std)
                i.bias.data.zero_()


class D(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        #img shape: [B, 28*28] -> [B, 1024]
        self.fc1 = nn.Linear(28 * 28, 1024)
        self.bn1 = nn.BatchNorm1d(1024)
        # label shape: [B, 10] -> [B, 1024]
        self.lfc1 = nn.Linear(10, 1024)
        self.lbn1 = nn.BatchNorm1d(1024)

        # fusion(img, label) shape: [B, 1024*2] -> [B, 512]
        self.fc2 = nn.Linear(2048, 512)
        self.bn2 = nn.BatchNorm1d(512)

        # [B, 512] -> [B, 256]
        # self.fc3 = nn.Linear(512, 256)
        # self.bn3 = nn.BatchNorm1d(256)

        # [B, 512] -> [B, 1]
        self.fc4 = nn.Linear(512, 1)

    def forward(self, input, label):
        x = F.leaky_relu(self.bn1(self.fc1(input)), negative_slope=0.2)
        y = F.leaky_relu(self.lbn1(self.lfc1(label)), negative_slope=0.2)
        # fusion x & y
        x = torch.cat([x, y], dim=1)
        x = F.leaky_relu(self.bn2(self.fc2(x)), negative_slope=0.2)
        # x = F.leaky_relu(self.bn3(self.fc3(x)), negative_slope=0.2)
        # output -> [0, 1]
        x = torch.sigmoid(self.fc4(x))
        return x

    def weight_init(self, mean, std):
        for i in self._modules:
            if isinstance(i, nn.Linear):
                i.weight.data.norm_(mean, std)
                i.bias.data.zero_()


def create_fake_label():
    # 用于预测生成指定图片的 label
    # img [B, 100], label [B, 10]
    noise_img = torch.rand(10, 100)
    noise_label = torch.arange(0, 10)  # shape [10, 1]
    noise_label_one_hot = F.one_hot(noise_label).type(torch.float32)
    print(f"noise_label: {noise_label}")
    return noise_img, noise_label_one_hot


def save_result(g_net, path='./result.jpg'):
    g_net.eval()
    noise_img, noise_label = create_fake_label()
    noise_img, noise_label = noise_img.to(device), noise_label.to(device)
    generated = g_net(noise_img, noise_label)
    a = generated.reshape(generated.shape[0], 1, 28, 28)
    print(f"save img to {path}")
    save_image(a, path)
    g_net.train()


def train_loop(g_net, d_net):
    """Train"""
    batch_size = 128
    lr = 0.0002
    epochs = 20
    output = './results'

    if not os.path.exists(output):
        os.mkdir(output)

    transform = transforms.Compose([
        transforms.ToTensor(),  # [0, 1]
        transforms.Normalize(mean=[0.5], std=[0.5])  # [-1, 1]
    ])
    train_dataloader = torch.utils.data.DataLoader(
        datasets.MNIST(root='data',
                       train=True,
                       download=True,
                       transform=transform),
        batch_size=batch_size,
        shuffle=True,
    )
    optG = optim.Adam(g_net.parameters(), lr=lr, betas=(0.5, 0.999))
    optD = optim.Adam(d_net.parameters(), lr=lr, betas=(0.5, 0.999))
    loss = nn.BCELoss()

    print("start training...")
    start_time = time.perf_counter()
    for epoch in range(epochs):
        G_losses = []
        D_losses = []
        epoch_start_time = time.perf_counter()

        if epoch == 30:
            optG.param_groups[0]['lr'] /= 10
            optD.param_groups[0]['lr'] /= 10
            print(f"lr: {str(optD.param_groups[0]['lr'])}")
        for img, label in train_dataloader:

            batch_size = img.shape[0] # 最后一个 batch_size 的样本数据可能会小于 dataloader 中设置的 batch_size
            # D 判别真假的标签
            real = torch.ones(batch_size, 1)
            fake = torch.zeros(batch_size, 1)

            # =============================================================================================
            # fix G, opt D, max V(D, G) = logD(x|y) + log[1-D(z|y)]
            # =============================================================================================
            optD.zero_grad()
            img = img.reshape(-1, 28 * 28)
            label = label.reshape(batch_size, 1)  # [batch] -> [batch, 1]
            label = torch.zeros(batch_size, 10).scatter(1, label, 1)  # one hot -> [128, 10]

            img, label, real = img.to(device), label.to(device), real.to(device)
            d_real_out = d_net(img, label)
            d_real_loss = loss(d_real_out, real)

            # 随机生成噪声以及标签
            fake_img = torch.rand(batch_size, 100)
            fake_label = (torch.rand(batch_size, 1) * 10).type(
                torch.int64)  # 随机生成 0-9 的标签, shape [batch_size, 1]
            fake_label = torch.zeros(batch_size, 10).scatter(1, fake_label, 1)  # [one hot] shape [batch_size, 10], dtype=float32

            fake_img, fake_label, fake = fake_img.to(device=device), fake_label.to(device), fake.to(device)
            d_fake_out = d_net(g_net(fake_img, fake_label), fake_label)
            d_fake_loss = loss(d_fake_out, fake)

            d_loss = d_real_loss + d_fake_loss
            d_loss.backward()
            optD.step()
            D_losses.append(d_loss.item())

            # ================================================================
            # fix D, opt G, min V(D, G) = log[1-D(z|y)]
            # ================================================================
            optG.zero_grad()
            fake_img = torch.rand(batch_size, 100)
            fake_label = (torch.rand(batch_size) * 10).type(torch.int64)  # shape [batch_size]
            fake_label = F.one_hot(fake_label, -1).type(torch.float32)  # [one hot] shape [batch_size, 10]
            fake_img, fake_label = fake_img.to(device), fake_label.to(device)
            d_out = d_net(g_net(fake_img, fake_label), fake_label)
            g_loss = loss(d_out, real)
            g_loss.backward()
            optG.step()
            G_losses.append(g_loss.item())

        print(f"epoch: {epoch}, G loss: {sum(G_losses)/len(G_losses)}, D loss: {sum(D_losses)/len(D_losses)}")
        file_path = f'./{output}/{epoch}-results.jpg'
        save_result(g_net, file_path)
        epoch_time = time.perf_counter() - epoch_start_time
        print(f"current epoch spent: {epoch_time}")

    dura_time = time.perf_counter() - start_time
    print(f"Total training spent: {dura_time}")


if __name__ == '__main__':
    g_net = G()
    g_net.to(device)
    g_net.weight_init(0, 0.02)
    d_net = D()
    d_net.to(device)
    d_net.weight_init(0, 0.2)
    train_loop(g_net, d_net)
