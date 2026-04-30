import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os

device = "cuda" if torch.cuda.is_available() else "cpu"
os.makedirs("gan_images", exist_ok=True)

# Dataset
transform = transforms.Compose([
    transforms.Resize(32),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

dataset = torchvision.datasets.MNIST(root="./data", train=True, transform=transform, download=True)
loader = DataLoader(dataset, batch_size=128, shuffle=True)

# Generator
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(100, 256, 4, 1, 0),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 1, 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, z):
        return self.net(z)

# Discriminator
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 64, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 1, 4, 1, 0),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x).view(-1)

G = Generator().to(device)
D = Discriminator().to(device)

opt_g = torch.optim.Adam(G.parameters(), lr=0.0002)
opt_d = torch.optim.Adam(D.parameters(), lr=0.0002)

def save_images(epoch):
    z = torch.randn(16, 100, 1, 1).to(device)
    fake = G(z).detach().cpu()
    grid = torchvision.utils.make_grid(fake, nrow=4, normalize=True)
    plt.imshow(grid.permute(1,2,0))
    plt.axis("off")
    plt.savefig(f"gan_images/epoch_{epoch}.png")
    plt.close()

# Training
for epoch in range(5):
    for real, _ in loader:
        real = real.to(device)
        batch = real.size(0)

        z = torch.randn(batch, 100, 1, 1).to(device)
        fake = G(z)

        # Train D
        loss_d = -torch.mean(torch.log(D(real)+1e-8) + torch.log(1-D(fake.detach())+1e-8))
        opt_d.zero_grad(); loss_d.backward(); opt_d.step()

        # Train G
        loss_g = -torch.mean(torch.log(D(fake)+1e-8))
        opt_g.zero_grad(); loss_g.backward(); opt_g.step()

    save_images(epoch)
    print(f"[GAN] Epoch {epoch} | D: {loss_d.item():.3f} | G: {loss_g.item():.3f}")
