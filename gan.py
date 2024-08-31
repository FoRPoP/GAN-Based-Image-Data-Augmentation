import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import os
import numpy as np
from PIL import Image

# Definisanje Generatora
class Generator(nn.Module):
    def __init__(self, noise_dim=100, img_channels=1, hidden_dim=128):
        super(Generator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(noise_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim * 4),
            nn.BatchNorm1d(hidden_dim * 4),
            nn.ReLU(),
            nn.Linear(hidden_dim * 4, img_channels * 28 * 28),
            nn.Tanh()
        )
        self.img_channels = img_channels

    def forward(self, z):
        out = self.net(z)
        out = out.view(out.size(0), self.img_channels, 28, 28)
        return out

# Definisanje Discriminatora
class Discriminator(nn.Module):
    def __init__(self, img_channels=1, hidden_dim=128):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(img_channels * 28 * 28, hidden_dim * 4),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim * 4, hidden_dim * 2),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, img):
        return self.net(img)

# Definisanje GAN modela
class GAN(nn.Module):
    def __init__(self, noise_dim=100, img_channels=1, hidden_dim=128, lr_disc=0.0002, lr_gen=0.0002):
        super(GAN, self).__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.noise_dim = noise_dim

        self.generator = Generator(noise_dim, img_channels, hidden_dim).to(self.device)
        self.discriminator = Discriminator(img_channels, hidden_dim).to(self.device)

        self.d_optimizer = optim.Adam(self.discriminator.parameters(), lr=lr_disc)
        self.g_optimizer = optim.Adam(self.generator.parameters(), lr=lr_gen)
        self.loss_function = nn.BCELoss()

        self.fixed_noise = torch.randn(25, noise_dim, device=self.device)

    def train_discriminator(self, real_data, fake_data):
        self.d_optimizer.zero_grad()

        prediction_real = self.discriminator(real_data)
        real_target = torch.ones(real_data.size(0), 1, device=self.device)
        loss_real = self.loss_function(prediction_real, real_target)

        prediction_fake = self.discriminator(fake_data)
        fake_target = torch.zeros(fake_data.size(0), 1, device=self.device)
        loss_fake = self.loss_function(prediction_fake, fake_target)

        loss = loss_real + loss_fake
        loss.backward()
        self.d_optimizer.step()

        return loss.item()

    def train_generator(self, noise_data):
        self.g_optimizer.zero_grad()

        fake_data = self.generator(noise_data)
        prediction = self.discriminator(fake_data)
        target = torch.ones(noise_data.size(0), 1, device=self.device)
        loss = self.loss_function(prediction, target)

        loss.backward()
        self.g_optimizer.step()

        return loss.item()

    def train(self, data_loader, num_epochs, print_interval=200):
        for epoch in range(num_epochs):
            disc_loss = 0
            gen_loss = 0
            for batch_idx, (real_data, _) in enumerate(data_loader):
                real_data = real_data.view(real_data.size(0), -1).to(self.device)

                noise = torch.randn(real_data.size(0), self.noise_dim, device=self.device)
                fake_data = self.generator(noise)

                disc_loss += self.train_discriminator(real_data, fake_data.detach())
                gen_loss += self.train_generator(noise)

                if batch_idx % print_interval == 0:
                    print(f'Epoch [{epoch + 1}/{num_epochs}] Batch {batch_idx}/{len(data_loader)} \
                           Discriminator Loss: {disc_loss / (batch_idx + 1):.4f}, Generator Loss: {gen_loss / (batch_idx + 1):.4f}')

            # Generisanje slika za prikazivanje
            if (epoch + 1) % 10 == 0:
                self.sample_images(epoch + 1)
                self.save_model(epoch + 1)

    def sample_images(self, epoch):
        fake_images = self.generator(self.fixed_noise).detach()
        fake_images = fake_images.view(fake_images.size(0), 1, 28, 28)

        grid = self.make_grid(fake_images)
        plt.imshow(grid.permute(1, 2, 0), cmap='gray')
        plt.title(f'Epoch {epoch}')
        os.makedirs('images', exist_ok=True)
        plt.savefig(f'images/epoch_{epoch}.png')
        plt.close()

    def make_grid(self, images, nrow=5):
        grid = torch.cat([images[i] for i in range(nrow)], dim=2)
        for i in range(1, len(images) // nrow):
            grid = torch.cat([grid, torch.cat([images[i * nrow + j] for j in range(nrow)], dim=2)], dim=1)
        return grid

    def save_model(self, epoch):
        os.makedirs('models', exist_ok=True)
        torch.save(self.generator.state_dict(), f'models/generator_epoch_{epoch}.pth')
        torch.save(self.discriminator.state_dict(), f'models/discriminator_epoch_{epoch}.pth')

    def generate_dataset(self, n, label=9):
        self.generator.eval()
        os.makedirs('generated_data', exist_ok=True)
        
        labels = []
        images = []

        for i in range(n):
            noise = torch.randn(1, self.noise_dim, device=self.device)
            with torch.no_grad():
                fake_image = self.generator(noise).detach().cpu()
            fake_image = fake_image.view(28, 28).numpy()

            img = Image.fromarray((fake_image * 225).astype(np.uint8), mode='L')
            img_path = f'generated_data/fake_image_{i}.png'
            img.save(img_path)

            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
            ])
            img_tensor = transform(Image.open(img_path))
            images.append(img_tensor)
            labels.append(label)

        # Sačuvamo sve labele u .txt fajl
        with open('generated_data/labels.txt', 'w') as f:
            for label in labels:
                f.write(f"{label}\n")

        print(f"Generated {n} images with label {label}.")

        images = torch.stack(images).numpy()
        labels = np.array(labels)
        
