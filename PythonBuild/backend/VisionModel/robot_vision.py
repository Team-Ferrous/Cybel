import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from PIL import Image
import matplotlib.pyplot as plt
import os

# Hyperparameters
latent_dim = 64
batch_size = 128
epochs = 50
learning_rate = 1e-3
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Model Definition
class Encoder(nn.Module):
    def __init__(self, latent_dim=64):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1)  # 32x32 -> 16x16
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)  # 16x16 -> 8x8
        self.conv3 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)  # 8x8 -> 4x4
        self.fc_mu = nn.Linear(128 * 4 * 4, latent_dim)
        self.fc_logvar = nn.Linear(128 * 4 * 4, latent_dim)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

class Decoder(nn.Module):
    def __init__(self, latent_dim=64):
        super(Decoder, self).__init__()
        self.fc = nn.Linear(latent_dim, 128 * 4 * 4)
        self.deconv1 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)  # 4x4 -> 8x8
        self.deconv2 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)  # 8x8 -> 16x16
        self.deconv3 = nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1)  # 16x16 -> 32x32

    def forward(self, z):
        z = self.fc(z)
        z = z.view(z.size(0), 128, 4, 4)
        z = F.relu(self.deconv1(z))
        z = F.relu(self.deconv2(z))
        z = torch.sigmoid(self.deconv3(z))  # Output in [0,1]
        return z

class VAE(nn.Module):
    def __init__(self, latent_dim=64):
        super(VAE, self).__init__()
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z)
        return recon, mu, logvar

def vae_loss(recon_x, x, mu, logvar):
    recon_loss = F.mse_loss(recon_x, x, reduction='sum')
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl_loss

# Training Function
def train_vae():
    # Dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Model
    model = VAE(latent_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.to(device)
            optimizer.zero_grad()
            recon, mu, logvar = model(data)
            loss = vae_loss(recon, data, mu, logvar)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f'Epoch {epoch+1}/{epochs}, Loss: {total_loss / len(train_loader):.4f}')

    # Save model
    os.makedirs('models', exist_ok=True)
    torch.save(model.state_dict(), 'models/vae_cifar10.pth')
    print('Model saved to models/vae_cifar10.pth')
    return model

# Inference Function
def load_model():
    model = VAE(latent_dim).to(device)
    model.load_state_dict(torch.load('models/vae_cifar10.pth', map_location=device))
    model.eval()
    return model

def process_image(model, image_path):
    """Process a single image: encode to latent space and reconstruct."""
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    img = Image.open(image_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        recon, mu, logvar = model(img_tensor)
        latent = model.reparameterize(mu, logvar)

    # Denormalize for display
    recon = recon.squeeze().cpu()
    recon = (recon * 0.5) + 0.5  # Unnormalize
    recon = torch.clamp(recon, 0, 1)

    return latent.cpu().numpy(), recon.permute(1, 2, 0).numpy()

def display_reconstruction(original_img, reconstructed_img):
    """Display original and reconstructed images."""
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(original_img)
    axes[0].set_title('Original')
    axes[0].axis('off')
    axes[1].imshow(reconstructed_img)
    axes[1].set_title('Reconstructed')
    axes[1].axis('off')
    plt.show()

# Main
if __name__ == "__main__":
    # Train the model
    print("Training VAE...")
    trained_model = train_vae()

    # Example inference
    model = load_model()
    image_path = 'path/to/your/image.jpg'  # Replace with actual path
    try:
        latent, recon_img = process_image(model, image_path)
        original_img = Image.open(image_path).convert('RGB').resize((32, 32))
        display_reconstruction(original_img, recon_img)
        print(f'Latent representation shape: {latent.shape}')
    except FileNotFoundError:
        print("Please provide a valid image path for inference.")