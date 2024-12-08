import torch
from torch import optim
from torchvision import datasets
from torch.utils.data import DataLoader
import argparse
from tqdm import tqdm
import os
from data_loading import load_dataset, get_transform
from model import Generator, Discriminator


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def save_model(model, save_path, name=""):
    torch.save(model.state_dict(), os.path.join(save_path, name + ".pth"))

def train(data_path, batch_size=64, lr=0.0001, epochs=10, latent_dim=100, save_path="../checkpoints", save_name=""):
    device = get_device()

    print("Loading dataset...")
    load_dataset(data_path)
    transform = get_transform()
    train_dataset = datasets.FashionMNIST(root=data_path, train=True, download=False, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    print("Dataset loaded !")

    print("Loading models...")
    G, D = Generator(latent_dim=latent_dim).to(device), Discriminator().to(device)
    optim_G = optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
    optim_D = optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))
    print("Models loaded !")

    print("Starting training...")
    for epoch in range(epochs):
        g_loss_epoch, d_loss_epoch = 0.0, 0.0
        with tqdm(total=len(train_loader), desc=f"Epoch {epoch + 1}/{epochs}") as pbar:
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(device), target.to(device)
                batch_size = data.size(0)
                real_label = torch.full((batch_size, 1), 1., device=device)
                fake_label = torch.full((batch_size, 1), 0., device=device)

                # Train G
                G.zero_grad()
                z_noise = torch.randn(batch_size, latent_dim, device=device)
                x_fake_labels = torch.randint(0, 10, (batch_size,), device=device)
                x_fake = G(z_noise, x_fake_labels)
                y_fake_g = D(x_fake, x_fake_labels)
                g_loss = D.loss(y_fake_g, real_label)
                g_loss.backward()
                optim_G.step()

                # Train D
                D.zero_grad()
                y_real = D(data, target)
                d_real_loss = D.loss(y_real, real_label)
                y_fake_d = D(x_fake.detach(), x_fake_labels)
                d_fake_loss = D.loss(y_fake_d, fake_label)
                d_loss = (d_real_loss + d_fake_loss) / 2
                d_loss.backward()
                optim_D.step()

                g_loss_epoch += g_loss.item()
                d_loss_epoch += d_loss.item()

                pbar.update(1)

        pbar.close()
        g_loss_epoch /= len(train_loader)
        d_loss_epoch /= len(train_loader)
        print(f'Epoch {epoch + 1} loss_D: {d_loss_epoch:.4f} loss_G: {g_loss_epoch:.4f}')
    print("Training finished !")

    print("Saving the model")
    save_model(G, save_path, "generator" + save_name)
    save_model(D, save_path, "discriminator" + save_name)
    print(f"Models saved : path {save_path} !")

    return

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="../data", help="Directory where the dataset is stored.")
    parser.add_argument("--save-dir", default="../checkpoints", help="Directory where to save the trained model.")
    parser.add_argument("--save-name",default="", help="Name of the saved model")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for training.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs to train.")
    parser.add_argument("--lr", type=float, default=0.0002, help="Learning rate.")
    parser.add_argument("--latent-dim", type=int, default=100, help="Dimension of the latent noise vector.")
    args = parser.parse_args()

    train(data_path=args.data_dir,
          batch_size=args.batch_size,
          lr=args.lr,
          epochs=args.epochs,
          latent_dim=args.latent_dim,
          save_name=args.save_name)

    return

if __name__ == "__main__":
    main()

