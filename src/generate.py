import torch
import matplotlib.pyplot as plt
from model import Generator, Discriminator
import argparse
from tqdm import tqdm

def load_models(generator_path, discriminator_path, device):
    # Load the trained models
    G, D = Generator().to(device), Discriminator().to(device)
    G.load_state_dict(torch.load(generator_path, map_location=device, weights_only=True))
    D.load_state_dict(torch.load(discriminator_path, map_location=device, weights_only=True))
    G.eval()
    D.eval()
    return G, D

def save_images(images, save_path):
    for i, img in enumerate(images):
        if img.shape[0] == 1:  # If the image is grayscale
            img = img.squeeze(0)  # Remove the channel dimension
        plt.imsave(f"{save_path}/{i}.png", img, cmap='gray')
    print(f"Images saved successfully : path {save_path}!")

def generate_and_save_images(generator_path, discriminator_path, latent_dim=100, num_images=10, save_path='../samples'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading the trained models...")
    G, D = load_models(generator_path, discriminator_path, device)
    print("Models loaded successfully !")

    # Generate images
    print("Generating images...")
    approved_images = []
    with tqdm(total=num_images, desc="Generating images") as pbar:
        while len(approved_images) < num_images:
            # Generate images
            z_noise = torch.randn(1, latent_dim, device=device)
            labels = torch.randint(0, 10, (1,), device=device)
            generated_images = G(z_noise, labels)
            # Check image validity with discriminator
            approvals = D(generated_images, labels)
            for img, approval in zip(generated_images, approvals):
                if approval > 0.5:
                    approved_images.append(img)
                    pbar.update(1)
                    if len(approved_images) >= num_images:
                        break
    # Convert images to numpy and save
    approved_images = torch.stack(approved_images).cpu().detach().numpy()
    approved_images = (approved_images + 1) / 2  # Rescale to [0, 1]
    print("Images generated successfully !")

    print("Saving images...")
    save_images(approved_images, save_path)
    print(f"Images saved successfully : path {save_path}!")
    return

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--generator-path", default="../checkpoints/generator.pth", help="Path to the trained generator model.")
    parser.add_argument("--discriminator-path", default="../checkpoints/discriminator.pth", help="Path to the trained discriminator model.")
    parser.add_argument("--save-path", default="../samples", help="Path to save the generated")
    parser.add_argument("--latent-dim", default=100, help="Noise dimension for the generator.")
    parser.add_argument("--num-images", default=10, help="Number of images to generate.")
    args = parser.parse_args()

    generate_and_save_images(generator_path=args.generator_path,
                             discriminator_path=args.discriminator_path,
                             latent_dim=args.latent_dim,
                             num_images=args.num_images,
                             save_path=args.save_path)

if __name__ == "__main__":
    main()