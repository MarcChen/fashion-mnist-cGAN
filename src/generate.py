import torch
from matplotlib import pyplot as plt
from tqdm import tqdm
import argparse
from model import Generator, Discriminator


def load_model(model, path):
    model.load_state_dict(torch.load(path), wheights_only=True)
    return model

def save_image(image, save_path, name):
    if image.shape[0] == 1:  # If the image is grayscale
        image.squeeze(0)  # Remove the channel dimension
    plt.imsave(save_path, f"{save_path}/{name}.png", cmap='gray')
    return

def generate_and_save_images(generator_path="../checkpoints/generator.pth",
                             discriminator_path="../checkpoints/discriminator.pth",
                             latent_dim=100,
                             num_images=10,
                             save_path='../samples'):
    print("Generating images...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the trained models
    G, D = Generator().to(device), Discriminator().to(device)
    try:
        G = load_model(G, generator_path)
        D = load_model(D, discriminator_path)
    except:
        print("Error loading the models. Please check the paths.")
        return

    nb_approved_images = 0
    generated_images = []
    with tqdm(total=num_images, desc="Generating images") as pbar:
        while nb_approved_images < num_images:
            # Generate an image
            z_noise = torch.randn(1, latent_dim, device=device)
            label = torch.randint(0, 10, (1,), device=device)
            generated_image = G(z_noise, label)

            # Check image validity with the discriminator
            approval = D(generated_image, label)
            if approval > 0.5:
                generated_images.append(generated_image)
                nb_approved_images += 1
                pbar.update(1)

    # Convert images to numpy and save
    generated_images = generated_images.cpu().detach().numpy()
    generated_images = (generated_images + 1) / 2  # Rescale to [0, 1]
    print("Images generated !")

    print("Saving images...")
    for i, img in enumerate(generated_images):
       save_image(img, save_path, i)
    print("Images saved !")

    return

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--generator-path", default="../checkpoints/generator.pth", help="Path to the generator model.")
    parser.add_argument("--discriminator-path", default="../checkpoints/discriminator.pth", help="Path to the discriminator model.")
    parser.add_argument("--latent-dim", default=100, help="Noise dimension for the generator.")
    parser.add_argument("--num-images", default=10, help="Number of images to generate.")
    parser.add_argument("--save-path", default='../samples', help="Path to save the generated images.")
    args = parser.parse_args()

    generate_and_save_images(generator_path=args.generator_path,
                            discriminator_path=args.discriminator_path,
                            latent_dim=args.latent_dim,
                            num_images=args.num_images,
                            save_path=args.save_path)
    return

if __name__ == "__main__":
    main()