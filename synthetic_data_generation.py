from multiprocessing import Process
import torch
import os
from classifier import *
from gan import *

def generate_synthetic_data() -> Tuple[list, list]:

    # creating a dir for models to be saved
    images, labels = [], []
    models_output_dir = 'saved_models'
    os.makedirs(models_output_dir, exist_ok=True)

    # Getting DataLoaders for every digit
    digit_loaders = get_digit_data_loaders()

    # Utilizing multiprocessing to speed up GAN training
    processes = []
    for digit, loader in digit_loaders.items():
        p = Process(target=train_gan_for_digit, args=(digit, loader, models_output_dir))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    # Loading all of the trained GANs
    for digit in digit_loaders.keys():
        model_path = os.path.join(models_output_dir, f'gan_{digit}.pth')

        gan: GAN = torch.load(model_path)
        gan.to(gan.device)

        # Generating a part of the dataset corresponding to the digit that the GAN was trained on
        images_digit, labels_digit = gan.generate_dataset(n=1000, label=digit)

        # Generating GAN image samples
        display_gan_samples(images_digit, digit)

        images.extend(images_digit)
        labels.extend(labels_digit)

    return images, labels

def display_gan_samples(images_digit: list, digit: int, num_samples: int = 5) -> None:
    """Prikaz uzoraka slika generisanih pomoću GAN-a."""
    fig, axs = plt.subplots(1, num_samples, figsize=(num_samples * 2, 2))
    for i in range(num_samples):
        axs[i].imshow(images_digit[i].squeeze(), cmap='gray')
        axs[i].set_title(f"Digit {digit}")
        axs[i].axis('off')
    plt.show()

def train_gan_for_digit(digit: int, loader: DataLoader, output_dir: str) -> None:

    # Training GAN only if the corresponding model doesn't already exist in the saved_models folder
    print(f"Training GAN for digit {digit}")
    if os.path.exists(os.path.join(output_dir, f'gan_{digit}.pth')):
        return
    
    gan = GAN()
    gan.to(gan.device)
    gan.train(loader, num_epochs=2000)

    model_path = os.path.join(output_dir, f'gan_{digit}.pth')
    torch.save(gan, model_path)
    print(f"Model for digit {digit} saved to {model_path}")

# Funkcija za dobijanje DataLoadera za svaki broj
def get_digit_data_loaders(batch_size=64):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    mnist_data = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    data_loaders = {}

    for digit in range(10):
        digit_data = [data for data in mnist_data if data[1] == digit]
        digit_loader = DataLoader(TensorDataset(torch.stack([x[0] for x in digit_data]), 
                                                torch.tensor([x[1] for x in digit_data])),
                                  batch_size=batch_size, shuffle=True)
        data_loaders[digit] = digit_loader

    return data_loaders