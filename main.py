from multiprocessing import Process
import torch
import os
from classifier import *
from gan import *

def main():

    images, labels = generate_synthetic_data()

    mnist_classifier = MNISTClassifier(lr=0.001, input_dim=784, output_dim=10, hidden_dim=300, dropout_rate=0.5)
    train_loader, validation_loader, test_loader = mnist_classifier.load_and_preprocess_data(validation_split=0.2)
    train_loader, validation_loader, test_loader = mnist_classifier.load_and_preprocess_data(train_data=images, train_labels=labels, validation_split=0.2)
    mnist_classifier.train_model(train_loader=train_loader, validation_loader=validation_loader, num_epochs=100)

    torch.save(mnist_classifier, 'saved_models/classifier.pth')

    true_labels, pred_labels = mnist_classifier.evaluate_model(test_loader)
    mnist_classifier.plot_confusion_matrix(true_labels, pred_labels)

def generate_synthetic_data() -> Tuple[list, list]:

    images, labels = [], []
    models_output_dir = 'saved_models'
    os.makedirs(models_output_dir, exist_ok=True)

    # Dobijanje DataLoader-a za svaki broj
    digit_loaders = get_digit_data_loaders()

    processes = []
    for digit, loader in digit_loaders.items():
        p = Process(target=train_gan_for_digit, args=(digit, loader, models_output_dir))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    for digit in digit_loaders.keys():
        model_path = os.path.join(models_output_dir, f'gan_{digit}.pth')

        gan: GAN = torch.load(model_path)
        gan.to(gan.device)

        images_digit, labels_digit = gan.generate_dataset(n=1000, label=digit)

        images.extend(images_digit)
        labels.extend(labels_digit)

    return images, labels

def train_gan_for_digit(digit: int, loader: DataLoader, output_dir: str) -> None:

    print(f"Training GAN for digit {digit}")
    gan = GAN()
    gan.to(gan.device)
    gan.train(loader, num_epochs=100)

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

if __name__ == "__main__":
    main()