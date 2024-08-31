import torch
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

    # Dobijanje DataLoader-a za svaki broj
    digit_loaders = get_digit_data_loaders()

    # Treniranje GAN-a za svaki broj i generisanje datasetova
    for digit, loader in digit_loaders.items():
        print(f"Training GAN for digit {digit}")
        gan = GAN()
        gan.to(gan.device)
        gan.train(loader, num_epochs=100)

        # Generisanje 1000 slika za trenutni broj
        images_digit, labels_digit = gan.generate_dataset(n=1000, label=digit)
        
        images.extend(images_digit)
        labels.extend(labels_digit)

        torch.save(gan, f'saved_models/gan_{digit}.pth')

    return images, labels

if __name__ == "__main__":
    main()