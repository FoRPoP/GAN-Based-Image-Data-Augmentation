from classifier import *
from gan import *

def main():

    gan = prepare_gan()

    # Generisanje i prikazivanje slika nakon treniranja
    #gan.sample_images(epoch=50)

    # Generisanje dataset-a
    images, labels = [], []
    for i in range(10):
        images_i, labels_i = gan.generate_dataset(n=1000, label=i)  # Generišemo 500 slika sa labelom i
        images.extend(images_i)
        labels.extend(labels_i)

    mnist_classifier = MNISTClassifier(lr=0.001, input_dim=784, output_dim=10, hidden_dim=300, dropout_rate=0.5)
    #train_loader, validation_loader, test_loader = mnist_classifier.load_and_preprocess_data(validation_split=0.2)
    train_loader, validation_loader, test_loader = mnist_classifier.load_and_preprocess_data(data=images, labels=labels, validation_split=0.2)
    mnist_classifier.train_model(train_loader=train_loader, validation_loader=validation_loader, num_epochs=50)
    true_labels, pred_labels = mnist_classifier.evaluate_model(test_loader)

    mnist_classifier.plot_confusion_matrix(true_labels, pred_labels)

def prepare_gan() -> GAN:

    # Učitavanje MNIST podataka
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    mnist_data = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    data_loader = DataLoader(mnist_data, batch_size=64, shuffle=True)

    # Postavljanje uređaja (CPU ili GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Treniranje GAN modela
    gan = GAN().to(device)
    gan.train(data_loader, num_epochs=50)

    return gan


if __name__ == "__main__":
    main()