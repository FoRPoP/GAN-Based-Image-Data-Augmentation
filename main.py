from classifier import *
from gan import *
from synthetic_data_generation import *

def main():

    images, labels = generate_synthetic_data()

    mnist_classifier = MNISTClassifier(lr=0.001, input_dim=784, output_dim=10, hidden_dim=300, dropout_rate=0.1)
    train_loader, validation_loader, test_loader = mnist_classifier.load_and_preprocess_data(validation_split=0.2)
    train_loader, validation_loader, test_loader = mnist_classifier.load_and_preprocess_data(train_data=images, train_labels=labels, validation_split=0.2)
    mnist_classifier.train_model(train_loader=train_loader, validation_loader=validation_loader, num_epochs=100)

    torch.save(mnist_classifier, 'saved_models/classifier.pth')

    true_labels, pred_labels = mnist_classifier.evaluate_model(test_loader)
    mnist_classifier.plot_confusion_matrix(true_labels, pred_labels)

if __name__ == "__main__":
    main()