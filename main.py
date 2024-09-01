from classifier import *
from gan import *
from synthetic_data_generation import *
from data_analysis import *

def main():

    analyze_class_distribution()
    calc_avg_pixel_brightness()

    # Generating synthetic data
    images, labels = generate_synthetic_data()

    # Going through the following ratios where the ratio = len(custom data) / len(mnist data)
    # Ratio of -1 means that only the custom data was used
    custom_data_ratios_to_classifiers = {-1: None, 0: None, 0.25: None, 0.5: None, 1: None, 2: None, 4: None}
    for ratio in custom_data_ratios_to_classifiers.keys():
        print(f'Training classifier for {ratio} ratio of custom data and MNIST data.')
        # Training the classifier only if the model doesn't already exist in the saved_models folder
        if not os.path.exists(f'saved_models/classifier_{ratio}.pth'): 
            max_acc = 0
            # Training 5 models and picking the best one by the final accuracy result
            for _ in range(5):
                classifier = MNISTClassifier(lr=0.001, input_dim=784, output_dim=10, hidden_dim=300, dropout_rate=0.1)
                #train_loader, validation_loader, test_loader = classifier.load_and_preprocess_data(validation_split=0.2)
                train_loader, validation_loader, test_loader = classifier.load_and_preprocess_data(train_data=images, train_labels=labels, validation_split=0.2, custom_data_ratio=ratio)
                classifier.train_model(train_loader=train_loader, validation_loader=validation_loader, num_epochs=200)

                _, _ = classifier.evaluate_model(test_loader)
                if classifier.acc > max_acc:
                    custom_data_ratios_to_classifiers[ratio] = classifier
                    max_acc = classifier.acc
            # Saving the best model
            torch.save(custom_data_ratios_to_classifiers[ratio], f'saved_models/classifier_{ratio}.pth')
        else:
            # Loading the model into the dictionary if it already exists
            custom_data_ratios_to_classifiers[ratio] = torch.load(f'saved_models/classifier_{ratio}.pth')
        print(f'Best classifier accuracy: {custom_data_ratios_to_classifiers[ratio].acc}.')

    # Getting a test loader for a test dataset
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0,), (1,))])
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=1000)

    # Going through all of the trained classifiers and printing their statistics and the confusion matrix
    for ratio, classifier in custom_data_ratios_to_classifiers.items():
        print(f'Classifier statistic for {ratio} ratio of custom data and MNIST data.')
        true_labels, pred_labels = classifier.evaluate_model(test_loader)
        classifier.plot_confusion_matrix(true_labels, pred_labels)

    # Plotting the comparison graph
    ratios = [str(ratio) for ratio in custom_data_ratios_to_classifiers.keys()]
    accs = [model.acc for model in custom_data_ratios_to_classifiers.values()]

    plt.figure(figsize=(10, 6))
    plt.bar(ratios, accs, color='skyblue')
    plt.title('Model Accuracy Comparison')
    plt.xlabel('Ratio of custom data and MNIST data')
    plt.ylabel('Accuracy')

    plt.yticks(np.arange(0, max(accs) + 0.1, 0.1))

    plt.show()

if __name__ == "__main__":
    main()