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

    # Going through all of the trained classifiers and printing their statistics and the confusion matrix
    for ratio, classifier in custom_data_ratios_to_classifiers.items():
        print(f'Classifier statistic for {ratio} ratio of custom data and MNIST data.')
        true_labels, pred_labels = classifier.evaluate_model(test_loader)
        classifier.plot_confusion_matrix(true_labels, pred_labels)

if __name__ == "__main__":
    main()