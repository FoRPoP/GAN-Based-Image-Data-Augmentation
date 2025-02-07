import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
from torchvision.datasets import MNIST


def analyze_class_distribution() -> None:
    #ANALIZA DISTRIBUCIJE KLASA 
    #Ovaj kod će prikazati histogram koji pokazuje broj primera svake cifre u MNIST skupu podataka. To će nam pomoći da identifikujemo potencijalne neuravnoteženosti.
    # Učitavanje MNIST skupa podataka
    mnist_train = MNIST(root='data', train=True, download=True)
    mnist_test = MNIST(root='data', train=False, download=True)

    # Spajanje svih labela iz trening i test skupa
    all_labels = [label for (image, label) in mnist_train] + [label for (image, label) in mnist_test]

    # Brojanje pojavljivanja svake klase
    class_counts = Counter(all_labels)

    # Prikazivanje distribucije klasa
    plt.bar(class_counts.keys(), class_counts.values())
    plt.xlabel('Cifra')
    plt.ylabel('Broj primera')
    plt.title('Distribucija klasa u MNIST skupu podataka')
    plt.show()


def calc_avg_pixel_brightness():
# Izračunavanje prosečne svetlosti (intenziteta) piksela za svaku sliku
    mnist_train = MNIST(root='data', train=True, download=True)
    avg_pixel_intensity = [np.mean(image) for (image, label) in mnist_train]

    # Prikazivanje histogram prosečne svetlosti piksela
    plt.hist(avg_pixel_intensity, bins=30)
    plt.xlabel('Prosečna svetlost piksela')
    plt.ylabel('Broj slika')
    plt.title('Distribucija prosečne svetlosti piksela u MNIST skupu podataka')
    plt.show()
