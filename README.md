# GAN-Based-Image-Data-Augmentation

## Pregled Projekta
Ovaj projekat obuhvata razvoj Generativne Adverzalne Mreže (GAN) za generisanje slika MNIST cifara, nakon čega sledi obuka neuronske mreže klasifikatora za prepoznavanje ovih cifara. GAN se trenira za svaku cifru pojedinačno (0-9) i koristi se za generisanje 1,000 sintetičkih slika po cifri. Zatim se klasifikator trenira na ovim generisanim slikama kako bi se procenila performansa i potencijalna poboljšanja u odnosu na tradicionalnu obuku na MNIST skupu podataka.

## Opis Skupa Podataka
- **Originalni Skup Podataka**: U projektu se koristi MNIST skup podataka, dobro poznat skup podataka sa slikama cifara (0-9) u sivim tonovima rezolucije 28x28 piksela. Svaka slika je označena odgovarajućom cifrom koju predstavlja.
- **Generisani Skup Podataka**: Za svaku cifru (0-9) generisano je 1,000 sintetičkih slika korišćenjem treniranog GAN-a. Ove generisane slike su sačuvane u direktorijumu `generated_data/`.

## Literatura
-  Liu, L., Hu, H. "Exploring GANs for Data Augmentation in MNIST Digit Classification." *CS229 Machine Learning Final Project Report*, Stanford University, 2020. Dostupno na: [https://cs229.stanford.edu/proj2020spr/report/Liu_Hu.pdf](https://cs229.stanford.edu/proj2020spr/report/Liu_Hu.pdf)
- **Materijali sa vežbi**: [Materijali sa vežbi - Matf ML 2024](https://github.com/matf-ml/materijali-sa-vezbi-2024)

## Instalacija i pokretanje
Prvo, klonirajte repozitorijum:

```bash
git clone https://github.com/FoRPoP/GAN-Based-Image-Data-Augmentation.git
cd GAN-Based-Image-Data-Augmentation 
```

Zatim instalirajte potrebne pakete:

```bash
pip install matplotlib numpy torch torchvision jupyter 
```
Projekat možete pokrenuti na sledeći način:
1. Pokretanje glavnog Python fajla:
```bash
python main.py
```
2. Pokretanje Jupyter Notebook-a:
```bash
jupyter notebook
```
Za pokretanje projekta potrebno je pratiti 1.ipynb jupyter notebook fajl.
Projekat koristi unapred istrenirane GAN i klasifikator modele.
Za ponovno treniranje modela potrebno je izbrisati zeljenje modele is saved_models foldera i ponovo pratiti 1.ipynb jupyter notebook fajl. Samo izbrisani modeli ce biti ponovo trenirani.
## Članovi Tima
- Marija Božić
- Ognjen Popović

