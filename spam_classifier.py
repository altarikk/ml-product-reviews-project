# Import potrebnih biblioteka
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import sys

# Ucitavanje podataka
try:
    df = pd.read_csv("messages.csv")
except FileNotFoundError:
    print("Fajl 'messages.csv' nije pronaden. Provjeri da li se nalazi u istom folderu kao i skripta.")
    sys.exit()

# Prikaz prvih par redova 
print("\nPrvih 5 redova skupa podataka:")
print(df.head())

# Provjera nedostajucih vrijednosti
print("\nProvjera da li ima nedostajucih vrijednosti:")
print(df.isnull().sum())

# Čišćenje podataka
df.dropna(subset=['message', 'category'], inplace=True)

# Pretvaranje svih vrijednosti u koloni 'category' u mala slova
df['category'] = df['category'].str.lower()

# Standardizovanje vrijednosti 
df = df[df['category'].isin(['spam', 'ham'])]

# Definisanje X (poruke) i y (kategorije)
X = df['message']
y = df['category']

# Transformacija teksta u numericki oblik – TF-IDF
vectorizer = TfidfVectorizer()
X_vectorized = vectorizer.fit_transform(X)

# Treniranje modela – koristimo Logistic Regression
model = LogisticRegression()
model.fit(X_vectorized, y)

print("\nModel uspjesno treniran!")

# Interaktivni dio – unos poruka iz konzole
print("\nUnesite poruku za klasifikaciju (ili 'exit' za izlaz):")

while True:
    user_input = input("\nUnesi poruku: ")

    if user_input.lower() == 'exit':
        print("Izlazak iz programa. Hvala!")
        break

    # Transformisanje korisnickih unosa u isti vektorski oblik kao trening skup
    input_vectorized = vectorizer.transform([user_input])

    # Predikcija
    prediction = model.predict(input_vectorized)[0]

    # Prikaz rezultata
    if prediction == 'spam':
        print("Ova poruka je SPAM.")
    else:
        print("Ova poruka NIJE spam.")
