#Zadatak za samoprocjenu
from sklearn import datasets
iris = datasets.load_iris()
print("Dataset name:", iris['target_names']) 
print("Feature names:", iris['feature_names']) 
print("Number of instances:", len(iris['data']))



#Obucavanje modela na oznacenim recenzijama
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
 
# Load the labeled dataset
df = pd.read_csv("data/reviews_labeled_cleaned.csv")
 
# Separate features and labels
X = df['review']
y = df['sentiment']

# Vectorize all input reviews 
vectorizer = TfidfVectorizer()
X_tfidf = vectorizer.fit_transform(X)

# Train the logistic regression model on the full dataset
model = LogisticRegression()
model.fit(X_tfidf, y)
print("\nModel is ready.")


#Na postojeći kod u kome se obavlja treniranje modela, dodaćemo jednostavnu petlju koja omogućava da direktno unesemo bilo koji tekst recenzije i odmah vidimo kako model procenjuje njen sentiment:
# Loop for user input
print("\nModel is ready. Enter a review to classify its sentiment.")
print("Type 'exit' to quit.\n")

while True: 
    user_input = input("Enter a review: ") 
    if user_input.lower() == 'exit': 
        print("Exiting sentiment classifier.") 
        break

    # Transform user input to match TF–IDF vector format
    user_tfidf = vectorizer.transform([user_input])
    prediction = model.predict(user_tfidf)[0]
    print(f"Predicted sentiment: {prediction}\n")


#Hajde da vidimo kako možemo nekoliko recenzija da pretvorimo u ovaj bag-of-words oblik:
from sklearn.feature_extraction.text import CountVectorizer

# Mini dataset of sample reviews
corpus = [
    "I love this product",
    "This product is not good",
    "Absolutely fantastic experience",
    "Terrible, I hate it",
    "Not great, not terrible" 
]

# Create and fit the vectorizer
bow_vectorizer = CountVectorizer()
X_bow = bow_vectorizer.fit_transform(corpus)
 
# Show feature names (vocabulary)
print("Vocabulary:", bow_vectorizer.get_feature_names_out()) 
# Convert sparse matrix to array for readability
print("\nBoW Matrix (Document-Term Matrix):\n", X_bow.toarray())


#Kod za TF–IDF vektorizaciju
from sklearn.feature_extraction.text import TfidfVectorizer
# Mini dataset of sample reviews
corpus = [
    "I love this product",
    "This product is not good",
    "Absolutely fantastic experience",
    "Terrible, I hate it",
    "Not great, not terrible"
]
# Create and fit the TF–IDF vectorizer
tfidf_vectorizer = TfidfVectorizer()
X_tfidf = tfidf_vectorizer.fit_transform(corpus)
 
# Show feature names
print("Vocabulary:", tfidf_vectorizer.get_feature_names_out())

# Convert to array and print
print("\nTF–IDF Matrix:\n", X_tfidf.toarray())


#Umesto da treniramo samo jedan model, koristićemo tri različita algoritma – na potpuno istim podacima. Tako odmah možemo da uporedimo njihove rezultate i naučimo kako svaki gleda na problem.
#Kod koji koristimo:
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier

# Load labeled dataset
df = pd.read_csv("data/reviews_labeled_cleaned.csv")
X = df['review'] 
y = df['sentiment']

# Convert text to TF-IDF
vectorizer = TfidfVectorizer() 
X_tfidf = vectorizer.fit_transform(X)

# Train different algorithms on the same data
models = { 
    "Logistic Regression": LogisticRegression(max_iter=1000), 
    "Naive Bayes": MultinomialNB(), 
    "Decision Tree": DecisionTreeClassifier()
}
trained_models = {}
for name, model in models.items():
    model.fit(X_tfidf, y)
    trained_models[name] = model
print("\nModels are trained.")


#Za početak, proširimo svoj primer kodom koji će obavljati testiranje kreiranih modela.
#Kod za testiranje:Za početak, proširimo svoj primer kodom koji će obavljati testiranje kreiranih modela.
#Kod za testiranje:
# Interactive testing
print("\nModels are trained. Type a review to classify it.")
print("Type 'exit' to stop.\n")

while True:
    user_input = input("Enter a review: ")
    if user_input.lower() == 'exit':
        print("Exiting.") 
        break
    user_tfidf = vectorizer.transform([user_input])

    for name, model in trained_models.items():
        prediction = model.predict(user_tfidf)[0]
        print(f"{name} → {prediction}")
    print("-" * 40)


#Ovde je to automatizovano i dodatno pametno – Python i scikit-learn vode računa o tome da raspodela klasa (npr. pozitivan/negativan sentiment) ostane uravnotežena i u trening i u test setu
import pandas as pd
from sklearn.model_selection import train_test_split
# Load labeled dataset
df = pd.read_csv("data/reviews_labeled_cleaned.csv")
 
# Separate features (X) and labels (y)
X = df["review"]
y = df["sentiment"]

# Split the data: 80% for training, 20% for testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y 
)
 
# Display sizes of the resulting sets
print("Training set size:", len(X_train))
print("Test set size:", len(X_test))
 
# Display class distribution in the training and test sets
print("Training set class distribution:")
print(y_train.value_counts())
print("Test set class distribution:")
print(y_test.value_counts())


#Model ćemo trenirati nad podacima namenjenim za treniranje, a zatim ćemo dobijene predikcije uporediti sa onim iz test skupa podataka.
#Kako u Pythonu koristimo trening i test setove podataka?
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import pandas as pd
 
# Load dataset
df = pd.read_csv("data/reviews_labeled_cleaned.csv")
X = df["review"]
y = df["sentiment"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y 
)

# TF-IDF vectorizatio 
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train logistic regression
model = LogisticRegression(max_iter=1000)
model.fit(X_train_tfidf, y_train)
 
# Predict on test set
y_pred = model.predict(X_test_tfidf)

# Evaluate
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

#Accuracy jednostavno meri odnos tačnih predikcija prema ukupnom broju predikcija:

#tačnost = (broj tačnih predikcija) / (ukupan broj predikcija)
#Naš primer:
Accuracy = 2183/2282
Accuracy = 0.9566

#preciznost = (tačno pozitivne predikcije) / (sve predikcije kao pozitivne)
#odziv = (tačno prepoznate pozitivne) / (sve stvarno pozitivne)

#Šta je F1 mera i zašto je važna?
#F1 je zbirna metrika koja kombinuje preciznost i odziv u jednu vrednost.
#Ne dopušta da nam model ispadne odličan samo zbog jednog kriterijuma.
#Formula: F1 = 2 × (preciznost × odziv) / (preciznost + odziv)

#vizualno prikazivanje matrice zabune:
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
cm = confusion_matrix(y_test, y_pred, labels=["negative", "positive"])
disp  = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
disp.plot()
plt.show()

#Za početak, potrebno je da učitamo skup podataka, podelimo ga na trening i test skup i 
#zatim izvršimo vektorizaciju tekstualnih podataka kako bi ih algoritmi mogli koristiti. 
#Zapravo, da iskombinujemo sve ono što smo naučili u prethodnim lekcijama.

from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import pandas as pd
# Load dataset
df = pd.read_csv("data/reviews_labeled_cleaned.csv")
X = df["review"]
y = df["sentiment"]
# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
# TF-IDF vectorization
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

#Ono što smo do sada radili sa jednim, sada ćemo primeniti nad više modela – 
#paralelno ćemo trenirati i evaluirati modele, ali na potpuno istom skupu podataka. 
#Na taj način, možemo fer i precizno da uporedimo njihove performanse.

# Initialize models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Naive Bayes": MultinomialNB(),
    "Decision Tree": DecisionTreeClassifier(),
    "Support Vector Machine": LinearSVC()
}
# Train, predict, and evaluate
for name, model in models.items():
    model.fit(X_train_vec, y_train)
    y_pred = model.predict(X_test_vec)
    print(f"\n{name} - Classification Report:")
    print(classification_report(y_test, y_pred))

