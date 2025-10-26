# Ucitavanje potrebnih biblioteka
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Ucitavanje i analiza skupa podataka
df = pd.read_csv('messages.csv')

# Prikaz prvih nekoliko redova
df.head()

# Provjera nedostajucih vrijednosti
df.isnull().sum()

# Prikaz jedinstvenih vrijednosti u koloni category
df['category'].unique()

# 2. Ciscenje i priprema podataka
df = df.dropna(subset=['message', 'category']) 
df['category'] = df['category'].str.lower()  
df['category'] = df['category'].map({'spam': 1, 'ham': 0})  

# 3. Podjela podataka
X = df['message'] 
y = df['category'] 

# Podjela na trening i test skup
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 4. Vektorizacija teksta
vectorizer = TfidfVectorizer(stop_words='english') 
X_train_tfidf = vectorizer.fit_transform(X_train)  
X_test_tfidf = vectorizer.transform(X_test)  

# 5. Treniranje modela
models = {
    'Logistic Regression': LogisticRegression(),
    'Naive Bayes': MultinomialNB(),
    'Decision Tree': DecisionTreeClassifier(),
    'Linear SVC': SVC(kernel='linear')
}

results = {}

# Treniranje i evaluacija modela
for model_name, model in models.items():
    model.fit(X_train_tfidf, y_train)
    y_pred = model.predict(X_test_tfidf)
    
    # Evaluacija modela
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    
    results[model_name] = {
        'accuracy': accuracy,
        'classification_report': report,
        'confusion_matrix': cm
    }

# 6. Prikaz rezultata
for model_name, result in results.items():
    print(f"Model: {model_name}")
    print(f"Accuracy: {result['accuracy']}")
    print(f"Classification Report:\n{result['classification_report']}")
    
    # Vizualizacija matrice zabune
    cm = result['confusion_matrix']
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'])
    plt.title(f"Confusion Matrix for {model_name}")
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

# 7. Zakljucak
# Na osnovu rezultata, odlucite koji model je najbolji. Fokusirati se na F1 skor, preciznost i odziv.
