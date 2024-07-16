import pandas as pd
import numpy as np
import pandas as pd
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

df = pd.read_csv("TuniziDataset.csv")
print(df.head())

# **Cleaning** data
clean_dataset = df.drop_duplicates()

# Checking the numbers of rows and columns
print(clean_dataset.shape)
# Counting the number of missing Values in the dataset
print(clean_dataset.isnull().sum())
# Checking the distribution of target column
print(clean_dataset["SentimentLabel"].value_counts())

# Separating the data and label
X = clean_dataset['InputText'].values
Y = clean_dataset['SentimentLabel'].values
print(X)
print(Y)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)
print(X.shape, X_train.shape, X_test.shape)
# Converting the textual data into numerical data
vectorizer = TfidfVectorizer()

X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)
print(X_train)
print(X_test)

# Logistic Regression
model = LogisticRegression(max_iter = 1000)
model.fit(X_train, Y_train)

# Accuracy score on the training data
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(Y_train, X_train_prediction)
print("Accuracy score on the training data :", training_data_accuracy)
# Accuracy score on the test data
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(Y_test, X_test_prediction)

print("Accuracy score on the training data :", test_data_accuracy)
