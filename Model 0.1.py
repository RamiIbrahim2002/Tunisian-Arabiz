import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE
import joblib

# Load datasets
df1 = pd.read_csv("TuniziDataset.csv")
df2 = pd.read_parquet("hf://datasets/arbml/Tunisian_Dialect_Corpus/data/train-00000-of-00001.parquet")

# Rename the columns of df2 to match df1
df2.columns = ['InputText', 'SentimentLabel']

# Changing the values of df2 (0: positive, 1: negative) => (1: positive, -1: negative)
df2['SentimentLabel'] = df2['SentimentLabel'].replace({0: 1, 1: -1})

# Concatenate the two DataFrames
df = pd.concat([df1, df2], ignore_index=True)

# Drop the first column if it's an unnecessary index or an ID column
df = df.drop(df.columns[0], axis=1)

# Remove duplicates
clean_dataset = df.drop_duplicates()

# Checking the distribution of the target column
print(clean_dataset["SentimentLabel"].value_counts())

# Separating the data and label
X = clean_dataset['InputText'].values
Y = clean_dataset['SentimentLabel'].values

# Splitting the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

# Converting the textual data into numerical data
vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

# Apply SMOTE to balance the dataset
smote = SMOTE(random_state=2)
X_train_resampled, Y_train_resampled = smote.fit_resample(X_train, Y_train)

# Logistic Regression with best parameters
best_C = 100
best_solver = 'liblinear'

model = LogisticRegression(C=best_C, solver=best_solver, max_iter=1000, class_weight='balanced')
model.fit(X_train_resampled, Y_train_resampled)

# Accuracy score on the training data
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(Y_train, X_train_prediction)
print("Accuracy score on the training data:", training_data_accuracy)

# Accuracy score on the test data
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(Y_test, X_test_prediction)
print("Accuracy score on the test data:", test_data_accuracy)

# Classification report on test data
print("Classification Report on Test Data:")
print(classification_report(Y_test, X_test_prediction))

# Save the model to a file
joblib.dump(model, 'tunisian_arabiz_sentiment_analysis_model.pkl')

# Optionally, save other components like vectorizer if needed
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')
