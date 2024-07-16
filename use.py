import joblib

# Load the model and vectorizer
model = joblib.load('tunisian_arabiz_sentiment_analysis_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Example usage: Predicting on new data
new_data = ["ennes el kol te3ba w ma3andhomch flous"]  # Example new data
transformed_data = vectorizer.transform(new_data)
predictions = model.predict(transformed_data)
probabilities = model.predict_proba(transformed_data)

# Output predictions
print("Predictions:", predictions)
print("Probabilities:", probabilities)

# Calculate confidence
confidence = max(probabilities[0])  # Assuming binary classification, taking the highest probability
print("Confidence:", confidence)

# Alternatively, calculate accuracy (if you have ground truth labels)
# ground_truth_labels = [0]  # Example ground truth labels
# accuracy = accuracy_score(ground_truth_labels, predictions)
# print("Accuracy:", accuracy)
