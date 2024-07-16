import gradio as gr
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the saved model and vectorizer
model = joblib.load('tunisian_arabiz_sentiment_analysis_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

def predict_sentiment(text):
    if not text.strip():
        return (
            "No input provided",
            "N/A",
            "Please enter some text to get a sentiment prediction."
        )
    
    text_vectorized = vectorizer.transform([text])
    prediction = model.predict(text_vectorized)[0]
    probabilities = model.predict_proba(text_vectorized)[0]
    confidence = max(probabilities)
    
    sentiment = "Positive" if prediction == 1 else "Negative"
    
    return (
        sentiment,
        f"{confidence:.2f}",
        f"The model predicts this text is {sentiment.lower()} with {confidence:.2%} confidence."
    )

# Function to get predictions for examples
def get_example_predictions(examples):
    return [predict_sentiment(ex[0]) for ex in examples]

# Example texts
examples = [
    ["3jebni barcha el film hedha"],
    ["ma7abitch el mekla mte3 el restaurant"],
    ["el jaw fi tounes a7la 7aja"],
    ["ennes el kol te3ba w ma3andhomch flous"],
    ["كان جات الدنيا دنيا راني ساهرة في دار حماتي"],
    ["مبابي مانستعرف بيه مدريدي كان مانشوفو مركى هاتريك بمريول الريال"]
]

# Get predictions for examples
example_predictions = get_example_predictions(examples)

# Create formatted examples with predictions
formatted_examples = [
    [ex[0], f"{pred[0]} (Confidence: {pred[1]})"] 
    for ex, pred in zip(examples, example_predictions)
]

# Create Gradio interface
iface = gr.Interface(
    fn=predict_sentiment,
    inputs=gr.Textbox(lines=3, placeholder="أدخل النص هنا... / Enter your text here..."),
    outputs=[
        gr.Label(label="Predicted Sentiment"),
        gr.Label(label="Confidence Score"),
        gr.Textbox(label="Explanation")
    ],
    examples=formatted_examples,
    title="Tunisian Arabiz Sentiment Analysis",
    description="""
    <p>This model predicts the sentiment of Tunisian text as either Positive or Negative. It works with both Tunisian Arabiz and standard Arabic script.</p>
        
    <h4>What is Tunisian Arabiz? / ما هي العربيزية التونسية؟</h4>
    <p>Tunisian Arabiz is a way of writing the Tunisian dialect using Latin characters and numbers. For example:</p>
    <ul>
        <li>"3ajbetni" means "I liked it""</li>
        <li>"7aja" means "thing" "</li>
        <li>"a3tini 9ahwa" means "give me a coffee""</li>
    </ul>
    
    <p>Try the examples below or enter your own text!</p>
    <p>!جرب الأمثلة أو أدخل نصك الخاص</p>
    """,
    article="""
    <h3>About the Model</h3>
    <p>This sentiment analysis model was trained on a combined dataset from TuniziDataset and the Tunisian Dialect Corpus. 
    It uses TF-IDF vectorization for feature extraction and Logistic Regression for classification.</p>
    
    <p>The model accepts Tunisian Arabiz written with Latin and Arabic script.</p>
    
    <h3>Limitations</h3>
    <p>Due to dataset limitations, neutral sentiment data was removed to achieve maximum performance. </p>
    <p>The model may not perform well on very colloquial expressions or new slang terms not present in the training data. 
    Sentiment can be nuanced and context-dependent, which may not always be captured accurately by this model.</p>
    <center>
    <h2>This model is open-source, and contributions of additional datasets are welcome to improve its capabilities.</h2>
    
    <h2>هذا النموذج مفتوح المصدر، ونرحب بمساهمات مجموعات البيانات الإضافية لتحسين قدراته.</h2>
    </center>
    """
)

# Launch the interface
iface.launch()
