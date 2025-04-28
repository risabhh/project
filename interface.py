import streamlit as st
import pickle
import re
import nltk
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords

# Download stopwords from nltk
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Load the saved Logistic Regression model and vectorizer
with open('best_model.pkl', 'rb') as f:
    best_logreg = pickle.load(f)



with open('tfidf_vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

# Text cleaning function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Keep only alphabetic characters
    words = text.split()
    words = [word for word in words if word not in stop_words]  # Remove stop words
    return ' '.join(words)

# Streamlit app UI
st.title("Sentiment Analysis Web App")
st.write("This web app predicts whether a review is Positive or Negative using Logistic Regression.")

# User input for text
user_input = st.text_area("Enter your review text here:", "I love this product! It's amazing!")

# Predict sentiment
if st.button("Predict Sentiment"):
    cleaned_text = clean_text(user_input)
    input_vectorized = vectorizer.transform([cleaned_text])
    prediction = best_logreg.predict(input_vectorized)

    # Show sentiment
    sentiment = "Positive" if prediction == 1 else "Negative"
    st.write(f"Prediction: {sentiment}")

    # Display results as a bar chart
    sentiment_counts = {"Positive": 0, "Negative": 0}
    sentiment_counts[sentiment] += 1
    
    # Plot the bar chart
    fig, ax = plt.subplots()
    ax.bar(sentiment_counts.keys(), sentiment_counts.values(), color=['green', 'red'])
    ax.set_title('Sentiment Distribution')
    ax.set_ylabel('Count')
    st.pyplot(fig)

# Add some example reviews and their predictions
st.subheader("Example Reviews")
example_reviews = [
    "This product is terrible, I regret buying it.",
    "Excellent! I am very satisfied with the quality.",
    "Not bad, it works as expected.",
    "I am not happy with the service. Will not buy again."
]

for review in example_reviews:
    st.write(f"Review: {review}")
    cleaned_example = clean_text(review)
    example_vectorized = vectorizer.transform([cleaned_example])
    example_prediction = best_logreg.predict(example_vectorized)
    example_sentiment = "Positive" if example_prediction == 1 else "Negative"
    st.write(f"Prediction: {example_sentiment}")

models = {
    "Logistic Regression": {
        "accuracy": 0.7973,
        "precision": 0.80,
        "recall": 1.00,
        "f1_score": 0.88
    },
    "Random Forest": {
        "accuracy": 0.8056,
        "precision": 0.80,
        "recall": 1.00,
        "f1_score": 0.89
    },
    "Support Vector Machine": {
        "accuracy": 0.7990,
        "precision": 0.80,
        "recall": 1.00,
        "f1_score": 0.89
    }
}

# Streamlit app UI
st.title("Sentiment Analysis Web App")
st.write("This web app predicts whether a review is Positive or Negative using various machine learning models.")

# Add evaluation metrics bar charts
st.subheader("Model Evaluation Metrics")

def plot_metrics(model_name, metrics):
    fig, ax = plt.subplots(figsize=(8, 5))
    
    st.subheader(f'{model_name} - Performance Metrics')
    metric_names = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    values = [metrics["accuracy"], metrics["precision"], metrics["recall"], metrics["f1_score"]]
    
    ax.bar(metric_names, values, color=['blue', 'green', 'orange', 'red'])
    ax.set_ylim([0, 1])
    ax.set_title(f'{model_name} - Performance Metrics')
    ax.set_ylabel('Score')
    st.pyplot(fig)

# Plot metrics for all models
for model_name, metrics in models.items():
    plot_metrics(model_name, metrics)

st.write("This web app uses a pre-trained Logistic Regression model with TF-IDF vectorization.")
st.write("You can enter your review text in the input box above to get a prediction.")
