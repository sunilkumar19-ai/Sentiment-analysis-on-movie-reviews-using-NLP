import streamlit as st
import pickle
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
from nltk.stem import SnowballStemmer


def preprocess_text(text):
    # Remove HTML tags
    clean_text = re.sub(r'<.*?>', '', text)
    # Remove URLs
    clean_text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '',
                        clean_text)
    # Tokenize into sentences
    sentences = sent_tokenize(clean_text)

    # Process each sentence
    processed_sentences = []
    for sentence in sentences:
        # Remove punctuation
        sentence = sentence.translate(str.maketrans('', '', string.punctuation))
        # Convert to lowercase
        sentence = sentence.lower()
        # Remove stopwords
        stop_words = set(stopwords.words('english'))
        words = sentence.split()
        filtered_words = [word for word in words if word.lower() not in stop_words]

        # Apply Snowball Stemmer to each word
        stemmer = SnowballStemmer('english')
        stemmed_words = [stemmer.stem(word) for word in filtered_words]

        processed_sentence = ' '.join(stemmed_words)
        processed_sentences.append(processed_sentence)

    # Join the processed sentences back into a single string
    clean_text = ' '.join(processed_sentences)

    # Correct text using TextBlob
    # Uncomment the following lines if you want to use TextBlob for text correction
    # txtblb = TextBlob(clean_text)
    # corrected_text = txtblb.correct()

    return clean_text


cv = pickle.load(open('vectorizer.pkl','rb'))
rc = pickle.load(open('model.pkl','rb'))

st.title("MOVIE REVIEW Classifier")

input_review = st.text_area("Enter the Review")


if st.button('Analysis'):

    # 1. preprocess
    transformed_review = preprocess_text(input_review)
    # 2. vectorize
    vector_input = cv.transform([transformed_review])
    # 3. predict
    result = rc.predict(vector_input)[0]
    # 4. Display
    if result == 1:
        st.header("POSITIVE")
    else:
        st.header("NEGATIVE")