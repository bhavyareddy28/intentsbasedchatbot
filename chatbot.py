import os
import nltk
import ssl
import random
import json
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

# Set up SSL context and NLTK data path
ssl._create_default_https_context = ssl._create_unverified_context
nltk.data.path.append(os.path.abspath("nltk_data"))
nltk.download('punkt')

# Load intents from JSON file
with open('intents.json', 'r') as file:
    intents = json.load(file)

# Prepare data for training
tags = []
patterns = []
for intent in intents:
    for pattern in intent['patterns']:
        tags.append(intent['tag'])
        patterns.append(pattern)

# Vectorize the patterns
vectorizer = TfidfVectorizer()
x = vectorizer.fit_transform(patterns)
y = tags

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Train the classifier
clf = RandomForestClassifier(n_estimators=200, random_state=42)
clf.fit(X_train, y_train)

# Define the chatbot function
def chat_bot(input_text):
    input_text = vectorizer.transform([input_text])
    tag = clf.predict(input_text)[0]
    for intent in intents:
        if intent['tag'] == tag:
            response = random.choice(intent['responses'])
            return response

# Initialize session state for conversation history
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []

# Streamlit UI
st.title("Intent Based Chatbot")

# Sidebar for conversation history
with st.sidebar:
    st.header("Conversation History")
    if st.session_state.conversation_history:
        for entry in st.session_state.conversation_history:
            st.text(f"You: {entry['user']}")
            st.text(f"Chatbot: {entry['bot']}")
            st.write("---")
    else:
        st.info("No conversation history yet.")

    # Button to clear conversation history
    if st.button("Clear History"):
        st.session_state.conversation_history = []
        st.success("Conversation history cleared!")

# Main chat interface
user_input = st.text_input("You: ", "")

if st.button("Send"):
    if user_input:
        # Get the chatbot's response
        response = chat_bot(user_input)

        # Update conversation history
        st.session_state.conversation_history.append({
            "user": user_input,
            "bot": response
        })

        # Display the chatbot's response
        st.text_area("Chatbot:", value=response, height=100, max_chars=None, key=None)
    else:
        st.warning("Please enter some text to chat with the bot.")