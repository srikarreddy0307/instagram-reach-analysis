import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
from sklearn.metrics.pairwise import cosine_similarity

# Load the data from CSV file
data = pd.read_csv("BankFAQs.csv")

# Define the vectorizer
vectorizer = CountVectorizer()

# Transform the text data into feature vectors
X = vectorizer.fit_transform(data['Question'])

# Define the labels
y = data['Class']

# Train the SVM model
svm_model = SVC(C=10, kernel='rbf', gamma=0.1, decision_function_shape='ovr')
svm_model.fit(X, y)

# Define the Streamlit app
st.set_page_config(page_title="Banking-Chatbot", page_icon=":robot:")

# Add a title
st.title("How can i help you?")

# Create a session state to persist data across user interactions
session_state = st.session_state
if not hasattr(session_state, 'previous_qa'):
    session_state.previous_qa = []

# Button to clear the history
clear_history = st.checkbox("Clear Chat History", key="clear_history_checkbox")

# Display chat history
st.header("Chat History")

# Display previous conversations in a more visually appealing way
for qa in session_state.previous_qa:
    st.markdown(
        f"<div style='background-color:#f2f2f2; padding:10px; border-radius:10px; margin-bottom:10px;'>"
        f"<p style='font-weight:bold; color:#1f405d;'>User: {qa['User']}</p>"
        f"<p style='color:#4d4d4d;'>Bot: {qa['Bot']}</p>"
        f"</div>",
        unsafe_allow_html=True
    )

# Get user input at the bottom of the page
user_input = st.text_input("You:", value="", key="user_input")

# Make predictions
if user_input:
    if clear_history:
        # Clear the chat history if the user wants to clear it
        session_state.previous_qa = []

    # Transform the input question into a feature vector
    input_vector = vectorizer.transform([user_input])

    # Predict the class of the input question
    predicted_class = svm_model.predict(input_vector)[0]

    # Find the answer of the predicted class that is most similar to the input question
    class_data = data[data['Class'] == predicted_class]
    class_vectors = vectorizer.transform(class_data['Question'])
    similarities = cosine_similarity(input_vector, class_vectors)
    most_similar_index = similarities.argmax()
    predicted_answer = class_data.iloc[most_similar_index]['Answer']

    # Display the user question and bot answer in the chat history
    session_state.previous_qa.append({"User": user_input, "Bot": predicted_answer})

    # Display the predicted answer from the bot in a more visually appealing way
    st.markdown(
        f"<div style='background-color:#f2f2f2; padding:10px; border-radius:10px; margin-bottom:10px;'>"
        f"<p style='font-weight:bold; color:#1f405d;'>Bot: {predicted_answer}</p>"
        f"</div>",
        unsafe_allow_html=True
    )
