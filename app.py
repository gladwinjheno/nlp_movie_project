import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Upgrade 1: Make the webpage wide and look more like an app
st.set_page_config(page_title="MovieBot", page_icon="🍿", layout="wide")
st.title("🍿 AI Movie Recommender Chatbot")

# Upgrade 2: Cache the data so the math runs lightning fast

@st.cache_data
def load_data():
    # Load the massive new dataset
    df = pd.read_csv("tmdb_5000_movies.csv")
    
    # Fill any missing text with blank spaces so our math doesn't crash
    df['overview'] = df['overview'].fillna('')
    df['genres'] = df['genres'].fillna('')
    
    # Combine the text. 
    # (Fun fact: TF-IDF is smart enough to extract the genre words even if they are wrapped in JSON brackets!)
    df['combined_features'] = df['genres'] + " " + df['overview']
    
    # Initialize TF-IDF, but tell it to ignore common English stop words
    vectorizer = TfidfVectorizer(stop_words='english')
    feature_vectors = vectorizer.fit_transform(df['combined_features'])
    
    # Calculate similarity across all 5000 movies
    similarity = cosine_similarity(feature_vectors)
    
    return df, similarity

# Load our math and data
df, similarity_matrix = load_data()

# Upgrade 3: Create a "memory" for the chatbot so it remembers the conversation
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hi! I am your AI movie recommender. Tell me a movie you love, and I'll find something similar!"}]

# Display all previous chat messages on the screen
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Upgrade 4: The Chat Input Box
if user_input := st.chat_input("Type a movie name here (e.g., The Matrix)"):
    
    # 1. Show what the user typed
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # 2. The Bot "Thinks" and Responds
    with st.chat_message("assistant"):
        # Search our dataset for what the user typed (ignoring capital letters)
        search_query = user_input.lower()
        match = df[df['title'].str.lower().str.contains(search_query)]
        
        if not match.empty:
            # We found a match! Let's do the NLP math.
            movie_index = match.index[0]
            movie_title = match.iloc[0]['title']
            distances = similarity_matrix[movie_index]
            
            # Get top 3 recommendations
            movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:4]
            
            response = f"Oh, **{movie_title}** is a great choice! Based on its plot and genre, I highly recommend:\n\n"
            for i in movies_list:
                rec_title = df.iloc[i[0]]['title']
                response += f"- 🎬 **{rec_title}**\n"
        else:
            # We didn't find the movie
            response = f"I'm sorry, I couldn't find a movie matching '{user_input}' in my current database. Try another one!"
        
        # Show the response and save it to memory
        st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
