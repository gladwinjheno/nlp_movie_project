import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import random

# --- UPGRADE 1: NETFLIX-STYLE PAGE CONFIG & CSS ---
st.set_page_config(page_title="Popcorn AI", page_icon="🍿", layout="wide")

# Injecting Custom CSS for a sleeker, modern look
st.markdown("""
    <style>
    /* Change the main header font, color, and add a drop shadow */
    h1 {
        color: #E50914 !important; /* Netflix Red */
        font-family: 'Arial Black', sans-serif;
        text-align: center;
        text-shadow: 2px 2px 4px #000000;
    }
    /* Style the chat input box to look more rounded and styled */
    [data-testid="stChatInput"] {
        border: 2px solid #E50914 !important;
        border-radius: 15px !important;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🍿 Popcorn AI: Your Movie Bestie")
st.markdown("<h4 style='text-align: center; color: #808080;'>Tell me what you love, and I'll find your next binge!</h4>", unsafe_allow_html=True)
st.write("---")

# --- THE NLP BRAIN (Unchanged) ---
@st.cache_data
def load_data():
    df = pd.read_csv("tmdb_5000_movies.csv")
    df['overview'] = df['overview'].fillna('')
    df['genres'] = df['genres'].fillna('')
    df['combined_features'] = df['genres'] + " " + df['overview']
    vectorizer = TfidfVectorizer(stop_words='english')
    feature_vectors = vectorizer.fit_transform(df['combined_features'])
    similarity = cosine_similarity(feature_vectors)
    return df, similarity

df, similarity_matrix = load_data()

# --- UPGRADE 2: FRIENDLY PERSONA & CUSTOM AVATARS ---
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hey there! 👋 I'm Popcorn AI. What's a movie you absolutely love? Tell me, and I'll find some hidden gems just for you!"}]

# Display chat messages with custom avatars (🧑‍💻 for you, 🍿 for the bot)
for msg in st.session_state.messages:
    avatar_icon = "🍿" if msg["role"] == "assistant" else "🧑‍💻"
    with st.chat_message(msg["role"], avatar=avatar_icon):
        st.markdown(msg["content"])

# --- UPGRADE 3: DYNAMIC CONVERSATION LOGIC ---
if user_input := st.chat_input("Type a movie name here (e.g., The Dark Knight)"):
    
    # 1. Show user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user", avatar="🧑‍💻"):
        st.markdown(user_input)

    # 2. Bot Response
    with st.chat_message("assistant", avatar="🍿"):
        search_query = user_input.lower()
        match = df[df['title'].str.lower().str.contains(search_query)]
        
        if not match.empty:
            movie_index = match.index[0]
            movie_title = match.iloc[0]['title']
            distances = similarity_matrix[movie_index]
            movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:4]
            
            # The bot randomly picks a friendly reaction so it feels alive!
            greetings = [
                f"Oh, wow! **{movie_title}** is an absolute masterpiece! 😍",
                f"Great taste! I totally love the vibe of **{movie_title}**. 🎬",
                f"**{movie_title}**? Say no more. That's a top-tier choice! 🔥"
            ]
            
            response = f"{random.choice(greetings)}\n\nSince you enjoyed that one, I dug through my database and I think you will be obsessed with these:\n\n"
            
            for i in movies_list:
                rec_title = df.iloc[i[0]]['title']
                response += f"✨ **{rec_title}**\n"
                
            response += "\n*Which one of these sounds good for tonight?*"
        else:
            response = f"Oh no! 🙈 I searched everywhere but I couldn't find '{user_input}' in my 5,000-movie database. Could you check the spelling or try another favorite?"
        
        st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
