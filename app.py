import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import random
import requests

# --- PAGE CONFIG & STYLING ---
st.set_page_config(page_title="Popcorn AI", page_icon="🍿", layout="wide")

st.markdown("""
    <style>
    h1 {
        color: #E50914 !important;
        font-family: 'Arial Black', sans-serif;
        text-align: center;
        text-shadow: 2px 2px 4px #000000;
    }
    [data-testid="stChatInput"] {
        border: 2px solid #E50914 !important;
        border-radius: 15px !important;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🍿 Popcorn AI: Your Movie Bestie")
st.markdown("<h4 style='text-align: center; color: #808080;'>Tell me what you love, and I'll find your next binge!</h4>", unsafe_allow_html=True)
st.write("---")

# --- THE NLP BRAIN ---
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

# --- THE TMDB API CONNECTION ---
def fetch_poster(movie_id):
    # Your personal TMDB API Key
    api_key = "f4534c927fd5aee9527299c5b2e7a88c"
    url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={api_key}&language=en-US"
    try:
        data = requests.get(url).json()
        poster_path = data.get('poster_path')
        if poster_path:
            return f"https://image.tmdb.org/t/p/w500/{poster_path}"
        else:
            return "https://via.placeholder.com/500x750?text=No+Poster+Found"
    except:
        return "https://via.placeholder.com/500x750?text=API+Error"

# --- CHATBOT MEMORY (Upgraded to handle images) ---
if "messages" not in st.session_state:
    st.session_state.messages = [{
        "role": "assistant", 
        "content": "Hey there! 👋 I'm Popcorn AI. What's a movie you absolutely love? Tell me, and I'll find some hidden gems just for you!",
        "images": [] # New memory slot for posters!
    }]

# Display chat history (Text + Images)
for msg in st.session_state.messages:
    avatar_icon = "🍿" if msg["role"] == "assistant" else "🧑‍💻"
    with st.chat_message(msg["role"], avatar=avatar_icon):
        st.markdown(msg["content"])
        # If this message has images, display them in neat columns
        if msg.get("images"):
            cols = st.columns(len(msg["images"]))
            for col, img_url in zip(cols, msg["images"]):
                with col:
                    st.image(img_url)

# --- CHATBOT LOGIC ---
if user_input := st.chat_input("Type a movie name here (e.g., The Dark Knight)"):
    
    # 1. Show user message
    st.session_state.messages.append({"role": "user", "content": user_input, "images": []})
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
            
            greetings = [
                f"Oh, wow! **{movie_title}** is an absolute masterpiece! 😍",
                f"Great taste! I totally love the vibe of **{movie_title}**. 🎬",
                f"**{movie_title}**? Say no more. That's a top-tier choice! 🔥"
            ]
            
            response_text = f"{random.choice(greetings)}\n\nSince you enjoyed that one, I dug through my database and I think you will be obsessed with these:\n\n"
            
            posters = [] # List to hold our fetched images
            
            # Grab titles and fetch posters using the 'id' column
            for i in movies_list:
                rec_title = df.iloc[i[0]]['title']
                rec_id = df.iloc[i[0]]['id']
                
                response_text += f"✨ **{rec_title}**\n"
                posters.append(fetch_poster(rec_id))
                
            response_text += "\n*Which one of these sounds good for tonight?*"
            
            # Display the text
            st.markdown(response_text)
            
            # Display the posters in columns
            cols = st.columns(len(posters))
            for col, img_url in zip(cols, posters):
                with col:
                    st.image(img_url)
            
            # Save the text AND images to the bot's memory
            st.session_state.messages.append({
                "role": "assistant", 
                "content": response_text, 
                "images": posters
            })
            
        else:
            response_text = f"Oh no! 🙈 I searched everywhere but I couldn't find '{user_input}' in my 5,000-movie database. Could you check the spelling or try another favorite?"
            st.markdown(response_text)
            st.session_state.messages.append({"role": "assistant", "content": response_text, "images": []})
