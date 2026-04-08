import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import random
import requests

# --- PAGE CONFIGURATION ---
# 'wide' layout uses the full screen, giving it a modern web-app feel
st.set_page_config(page_title="Popcorn AI", page_icon="🍿", layout="wide")

# --- ADVANCED RESPONSIVE CSS INJECTION ---
st.markdown("""
    <style>
    /* 1. Netflix Dark Theme Background */
    .stApp {
        background-color: #141414;
        color: #E5E5E5;
    }
    
    /* 2. Main Title Styling */
    h1 {
        color: #E50914 !important;
        font-family: 'Arial Black', sans-serif;
        text-align: center;
        text-transform: uppercase;
        letter-spacing: 2px;
        margin-bottom: 0px;
    }
    
    /* 3. Subtitle Styling */
    .subtitle {
        text-align: center;
        color: #808080;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    
    /* 4. Chat Input Box Modernization */
    [data-testid="stChatInput"] {
        border: 1px solid #333 !important;
        border-radius: 25px !important;
        background-color: #2b2b2b !important;
    }
    [data-testid="stChatInput"] textarea {
        color: white !important;
    }
    
    /* 5. Responsive Movie Posters with Hover Animation */
    /* This makes the posters gently zoom in when the user hovers over them */
    img {
        border-radius: 12px;
        box-shadow: 0 8px 16px rgba(0,0,0,0.8);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        width: 100%; /* Ensures images resize to fit their columns responsibly */
    }
    img:hover {
        transform: scale(1.05);
        box-shadow: 0 12px 24px rgba(229, 9, 20, 0.4); /* Subtle red glow */
    }
    
    /* 6. Chat Message Bubble Styling */
    [data-testid="stChatMessage"] {
        background-color: rgba(255, 255, 255, 0.05);
        border-radius: 15px;
        padding: 15px;
        margin-bottom: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# --- UI HEADER ---
st.title("🍿 Popcorn AI")
st.markdown("<div class='subtitle'>Your Personal Cinematic Bestie</div>", unsafe_allow_html=True)

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
    api_key = "f4534c927fd5aee9527299c5b2e7a88c"
    url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={api_key}&language=en-US"
    try:
        data = requests.get(url).json()
        poster_path = data.get('poster_path')
        if poster_path:
            return f"https://image.tmdb.org/t/p/w500/{poster_path}"
        else:
            return "https://via.placeholder.com/500x750/141414/808080?text=No+Poster+Found"
    except:
        return "https://via.placeholder.com/500x750/141414/808080?text=API+Error"

# --- CHATBOT MEMORY ---
if "messages" not in st.session_state:
    st.session_state.messages = [{
        "role": "assistant", 
        "content": "Hey there! 👋 I'm Popcorn AI. Tell me a movie you absolutely love, and I'll find your next binge!",
        "images": [] 
    }]

# --- DISPLAY CHAT HISTORY ---
for msg in st.session_state.messages:
    avatar_icon = "🍿" if msg["role"] == "assistant" else "🧑‍💻"
    with st.chat_message(msg["role"], avatar=avatar_icon):
        st.markdown(msg["content"])
        
        # Responsive Image Grid
        if msg.get("images"):
            # st.columns automatically stacks vertically on mobile screens!
            cols = st.columns(len(msg["images"]))
            for col, img_url in zip(cols, msg["images"]):
                with col:
                    st.image(img_url)

# --- CHATBOT LOGIC ---
if user_input := st.chat_input("Type a movie name here (e.g., Inception, The Notebook)"):
    
    st.session_state.messages.append({"role": "user", "content": user_input, "images": []})
    with st.chat_message("user", avatar="🧑‍💻"):
        st.markdown(user_input)

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
            
            response_text = f"{random.choice(greetings)}\n\nBased on the genres and plot, I dug through my database and found these perfect matches:\n\n"
            
            posters = []
            
            for i in movies_list:
                rec_title = df.iloc[i[0]]['title']
                rec_id = df.iloc[i[0]]['id']
                
                response_text += f"✨ **{rec_title}**\n"
                posters.append(fetch_poster(rec_id))
                
            response_text += "\n*Which one of these sounds good for tonight?*"
            
            st.markdown(response_text)
            
            # Display posters in a responsive layout
            cols = st.columns(len(posters))
            for col, img_url in zip(cols, posters):
                with col:
                    st.image(img_url)
            
            st.session_state.messages.append({
                "role": "assistant", 
                "content": response_text, 
                "images": posters
            })
            
        else:
            response_text = f"Oh no! 🙈 I searched everywhere but I couldn't find '{user_input}' in my 5,000-movie database. Could you check the spelling or try another favorite?"
            st.markdown(response_text)
            st.session_state.messages.append({"role": "assistant", "content": response_text, "images": []})
