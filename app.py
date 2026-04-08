import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import random
import requests

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Popcorn AI", page_icon="🍿", layout="wide")

# --- ADVANCED HOVER-CARD & SIDEBAR CSS INJECTION ---
st.markdown("""
    <style>
    /* Netflix Dark Theme */
    .stApp { background-color: #141414; color: #E5E5E5; }
    h1 { color: #E50914 !important; font-family: 'Arial Black', sans-serif; text-align: center; text-transform: uppercase; letter-spacing: 2px; margin-bottom: 0px; }
    .subtitle { text-align: center; color: #808080; font-size: 1.2rem; margin-bottom: 2rem; }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] { background-color: #000000; border-right: 1px solid #333; }
    div.stButton > button { background-color: #141414; color: #E5E5E5; border: 1px solid #333; border-radius: 8px; transition: 0.3s; }
    div.stButton > button:hover { border-color: #E50914; color: #E50914; }
    
    /* Chat Input Box Modernization */
    [data-testid="stChatInput"] { border: 1px solid #333 !important; border-radius: 25px !important; background-color: #2b2b2b !important; }
    [data-testid="stChatInput"] textarea { color: white !important; }
    [data-testid="stChatMessage"] { background-color: rgba(255, 255, 255, 0.05); border-radius: 15px; padding: 15px; margin-bottom: 10px; }
    
    /* THE NEW MOVIE CARD HOVER EFFECT */
    .movie-card {
        position: relative;
        overflow: hidden;
        border-radius: 12px;
        box-shadow: 0 8px 16px rgba(0,0,0,0.8);
        cursor: pointer;
    }
    .movie-card img {
        width: 100%;
        display: block;
        transition: transform 0.3s ease;
    }
    .movie-card:hover img {
        transform: scale(1.05);
    }
    
    /* The hidden text that appears on hover */
    .hover-info {
        position: absolute;
        bottom: 0;
        left: 0;
        right: 0;
        background: linear-gradient(to top, rgba(20,20,20,1) 10%, rgba(20,20,20,0.9) 60%, transparent 100%);
        color: #fff;
        padding: 15px;
        font-size: 0.85rem;
        opacity: 0; 
        transition: opacity 0.3s ease;
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: flex-end;
    }
    .movie-card:hover .hover-info {
        opacity: 1; 
    }
    .hover-title {
        font-weight: bold;
        color: #E50914;
        margin-bottom: 2px;
        font-size: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

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
    except:
        pass
    return "https://via.placeholder.com/500x750/141414/808080?text=No+Poster+Found"

# --- STATE MANAGEMENT & MEMORY ---
if "messages" not in st.session_state:
    st.session_state.messages = [{
        "role": "assistant", 
        "content": "Hey there! 👋 I'm Popcorn AI. Tell me a movie you love, or ask for a specific genre like 'Romance', and I'll find your next binge!",
        "movie_data": [] 
    }]
if "clicked_genre" not in st.session_state:
    st.session_state.clicked_genre = None

# --- SIDEBAR: GENRE COLUMN ---
with st.sidebar:
    st.markdown("<h2 style='color: #E50914; text-align: center;'>🎭 Browse Genres</h2>", unsafe_allow_html=True)
    st.write("Not sure what you want? Pick a vibe below!")
    
    genre_list = ["Action", "Comedy", "Drama", "Sci-Fi", "Horror", "Romance", "Thriller", "Fantasy", "Animation", "Adventure"]
    
    for g in genre_list:
        if st.button(g, use_container_width=True):
            st.session_state.clicked_genre = g

# --- UI HEADER ---
st.title("🍿 Popcorn AI")
st.markdown("<div class='subtitle'>Your Personal Cinematic Bestie</div>", unsafe_allow_html=True)

# --- DISPLAY CHAT HISTORY ---
for msg in st.session_state.messages:
    avatar_icon = "🍿" if msg["role"] == "assistant" else "🧑‍💻"
    with st.chat_message(msg["role"], avatar=avatar_icon):
        st.markdown(msg["content"])
        
        if msg.get("movie_data"):
            cols = st.columns(len(msg["movie_data"]))
            for col, movie in zip(cols, msg["movie_data"]):
                with col:
                    card_html = f"""
                    <div class="movie-card">
                        <img src="{movie['poster']}" alt="{movie['title']}">
                        <div class="hover-info">
                            <div class="hover-title">{movie['title']}</div>
                            <div>{movie['overview'][:150]}...</div>
                        </div>
                    </div>
                    """
                    st.markdown(card_html, unsafe_allow_html=True)

# --- CHATBOT INPUT LOGIC (UPGRADED WITH NLP INTENT SCANNER) ---
user_typed = st.chat_input("Type a movie name or ask for a genre (e.g., 'give me romance')")
query_text = None
is_genre_mode = False
target_genre = ""

if st.session_state.clicked_genre:
    # Triggered if they clicked a sidebar button
    query_text = f"Show me some great {st.session_state.clicked_genre} movies!"
    is_genre_mode = True
    target_genre = st.session_state.clicked_genre
    st.session_state.clicked_genre = None 
    
elif user_typed:
    # Triggered if they typed something in the chat box
    query_text = user_typed
    typed_lower = user_typed.lower()
    
    # 1. Scan the input text for known genre keywords
    known_genres = ["action", "comedy", "drama", "sci-fi", "horror", "romance", "thriller", "fantasy", "animation", "adventure"]
    
    for g in known_genres:
        if g in typed_lower:
            # If a genre is found in the text, intercept the search and switch to genre mode!
            is_genre_mode = True
            target_genre = g.capitalize() # Capitalize it (e.g., 'romance' -> 'Romance') for the search
            break

# --- BOT RESPONSE GENERATOR ---
if query_text:
    # 1. User Message Display
    st.session_state.messages.append({"role": "user", "content": query_text, "movie_data": []})
    with st.chat_message("user", avatar="🧑‍💻"):
        st.markdown(query_text)

    # 2. AI Response Display
    with st.chat_message("assistant", avatar="🍿"):
        current_movie_data = []
        response_text = ""
        
        # --- PATH A: GENRE MODE (Triggered by sidebar OR typed keywords) ---
        if is_genre_mode:
            match = df[df['genres'].str.contains(target_genre, case=False, na=False)]
            
            if not match.empty:
                sample_size = 3 if len(match) >= 3 else len(match)
                selected_movies = match.sample(sample_size)
                
                response_text = f"You got it! 🎬 Here are three fantastic **{target_genre}** movies pulled right from the archives:\n\n"
                
                for _, row in selected_movies.iterrows():
                    rec_title = row['title']
                    rec_id = row['id']
                    rec_overview = row['overview']
                    short_plot = rec_overview[:120].strip() + "..." if len(rec_overview) > 120 else rec_overview
                    
                    response_text += f"🎯 **{rec_title}**\n> *\"{short_plot}\"*\n\n"
                    current_movie_data.append({
                        "title": rec_title, "poster": fetch_poster(rec_id), "overview": rec_overview
                    })
                response_text += "\n*(Hover over the posters to read the full plot!)*"
            else:
                response_text = f"I couldn't find any {target_genre} movies in my database right now!"

        # --- PATH B: MOVIE TITLE SEARCH (TF-IDF Similarity) ---
        else:
            search_query = query_text.lower()
            match = df[df['title'].str.lower().str.contains(search_query)]
            
            if not match.empty:
                movie_index = match.index[0]
                movie_title = match.iloc[0]['title']
                distances = similarity_matrix[movie_index]
                movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:4]
                
                intros = [
                    f"Oh, **{movie_title}** is a phenomenal pick. 🍿",
                    f"I see you! **{movie_title}** has such a specific vibe. ✨",
                    f"Excellent choice. If you liked the pacing and themes of **{movie_title}**, you are in for a treat. 🎬",
                    f"**{movie_title}**? A total classic. Let me scan the archives for something with that same energy... 🔍"
                ]
                
                transitions = [
                    "I ran the numbers on the plot and genres, and these three films are a mathematical perfect match:\n\n",
                    "Based on the underlying themes of that movie, I think you'll be obsessed with these recommendations:\n\n",
                    "Here are three hidden gems and blockbusters that share the exact same cinematic DNA:\n\n",
                    "I calculated the TF-IDF text vectors (my favorite AI trick 🤫), and these are your top matches:\n\n"
                ]
                
                closings = [
                    "\n\n*(Hover over the posters to read the plot! Which one is calling your name?)*",
                    "\n\n*(Take a look at the plots by hovering over the posters. See anything you like?)*",
                    "\n\n*(Hover to reveal the descriptions. Grab some popcorn!)*"
                ]
                
                response_text = f"{random.choice(intros)}\n\n{random.choice(transitions)}"
                
                for i in movies_list:
                    rec_title = df.iloc[i[0]]['title']
                    rec_id = df.iloc[i[0]]['id']
                    rec_overview = df.iloc[i[0]]['overview']
                    short_plot = rec_overview[:120].strip() + "..." if len(rec_overview) > 120 else rec_overview
                    
                    response_text += f"🎯 **{rec_title}**\n> *\"{short_plot}\"*\n\n"
                    current_movie_data.append({
                        "title": rec_title, "poster": fetch_poster(rec_id), "overview": rec_overview
                    })
                    
                response_text += random.choice(closings)
                
            else:
                response_text = f"Oh no! 🙈 I searched everywhere but couldn't find '{query_text}'. Could you check the spelling?"

        # --- RENDER THE FINAL HTML & SAVE TO MEMORY ---
        st.markdown(response_text)
        
        if current_movie_data:
            cols = st.columns(len(current_movie_data))
            for col, movie in zip(cols, current_movie_data):
                with col:
                    card_html = f"""
                    <div class="movie-card">
                        <img src="{movie['poster']}" alt="{movie['title']}">
                        <div class="hover-info">
                            <div class="hover-title">{movie['title']}</div>
                            <div>{movie['overview'][:150]}...</div>
                        </div>
                    </div>
                    """
                    st.markdown(card_html, unsafe_allow_html=True)
        
        st.session_state.messages.append({
            "role": "assistant", 
            "content": response_text, 
            "movie_data": current_movie_data
        })
