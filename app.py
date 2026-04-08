import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.title("🎬 My NLP Movie Recommendation System")
st.write("Welcome to my 3rd Internal Project!")

# Load the dataset
df = pd.read_csv("movies.csv")

# Combine the text features for NLP processing
df['combined_features'] = df['genres'] + " " + df['plot']

# Display the data
st.write("### Our Movie Database")
st.dataframe(df)

# --- NEW NLP LOGIC BELOW ---
st.write("### ⚙️ System Status")

# 1. Initialize the TF-IDF Vectorizer
vectorizer = TfidfVectorizer()

# 2. Fit and transform the combined text into a matrix of numbers
feature_vectors = vectorizer.fit_transform(df['combined_features'])

# 3. Calculate the Cosine Similarity between all movies
similarity_matrix = cosine_similarity(feature_vectors)

# Show a success message on the website
st.success("✅ NLP Model trained successfully! TF-IDF and Cosine Similarity calculated.") 

# --- STEP 7: INTERACTIVE UI ---
st.write("---")
st.write("### 🔍 Find Your Next Movie")

# 1. Create a dropdown menu with all the movie titles
selected_movie = st.selectbox("Choose a movie you like:", df['title'].values)

# 2. Create a button to trigger the recommendation
if st.button("Recommend"):
    
    # Find the index number of the movie the user selected
    movie_index = df[df['title'] == selected_movie].index[0]
    
    # Get the similarity scores for that specific movie
    distances = similarity_matrix[movie_index]
    
    # Sort the scores from highest to lowest, and grab the top 3 (ignoring the first one, which is the exact same movie)
    movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:4]
    
    # Display the results
    st.write(f"**Because you liked '{selected_movie}', you might also love:**")
    
    for i in movies_list:
        # Get the title from the original dataframe using the index
        recommended_title = df.iloc[i[0]]['title']
        st.info(f"🍿 {recommended_title}")