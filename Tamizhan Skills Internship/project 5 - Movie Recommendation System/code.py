# Step 1: Import libraries
import pandas as pd
import ast
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Step 2: Load datasets
movies = pd.read_csv('movies.csv')
credits = pd.read_csv('credits.csv')

# Step 3: Merge datasets
movies = movies.merge(credits, left_on='id', right_on='movie_id')

# ✅ FIX: Use 'original_title' as 'title'
movies.rename(columns={'original_title': 'title'}, inplace=True)

# Step 4: Select relevant columns
movies = movies[['id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']]

# Step 5: Convert stringified JSON to lists/values
def convert(obj):
    try:
        return [i['name'] for i in ast.literal_eval(obj)]
    except:
        return []

def get_director(obj):
    try:
        for i in ast.literal_eval(obj):
            if i['job'] == 'Director':
                return i['name']
        return ''
    except:
        return ''

movies['genres'] = movies['genres'].apply(convert)
movies['keywords'] = movies['keywords'].apply(convert)
movies['cast'] = movies['cast'].apply(lambda x: convert(x)[:3])
movies['crew'] = movies['crew'].apply(get_director)
movies['overview'] = movies['overview'].fillna('')

# Step 6: Create 'tags'
movies['tags'] = movies['overview'] + ' ' + \
                 movies['genres'].apply(lambda x: " ".join(x)) + ' ' + \
                 movies['keywords'].apply(lambda x: " ".join(x)) + ' ' + \
                 movies['cast'].apply(lambda x: " ".join(x)) + ' ' + \
                 movies['crew']

# Step 7: Lowercase the tags
movies['tags'] = movies['tags'].str.lower()

# Step 8: Vectorization
cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(movies['tags']).toarray()

# Step 9: Cosine Similarity
similarity = cosine_similarity(vectors)

# Step 10: Recommendation function
def recommend(movie):
    movie = movie.lower()
    if movie not in movies['title'].str.lower().values:
        print("Movie not found.")
        return
    index = movies[movies['title'].str.lower() == movie].index[0]
    distances = list(enumerate(similarity[index]))
    movies_list = sorted(distances, key=lambda x: x[1], reverse=True)[1:6]
    
    print(f"\nTop 5 recommendations for '{movie.title()}':")
    for i in movies_list:
        print(movies.iloc[i[0]].title)

# Example:
recommend("Avatar")
