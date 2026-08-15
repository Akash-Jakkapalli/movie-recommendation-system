import pandas as pd
import ast

from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# -----------------------------------
# 1. Helper Functions
# -----------------------------------

def convert(text):
    items = ast.literal_eval(text)
    names = []

    for item in items:
        names.append(item['name'])

    return names


def convert_cast(text):
    items = ast.literal_eval(text)
    names = []

    for item in items[:3]:
        names.append(item['name'])

    return names


def fetch_director(text):
    items = ast.literal_eval(text)

    for item in items:
        if item['job'] == 'Director':
            return item['name']

    return ''


# -----------------------------------
# 2. Create Stemmer
# -----------------------------------

ps = PorterStemmer()


# -----------------------------------
# 3. Load Dataset
# -----------------------------------

movies = pd.read_csv("data/tmdb_5000_movies.csv")
credits = pd.read_csv("data/tmdb_5000_credits.csv")


# -----------------------------------
# 4. Merge Movies and Credits
# -----------------------------------

movies = movies.merge(
    credits,
    left_on="id",
    right_on="movie_id"
)


# -----------------------------------
# 5. Select Important Columns
# -----------------------------------

movies = movies[
    ['title_x', 'overview', 'genres', 'keywords', 'cast', 'crew']
]

movies.rename(
    columns={'title_x': 'title'},
    inplace=True
)


# -----------------------------------
# 6. Handle Missing Values
# -----------------------------------

movies['overview'] = movies['overview'].fillna('')
movies['genres'] = movies['genres'].fillna('[]')
movies['keywords'] = movies['keywords'].fillna('[]')
movies['cast'] = movies['cast'].fillna('[]')
movies['crew'] = movies['crew'].fillna('[]')


# -----------------------------------
# 7. Convert Genres and Keywords
# -----------------------------------

movies['genres'] = movies['genres'].apply(convert)

movies['keywords'] = movies['keywords'].apply(convert)


# -----------------------------------
# 8. Extract Top 3 Actors
# -----------------------------------

movies['cast'] = movies['cast'].apply(convert_cast)


# -----------------------------------
# 9. Extract Director
# -----------------------------------

movies['crew'] = movies['crew'].apply(fetch_director)

movies.rename(
    columns={'crew': 'director'},
    inplace=True
)


# -----------------------------------
# 10. Prepare Overview
# -----------------------------------

movies['overview'] = movies['overview'].apply(
    lambda x: x.split()
)


# -----------------------------------
# 11. Prepare Director
# -----------------------------------

movies['director'] = movies['director'].apply(
    lambda x: [x]
)


# -----------------------------------
# 12. Create Tags
# -----------------------------------

movies['tags'] = (
    movies['overview']
    + movies['genres']
    + movies['keywords']
    + movies['cast']
    + movies['director']
)


# -----------------------------------
# 13. Convert Tags to Text
# -----------------------------------

movies['tags'] = movies['tags'].apply(
    lambda x: " ".join(x)
)


# -----------------------------------
# 14. Convert to Lowercase
# -----------------------------------

movies['tags'] = movies['tags'].apply(
    lambda x: x.lower()
)


# -----------------------------------
# 15. Stemming
# -----------------------------------

def stem(text):
    return " ".join(
        [ps.stem(word) for word in text.split()]
    )


movies['tags'] = movies['tags'].apply(stem)


# -----------------------------------
# 16. TF-IDF
# -----------------------------------

vectorizer = TfidfVectorizer(
    max_features=5000,
    stop_words='english'
)

vectors = vectorizer.fit_transform(
    movies['tags']
)


# -----------------------------------
# 17. Cosine Similarity
# -----------------------------------

similarity = cosine_similarity(vectors)


# -----------------------------------
# 18. Check Results
# -----------------------------------

print("Movie Recommendation System")
print("--------------------------------")

print("Number of movies:", len(movies))

print("Vector shape:", vectors.shape)

print("Similarity matrix shape:", similarity.shape)


# -----------------------------------
# 19. Recommendation Function
# -----------------------------------

def recommend(movie):

    movie = movie.lower().strip()

    matching_movies = movies[
        movies['title'].str.lower().str.strip() == movie
    ]

    if matching_movies.empty:
        print("\nMovie not found.")
        print("Please check the movie title and try again.")
        return

    movie_index = matching_movies.index[0]

    distances = similarity[movie_index]

    movie_list = sorted(
        list(enumerate(distances)),
        reverse=True,
        key=lambda x: x[1]
    )

    print(
        "\nRecommended movies for:",
        movies.iloc[movie_index]['title']
    )

    print("--------------------------------")

    for index, i in enumerate(movie_list[1:6], start=1):

        movie_title = movies.iloc[i[0]]['title']
        score = i[1]

        print(
            f"{index}. {movie_title} | Similarity: {score:.2f}"
        )