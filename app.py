from flask import Flask, render_template, request

from recommender import movies, similarity


app = Flask(__name__)


def get_recommendations(movie):
    movie = movie.lower().strip()

    matching_movies = movies[
        movies['title'].str.lower().str.strip() == movie
    ]

    if matching_movies.empty:
        return []

    movie_index = matching_movies.index[0]

    distances = similarity[movie_index]

    movie_list = sorted(
        list(enumerate(distances)),
        reverse=True,
        key=lambda x: x[1]
    )

    recommendations = []

    for i in movie_list[1:6]:
        recommendations.append({
            "title": movies.iloc[i[0]]['title'],
            "score": round(float(i[1]), 2)
        })

    return recommendations


@app.route("/", methods=["GET", "POST"])
def home():

    recommendations = []
    movie_name = ""
    error = ""

    if request.method == "POST":

        movie_name = request.form.get("movie", "").strip()

        if movie_name:
            recommendations = get_recommendations(movie_name)

            if not recommendations:
                error = "Movie not found. Please check the movie title."

        else:
            error = "Please enter a movie name."

    return render_template(
        "index.html",
        recommendations=recommendations,
        movie_name=movie_name,
        error=error
    )


if __name__ == "__main__":
    app.run(debug=True)