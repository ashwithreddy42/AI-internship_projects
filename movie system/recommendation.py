import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

df = pd.read_csv("data/movies_clean.csv")

df["tags"] = df["tags"].fillna("")

cv = CountVectorizer(max_features=5000, stop_words="english")
vectors = cv.fit_transform(df["tags"]).toarray()

similarity = cosine_similarity(vectors)

def recommend(movie_name):

    movie_name = movie_name.lower()

    matches = df[df["title"].str.lower() == movie_name]

    if matches.empty:
        print("Movie not found!")
        return

    movie_index = matches.index[0]

    distances = similarity[movie_index]

    movie_list = sorted(
        list(enumerate(distances)),
        reverse=True,
        key=lambda x: x[1]
    )[1:6]

    print(f"\nRecommended movies for {matches.iloc[0]['title']}:\n")

    for movie in movie_list:
        print(df.iloc[movie[0]].title)

while True:

    movie_name = input(
        "\nEnter movie name (or 'exit' to quit): "
    )

    if movie_name.lower() == "exit":
        break

    recommend(movie_name)