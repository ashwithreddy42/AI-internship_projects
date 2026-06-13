import pandas as pd
import ast

# ============================================================
# Load Datasets
# ============================================================

movies_df = pd.read_csv("data/tmdb_5000_movies.csv")
credits_df = pd.read_csv("data/tmdb_5000_credits.csv")

# ============================================================
# Merge Datasets
# ============================================================

movies = movies_df.merge(credits_df, on="title")

# ============================================================
# Keep Required Columns
# ============================================================

movies = movies[
    ["movie_id", "title", "genres", "keywords", "cast", "crew"]
]

# ============================================================
# Helper Functions
# ============================================================

def extract_names(text):
    try:
        items = ast.literal_eval(text)
        return [item["name"] for item in items]
    except:
        return []


def extract_top3_cast(text):
    try:
        items = ast.literal_eval(text)
        return [item["name"] for item in items[:3]]
    except:
        return []


def extract_director(text):
    try:
        items = ast.literal_eval(text)

        for item in items:
            if item.get("job") == "Director":
                return [item["name"]]

        return []

    except:
        return []

# ============================================================
# Apply Transformations
# ============================================================

movies["genres"] = movies["genres"].apply(extract_names)

movies["keywords"] = movies["keywords"].apply(extract_names)

movies["cast"] = movies["cast"].apply(extract_top3_cast)

movies["crew"] = movies["crew"].apply(extract_director)

# ============================================================
# Remove Spaces
# ============================================================

for col in ["genres", "keywords", "cast", "crew"]:
    movies[col] = movies[col].apply(
        lambda x: [i.replace(" ", "") for i in x]
    )

# ============================================================
# Create Tags Column
# ============================================================

movies["tags"] = (
    movies["genres"]
    + movies["keywords"]
    + movies["cast"]
    + movies["crew"]
)

movies["tags"] = movies["tags"].apply(lambda x: " ".join(x))

movies["tags"] = movies["tags"].str.lower()

# ============================================================
# Final DataFrame
# ============================================================

new_df = movies[["movie_id", "title", "tags"]]

# ============================================================
# Save Clean Dataset
# ============================================================

new_df.to_csv("data/movies_clean.csv", index=False)

print("✅ movies_clean.csv created successfully!")
print(f"Rows: {new_df.shape[0]}")
print(f"Columns: {new_df.shape[1]}")