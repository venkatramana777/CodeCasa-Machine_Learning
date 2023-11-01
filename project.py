import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity

# Load movie ratings data (user, movie, rating)
ratings_data = pd.read_csv('ratings.csv')

# Create a user-item matrix
user_item_matrix = ratings_data.pivot(index='user_id', columns='movie_id', values='rating').fillna(0)

# Split the data into training and testing sets
train_data, test_data = train_test_split(user_item_matrix, test_size=0.2)

# Calculate cosine similarity between items (movies)
item_similarity = cosine_similarity(train_data.T)

# Create a movie recommendation function
def get_movie_recommendations(movie_id, user_ratings, item_similarity):
    similar_scores = item_similarity[movie_id]
    weighted_scores = user_ratings.dot(similar_scores)
    normalized_scores = weighted_scores / similar_scores.sum()
    recommendations = normalized_scores.sort_values(ascending=False)
    return recommendations

# Get movie recommendations for a user
user_id = 1  # Replace with the user's ID
user_ratings = train_data.loc[user_id]
movie_id = 42  # Replace with the movie for which you want recommendations

recommendations = get_movie_recommendations(movie_id, user_ratings, item_similarity)

# Print the top N recommendations
top_n = 10
top_movies = recommendations.head(top_n)

# Display the recommended movies
recommended_movie_ids = top_movies.index
recommended_movies = ratings_data[ratings_data['movie_id'].isin(recommended_movie_ids)]['movie_title'].unique()
print(f"Top {top_n} movie recommendations for user {user_id}:")
for i, movie in enumerate(recommended_movies):
    print(f"{i + 1}. {movie}")
