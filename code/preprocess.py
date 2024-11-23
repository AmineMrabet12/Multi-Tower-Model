import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MultiLabelBinarizer
from sentence_transformers import SentenceTransformer
from tqdm import tqdm  # For progress bar

# Load the dataset
# data = pd.read_csv("../data/movielens.csv")

# Initialize transformers
genre_encoder = MultiLabelBinarizer()
occupation_encoder = OneHotEncoder(sparse=False)
scaler = StandardScaler()
title_transformer = SentenceTransformer('all-MiniLM-L6-v2')

genre_map = {
    0: 'Unknown',
    1: 'Action',
    2: 'Adventure',
    3: 'Animation',
    4: 'Children',
    5: 'Comedy',
    6: 'Crime',
    7: 'Documentary',
    8: 'Drama',
    9: 'Fantasy',
    10: 'Film-Noir',
    11: 'Horror',
    12: 'Musical',
    13: 'Mystery',
    14: 'Romance',
    15: 'Sci-Fi',
    16: 'Thriller',
    17: 'War',
    18: 'Western',
    19: 'Other'
}


# Process user features
def preprocess_user_features(data):
    # Initialize tqdm for the processing steps
    with tqdm(total=3, desc="Preprocessing User Features", unit="step") as pbar:
        # Normalize bucketized_user_age
        bucketized_user_age = scaler.fit_transform(data[['bucketized_user_age']])
        pbar.update(1)

        # Encode user_gender as binary (0: False, 1: True)
        user_gender = data['user_gender'].astype(int).values.reshape(-1, 1)
        pbar.update(1)

        # One-hot encode user_occupation_label
        occupation_encoded = occupation_encoder.fit_transform(data[['user_occupation_text']])
        pbar.update(1)

    # Return user feature vector
    user_features = np.hstack([bucketized_user_age, user_gender, occupation_encoded])
    return user_features


# Process movie features
def preprocess_movie_features(data):
    """
    Preprocess movie features for model input.

    Args:
        data (pd.DataFrame): DataFrame containing movie features, including 'movie_genres' and 'movie_title'.
        title_transformer: Sentence transformer model for title embeddings.
        genre_encoder: MultiLabelBinarizer for encoding genres.

    Returns:
        np.ndarray: Preprocessed movie feature vectors.
    """
    # Initialize tqdm for movie genres and titles
    with tqdm(total=2, desc="Preprocessing Movie Features", unit="step") as pbar:
        # Map genres to their string values and one-hot encode them
        data["movie_genres"] = data["movie_genres"].apply(
            lambda ids: [genre_map[int(id)] for id in ids.strip('[]').split()]
        )
        genre_encoded = genre_encoder.fit_transform(data["movie_genres"])
        pbar.update(1)

        # Encode movie titles with sentence embeddings
        title_embeddings = np.array(title_transformer.encode(data['movie_title'].tolist()))
        pbar.update(1)

    # Return movie feature vector
    movie_features = np.hstack([genre_encoded, title_embeddings])
    return movie_features


# Prepare data
# user_features = preprocess_user_features(data)
# movie_features = preprocess_movie_features(data)

# # Ensure user_features and movie_features are ready for model input
# print("User Features Shape:", user_features.shape)
# print("Movie Features Shape:", movie_features.shape)
