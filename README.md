# Data Science Project: Recommender System using Multi-Tower-Model

This project involves processing a movie dataset for various data analysis and machine learning tasks, including genre encoding, user data handling, embeddings integration for movie titles, and the use of a Multi-Tower-Model for movie recommendations.

## Project Overview

In this project, we focused on handling a movie dataset with the following objectives:

1. **Data Preprocessing:** Cleaning and transforming data columns.

2. **Movie Genre Encoding:** Applying one-hot encoding to the movie genre columns.

3. **Embedding Generation:** Generating and integrating sentence embeddings for movie titles.

4. **Gender Mapping:** Mapping boolean gender values to categorical labels.

5. **Recommendation System:** Implementing a Multi-Tower-Model for movie recommendations based on user data and movie attributes.

### Table of Contents

- [Project Overview](#project-overview)
- [Environment Setup](#environment-setup)
- [Data Preprocessing](#data-preprocessing)
- [One-Hot Encoding](#one-hot-encoding)
- [Generating Embeddings](#generating-embeddings)
- [Gender Mapping](#gender-mapping)
- [Multi-Tower-Model for Recommendations](#multi-tower-model-for-recommendations)
- [Usage](#usage)
- [License](#license)



## Environment Setup

Before running the project, ensure you have the necessary dependencies installed. You can install them using `pip`:

```bash
pip install -r requirements.txt
```

## Data Preprocessing

The dataset contains the following columns:

- **`bucketized_user_age`**: User's age bucket.
- **`movie_genres`**: Genres of the movie (encoded as a list of integers).
- **`movie_id`**: Unique identifier for the movie.
- **`movie_title`**: Title of the movie.
- **`raw_user_age`**: Actual user age.
- **`timestamp`**: Timestamp of the movie rating.
- **`user_gender`**: Boolean value indicating the user's gender.
- **`user_id`**: Unique identifier for the user.
- **`user_occupation_label`**: Occupation label of the user.
- **`user_occupation_text`**: Occupation description.
- **`user_rating`**: Rating provided by the user.
- **`user_zip_code`**: Zip code of the user.
