import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import numpy as np


def model_train(user_features, movie_features, data, epochs, batch_size):
    X_users = user_features
    X_movies = movie_features
    y = data['user_rating'].values  # Labels (e.g., binary rating or numeric)

    rating_to_int = {1.: 0, 2.: 0, 3.: 1, 4.: 2, 5.: 2}  # Mapping ratings to integers
    y_encoded = np.array([rating_to_int[rating] for rating in y])

    # Train/test split
    from sklearn.model_selection import train_test_split

    X_users_train, X_users_test, X_movies_train, X_movies_test, y_train, y_test = train_test_split(
        X_users, X_movies, y_encoded, test_size=0.2, random_state=42
    )
    # Convert to TensorFlow tensors

    X_users_train = tf.convert_to_tensor(X_users_train, dtype=tf.float32)
    X_movies_train = tf.convert_to_tensor(X_movies_train, dtype=tf.float32)
    y_train = tf.convert_to_tensor(y_train, dtype=tf.float32)

    X_users_test = tf.convert_to_tensor(X_users_test, dtype=tf.float32)
    X_movies_test = tf.convert_to_tensor(X_movies_test, dtype=tf.float32)
    y_test = tf.convert_to_tensor(y_test, dtype=tf.float32)
    y_train_encoded = to_categorical(y_train, num_classes=3)
    y_test_encoded = to_categorical(y_test, num_classes=3)

    # Define the dimensions of the input features
    user_feature_dim = user_features.shape[1]  # Adjust based on your data
    movie_feature_dim = movie_features.shape[1]  # Adjust based on your data

    # User input and embedding
    user_input = layers.Input(shape=(user_feature_dim,), name="user_input")
    user_embedding = layers.Dense(16, activation='relu')(user_input)
    user_embedding = layers.BatchNormalization()(user_embedding)

    # Movie input and embedding
    movie_input = layers.Input(shape=(movie_feature_dim,), name="movie_input")
    movie_embedding = layers.Dense(16, activation='relu')(movie_input)
    movie_embedding = layers.BatchNormalization()(movie_embedding)

    # Combine user and movie embeddings
    combined = layers.Concatenate(name="concatenate")([user_embedding, movie_embedding])
    # combined = layers.Dense(128, activation='relu')(combined)
    # combined = layers.Dropout(0.3)(combined)  # Dropout for regularization
    # combined = layers.Dense(64, activation='relu')(combined)

    # Output layer with 5 classes
    output = layers.Dense(3, activation='softmax', name="output")(combined)

    # Final model
    model = Model(inputs=[user_input, movie_input], outputs=output)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    model.summary()

    checkpoint_callback = ModelCheckpoint('../models/best_model.keras',
                                          monitor='val_loss', save_best_only=True, mode='min', verbose=1)
    early_stopping_callback = EarlyStopping(monitor='val_loss',
                                            patience=5, restore_best_weights=True, verbose=1)

    model.fit(
        [X_users_train, X_movies_train], y_train_encoded,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=([X_users_test, X_movies_test], y_test_encoded),  # Use one-hot encoded labels
        callbacks=[checkpoint_callback, early_stopping_callback],
        verbose=1  # Show training progress
    )

    print("\n\n")
    test_loss, test_acc = model.evaluate([X_users_test, X_movies_test], y_test_encoded)  # Test data should be prepared
    print(f"Test Loss: {test_loss}, Test Accuracy: {test_acc}")

    model.save('../models/MT-Model.h5')
    print("Model saved to models/MT-Model.h5")

    return f"Test Loss: {test_loss}, Test Accuracy: {test_acc}"
