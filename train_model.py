import os
import numpy as np
import librosa

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.layers import Dense, Dropout, Input

DATASET_PATH = "Data"

features = []
labels = []

# Extract MFCC features
def extract_mfcc(file_path):
    audio, sr = librosa.load(file_path, duration=30)

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_mfcc=40
    )

    mfcc = np.resize(mfcc, (40, 130))

    return mfcc

print("Loading audio files...\n")

# Read dataset
for genre in os.listdir(DATASET_PATH):

    genre_path = os.path.join(DATASET_PATH, genre)

    if os.path.isdir(genre_path):

        for file in os.listdir(genre_path):

            if file.endswith(".wav"):

                file_path = os.path.join(genre_path, file)

                try:
                    mfcc = extract_mfcc(file_path)

                    features.append(mfcc)
                    labels.append(genre)

                    print("Loaded:", file_path)

                except Exception as e:

                    print("Skipped invalid audio:", file_path)

# Check dataset
if len(features) == 0:

    print("\nNo valid wav files found!")
    exit()

# Convert to arrays
X = np.array(features)
y = np.array(labels)

# Reshape for CNN
X = X.reshape(X.shape[0], 40, 130, 1)

# Encode labels
encoder = LabelEncoder()

y = encoder.fit_transform(y)

y = to_categorical(y)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Number of genres
num_classes = y.shape[1]

# CNN Model
model = Sequential([

    Input(shape=(40, 130, 1)),

    Conv2D(32, (3, 3), activation="relu"),
    MaxPooling2D((2, 2)),

    Conv2D(64, (3, 3), activation="relu"),
    MaxPooling2D((2, 2)),

    Conv2D(128, (3, 3), activation="relu"),
    MaxPooling2D((2, 2)),

    Flatten(),

    Dense(128, activation="relu"),

    Dropout(0.3),

    Dense(num_classes, activation="softmax")
])

# Compile model
model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

print("\nTraining started...\n")

# Train model
model.fit(
    X_train,
    y_train,
    epochs=10,
    batch_size=32,
    validation_data=(X_test, y_test)
)

# Create model folder
os.makedirs("model", exist_ok=True)

# Save model
model.save("model/genre_cnn_model.h5")

# Save classes
np.save("model/classes.npy", encoder.classes_)

print("\nModel trained and saved successfully!")