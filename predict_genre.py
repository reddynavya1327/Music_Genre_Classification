import numpy as np
import librosa
from tensorflow.keras.models import load_model

MODEL_PATH = "model/genre_cnn_model.h5"
CLASSES_PATH = "model/classes.npy"

model = load_model(MODEL_PATH)
classes = np.load(CLASSES_PATH, allow_pickle=True)

def extract_mfcc(file_path):
    audio, sr = librosa.load(file_path, duration=30)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    mfcc = np.resize(mfcc, (40, 130))
    return mfcc

audio_path = input("Enter audio file path: ")

mfcc = extract_mfcc(audio_path)
mfcc = mfcc.reshape(1, 40, 130, 1)

prediction = model.predict(mfcc)

index = np.argmax(prediction)
genre = classes[index]
confidence = prediction[0][index] * 100

print("Predicted Genre:", genre)
print("Confidence Score:", round(confidence, 2), "%")