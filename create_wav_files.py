import os
import numpy as np
from scipy.io.wavfile import write

sample_rate = 44100
duration = 5

folders = {
    "rock": 440,
    "pop": 550,
    "jazz": 660
}

base_path = "Data/genres_original"

for genre, freq in folders.items():

    folder_path = os.path.join(base_path, genre)

    os.makedirs(folder_path, exist_ok=True)

    t = np.linspace(0, duration, int(sample_rate * duration))

    audio = np.sin(2 * np.pi * freq * t)

    audio = (audio * 32767).astype(np.int16)

    file_path = os.path.join(folder_path, f"{genre}1.wav")

    write(file_path, sample_rate, audio)

    print("Created:", file_path)

print("\nValid WAV files created successfully!")