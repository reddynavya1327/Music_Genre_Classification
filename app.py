import os
import sqlite3
import numpy as np
import librosa
import librosa.display
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from flask import Flask, render_template, request, redirect, session, url_for, send_file
from tensorflow.keras.models import load_model
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)
app.secret_key = "music_secret_key"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
SPECTROGRAM_FOLDER = os.path.join(BASE_DIR, "static", "spectrograms")
DB_PATH = os.path.join(BASE_DIR, "database", "users.db")
MODEL_PATH = os.path.join(BASE_DIR, "model", "genre_cnn_model.h5")
CLASSES_PATH = os.path.join(BASE_DIR, "model", "classes.npy")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(SPECTROGRAM_FOLDER, exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, "database"), exist_ok=True)

model = load_model(MODEL_PATH)
classes = np.load(CLASSES_PATH, allow_pickle=True)


def init_db():
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS users(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE,
            password TEXT
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS history(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT,
            filename TEXT,
            genre TEXT,
            confidence REAL
        )
    """)

    con.commit()
    con.close()


def extract_mfcc(file_path):
    audio, sr = librosa.load(file_path, duration=30)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    mfcc = np.resize(mfcc, (40, 130))
    return mfcc.reshape(1, 40, 130, 1)


def create_spectrogram(file_path, filename):
    audio, sr = librosa.load(file_path, duration=30)

    plt.figure(figsize=(10, 4))
    spectrogram = librosa.amplitude_to_db(
        np.abs(librosa.stft(audio)),
        ref=np.max
    )

    librosa.display.specshow(
        spectrogram,
        sr=sr,
        x_axis="time",
        y_axis="hz"
    )

    plt.colorbar(format="%+2.0f dB")
    plt.title("Audio Spectrogram")
    plt.tight_layout()

    image_name = filename + "_spectrogram.png"
    image_path = os.path.join(SPECTROGRAM_FOLDER, image_name)

    plt.savefig(image_path)
    plt.close()

    return image_name


def create_waveform(file_path, filename):
    audio, sr = librosa.load(file_path, duration=30)

    plt.figure(figsize=(10, 3))
    librosa.display.waveshow(audio, sr=sr)

    plt.title("Audio Waveform")
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.tight_layout()

    image_name = filename + "_waveform.png"
    image_path = os.path.join(SPECTROGRAM_FOLDER, image_name)

    plt.savefig(image_path)
    plt.close()

    return image_name


@app.route("/")
def home():
    return redirect(url_for("login"))


@app.route("/register", methods=["GET", "POST"])
def register():
    message = ""

    if request.method == "POST":
        username = request.form["username"]
        password = generate_password_hash(request.form["password"])

        try:
            con = sqlite3.connect(DB_PATH)
            cur = con.cursor()

            cur.execute(
                "INSERT INTO users(username, password) VALUES (?, ?)",
                (username, password)
            )

            con.commit()
            con.close()

            return redirect(url_for("login"))

        except:
            message = "Username already exists!"

    return render_template("register.html", message=message)


@app.route("/login", methods=["GET", "POST"])
def login():
    message = ""

    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]

        con = sqlite3.connect(DB_PATH)
        cur = con.cursor()

        cur.execute(
            "SELECT password FROM users WHERE username=?",
            (username,)
        )

        user = cur.fetchone()
        con.close()

        if user and check_password_hash(user[0], password):
            session["username"] = username
            return redirect(url_for("dashboard"))
        else:
            message = "Invalid username or password!"

    return render_template("login.html", message=message)


@app.route("/dashboard", methods=["GET", "POST"])
def dashboard():
    if "username" not in session:
        return redirect(url_for("login"))

    if request.method == "POST":
        file = request.files["audio"]

        if file:
            filename = file.filename
            file_path = os.path.join(UPLOAD_FOLDER, filename)
            file.save(file_path)

            if not filename.lower().endswith(".wav"):
                return render_template(
                    "result.html",
                    filename=filename,
                    genre="Only WAV files are allowed",
                    confidence=0,
                    audio_file=None,
                    spectrogram=None,
                    waveform=None,
                    top_predictions=[]
                )

            try:
                mfcc = extract_mfcc(file_path)
                prediction = model.predict(mfcc)

                index = np.argmax(prediction)
                genre = classes[index]
                confidence = round(prediction[0][index] * 100, 2)

                top_indices = prediction[0].argsort()[-3:][::-1]
                top_predictions = []

                for i in top_indices:
                    top_predictions.append({
                        "genre": classes[i],
                        "confidence": round(prediction[0][i] * 100, 2)
                    })

                spectrogram = create_spectrogram(file_path, filename)
                waveform = create_waveform(file_path, filename)

                con = sqlite3.connect(DB_PATH)
                cur = con.cursor()

                cur.execute(
                    """
                    INSERT INTO history(username, filename, genre, confidence)
                    VALUES (?, ?, ?, ?)
                    """,
                    (
                        session["username"],
                        filename,
                        genre,
                        confidence
                    )
                )

                con.commit()
                con.close()

                return render_template(
                    "result.html",
                    filename=filename,
                    genre=genre,
                    confidence=confidence,
                    audio_file=filename,
                    spectrogram=spectrogram,
                    waveform=waveform,
                    top_predictions=top_predictions
                )

            except Exception as e:
                print("Prediction error:", e)

                return render_template(
                    "result.html",
                    filename=filename,
                    genre="Invalid Audio File",
                    confidence=0,
                    audio_file=None,
                    spectrogram=None,
                    waveform=None,
                    top_predictions=[]
                )

    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()

    cur.execute(
        "SELECT COUNT(*), AVG(confidence) FROM history WHERE username=?",
        (session["username"],)
    )

    stats = cur.fetchone()

    cur.execute(
        """
        SELECT genre
        FROM history
        WHERE username=?
        ORDER BY id DESC
        LIMIT 1
        """,
        (session["username"],)
    )

    last = cur.fetchone()
    con.close()

    total_predictions = stats[0] if stats[0] else 0
    average_confidence = round(stats[1], 2) if stats[1] else 0
    last_genre = last[0] if last else "No prediction yet"

    return render_template(
        "dashboard.html",
        username=session["username"],
        total_predictions=total_predictions,
        average_confidence=average_confidence,
        last_genre=last_genre
    )


@app.route("/history")
def history():
    if "username" not in session:
        return redirect(url_for("login"))

    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()

    cur.execute(
        """
        SELECT filename, genre, confidence
        FROM history
        WHERE username=?
        """,
        (session["username"],)
    )

    records = cur.fetchall()
    con.close()

    return render_template("history.html", records=records)


@app.route("/delete_history")
def delete_history():
    if "username" not in session:
        return redirect(url_for("login"))

    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()

    cur.execute(
        "DELETE FROM history WHERE username=?",
        (session["username"],)
    )

    con.commit()
    con.close()

    return redirect(url_for("history"))


@app.route("/download_report/<filename>/<genre>/<confidence>")
def download_report(filename, genre, confidence):
    report_path = os.path.join(BASE_DIR, "static", "prediction_report.txt")

    with open(report_path, "w") as file:
        file.write("Music Genre Classification Report\n")
        file.write("--------------------------------\n")
        file.write(f"Uploaded File: {filename}\n")
        file.write(f"Predicted Genre: {genre}\n")
        file.write(f"Confidence Score: {confidence}%\n")

    return send_file(report_path, as_attachment=True)


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


if __name__ == "__main__":
    init_db()
    print("Starting Music Genre Website...")
    app.run(debug=False, use_reloader=False)