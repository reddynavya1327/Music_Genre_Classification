Music Genre Classification Using Deep Learning

Introduction
Music Genre Classification is a Deep Learning and Audio Processing project that predicts the genre of a music file automatically. The system analyzes audio signals using Machine Learning techniques and classifies songs into genres such as Rock, Jazz, HipHop, Pop, Classical, Blues, Country, Disco, Metal, and Reggae. The project also provides a web-based interface where users can upload audio files and view prediction results with confidence scores and visualizations.

Problem Statement
With the huge growth of digital music platforms, manually organizing songs based on genres becomes difficult and time-consuming. Music streaming applications require automatic genre classification systems to improve music recommendation, playlist generation, and audio organization. Traditional manual classification methods are inefficient and less scalable. Therefore, an intelligent system is required to automatically detect music genres from audio files.

Proposed Solution
The proposed system uses a Convolutional Neural Network (CNN) model to classify music genres from WAV audio files. Audio features are extracted using MFCC (Mel Frequency Cepstral Coefficients), which capture important sound characteristics. The extracted features are passed into a trained CNN model that predicts the genre with confidence scores. The system also generates spectrogram and waveform visualizations for better audio analysis.


The project includes:
* User Login and Registration
* Audio Upload System
* Genre Prediction with Confidence Score
* Top 3 Genre Predictions
* Audio Player
* Spectrogram Visualization
* Waveform Visualization
* Prediction History
* Downloadable Prediction Report
  
Technologies Used

 Technology         - Purpose                                 
 Python             - Core programming language               
 Flask              - Web application framework               
 TensorFlow / Keras - Deep Learning model development         
 Librosa            - Audio processing and feature extraction 
 NumPy              - Numerical operations                    
 Matplotlib         - Spectrogram and waveform visualization  
 SQLite             - Database management                     
 HTML/CSS           - Frontend user interface                 

How the Project Works

Step 1: Audio Upload
The user uploads a WAV music file through the website dashboard.

Step 2: Feature Extraction
The system uses Librosa to extract MFCC audio features from the uploaded file.

Step 3: Deep Learning Prediction
The extracted features are passed into the trained CNN model. The model analyzes audio patterns and predicts the most suitable music genre.

Step 4: Result Generation
The system displays:

* Predicted Genre
* Confidence Score
* Top 3 Predictions
* Audio Player
* Spectrogram Graph
* Waveform Graph

Step 5: History Storage
Prediction details are stored in the SQLite database for future reference.

Step 6: Report Download
Users can download the prediction report containing file details, genre, and confidence score.



Advantages of the Project

* Automatic music classification
* Fast and accurate predictions
* User-friendly web interface
* Audio visualization support
* Useful for music streaming platforms
* Reduces manual effort in organizing music



Applications
* Music Streaming Platforms
* Audio Recommendation Systems
* Smart Playlist Generation
* Digital Music Libraries
* Audio Analysis Systems


Conclusion

The Music Genre Classification project demonstrates the application of Deep Learning and Audio Signal Processing in real-world music analysis. By combining CNN models with MFCC feature extraction, the system can accurately classify music genres and provide interactive visualizations through a web-based platform.
