# HEAR_IT

**HEAR_IT** is a voice emotion recognition system that identifies emotional states from speech input. The system uses machine learning and audio processing techniques to classify emotions such as happy, sad, angry, calm, and neutral based on voice characteristics. Developed as a final-year college project in collaboration with others, the project aims to provide real-time emotion classification, making it suitable for applications such as virtual assistants, customer service bots, and mental health applications.

## Description
HEAR_IT leverages various machine learning and audio processing techniques to classify emotions in speech input. The system provides a simple interface for users to upload voice samples and receive emotion predictions in real-time. This project was built using Python and libraries like Flask, TensorFlow, and Librosa.

### Key Features:
- Emotion classification based on speech input.
- Supports multiple emotions including happy, sad, angry, calm, and neutral.
- Real-time audio processing and classification.
- A simple Flask web interface to interact with the system.

## Getting Started

To get started with the HEAR_IT project, follow the instructions below to clone the repository and set up your environment.

### Clone the repository:
```bash
git clone https://github.com/harishnaidugaddam/HEAR_IT_FinalVersion.git
```

### Navigate to the project directory:
```bash
cd HEAR_IT_FinalVersion
```

### Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Installation

- Ensure that Python 3.x is installed on your system.
- Install necessary dependencies using the `requirements.txt` file.
- Set up a virtual environment to avoid conflicts with other projects:
  ```bash
  python -m venv venv
  source venv/bin/activate  # On Windows use venv\Scripts\activate
  ```

### Dependencies:
- Flask
- numpy
- scipy
- librosa
- tensorflow (or other machine learning libraries)
- pyaudio

## Features

- **Emotion Detection**: The model detects various emotions like angry, calm, happy, sad, and neutral from speech.
- **Web Interface**: The Flask application allows users to upload an audio file and receive emotion predictions.
- **Real-Time Processing**: Audio files are processed in real-time, providing instant emotion classification.

## Files and Directories

- **`app.py`**: The main Flask application that serves the web interface.
- **`SVC.py`**: Script for training and evaluating the Support Vector Classifier (SVC) model for emotion recognition.
- **`convert_wavs.py`**: Helper script to convert audio files to a suitable format for processing.
- **`test.py`**: Testing script for validating the model’s performance.
- **`ser.py`**: Emotion classification using the trained model.
- **`utils.py`**: Utility functions for data preprocessing.
- **`data/`**: Directory containing training data files.
- **`templates/`**: HTML files for the Flask web application interface.
- **`static/`**: Static files such as images or CSS for the web app.

## Usage

1. Start the Flask app:
   ```bash
   python app.py
   ```

2. Navigate to `http://127.0.0.1:5000/` in your web browser.
3. Upload a `.wav` file containing a sample of speech.
4. The system will classify the emotion of the speaker based on the uploaded audio file.

## Contributing

If you'd like to contribute to HEAR_IT, follow these steps:

1. Fork the repository.
2. Create a new branch for your feature.
3. Make your changes and commit them.
4. Push your changes to your fork.
5. Create a pull request to the original repository.

## Acknowledgements

- The development of this project is based on research and studies in emotion recognition using speech.
- Special thanks to the collaborators for their contributions to the project.
