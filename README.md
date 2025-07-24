# 🖐️ Korean Sign Language Recognition

Welcome to a lightweight prototype for recognizing Korean sign language. This repo contains a set of Python scripts built on top of [MediaPipe](https://mediapipe.dev/) and TensorFlow to collect data, train an LSTM model and test it in real time.

---

## Table of Contents
1. [Repository Layout](#repository-layout)
2. [Quick Start](#quick-start)
3. [Data Collection](#data-collection)
4. [Training](#training)
5. [Real‑Time Demo](#real-time-demo)
6. [Notes](#notes)

---

## Repository Layout

- **`modules.py`** – common imports shared by most scripts
- **`folder_setup.py`** – creates the data directory structure and defines the list of signs
- **`action_features_extraction.py`** – captures webcam frames, extracts landmarks and stores them as NumPy arrays
- **`Data_Preprocessing.py`** – loads the saved sequences and splits them for training/testing
- **`LSTM_Model.py`** – defines and trains the LSTM network
- **`realtime_testing.py`** – performs real-time recognition, uses OpenAI API to form sentences, and optionally speaks them
- **`api_chat.py`** – helper functions for OpenAI calls and text-to-speech
- **Other utilities** – drawing and detection helpers using MediaPipe

A small pre-trained model `lstm_model_fin3.h5` is provided for quick experimentation.

---

## Quick Start

1. Install dependencies: `pip install tensorflow opencv-python mediapipe gtts pydub openai`
2. Run `folder_setup.py` to create the `Sign_data_finfin` directory
3. Capture data using `action_features_extraction.py`
4. Process the data with `Data_Preprocessing.py` and train with `LSTM_Model.py`
5. Test live recognition via `realtime_testing.py`

Set your OpenAI API key in `api_chat.py` and `realtime_testing.py` if you want sentence generation and speech synthesis.

---

## Data Collection

Each sign is recorded as a sequence of 30 frames. The scripts save these sequences under `Sign_data_finfin/<SIGN>/<SEQ_NUM>/`. You can adjust `actions`, `number_sequences` and `sequence_length` in `folder_setup.py`.

## Training

`LSTM_Model.py` builds a simple LSTM network using TensorFlow/Keras. After training it saves a model file (`.h5`) that can be loaded for inference.

## Real-Time Demo

`realtime_testing.py` loads the trained model and listens to your webcam. Detected signs are combined into natural sentences using the OpenAI API and may be spoken aloud via `gTTS`.

---

## Notes

This project is meant as a minimal prototype for experimenting with sign language recognition. Feel free to tailor the scripts and model to your own dataset and requirements.
