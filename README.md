# Semantic Analysis and Emotion Detection using Python

## Overview
This Jupyter notebook demonstrates semantic analysis and emotion detection using machine learning techniques in Python. It utilizes Natural Language Processing (NLP) tools and various classifiers to detect emotions from text data. The notebook covers data preprocessing, model training, evaluation, and inference on custom samples.

## Prerequisites
- Python 3.x
- Jupyter Notebook environment
- Libraries:
  - NLTK
  - NumPy
  - Pandas
  - Seaborn
  - Matplotlib
  - Scikit-learn
  - TensorFlow (Keras)
  
## Usage
1. Clone or download the notebook file (`Semantic_analysis_and_emotion_detection.ipynb`).
2. Open the notebook in a Jupyter Notebook environment.
3. Install the necessary libraries if not already installed (`pip install nltk numpy pandas seaborn matplotlib scikit-learn tensorflow`).
4. Execute each cell in the notebook to run the code sequentially.
5. Follow the instructions/comments provided within the notebook.
6. Customize parameters, models, and methods as needed.

## Features
- **Data Preprocessing**:
  - Text cleaning (removing stopwords, punctuation, URLs, numbers, etc.).
  - Lemmatization and lowercasing of text.
  - Handling duplicates and imbalanced data.
- **Feature Engineering**:
  - Tokenization and padding of text sequences.
  - Word embeddings using pre-trained GloVe vectors.
- **Model Training**:
  - Utilizes various classifiers such as Logistic Regression, Decision Tree, Support Vector Machine, and Random Forest.
  - Deep learning model architecture using Bidirectional LSTM layers.
- **Model Evaluation**:
  - Accuracy, F1 score, and classification report.
  - Visualization of loss and accuracy during training.
- **Inference**:
  - Predicting emotions from custom text samples.

## Dataset
- The dataset consists of labeled text samples categorized into different emotions.
- It includes separate files for training, validation, and testing.

## Acknowledgments
- The notebook utilizes pre-trained word embeddings from the GloVe project.
- It also makes use of libraries such as NLTK, TensorFlow, and scikit-learn for NLP and machine learning tasks.

## License
This project is licensed under the [MIT License](LICENSE).
