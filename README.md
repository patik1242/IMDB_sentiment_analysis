🎬 Sentiment Analysis of Movie Reviews
📌 Project Overview

This project focuses on sentiment analysis of movie reviews using machine learning techniques.
The goal is to classify reviews as positive or negative based on their textual content.

The project includes:

data preprocessing and cleaning
feature extraction using TF-IDF
training and evaluation of multiple ML models
statistical comparison of models
visualization of results
deployment as a simple web application (Flask)
📊 Dataset

The model is trained on the IMDB Dataset containing movie reviews and sentiment labels.

Data processing includes:

removing duplicates
handling missing values
cleaning text (HTML removal, normalization)
mapping labels (positive → 1, negative → 0)
⚙️ Data Preprocessing

Text preprocessing is performed in two stages:

🔹 Basic preprocessing
removal of HTML tags
normalization of punctuation
removal of unwanted characters
lowercasing
🔹 Advanced preprocessing
lemmatization using NLTK
token normalization

This ensures consistent input for the ML models

🧠 Feature Engineering

Text data is transformed using:

TF-IDF Vectorization
n-grams: (1,2)
max features: 120,000
filtering rare and overly common words

🤖 Models Used

The project trains and compares multiple classification models:

Logistic Regression
Linear SVM (LinearSVC)
Ridge Classifier
Multinomial Naive Bayes
📈 Model Evaluation

Models are evaluated using:

Accuracy
Precision
Recall
F1-score (main metric)
ROC AUC

Additional analysis includes:

confusion matrix visualization
ROC curves
learning curves
analysis of false positives and false negatives
McNemar statistical test for model comparison
🏆 Model Selection

The best model is selected based on:

highest F1-score
penalty for overfitting

The final model and vectorizer are saved for reuse

🌐 Web Application

A simple Flask app allows users to test the model:

input: movie review
output: predicted sentiment (Positive / Negative)

📂 Project Structure
project/
│
├── data/                     # dataset
├── models/                   # saved model and vectorizer
├── charts/                   # generated plots
├── analysis/                 # error analysis outputs
├── results/                  # evaluation results (JSON)
│
├── main.py                   # entry point
├── main_train.py             # training pipeline
├── clean_data.py             # data cleaning
├── preprocessing.py          # text preprocessing
├── training_and_calculating_metrics.py
├── analysis.py               # evaluation and plots
├── app.py                    # Flask app
├── save_to_json.py           # saving results
├── requirements.txt
🚀 How to Run
1. Install dependencies
pip install -r requirements.txt
2. Train the model
python main.py
3. Run the web app
python app.py

Then open:

http://127.0.0.1:5000
📦 Requirements
Python 3.x
pandas
numpy
scikit-learn
matplotlib
seaborn
nltk
flask
scipy
📌 Key Features

✔ End-to-end ML pipeline
✔ Advanced text preprocessing
✔ Multiple model comparison
✔ Statistical validation (McNemar test)
✔ Visualization of performance
✔ Ready-to-use web interface

🔮 Future Improvements
deep learning models (LSTM / Transformers)
hyperparameter tuning
deployment (Docker / cloud)
real-time sentiment API
support for multiple languages
