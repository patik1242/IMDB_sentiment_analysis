import pickle
from flask import Flask, request, render_template
from preprocessing import preprocess_vector

app = Flask(__name__, template_folder='templates')
vectorizer = None 

with open("models/model.pkl", "rb") as f:
    model = pickle.load(f)

with open("models/vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f) 

@app.route('/', methods=['GET', 'POST'])
def index():
    sentiment = None
    text = ""
    if request.method =="POST":
        text = request.form.get("sentiment", "")

        if text.strip():
            clean_text = preprocess_vector(text)
            X_vec = vectorizer.transform([clean_text])
            pred = model.predict(X_vec)
            sentiment = "Positive" if pred[0]==1 else "Negative"

    return render_template("index.html", sentiment=sentiment, text=text)

if __name__ == "__main__":
    app.run(debug=True)