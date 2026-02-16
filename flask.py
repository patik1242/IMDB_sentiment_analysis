import flask, pickle
from preprocessing import preprocess_vector

app = flask.Flask(__name__, template_folder='templates')
vectorizer = None 

with open("models/model.pkl", "rb") as f:
    model = pickle.load(f)

with open("models/vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f) 

@app.route('/', methods=['GET', 'POST'])
def index():
    sentiment = None

    if flask.request.method =="POST":
        text = flask.request.form["sentiment"]
        clean_text = preprocess_vector(text)
        
        X_vec = vectorizer.transform([clean_text])

        pred = model.predict(X_vec)

        sentiment = "Positive" if pred[0]==1 else "Negative"

    return flask.render_template("index.html", sentiment=sentiment)

if __name__ == "__main__":
    app.run(debug=True)