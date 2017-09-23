__author__ = 'xead'
from sentiment_classifier import SentimentClassifier
from codecs import open
import time
from flask import Flask, render_template, redirect, request, jsonify 
app = Flask(__name__)
contextss = [""]

print("Preparing classifier")
start_time = time.time()
classifier = SentimentClassifier()
print("Classifier is ready")
print(time.time() - start_time, "seconds")

@app.route('/', methods=['GET', 'POST'])
def index_page(text="", prediction_message=""):
    if request.method == "POST":
        text = request.form['text']
    prediction_message = classifier.get_prediction_message([text])
    return render_template('sentiment.html', text=text, prediction_message = prediction_message[0],
                                                             positive = prediction_message[1],
                                                             negative = prediction_message[2])                                                                                                            

if __name__ == "__main__":
    app.run()#(host="0.0.0.0", port="1995", threaded=True)
