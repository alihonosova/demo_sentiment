#!/usr/bin/python

# Flask is required
import numpy as np
import sys

from flask import Flask, render_template, request, redirect
from sklearn.externals import joblib
from sentiment_classifier import SentimentClassifier                                                                                        
app = Flask(__name__)

# Load classifier from file
classifier = SentimentClassifier()         
# Entry point -> Template
@app.route('/')
def index_page():
    return render_template('sentiment.html')

# API for 
@app.route('/predict')
def predict(text=""):
    text = request.args.get("text")
    prediction_message = classifier.get_prediction_message([text])
    
    return ','.join([i for i in prediction_message])

# redirect other requests to server to /-page
@app.route('/<path:path>')
def other_page(path = ''):
    return redirect("/", code=302)

# start Flask application
if __name__ == "__main__":
    port = 8080
    if len(sys.argv) > 1:
      port = int(sys.argv[1])
    app.run(host = '0.0.0.0', port = port, debug = False)
