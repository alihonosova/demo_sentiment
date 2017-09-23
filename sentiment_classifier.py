__author__ = 'xead'
from sklearn.externals import joblib

class SentimentClassifier(object):
    def __init__(self):
        self.model = joblib.load("./models/LogisticReg.pkl")
        self.count_vect = joblib.load("./models/CountVect.pkl")
        self.tfidf = joblib.load("./models/TFIDFTransf.pkl")
        self.trunc = joblib.load("./models/SVDTrunc.pkl")
        self.classes_dict = {0: "negative", 1: "positive", -1: "prediction error"}

    @staticmethod
    def get_probability_words(probability):
        if probability < 0.55:
            return "neutral or uncertain"
        if probability < 0.7:
            return "probably"
        if probability > 0.95:
            return "certain"
        else:
            return ""

    def predict_text(self, text):
        try:
            vectorized = self.count_vect.transform([text])
            transformed = self.tfidf.transform(vectorized)
            truncated = self.trunc.transform(transformed)
            return self.model.predict(truncated)[0],\
                   self.model.predict_proba(truncated)[0].max(),\
                   self.model.predict_proba(truncated)[0]
        except:
            print("prediction error")
            return -1, 0.8

    def predict_list(self, list_of_texts):
        try:
            vectorized = self.count_vect.transform(list_of_texts)
            transformed = self.tfidf.transform(vectorized)
            truncated = self.trunc.transform(transformed)
            return self.model.predict(vectorized),\
                   self.model.predict_proba(vectorized)
        except:
            print('prediction error')
            return None

    def get_prediction_message(self, text):
        prediction = self.predict_text(text)
        class_prediction = prediction[0]
        prediction_probability = prediction[1]
        prediction_positive = prediction[2]
        return [self.get_probability_words(prediction_probability) + " " + self.classes_dict[class_prediction],
                str(round(prediction_positive[1],2)),
                str(round(1-prediction_positive[1],2))]
