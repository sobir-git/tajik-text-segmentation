import numpy as np


class DummyModel:
    def predict(self, tokens):
        # produce a random binary 2-d array
        prediction = np.random.uniform(0,1,(len(tokens), 2)) > 0.8
        return prediction

class DummyPredictor:
    def predict(self, tokens):
        # produce a random binary 2-d array
        probs = np.random.uniform(0,1,(len(tokens), 2))
        preds = probs > 0.8
        return {'probs': probs, 'preds': preds}