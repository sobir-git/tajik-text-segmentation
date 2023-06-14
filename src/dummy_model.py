import numpy as np


class DummyModel:
    def predict(self, tokens):
        # produce a random binary 2-d array
        prediction = np.random.uniform(0,1,(len(tokens), 2)) > 0.8
        return prediction

        