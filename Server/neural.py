from keras.models import load_model
import numpy as np

class Neural:
    def __init__(self):
        self.model = load_model("model.h5", compile=False)
        
    def predict(self, frame):
        preds = int(np.argmax(self.model.predict(np.asarray(frame))))
        print(preds)
        print(len(str(preds)))
        return preds