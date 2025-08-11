import joblib
import numpy as np

model = joblib.load('model/model.joblib')

def predict(input_data):
    input_array = np.array(input_data)
    prediction = model.predict(input_array)
    return prediction.tolist()
