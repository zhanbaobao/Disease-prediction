import numpy as np
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import joblib

model = load_model('/Data4TB/zhan/mimiciv_diabete/diabete.h5')

scaler = joblib.load('/Data4TB/zhan/mimiciv_diabete/scaler.pkl')  

def predict_icd_code(glucose, age, gender, BMI, weight, height):
    X_new = np.array([glucose, age, gender, BMI, weight, height]).reshape(1, -1)
    X_new_scaled = scaler.transform(X_new)
    X_new_scaled = np.expand_dims(X_new_scaled, axis=2)
    prediction = model.predict(X_new_scaled)
    predicted_icd_code = np.argmax(prediction, axis=1)
    return predicted_icd_code

glucose = float(input("Enter glucose values: "))
age = float(input("Enter age: "))
gender = int(input("Enter gender (0 for male, 1 for female): "))
BMI = float(input("Enter BMI: "))
weight = float(input("Enter weight (in kilograms): "))
height = float(input("Enter height (in centimeters): "))

predicted_icd_code = predict_icd_code(glucose, age, gender, BMI, weight, height)
if predicted_icd_code == 0:
    print("Diabetes: No")
else:
    print("Diabetes: Yes")
