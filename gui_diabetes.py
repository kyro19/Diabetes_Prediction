import tkinter as tk
from tkinter import messagebox
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Load the dataset
data = pd.read_csv('diabetes.csv')

# Create the GUI
window = tk.Tk()
window.title("Diabetes Prediction")

# Function to perform prediction
def predict_diabetes():
    # Load user input values
    pregnancies = float(pregnancies_entry.get())
    glucose = float(glucose_entry.get())
    blood_pressure = float(blood_pressure_entry.get())
    skin_thickness = float(skin_thickness_entry.get())
    insulin = float(insulin_entry.get())
    bmi = float(bmi_entry.get())
    dpf = float(dpf_entry.get())
    age = float(age_entry.get())

    # Data Preprocessing
    data['Glucose'].fillna(data['Glucose'].mean(), inplace=True)
    data['BloodPressure'].fillna(data['BloodPressure'].mean(), inplace=True)
    data['SkinThickness'].fillna(data['SkinThickness'].median(), inplace=True)
    data['Insulin'].fillna(data['Insulin'].median(), inplace=True)
    data['BMI'].fillna(data['BMI'].median(), inplace=True)

    # Data Normalization
    sc_X = StandardScaler()
    X = pd.DataFrame(sc_X.fit_transform(data.drop(["Outcome"], axis=1)),
                 columns=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'])
    Y = data['Outcome']

    # Prepare the input for prediction
    input_data = [[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]]
    input_data_scaled = sc_X.transform(input_data)

    # Load the trained model
    model = RandomForestClassifier(n_estimators=200)
    model.fit(X, Y)

    # Perform prediction
    prediction = model.predict(input_data_scaled)

    if prediction[0]==1:
        result = " Diabetic"
    else:
        result = " Non Diabetic"
    # Display the prediction
    messagebox.showinfo("Diabetes Prediction Result", f"The predicted outcome is: {prediction[0]}-{result}")

# Create input labels and entry fields
pregnancies_label = tk.Label(window, text="Pregnancies:")
pregnancies_entry = tk.Entry(window)
pregnancies_label.grid(row=0, column=0, padx=10, pady=5)
pregnancies_entry.grid(row=0, column=1, padx=10, pady=5)

glucose_label = tk.Label(window, text="Glucose:")
glucose_entry = tk.Entry(window)
glucose_label.grid(row=1, column=0, padx=10, pady=5)
glucose_entry.grid(row=1, column=1, padx=10, pady=5)

blood_pressure_label = tk.Label(window, text="Blood Pressure:")
blood_pressure_entry = tk.Entry(window)
blood_pressure_label.grid(row=2, column=0, padx=10, pady=5)
blood_pressure_entry.grid(row=2, column=1, padx=10, pady=5)

skin_thickness_label = tk.Label(window, text="Skin Thickness:")
skin_thickness_entry = tk.Entry(window)
skin_thickness_label.grid(row=3, column=0, padx=10, pady=5)
skin_thickness_entry.grid(row=3, column=1, padx=10, pady=5)

insulin_label = tk.Label(window, text="Insulin:")
insulin_entry = tk.Entry(window)
insulin_label.grid(row=4, column=0, padx=10, pady=5)
insulin_entry.grid(row=4, column=1, padx=10, pady=5)

bmi_label = tk.Label(window, text="BMI:")
bmi_entry = tk.Entry(window)
bmi_label.grid(row=5, column=0, padx=10, pady=5)
bmi_entry.grid(row=5, column=1, padx=10, pady=5)

dpf_label = tk.Label(window, text="Diabetes Pedigree Function:")
dpf_entry = tk.Entry(window)
dpf_label.grid(row=6, column=0, padx=10, pady=5)
dpf_entry.grid(row=6, column=1, padx=10, pady=5)

age_label = tk.Label(window, text="Age:")
age_entry = tk.Entry(window)
age_label.grid(row=7, column=0, padx=10, pady=5)
age_entry.grid(row=7, column=1, padx=10, pady=5)

# Create the predict button
predict_button = tk.Button(window, text="Predict", command=predict_diabetes)
predict_button.grid(row=8, column=0, columnspan=2, padx=10, pady=10)

# Start the GUI event loop
window.mainloop()