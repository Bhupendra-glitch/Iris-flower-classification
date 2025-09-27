import streamlit as st
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Load dataset
iris = load_iris()
X = iris.data
y = iris.target

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Streamlit UI
st.title("🌸 Iris Flower Classification App")
st.write("Enter the flower features to predict the Iris species:")

# Input sliders
sepal_length = st.slider("Sepal Length (cm)", 4.0, 8.0, 5.0)
sepal_width = st.slider("Sepal Width (cm)", 2.0, 4.5, 3.0)
petal_length = st.slider("Petal Length (cm)", 1.0, 7.0, 4.0)
petal_width = st.slider("Petal Width (cm)", 0.1, 2.5, 1.0)

# Prediction
if st.button("Predict"):
    input_data = [[sepal_length, sepal_width, petal_length, petal_width]]
    prediction = model.predict(input_data)
    predicted_class = iris.target_names[prediction[0]]

    st.success(f"🌼 The predicted Iris species is: **{predicted_class}**")
