import streamlit as st
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend
import matplotlib.pyplot as plt
matplotlib.use('Agg')  # Use a non-interactive backend
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Streamlit app
st.title("Simple Linear Regression App")

st.write("""
Upload a CSV file, select the feature and target columns, 
and see the regression line with predictions.
""")

# File upload
uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file:
    # Load data
    data = pd.read_csv(uploaded_file)
    st.write("Data Preview:")
    st.write(data.head())

    # Column selection
    feature_column = st.selectbox("Select the feature column (X):", data.columns)
    target_column = st.selectbox("Select the target column (Y):", data.columns)

    if st.button("Run Regression"):
        try:
            # Prepare data
            X = data[[feature_column]]
            y = data[target_column]

            # Fit linear regression model
            model = LinearRegression()
            model.fit(X, y)

            # Predictions
            y_pred = model.predict(X)

            # Display results
            st.write("Regression Coefficients:")
            st.write(f"Intercept: {model.intercept_}")
            st.write(f"Coefficient: {model.coef_[0]}")
            st.write(f"Mean Squared Error: {mean_squared_error(y, y_pred):.4f}")

            # Plot
            fig, ax = plt.subplots()
            ax.scatter(X, y, color='blue', label='Data points')
            ax.plot(X, y_pred, color='red', label='Regression line')
            ax.set_xlabel(feature_column)
            ax.set_ylabel(target_column)
            ax.legend()
            st.pyplot(fig)

        except Exception as e:
            st.error(f"An error occurred: {e}")
