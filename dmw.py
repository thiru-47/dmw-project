import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="DMW Dashboard", layout="wide")

# ---------------- CUSTOM STYLE ----------------
st.markdown("""
    <style>
    .main {
        background-color: #0E1117;
        color: white;
    }
    h1, h2, h3 {
        color: #00C9A7;
    }
    </style>
""", unsafe_allow_html=True)

# ---------------- SIDEBAR ----------------
st.sidebar.title("📌 Dashboard")
option = st.sidebar.radio("Navigation", 
["🏠 Home", "📂 Dataset", "📊 Visualization", "🤖 Model", "🎯 Prediction"])

# ---------------- LOAD DATA ----------------
df = pd.read_csv("Student_Marks.csv")

# Preprocessing
df['time_study'] = df['time_study'].fillna(df['time_study'].mean())
df['Marks'] = df['Marks'].fillna(df['Marks'].mean())

# Model
X = df[['number_courses', 'time_study']]
y = df['Marks']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = LinearRegression()
model.fit(X_train, y_train)
pred = model.predict(X_test)

# ---------------- HOME ----------------
if option == "🏠 Home":
    st.title("🎓 Student Performance Dashboard")

    st.markdown("### 📌 Project Overview")
    st.info("Predict student marks using Machine Learning (Linear Regression).")

    col1, col2, col3 = st.columns(3)

    col1.metric("📊 Data Size", len(df))
    col2.metric("📚 Features", len(df.columns))
    col3.metric("🎯 Accuracy (R2)", round(r2_score(y_test, pred), 2))

# ---------------- DATASET ----------------
elif option == "📂 Dataset":
    st.title("📂 Dataset Analysis")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Preview")
        st.dataframe(df)

    with col2:
        st.subheader("Statistics")
        st.write(df.describe())

# ---------------- VISUALIZATION ----------------
elif option == "📊 Visualization":
    st.title("📊 Data Visualization")

    # Scatter Plot
    fig1, ax1 = plt.subplots()
    ax1.scatter(df['time_study'], df['Marks'])
    ax1.set_title("Study Time vs Marks")
    ax1.set_xlabel("Study Time")
    ax1.set_ylabel("Marks")
    st.pyplot(fig1)

    # Heatmap
    st.subheader("🔥 Correlation Heatmap")
    fig2, ax2 = plt.subplots()
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm", ax=ax2)
    st.pyplot(fig2)

# ---------------- MODEL ----------------
elif option == "🤖 Model":
    st.title("🤖 Model Performance")

    mse = mean_squared_error(y_test, pred)
    r2 = r2_score(y_test, pred)

    col1, col2 = st.columns(2)

    col1.metric("📉 MSE", round(mse, 2))
    col2.metric("📊 R2 Score", round(r2, 2))

    st.success("✅ Model trained successfully!")

    # Actual vs Predicted
    st.subheader("Actual vs Predicted")
    fig3, ax3 = plt.subplots()
    ax3.scatter(y_test, pred)
    ax3.set_xlabel("Actual")
    ax3.set_ylabel("Predicted")
    st.pyplot(fig3)

# ---------------- PREDICTION ----------------
elif option == "🎯 Prediction":
    st.title("🎯 Predict Marks")

    courses = st.slider("📚 Number of Courses", 1, 10, 4)
    study_time = st.slider("⏱ Study Time (hours)", 0.0, 10.0, 5.0)

    # Buttons
    col1, col2 = st.columns(2)

    with col1:
        predict_btn = st.button("🎯 Predict")

    with col2:
        save_btn = st.button("💾 Predict & Save")

    # Predict Only
    if predict_btn:
        user_data = pd.DataFrame([[courses, study_time]],
                                 columns=['number_courses', 'time_study'])

        predicted_marks = model.predict(user_data)[0]
        st.success(f"📊 Predicted Marks: {predicted_marks:.2f}")

    # Predict + Save
    if save_btn:
        user_data = pd.DataFrame([[courses, study_time]],
                                 columns=['number_courses', 'time_study'])

        predicted_marks = model.predict(user_data)[0]

        st.success(f"📊 Predicted Marks: {predicted_marks:.2f}")

        # Create new row
        new_row = pd.DataFrame([[courses, study_time, predicted_marks]],
                               columns=['number_courses', 'time_study', 'Marks'])

        # Append and save
        df_updated = pd.concat([df, new_row], ignore_index=True)
        df_updated.to_csv("Student_Marks.csv", index=False)

        st.info("✅ Data saved successfully!")

        # Show last rows
        st.subheader("📂 Updated Dataset (Last Entries)")
        st.write(df_updated.tail())

# ---------------- FOOTER ----------------
st.markdown("---")
st.markdown("✨ Developed by Thirunarayanan K | DMW Project")