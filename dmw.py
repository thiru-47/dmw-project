import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.cluster import KMeans

st.set_page_config(page_title="Student Performance App", layout="wide")

st.title("📊 Student Performance Prediction System")

# =========================
# DATA LOADING
# =========================
df = pd.read_csv("Student_Marks.csv")

st.subheader("📂 Dataset Preview")
st.write(df.head())

# =========================
# DATA PREPROCESSING
# =========================
st.subheader("🧹 Data Preprocessing")

df['time_study'] = df['time_study'].fillna(df['time_study'].mean())
df['Marks'] = df['Marks'].fillna(df['Marks'].mean())

st.write("Null Values After Cleaning:")
st.write(df.isnull().sum())

# =========================
# VISUALIZATION
# =========================
st.subheader("📊 Data Visualization")

fig1, ax1 = plt.subplots()
ax1.scatter(df['time_study'], df['Marks'])
ax1.set_xlabel("Study Time")
ax1.set_ylabel("Marks")
ax1.set_title("Study Time vs Marks")
st.pyplot(fig1)

# =========================
# ML MODEL
# =========================
st.subheader("🤖 Model Training")

X = df[['number_courses', 'time_study']]
y = df['Marks']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = LinearRegression()
model.fit(X_train, y_train)

pred = model.predict(X_test)

mse = mean_squared_error(y_test, pred)
r2 = r2_score(y_test, pred)

st.write(f"📉 MSE: {mse:.2f}")
st.write(f"📈 R² Score: {r2:.2f}")

# =========================
# ACTUAL VS PREDICTED
# =========================
fig2, ax2 = plt.subplots()
ax2.scatter(y_test, pred)
ax2.set_xlabel("Actual Marks")
ax2.set_ylabel("Predicted Marks")
ax2.set_title("Actual vs Predicted")
st.pyplot(fig2)

# =========================
# PREDICTION
# =========================
st.subheader("🎯 Predict Marks")

courses = st.number_input("Enter number of courses", min_value=1)
study_time = st.number_input("Enter study time (hours)", min_value=0.0)

if st.button("Predict"):
    user_data = pd.DataFrame([[courses, study_time]],
                             columns=['number_courses', 'time_study'])
    predicted_marks = model.predict(user_data)
    st.success(f"Predicted Marks: {predicted_marks[0]:.2f}")

# SAVE OPTION
if st.button("Predict & Save"):
    user_data = pd.DataFrame([[courses, study_time]],
                             columns=['number_courses', 'time_study'])
    predicted_marks = model.predict(user_data)

    new_row = pd.DataFrame([[courses, study_time, predicted_marks[0]]],
                           columns=['number_courses', 'time_study', 'Marks'])

    df = pd.concat([df, new_row], ignore_index=True)
    df.to_csv("Student_Marks.csv", index=False)

    st.success("✅ Prediction saved to dataset!")

# =========================
# CORRELATION
# =========================
st.subheader("📊 Correlation Analysis")

corr = df.corr()

fig3, ax3 = plt.subplots()
sns.heatmap(corr, annot=True, ax=ax3)
st.pyplot(fig3)

# =========================
# BOXPLOT
# =========================
fig4, ax4 = plt.subplots()
sns.boxplot(data=df, ax=ax4)
ax4.set_title("Boxplot for Outliers")
st.pyplot(fig4)

# =========================
# K-MEANS CLUSTERING
# =========================
st.subheader("📌 K-Means Clustering")

X_cluster = df[['time_study', 'Marks']]

kmeans = KMeans(n_clusters=3)
df['Cluster'] = kmeans.fit_predict(X_cluster)

fig5, ax5 = plt.subplots()
scatter = ax5.scatter(df['time_study'], df['Marks'], c=df['Cluster'])
ax5.set_xlabel("Study Time")
ax5.set_ylabel("Marks")
ax5.set_title("K-Means Clustering")
st.pyplot(fig5)

# =========================
# OLAP OPERATIONS
# =========================
st.subheader("📊 Data Warehouse (OLAP)")

# Roll-up
rollup = df.groupby('number_courses').agg(
    avg_marks=('Marks', 'mean'),
    avg_study=('time_study', 'mean'),
    count=('Marks', 'count')
).reset_index()

st.write("🔼 Roll-up Analysis")
st.write(rollup)

# Drill-down
df['study_bin'] = pd.cut(df['time_study'],
                        bins=[0, 2, 4, 6, 8],
                        labels=['0-2h', '2-4h', '4-6h', '6-8h'])

drilldown = df.groupby(['number_courses', 'study_bin'])['Marks'].mean().unstack()

st.write("🔽 Drill-down Analysis")
st.write(drilldown)

# Slice
slice_df = df[df['time_study'] >= 4]
st.write(f"📌 Slice (study >= 4h): {len(slice_df)} students")
st.write(f"Average Marks: {slice_df['Marks'].mean():.2f}")

# Rollup graph
fig6, ax6 = plt.subplots()
rollup.plot(x='number_courses', y='avg_marks', kind='bar', ax=ax6)
st.pyplot(fig6)

# =========================
# DOWNLOAD UPDATED DATA
# =========================
st.subheader("⬇️ Download Updated Dataset")

st.download_button(
    label="Download CSV",
    data=df.to_csv(index=False),
    file_name="Updated_Student_Marks.csv",
    mime="text/csv"
)

# =========================
# FOOTER
# =========================
st.markdown("---")
st.write("Developed by Thirunarayanan K")