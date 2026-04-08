# 🌍 Air Quality Analysis and Prediction using Python

### INT375 – Introduction to Python for Data Science Project

---

## 📌 Project Overview

This project focuses on analyzing real-world air pollution data to understand trends, patterns, and factors affecting air quality across different cities in India.

The project applies **data science techniques** including data cleaning, visualization, exploratory data analysis (EDA), statistical analysis, and machine learning to predict the **Air Quality Index (AQI)**.

---

## 🎯 Objectives

* Analyze air pollution levels across multiple cities
* Perform data cleaning and preprocessing
* Handle missing values effectively
* Conduct Exploratory Data Analysis (EDA)
* Visualize data using graphs and charts
* Apply statistical analysis techniques
* Build a machine learning model to predict AQI

---

## 📊 Dataset Information

* **Source:** Kaggle – Air Quality Data in India
* **File Used:** `city_day.csv`
* **Size:** 10,000+ records

### 📌 Features in Dataset

* City
* Date
* AQI (Air Quality Index)
* PM2.5
* PM10
* NO2
* SO2
* CO
* O3

---

## ⚙️ Technologies & Tools Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* Visual Studio Code

---

## 🔄 Project Workflow

### 1️⃣ Data Collection

* Dataset downloaded from Kaggle
* Loaded using Pandas

### 2️⃣ Data Cleaning & Preparation

* Handled missing values
* Converted date formats
* Filtered required columns

### 3️⃣ Data Visualization

* AQI Distribution
* AQI Trends over Time
* City-wise Pollution Levels
* Correlation Heatmap

### 4️⃣ Exploratory Data Analysis (EDA)

* Summary statistics
* Correlation analysis
* Outlier detection

### 5️⃣ Statistical Analysis

* Mean, median, variance
* Distribution analysis
* Relationship between pollutants

### 6️⃣ Machine Learning

* Model: Linear Regression
* Target Variable: AQI
* Features: PM2.5, PM10
* Model evaluation using train-test split

---

## 📈 Key Visualizations

* Histogram of AQI distribution
* Line plot of AQI trends over time
* Bar chart of top polluted cities
* Heatmap showing correlation between pollutants

---

## 🤖 Machine Learning Model

A **Linear Regression model** is used to predict AQI based on pollution parameters such as PM2.5 and PM10.

This helps in understanding how different pollutants contribute to air quality.

---

## 📁 Project Structure

```
INT375_Project/
│
├── project.py          # Main Python code
├── city_day.csv        # Dataset
├── README.md           # Project documentation
├── aqi_distribution.png
├── aqi_trend.png
├── correlation.png
└── top_cities.png
```

---

## 📌 Results & Insights

* AQI varies significantly across cities
* PM2.5 and PM10 have strong impact on AQI
* Some cities consistently show higher pollution levels
* Correlation analysis helps identify key contributing pollutants

---

## 🧠 Learning Outcomes

* Hands-on experience with real-world data
* Understanding of data preprocessing techniques
* Strong foundation in EDA and visualization
* Basic implementation of machine learning
* Improved problem-solving using Python

---

## 🚀 Future Improvements

* Use real-time API data (AQI API)
* Build a web dashboard for visualization
* Apply advanced ML models (Random Forest, XGBoost)
* Deploy project as a web application

---

## 👨‍💻 Author

**Pulikanti Vishnuvardhan Reddy**
B.Tech CSE, Lovely Professional University

---

## ⭐ Acknowledgment

Dataset sourced from Kaggle and inspired by real-world environmental challenges.

---

## 📢 Note

This project is developed as part of the INT375 course to demonstrate practical implementation of data science concepts using Python.

---
