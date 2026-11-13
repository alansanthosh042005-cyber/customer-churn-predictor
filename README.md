# Customer Churn Prediction using Machine Learning

## 📑 Table of Contents
1. [Project Overview](#project-overview)
2. [Dataset Description](#dataset-description)
3. [Methodology & Workflow](#methodology--workflow)
4. [Machine Learning Models](#machine-learning-models)
5. [Technologies & Libraries](#technologies--libraries)
6. [Project Structure](#project-structure)
7. [Installation & Usage](#installation--usage)
8. [Results & Evaluation](#results--evaluation)

---

## 📖 Project Overview
Customer churn is a critical challenge for subscription-based businesses. Acquiring a new customer is significantly more expensive than retaining an existing one. This project encompasses a complete, end-to-end Machine Learning pipeline designed to analyze customer data and predict the likelihood of a customer churning (canceling their subscription). 

By leveraging various classification algorithms and hyperparameter tuning techniques, this project identifies key behavioral and demographic patterns that contribute to customer attrition, providing actionable insights for retention strategies.

---

## 📊 Dataset Description
The project utilizes a proprietary telecommunications/subscription dataset (`Churn_Data.csv`) containing 5,000 customer records. 

**Features:**
* `CustomerID`: Unique identifier (Dropped during preprocessing as it holds no predictive value).
* `Age`: Age of the customer.
* `Gender`: Gender of the customer (Categorical).
* `Location`: Geographical location of the customer (Categorical).
* `SubscriptionType`: Type of subscription plan (Basic, Standard, Premium).
* `ContractLength`: Duration of the contract (Monthly, 1-Year, 2-Year).
* `MonthlyBill`: The monthly billing amount.
* `TotalDataUsageGB`: Total data consumed in Gigabytes.
* `CustomerSupportCalls`: Number of times the customer contacted support.

**Target Variable:**
* `Churn`: Whether the customer churned (`Yes`) or retained (`No`).

---

## 🛠️ Methodology & Workflow
The machine learning pipeline is structured into the following sequential phases:

1. **Data Ingestion & Inspection:** Loading the dataset and performing initial inspections using `.info()`, `.head()`, and checking for missing values and duplicates.
2. **Data Preprocessing:** 
   * Dropping irrelevant identifier columns (`CustomerID`).
   * Encoding categorical variables (`Gender`, `Location`, `SubscriptionType`, `ContractLength`) into numerical formats using `LabelEncoder`.
   * Mapping the target variable (`Churn`) to binary values (1 for 'Yes', 0 for 'No').
3. **Exploratory Data Analysis (EDA):** Generating boxplots to detect outliers and correlation heatmaps to understand feature relationships.
4. **Feature Scaling:** Applying `StandardScaler` to normalize numerical features, ensuring optimal convergence for distance-based and gradient-based algorithms.
5. **Data Splitting:** Dividing the dataset into training (80%) and testing (20%) sets using `train_test_split`.
6. **Model Training & Evaluation:** Training multiple baseline classification models and evaluating them using standard metrics.
7. **Hyperparameter Tuning:** Utilizing `GridSearchCV` with 5-fold cross-validation to find the optimal parameters for a K-Nearest Neighbors (KNN) classifier.
8. **Inference:** Making custom predictions on newly scaled, unseen data points.

---

## 🤖 Machine Learning Models
The following classification algorithms were implemented and compared:
* **Logistic Regression:** Used as a baseline linear model for binary classification.
* **Decision Tree Classifier:** A non-linear model capable of capturing complex, hierarchical decision rules.
* **Support Vector Machine (SVM):** Utilized for finding the optimal hyperplane to separate churned and non-churned customers.
* **K-Nearest Neighbors (KNN):** Implemented with `GridSearchCV` to optimize the number of neighbors, distance weights, and distance metrics (Euclidean vs. Manhattan).

---

## 💻 Technologies & Libraries
This project is built using Python and the following core libraries:
* **Data Manipulation:** `pandas`, `numpy`
* **Data Visualization:** `matplotlib`, `seaborn`
* **Machine Learning:** `scikit-learn` (Modeling, Preprocessing, Evaluation, and Tuning)

---

## 📁 Project Structure
```text
├── Churn_Data.csv          # Raw dataset
├── churn_prediction.py     # Main Python script
└── README.md               # Project documentation
