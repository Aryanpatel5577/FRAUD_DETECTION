## ⚠️ **NOTE**

> **To download the `creditcard.csv` dataset required for this project, please follow this link:**
> 🔗 **[https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)**
# CREDIT CARD FRAUD DETECTION

### Exploratory Data Analysis, Feature Engineering & Machine Learning

A Python-based machine learning project for analyzing credit-card transactions and identifying transactions that may be fraudulent. The project combines exploratory data analysis, feature engineering, imbalanced classification, model comparison, threshold tuning, and automated transaction flagging.

---

## 📌 Overview

Credit-card fraud detection is a challenging machine learning problem because fraudulent transactions represent only a very small portion of all transactions.

This project explores a large transaction dataset, analyzes patterns between legitimate and fraudulent transactions, engineers additional features, trains multiple classification models, evaluates them using metrics appropriate for imbalanced data, and generates a list of transactions flagged as potentially fraudulent.

The project focuses not only on model training, but also on understanding the data and evaluating how different modeling decisions affect fraud detection.

---

## 🎯 Objectives

The main objectives of this project are to:

* Analyze credit-card transaction data using exploratory data analysis.
* Understand the imbalance between legitimate and fraudulent transactions.
* Examine transaction amounts and time-based patterns.
* Create additional features that may help distinguish unusual transactions.
* Prepare and scale data for machine learning.
* Train and compare multiple classification models.
* Evaluate models using precision, recall, F1-score, Average Precision, ROC-AUC, and confusion matrices.
* Analyze Precision-Recall curves for an imbalanced classification problem.
* Tune the classification threshold based on F1-score.
* Generate fraud scores for transactions.
* Flag transactions that cross the selected fraud threshold.
* Save the trained model and flagged transactions for future use.

---

## 🧠 Problem Statement

The task is to classify each transaction into one of two classes:

| Class | Meaning                |
| ----- | ---------------------- |
| `0`   | Legitimate transaction |
| `1`   | Fraudulent transaction |

This is a **binary classification** problem.

A major challenge is the severe class imbalance in the dataset. The notebook contains **284,807 transactions**, of which **284,315 are legitimate and 492 are fraudulent**, making fraud approximately **0.173%** of the dataset.

Because fraudulent transactions are so rare, accuracy alone would not be a reliable measure of model performance.

---

## 📊 Dataset

The project expects a file named:

```text
creditcard.csv
```

The dataset contains **284,807 rows and 31 columns**.

The available fields include:

* `Time`
* `V1` through `V28`
* `Amount`
* `Class`

The `Class` column is used as the target variable.

The notebook also checks the dataset for missing values and reports zero missing values across the displayed columns.

> **Note:** The notebook itself does not document the external source of the dataset. If the dataset source is added later, it should be credited here.

---

## 🔎 Exploratory Data Analysis

Before training machine learning models, the project examines the structure and characteristics of the dataset.

### Data inspection

The project checks:

* Number of rows and columns
* Data types
* Missing values
* Target distribution
* Transaction amounts
* Time-based patterns

The initial dataset inspection confirms that the transaction features are numerical and that `Class` is an integer target variable.

### Class distribution

The project analyzes the number of legitimate and fraudulent transactions to understand the imbalance in the target variable.

This is particularly important because a model can achieve high overall accuracy while still performing poorly on the minority fraud class.

### Transaction amount analysis

The distribution of transaction amounts is visualized to understand:

* Typical transaction sizes
* Large transactions
* Potential outliers
* Differences between legitimate and fraudulent transactions

### Time analysis

The original `Time` feature is converted into an `Hour` feature to explore transaction activity across different times of day.

### Correlation analysis

A correlation matrix is generated to explore relationships among the numerical variables.

### Fraud and transaction behavior

The project also compares transaction amounts and examines the number of fraudulent transactions across different hours and between day/night periods.

---

## 🛠️ Feature Engineering

The project creates additional features from the existing transaction information.

### `Hour`

The original `Time` feature is transformed into an hour-of-day representation:

```python
Hour = (Time // 3600) % 24
```

This provides a more interpretable representation of transaction timing.

---

### `log_amount`

The transaction amount is transformed using:

```python
log_amount = np.log1p(Amount)
```

This reduces the effect of very large transaction values and provides a compressed representation of the amount.

---

### `amount_to_mean`

A relative transaction-size feature is created by comparing each transaction amount with the overall mean transaction amount.

This provides the model with additional information about how large a transaction is relative to the dataset's average transaction.

---

### `is_high_amount`

The project calculates the 99th percentile of transaction amounts and creates a binary indicator:

```text
0 → Not above the 99th percentile
1 → Above the 99th percentile
```

This allows the model to identify unusually large transactions as a separate signal.

---

### `is_night`

Transactions occurring during selected nighttime hours are represented using a binary feature:

```text
0 → Day
1 → Night
```

This allows the model to consider time-of-day behavior.

---

## ⚙️ Data Preparation

The target variable is separated from the input features:

```text
X → Transaction features
y → Class
```

The data is divided into training and testing sets using an **80/20 split**.

Stratification is used to preserve the relative distribution of the fraud and legitimate classes between the two sets.

A fixed random state is also used to make the split reproducible.

---

## 📏 Feature Scaling

The project uses `RobustScaler` for numerical feature scaling.

Robust scaling is useful when working with data that may contain extreme values because it is less sensitive to outliers than some conventional scaling approaches.

The scaler is fitted using the training data and then applied to the corresponding data transformations.

The raw `Time` feature is removed from the final model inputs while the engineered `Hour` feature is retained.

---

## 🤖 Machine Learning Models

The project is designed to compare multiple classification approaches.

### 1. Logistic Regression

Logistic Regression is used as a baseline classification model.

It provides a relatively simple reference point against which more complex models can be compared.

The model uses:

```python
class_weight="balanced"
```

to give greater consideration to the minority fraud class.

---

### 2. Random Forest

Random Forest is used as an ensemble classification model.

The project configures the model with multiple decision trees and balanced class weights.

Random Forest can capture nonlinear relationships between transaction features and fraud labels.

---

### 3. XGBoost

The notebook contains optional support for XGBoost.

When the package is available, the project can use XGBoost with positive-class weighting to address class imbalance.

The current notebook execution output shows that XGBoost was **not available in the recorded environment**, so it should be treated as an optional component rather than a guaranteed trained model.

---

### LightGBM

The notebook also contains an optional LightGBM import.

However, the recorded execution shows that LightGBM was unavailable, and it is not presented as a successfully trained model in the notebook.

---

## ⚖️ Handling Class Imbalance

Class imbalance is one of the central challenges of this project.

There are far more legitimate transactions than fraudulent transactions.

To address this, the project uses techniques such as:

* `class_weight="balanced"` for applicable models.
* Positive-class weighting for optional XGBoost training.
* Precision and recall analysis.
* F1-score evaluation.
* Average Precision.
* Precision-Recall curves.
* Threshold optimization.

This ensures that model evaluation focuses on the ability to identify the minority fraud class rather than relying solely on overall accuracy.

---

## 📈 Model Evaluation

The project evaluates models using several metrics.

### Precision

Precision measures how many transactions predicted as fraud were actually fraudulent.

```text
Precision =
True Positives /
(True Positives + False Positives)
```

High precision means fewer legitimate transactions are incorrectly flagged.

---

### Recall

Recall measures how many actual fraudulent transactions were successfully identified.

```text
Recall =
True Positives /
(True Positives + False Negatives)
```

Recall is particularly important in fraud detection because missing a fraudulent transaction can be costly.

---

### F1-Score

F1-score balances precision and recall using their harmonic mean.

```text
F1 =
2 × Precision × Recall /
(Precision + Recall)
```

The project uses F1-score during threshold analysis.

---

### Average Precision

Average Precision summarizes performance across different precision-recall thresholds.

This is especially useful for highly imbalanced classification problems.

The project uses Average Precision as an important metric for comparing models.

---

### ROC-AUC

ROC-AUC evaluates how effectively the model separates the two classes across different thresholds.

---

### Confusion Matrix

The confusion matrix provides four important outcomes:

|                       | Predicted Legitimate | Predicted Fraud |
| --------------------- | -------------------: | --------------: |
| **Actual Legitimate** |        True Negative |  False Positive |
| **Actual Fraud**      |       False Negative |   True Positive |

This makes it possible to understand the types of classification errors made by the model.

---

## 📉 Precision-Recall Analysis

The project generates Precision-Recall curves to study the relationship between:

* Precision
* Recall
* Classification threshold

This is particularly relevant because fraudulent transactions are extremely rare in the dataset.

Instead of assuming that a default classification threshold is always optimal, the project investigates how changing the threshold affects fraud detection performance.

---

## 🏆 Model Comparison

Each available model is evaluated using the same evaluation process.

The results are organized and compared using metrics including:

* Average Precision
* ROC-AUC

The models are then ordered by Average Precision to identify the strongest candidate according to the project's chosen comparison metric.

This makes the model-selection process more systematic than choosing an algorithm without evaluating alternatives.

---

## 🎚️ Threshold Optimization

A major part of the project is threshold tuning.

Machine-learning classifiers often produce a score or probability rather than immediately producing a final class.

The project initially considers a default threshold of:

```text
0.5
```

but also searches across thresholds to identify the threshold that produces the strongest F1-score.

Conceptually:

```text
Model Score
     ↓
Classification Threshold
     ↓
┌───────────────┐
│               │
Legitimate    Fraud
```

Changing the threshold creates a trade-off between precision and recall.

This allows the project to consider the practical consequences of different fraud-detection thresholds rather than treating `0.5` as universally optimal.

---

## 🚨 Fraud Flagging

After selecting the best available model and threshold, the project generates a `fraud_score` for transactions.

It then creates:

```text
pred_fraud
```

where:

```text
0 → Not flagged
1 → Flagged as potentially fraudulent
```

Transactions predicted as fraud are extracted and sorted by their fraud score so that higher-scoring transactions can be reviewed first.

---

## 💾 Output Files

The project generates two important output artifacts.

### `best_fraud_model.pkl`

A serialized version of the selected machine-learning model.

This allows the trained model to be saved and reused without retraining it from scratch.

### `flagged_transactions.csv`

A CSV containing transactions that crossed the selected fraud-detection threshold.

The output includes the model's fraud score and predicted fraud label along with the transaction information.

---

## 🧰 Technologies Used

| Technology           | Purpose                                         |
| -------------------- | ----------------------------------------------- |
| **Python**           | Core programming language                       |
| **Pandas**           | Data loading and manipulation                   |
| **NumPy**            | Numerical operations                            |
| **Matplotlib**       | Data visualization                              |
| **Scikit-learn**     | Machine learning, preprocessing, and evaluation |
| **XGBoost**          | Optional gradient-boosting model                |
| **LightGBM**         | Optional model support                          |
| **Joblib**           | Saving the trained model                        |
| **Jupyter Notebook** | Interactive development and analysis            |

The notebook imports the core data-science and machine-learning libraries directly.

---

## 📂 Project Structure

A clean version of the project can be organized as:

```text
CREDIT-CARD-FRAUD-DETECTION/
│
├── creditcard.csv
├── Credit_Card_Fraud_Detection.ipynb
├── best_fraud_model.pkl
├── flagged_transactions.csv
└── README.md
```

### File descriptions

**`creditcard.csv`**
Input transaction dataset.

**`Credit_Card_Fraud_Detection.ipynb`**
Main notebook containing the complete analysis, feature engineering, model training, evaluation, and fraud-flagging workflow.

**`best_fraud_model.pkl`**
Saved trained model.

**`flagged_transactions.csv`**
Transactions flagged by the selected model and threshold.

**`README.md`**
Project documentation.

---

## ▶️ How to Run

### 1. Clone the repository

```bash
git clone <your-repository-url>
cd CREDIT-CARD-FRAUD-DETECTION
```

### 2. Install the required libraries

```bash
pip install numpy pandas matplotlib scikit-learn joblib jupyter
```

Optional model libraries:

```bash
pip install xgboost lightgbm
```

### 3. Place the dataset

Make sure:

```text
creditcard.csv
```

is located in the same directory as the notebook.

### 4. Launch Jupyter Notebook

```bash
jupyter notebook
```

### 5. Open the notebook

Open:

```text
Credit_Card_Fraud_Detection.ipynb
```

and run the cells from top to bottom.

---

## 🔄 Workflow

The complete workflow can be summarized as:

```text
Dataset
   ↓
Data Loading
   ↓
Exploratory Data Analysis
   ↓
Class Imbalance Analysis
   ↓
Feature Engineering
   ↓
Train/Test Split
   ↓
Robust Scaling
   ↓
Model Training
   ↓
Model Evaluation
   ↓
Model Comparison
   ↓
Best Model Selection
   ↓
Threshold Optimization
   ↓
Fraud Scoring
   ↓
Transaction Flagging
   ↓
Saved Model + CSV Output
```

---

## 💡 Key Machine Learning Concepts

This project provided practical experience with:

* Binary classification
* Imbalanced datasets
* Exploratory data analysis
* Feature engineering
* Data preprocessing
* Robust scaling
* Train/test splitting
* Ensemble learning
* Logistic regression
* Random forests
* Gradient boosting


---

## ⚠️ Limitations

This project is designed as a machine-learning analysis and prediction pipeline rather than a production financial fraud-detection system.

Some important limitations include:

* The notebook does not implement a live transaction-processing system.
* It does not integrate with a banking or payment platform.
* Optional XGBoost and LightGBM support depends on whether those packages are installed.
* The current notebook tunes the classification threshold using test-set performance, which can make the final evaluation less independent than a separate validation/test workflow.
* Some engineered statistics are calculated before the train/test split and could be further improved to avoid subtle information leakage.
* The model's fraud score should be interpreted as a model-generated score/probability estimate rather than a guaranteed real-world probability.

These limitations provide clear directions for future improvement.

---

## 🚀 Future Improvements

Potential extensions include:

* Introduce a separate validation set for threshold selection.
* Use cross-validation for more robust model comparison.
* Move feature-engineering statistics inside a leakage-safe preprocessing pipeline.
* Perform systematic hyperparameter tuning.
* Calibrate model probabilities.
* Experiment with additional ensemble methods.
* Add explainability using feature importance or SHAP.
* Build a Streamlit or Flask interface for interactive predictions.
* Create a real-time prediction API.


---

## 📌 What I Learned

Through this project, I explored the complete workflow of a supervised machine-learning problem rather than focusing only on model training.

The most important lesson was that **choosing a model is only one part of building a useful machine-learning system**. Understanding the data, handling class imbalance, engineering meaningful features, selecting appropriate evaluation metrics, and choosing a suitable decision threshold can be equally important.

Working with a highly imbalanced fraud-detection problem also helped me understand why a model with high overall accuracy is not necessarily a good classifier.

---

## 🔐 Responsible Use

This project is intended for educational and experimental purposes.

Fraud detection in real financial systems involves additional requirements such as:

* Data security
* Privacy
* Regulatory compliance
* Human review
* Model monitoring
* Bias evaluation
* Robust validation
* Real-time infrastructure
* Financial risk controls

A machine-learning prediction should therefore be treated as a decision-support signal rather than an unquestionable determination of fraud.

---

## 👤 Author

**Aryan Patel**

A student project exploring machine learning, data analysis, and practical applications of Python.

---

## ⭐ Project Focus

**Data Analysis → Feature Engineering → Imbalanced Classification → Model Comparison → Threshold Optimization → Fraud Detection**

---
