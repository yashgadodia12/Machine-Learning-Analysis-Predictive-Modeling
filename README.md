# Applied Machine Learning & Predictive Modeling

A collection of end-to-end machine learning analyses covering **classification, regression, and natural language processing**.

The project demonstrates practical ML workflows including exploratory data analysis, preprocessing, feature engineering, model selection, hyperparameter tuning, evaluation, clustering, topic modelling, and visualization using Python and scikit-learn.

---

## Project Overview

This repository contains three machine learning projects built around different types of data and modelling problems:

| Project | Problem | Methods |
|---|---|---|
| Road Accident Severity Prediction | Classification | KNN, Decision Tree, Feature Selection, GridSearchCV |
| Bacterial Growth Prediction | Regression | Linear Regression, Random Forest, Hyperparameter Tuning |
| News Article Topic Discovery | NLP / Unsupervised Learning | TF-IDF, Bag of Words, K-Means, LDA, PCA |

---

## 1. Road Accident Severity Prediction

A classification pipeline built using the **FARS road-traffic accident dataset**, containing more than 100,000 accident records.

The objective is to predict accident injury severity using selected characteristics from the dataset.

### Workflow

- Exploratory Data Analysis
- Target distribution analysis
- Correlation analysis
- Min-Max normalization
- Feature selection using `SelectKBest`
- Train/test splitting
- K-Nearest Neighbors classification
- Decision Tree classification
- Hyperparameter tuning with `GridSearchCV`
- Accuracy, precision, recall and F1-score evaluation

### Selected Features

Feature selection identified:

- Age
- Alcohol test result
- Drug test result 1
- Drug test result 2
- Drug test result 3

### Model Results

| Model | Cross-Validation Accuracy | Test Accuracy |
|---|---:|---:|
| K-Nearest Neighbors | 44.09% | 44.48% |
| Decision Tree | 47.81% | 48.17% |

The **Decision Tree classifier produced the strongest overall result**, although both models showed difficulty predicting less frequent severity classes.

---

## 2. Bacterial Growth Prediction

A regression analysis designed to predict two bacterial growth parameters:

- `a` — maximum bacterial population
- `mu` — bacterial growth rate

The experimental features include:

- Initial cyanobacteria population
- CO₂ availability
- Light availability
- Sucrose production ratio
- Experiment number

### Workflow

- Exploratory Data Analysis
- Correlation analysis
- Feature visualization
- Min-Max normalization
- Train / validation / test split
- Linear Regression
- Random Forest Regression
- Cross-validation
- Hyperparameter tuning with `GridSearchCV`
- Evaluation using MSE and R²

### Final Results

#### Prediction of `a`

| Model | Test MSE | R² |
|---|---:|---:|
| Linear Regression | 128.27 | 0.28 |
| Random Forest | **4.58** | **0.97** |

#### Prediction of `mu`

| Model | Test MSE | R² |
|---|---:|---:|
| Linear Regression | 67.75 | 0.29 |
| Random Forest | **2.61** | **0.97** |

The **Random Forest Regressor significantly outperformed Linear Regression** for both target variables, indicating that the relationships within the experimental data are strongly non-linear.

---

## 3. News Article Clustering & Topic Modeling

An unsupervised NLP analysis of **2,225 news articles** designed to identify patterns and underlying topics without predefined labels.

### Workflow

- Text exploratory analysis
- Custom tokenization
- Stop-word removal
- Word-frequency analysis
- Word-count distribution
- Word cloud visualization
- Bag of Words vectorization
- TF-IDF transformation
- K-Means clustering
- Latent Dirichlet Allocation
- PCA dimensionality reduction
- Silhouette Score evaluation
- Topic distribution visualization

### Topic Discovery

LDA was configured to identify five broad topic groups, including themes associated with:

- Aviation
- Corporate activity
- Economic trends
- Year-based patterns
- Sales-related discussions

K-Means was also used to create five document clusters.

The resulting Silhouette Score was approximately:

```text
0.0067
