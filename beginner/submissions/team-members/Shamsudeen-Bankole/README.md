# 🟢 GlucoTrack: Beginner Track

## 📌 Overview
This project leverages supervised machine learning algorithms to predict the risk of diabetes using the CDC Diabetes Health Indicators dataset. By analysing a wide range of health and lifestyle features such as BMI, physical activity, blood pressure, smoking status, and general health condition, the project aims to identify patterns and risk factors associated with diabetes to allow early detection and support of public health decision-making.

## Goal
To classify individuals as diabetic (1) or non-diabetic (0)

## 🚀 Features

| Predictive Features | Semantic Type | Measurement Scale |

|--------|----------------|-------------------|

| HighBP, HighChol, CholCheck, Smoker, Stroke, HeartDiseaseorAttack, PhysActivity, HvyAlcoholConsump, AnyHealthcare, NoDocbcCost, DiffWalk | Health | Binary |

| BMI | Physiological Metric | Continuous |

| Fruits, Veggies | Lifestyle | Binary |

| GenHlth, MentHlth, PhysHlth | Health | Ordinal |

| Sex | Demographic | Binary |

| Age | Demographic | Ordinal |

| Education | Socioeconomic | Ordinal |

| Income | Socioeconomic | Ordinal |

|Target Feature | Semantic Type |

|--------| Measurement scale|

|Diabetes_binary | Binary |

### 📊 **Dataset**: CDC Diabetes Health Indicators contained in the 2014 BRFSS Survey Data and Documentation can be accessed through the [CDC Diabetes Health Indicators](https://archive.ics.uci.edu/dataset/891/cdc+diabetes+health+indicators)

## 🎓 Weekly Breakdown

### ✅ Week 1: Exploratory Data Analysis (EDA)

- Check for missing, duplicate, or invalid values
- Review column data types and formats
- Understand class balance of the target variable
- Visualize distributions, correlations, and feature-to-target patterns
- Document top 3–5 insights in your report

### ✅ Week 2: Feature Engineering & Preprocessing
- Encode binary, ordinal, and nominal features appropriately
- Create new features (e.g., BMI categories, health scores)
- Handle data imbalance using techniques like SMOTE or class weights
- Normalize numerical features
- Split data using stratified train/validation/test sets

### ✅ Week 3: Model Development & Experimentation
- Train baseline models: Logistic Regression, Decision Trees, Naive Bayes
- Track experiments using MLflow
- Evaluate models on validation set with metrics: Accuracy, Precision, Recall, F1-score
- Perform error analysis using confusion matrices

### ✅ Week 4: Model Tuning & Finalization
- Perform hyperparameter tuning (grid/random search)
- Use cross-validation to validate model stability
- Finalize best model with full training set
- Evaluate on test set and interpret feature importance

### ✅ Week 5: Deployment
- Build a Streamlit app for model inference
- Design user-friendly form inputs
- Display prediction results and risk messages
- Deploy the app to Streamlit Community Cloud

---

## 🗒️ Project Timeline Overview

| Phase                           | General Activities                                                     |

| ------------------------------- | ---------------------------------------------------------------------- |
| **Week 1: Setup + EDA**         | Clean, explore, and visualize the data                                 |
| **Week 2: Feature Engineering** | Transform features, encode variables, handle imbalance, prepare splits |
| **Week 3: Model Development**   | Train ML or DL models and evaluate performance                         |
| **Week 4: Model Optimization**  | Tune models, improve generalization, and interpret results             |
| **Week 5: Deployment**          | Deploy models via Streamlit or API-based solutions                     |

---

