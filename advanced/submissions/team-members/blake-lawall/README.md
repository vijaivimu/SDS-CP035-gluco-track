# 🔴 GlucoTrack Advanced Track - Blake Lawall

## 📋 Project Overview

This repository contains my implementation of the GlucoTrack Advanced Track project, focusing on deep learning approaches for diabetes risk prediction using the CDC Diabetes Health Indicators dataset.

## 🎯 Project Goals

- Build a Feedforward Neural Network (FFNN) for diabetes classification
- Implement advanced feature engineering techniques
- Apply model explainability using SHAP and LIME
- Deploy the model using modern deployment strategies
- Track experiments using MLflow

## 📁 Project Structure

```
blake-lawall/
├── notebooks/                 # Jupyter notebooks for each week
│   ├── week1_eda.ipynb       # Exploratory Data Analysis
│   ├── week2_feature_engineering.ipynb
│   ├── week3_neural_network.ipynb
│   ├── week4_model_tuning.ipynb
│   └── week5_deployment.ipynb
├── src/                      # Source code modules
│   ├── data/                 # Data processing utilities
│   ├── models/               # Neural network architectures
│   ├── training/             # Training utilities
│   └── deployment/           # Deployment scripts
├── data/                     # Dataset storage
├── models/                   # Saved model artifacts
├── reports/                  # Generated reports and visualizations
├── tests/                    # Unit tests
└── README.md                 # This file
```

## 🚀 Setup Instructions

### 1. Environment Setup

```bash
# Navigate to the blake-lawall directory
cd advanced/submissions/team-members/blake-lawall

# Create virtual environment using uv
uv venv .venv

# Activate virtual environment
source .venv/bin/activate  # On macOS/Linux
# or
.venv\Scripts\activate     # On Windows

# Install dependencies
uv pip install torch torchvision tensorflow scikit-learn pandas numpy matplotlib seaborn plotly xgboost lightgbm shap lime mlflow streamlit fastapi uvicorn pydantic python-dotenv requests tqdm jupyter ipykernel category-encoders imbalanced-learn

# Install Jupyter kernel
python -m ipykernel install --user --name=glucotrack-advanced --display-name='GlucoTrack Advanced'
```

### 2. Generate Notebooks

```bash
# Run the notebook generator (from blake-lawall directory)
python create_notebooks.py
```

### 3. Start Jupyter

```bash
# Start Jupyter Lab
jupyter lab

# Or start Jupyter Notebook
jupyter notebook
```

## 📊 Dataset

- **Source**: [CDC Diabetes Health Indicators](https://archive.ics.uci.edu/dataset/891/cdc+diabetes+health+indicators)
- **Target**: Binary classification (diabetic vs non-diabetic)
- **Features**: Health, lifestyle, and demographic indicators

## 🗓️ Weekly Progress

### Week 1: Exploratory Data Analysis (EDA)
- [ ] Data loading and initial exploration
- [ ] Data integrity assessment
- [ ] Target variable analysis
- [ ] Feature distribution analysis
- [ ] Correlation analysis
- [ ] EDA summary and preprocessing plan

### Week 2: Feature Engineering & Deep Learning Prep
- [ ] Data preprocessing pipeline
- [ ] Categorical feature encoding
- [ ] Numerical feature scaling
- [ ] Class imbalance handling
- [ ] Train/validation/test split
- [ ] PyTorch DataLoader creation

### Week 3: Neural Network Design & Baseline Training
- [ ] Feedforward Neural Network architecture
- [ ] Model training setup
- [ ] Training loop implementation
- [ ] Model evaluation metrics
- [ ] MLflow experiment tracking
- [ ] Baseline results analysis

### Week 4: Model Tuning & Explainability
- [ ] Hyperparameter tuning
- [ ] Architecture optimization
- [ ] Early stopping implementation
- [ ] SHAP explainability integration
- [ ] Feature importance analysis
- [ ] Model interpretability assessment

### Week 5: Model Deployment
- [ ] Model serialization
- [ ] Streamlit app development
- [ ] FastAPI backend creation
- [ ] Docker containerization
- [ ] Cloud deployment
- [ ] Performance monitoring

## 🛠️ Technologies Used

- **Deep Learning**: PyTorch, TensorFlow
- **Machine Learning**: Scikit-learn, XGBoost, LightGBM
- **Explainability**: SHAP, LIME
- **Experiment Tracking**: MLflow, Weights & Biases
- **Deployment**: Streamlit, FastAPI, Docker
- **Data Processing**: Pandas, NumPy, Category Encoders
- **Visualization**: Matplotlib, Seaborn, Plotly

## 📈 Key Metrics

- **Accuracy**: TBD
- **Precision**: TBD
- **Recall**: TBD
- **F1-Score**: TBD
- **AUC-ROC**: TBD

## 🔍 Model Explainability

The project includes comprehensive model explainability using:
- **SHAP (SHapley Additive exPlanations)**: For global and local feature importance
- **LIME (Local Interpretable Model-agnostic Explanations)**: For local explanations
- **Feature importance analysis**: Understanding model decisions

## 🚀 Deployment Strategy

The model will be deployed using one of the following approaches:
1. **Streamlit Cloud**: Simple web application
2. **FastAPI + Docker**: RESTful API with containerization
3. **Cloud Platform**: Deployment to platforms like Railway, Render, or GCP

## 📝 Documentation

- [Advanced Track Requirements](../README.md)
- [Report Template](../REPORT.md)
- [Contributing Guidelines](../../../CONTRIBUTING.md)

## 🤝 Contributing

This is a personal submission for the GlucoTrack Advanced Track. For general project contributions, please refer to the main [CONTRIBUTING.md](../../../CONTRIBUTING.md) file.

## 📄 License

This project is part of the SuperDataScience Community Project. Please refer to the main project license.

---

**Author**: Blake Lawall  
**Project**: GlucoTrack Advanced Track  
**Date**: 2024 