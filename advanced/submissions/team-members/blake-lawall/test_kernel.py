#!/usr/bin/env python3
"""
Test script to verify kernel and imports work correctly
"""

import sys
print(f"Python executable: {sys.executable}")
print(f"Python version: {sys.version}")

# Test all the imports from the notebook
try:
    import numpy as np
    print("✅ NumPy imported successfully")
except ImportError as e:
    print(f"❌ NumPy import failed: {e}")

try:
    import pandas as pd
    print("✅ Pandas imported successfully")
except ImportError as e:
    print(f"❌ Pandas import failed: {e}")

try:
    import matplotlib.pyplot as plt
    print("✅ Matplotlib imported successfully")
except ImportError as e:
    print(f"❌ Matplotlib import failed: {e}")

try:
    import seaborn as sns
    print("✅ Seaborn imported successfully")
except ImportError as e:
    print(f"❌ Seaborn import failed: {e}")

try:
    import plotly.express as px
    print("✅ Plotly Express imported successfully")
except ImportError as e:
    print(f"❌ Plotly Express import failed: {e}")

try:
    import plotly.graph_objects as go
    print("✅ Plotly Graph Objects imported successfully")
except ImportError as e:
    print(f"❌ Plotly Graph Objects import failed: {e}")

try:
    from plotly.subplots import make_subplots
    print("✅ Plotly Subplots imported successfully")
except ImportError as e:
    print(f"❌ Plotly Subplots import failed: {e}")

try:
    import torch
    print("✅ PyTorch imported successfully")
except ImportError as e:
    print(f"❌ PyTorch import failed: {e}")

try:
    import torch.nn as nn
    print("✅ PyTorch NN imported successfully")
except ImportError as e:
    print(f"❌ PyTorch NN import failed: {e}")

try:
    import torch.optim as optim
    print("✅ PyTorch Optim imported successfully")
except ImportError as e:
    print(f"❌ PyTorch Optim import failed: {e}")

try:
    from torch.utils.data import Dataset, DataLoader
    print("✅ PyTorch Utils imported successfully")
except ImportError as e:
    print(f"❌ PyTorch Utils import failed: {e}")

try:
    from sklearn.model_selection import train_test_split
    print("✅ Scikit-learn Model Selection imported successfully")
except ImportError as e:
    print(f"❌ Scikit-learn Model Selection import failed: {e}")

try:
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    print("✅ Scikit-learn Preprocessing imported successfully")
except ImportError as e:
    print(f"❌ Scikit-learn Preprocessing import failed: {e}")

try:
    from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
    print("✅ Scikit-learn Metrics imported successfully")
except ImportError as e:
    print(f"❌ Scikit-learn Metrics import failed: {e}")

try:
    import shap
    print("✅ SHAP imported successfully")
except ImportError as e:
    print(f"❌ SHAP import failed: {e}")

try:
    import lime
    print("✅ LIME imported successfully")
except ImportError as e:
    print(f"❌ LIME import failed: {e}")

try:
    import lime.lime_tabular
    print("✅ LIME Tabular imported successfully")
except ImportError as e:
    print(f"❌ LIME Tabular import failed: {e}")

try:
    import mlflow
    print("✅ MLflow imported successfully")
except ImportError as e:
    print(f"❌ MLflow import failed: {e}")

try:
    import mlflow.pytorch
    print("✅ MLflow PyTorch imported successfully")
except ImportError as e:
    print(f"❌ MLflow PyTorch import failed: {e}")

print("\n🎉 All imports completed!") 