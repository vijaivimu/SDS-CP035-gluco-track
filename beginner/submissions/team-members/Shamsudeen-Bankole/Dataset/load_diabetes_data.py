import pandas as pd
import numpy as np

# Load the diabetes dataset
print("Loading diabetes_binary_health_indicators_BRFSS2015.csv...")
df = pd.read_csv('diabetes_binary_health_indicators_BRFSS2015.csv')

# Display basic information about the dataset
print("\n=== Dataset Information ===")
print(f"Shape: {df.shape}")
print(f"Number of rows: {df.shape[0]}")
print(f"Number of columns: {df.shape[1]}")

print("\n=== Column Names ===")
print(df.columns.tolist())

print("\n=== Data Types ===")
print(df.dtypes)

print("\n=== First 5 rows ===")
print(df.head())

print("\n=== Basic Statistics ===")
print(df.describe())

print("\n=== Missing Values ===")
print(df.isnull().sum())

print("\n=== Dataset loaded successfully! ===")
print(f"The DataFrame 'df' is now available with {df.shape[0]} rows and {df.shape[1]} columns.") 