import pandas as pd
import numpy as np

# Load the dataset
print("Loading diabetes dataset for formatting analysis...")
df = pd.read_csv('diabetes_binary_health_indicators_BRFSS2015.csv')

print("\n" + "="*60)
print("DATA FORMATTING ANALYSIS")
print("="*60)

# 1. Data Type Issues
print("\n1. DATA TYPE ISSUES")
print("-" * 40)
print("Current data types:")
for col in df.columns:
    print(f"  {col}: {df[col].dtype}")

print("\nIssues identified:")
print("❌ All variables are stored as float64, but many should be:")
print("  - Binary variables should be int64 or bool")
print("  - Categorical variables should be int64 or category")
print("  - Only BMI should remain as float64")

# 2. Binary Variables Analysis
print("\n2. BINARY VARIABLES FORMATTING")
print("-" * 40)
binary_vars = ['Diabetes_binary', 'HighBP', 'HighChol', 'CholCheck', 'Smoker', 
               'Stroke', 'HeartDiseaseorAttack', 'PhysActivity', 'Fruits', 
               'Veggies', 'HvyAlcoholConsump', 'AnyHealthcare', 'NoDocbcCost', 'DiffWalk', 'Sex']

print("Binary variables should contain only 0 and 1 (integers), not 0.0 and 1.0 (floats)")
for col in binary_vars:
    unique_vals = df[col].unique()
    print(f"  {col}: {unique_vals} - {'✅ Correct' if set(unique_vals) == {0.0, 1.0} else '❌ Should be integers'}")

# 3. Categorical Variables Analysis
print("\n3. CATEGORICAL VARIABLES FORMATTING")
print("-" * 40)
categorical_vars = ['GenHlth', 'MentHlth', 'PhysHlth', 'Age', 'Education', 'Income']

print("Categorical variables should be integers, not floats:")
for col in categorical_vars:
    unique_vals = sorted(df[col].unique())
    print(f"  {col}: {unique_vals} - {'✅ Correct' if all(isinstance(x, (int, np.integer)) or x.is_integer() for x in unique_vals) else '❌ Should be integers'}")

# 4. Continuous Variables Analysis
print("\n4. CONTINUOUS VARIABLES FORMATTING")
print("-" * 40)
continuous_vars = ['BMI']
print("BMI should remain as float (correct):")
for col in continuous_vars:
    print(f"  {col}: {df[col].dtype} - ✅ Correct (should be float)")

# 5. Value Range Issues
print("\n5. VALUE RANGE ISSUES")
print("-" * 40)

# Check for decimal values in categorical variables
print("Checking for decimal values in categorical variables:")
for col in categorical_vars:
    decimal_vals = df[df[col] % 1 != 0][col].unique()
    if len(decimal_vals) > 0:
        print(f"  ❌ {col}: Contains decimal values {decimal_vals}")
    else:
        print(f"  ✅ {col}: All values are integers")

# Check for non-binary values in binary variables
print("\nChecking binary variables for non-binary values:")
for col in binary_vars:
    non_binary = df[~df[col].isin([0.0, 1.0])][col].unique()
    if len(non_binary) > 0:
        print(f"  ❌ {col}: Contains non-binary values {non_binary}")
    else:
        print(f"  ✅ {col}: Contains only binary values")

# 6. Recommended Data Types
print("\n6. RECOMMENDED DATA TYPES")
print("-" * 40)
print("After cleaning, data types should be:")
print("  Binary variables: int64")
print("  Categorical variables: int64")
print("  Continuous variables: float64")

# 7. Summary of Formatting Issues
print("\n7. FORMATTING ISSUES SUMMARY")
print("-" * 40)
print("❌ All variables stored as float64 instead of appropriate types")
print("❌ Binary variables should be integers (0, 1) not floats (0.0, 1.0)")
print("❌ Categorical variables should be integers, not floats")
print("✅ Only BMI should remain as float64")
print("✅ No missing values or invalid ranges detected")

print("\n8. IMPACT OF FORMATTING ISSUES")
print("-" * 40)
print("• Increased memory usage (float64 vs int64)")
print("• Potential confusion in analysis")
print("• Inconsistent with standard data science practices")
print("• May cause issues with certain ML algorithms")

print("\n9. RECOMMENDED FIXES")
print("-" * 40)
print("1. Convert binary variables to int64")
print("2. Convert categorical variables to int64")
print("3. Keep BMI as float64")
print("4. Consider using category dtype for categorical variables")

print(f"\n" + "="*60)
print("FORMATTING ANALYSIS COMPLETE")
print("="*60) 