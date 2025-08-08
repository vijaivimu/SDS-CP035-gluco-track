import pandas as pd
import numpy as np

# Load the dataset
print("Loading diabetes dataset for duplicate analysis...")
df = pd.read_csv('diabetes_binary_health_indicators_BRFSS2015.csv')

print("\n" + "="*60)
print("DUPLICATE RECORDS ANALYSIS")
print("="*60)

# Basic duplicate info
total_records = len(df)
duplicate_count = df.duplicated().sum()
unique_records = total_records - duplicate_count

print(f"\n1. DUPLICATE STATISTICS")
print("-" * 40)
print(f"Total records: {total_records:,}")
print(f"Unique records: {unique_records:,}")
print(f"Duplicate records: {duplicate_count:,} ({duplicate_count/total_records*100:.2f}%)")

# Analyze duplicates by target variable
print(f"\n2. DUPLICATES BY DIABETES STATUS")
print("-" * 40)
diabetes_duplicates = df[df.duplicated()]['Diabetes_binary'].value_counts()
print("Duplicates by diabetes status:")
for status, count in diabetes_duplicates.items():
    status_name = "Diabetic" if status == 1.0 else "Non-diabetic"
    print(f"  {status_name}: {count:,} records")

# Check if duplicates are exact or near-duplicates
print(f"\n3. DUPLICATE TYPE ANALYSIS")
print("-" * 40)
exact_duplicates = df.duplicated().sum()
print(f"Exact duplicates: {exact_duplicates:,}")

# Sample of duplicate records
print(f"\n4. SAMPLE DUPLICATE RECORDS")
print("-" * 40)
duplicate_sample = df[df.duplicated(keep=False)].head(10)
print("First 10 duplicate records:")
print(duplicate_sample[['Diabetes_binary', 'HighBP', 'BMI', 'Age', 'Sex']].to_string())

# Impact on class balance
print(f"\n5. IMPACT ON CLASS BALANCE")
print("-" * 40)
print("Before removing duplicates:")
original_balance = df['Diabetes_binary'].value_counts()
for status, count in original_balance.items():
    status_name = "Diabetic" if status == 1.0 else "Non-diabetic"
    print(f"  {status_name}: {count:,} ({count/len(df)*100:.2f}%)")

print("\nAfter removing duplicates:")
df_no_duplicates = df.drop_duplicates()
balanced_balance = df_no_duplicates['Diabetes_binary'].value_counts()
for status, count in balanced_balance.items():
    status_name = "Diabetic" if status == 1.0 else "Non-diabetic"
    print(f"  {status_name}: {count:,} ({count/len(df_no_duplicates)*100:.2f}%)")

# Recommendation
print(f"\n6. RECOMMENDATION")
print("-" * 40)
print("Consider the following factors:")

if duplicate_count/total_records > 0.05:
    print("⚠️  High duplicate rate (>5%) detected")
    print("   - May indicate data collection issues")
    print("   - Could bias your model training")
    print("   - Consider removing duplicates for ML tasks")
else:
    print("✅ Duplicate rate is acceptable")

# Check if duplicates are systematic
duplicate_pairs = df[df.duplicated(keep=False)].groupby(df.columns.tolist()).size()
if len(duplicate_pairs) < duplicate_count/2:
    print("⚠️  Duplicates appear to be systematic (many identical records)")
    print("   - Likely data collection/processing issue")
    print("   - Strongly recommend removing duplicates")
else:
    print("✅ Duplicates appear to be random")
    print("   - May represent legitimate repeated measurements")
    print("   - Consider keeping for analysis context")

print(f"\n7. FINAL RECOMMENDATION")
print("-" * 40)
print("For Machine Learning Tasks:")
print("  → DROP DUPLICATES (recommended)")
print("  - Prevents model from learning the same pattern multiple times")
print("  - Ensures fair evaluation metrics")
print("  - Reduces computational overhead")

print("\nFor Exploratory Data Analysis:")
print("  → KEEP DUPLICATES (if they represent real repeated measurements)")
print("  - Maintains data collection context")
print("  - Preserves original sample size")
print("  - But be aware of potential bias in statistical tests")

print(f"\n" + "="*60)
print("ANALYSIS COMPLETE")
print("="*60) 