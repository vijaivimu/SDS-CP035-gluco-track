# 🟢 GlucoTrack – Beginner Track

## ✅ Week 1: Exploratory Data Analysis (EDA)

---

### 📦 1. Data Integrity & Structure

Q: Are there any missing, duplicate, or incorrectly formatted entries in the dataset?  
A:  

Q: Are all data types appropriate (e.g., numeric, categorical)?  
A:  

Q: Did you detect any constant, near-constant, or irrelevant features?  
A:  

---

### 🎯 2. Target Variable Assessment

Q: What is the distribution of `Diabetes_binary`?  
A:  

Q: Is there a class imbalance? If so, how significant is it?  
A:  

Q: How might this imbalance influence your choice of evaluation metrics or model strategy?  
A:  

---

### 📊 3. Feature Distribution & Quality

Q: Which numerical features are skewed or contain outliers?  
A:  

Q: Did any features contain unrealistic or problematic values?  
A:  

Q: What transformation methods (if any) might improve these feature distributions?  
A:  

---

### 📈 4. Feature Relationships & Patterns

Q: Which categorical features (e.g., `GenHealth`, `PhysicalActivity`, `Smoking`) show visible patterns in relation to `Diabetes_binary`?  
A:  

Q: Are there any strong pairwise relationships or multicollinearity between features?  
A:  

Q: What trends or correlations stood out during your analysis?  
A:  

---

### 🧰 5. EDA Summary & Preprocessing Plan

Q: What are your 3–5 biggest takeaways from EDA?  
A:  

Q: Which features will you scale, encode, or exclude in preprocessing?  
A:  

Q: What does your cleaned dataset look like (rows, columns, shape)?  
A:  

---

## ✅ Week 2: Feature Engineering & Preprocessing

---

### 🏷️ 1. Feature Encoding

Q: Identify the binary (`0` or `1`) categorical features and apply a simple mapping or encoder. Which features did you encode?  
A:  

Q: The `GenHealth` and `Education` features are ordinal. Apply a custom mapping that preserves their inherent order and justify the order you chose.  
A:  

Q: For any remaining nominal categorical features, apply one-hot encoding. Why is this method more suitable for nominal data than a simple integer label?  
A:  

---

### ✨ 2. Feature Creation

Q: Create a new feature for BMI categories (e.g., Underweight, Normal, Overweight, Obese) from the `BMI` column. Display the value counts for your new categories.  
A:  

Q: Create a new feature named `TotalHealthDays` by combining `PhysHlth` and `MentHlth`. What is the rationale behind creating this feature?  
A:  

---

### ✂️ 3. Data Splitting

Q: Split your dataset into training and testing sets (an 80/20 split is recommended). Use stratification on the `Diabetes_binary` target variable.  
A:  

Q: Why is it critical to split the data *before* applying techniques like SMOTE or scaling?  
A:  

Q: Show the shape of your `X_train`, `X_test`, `y_train`, and `y_test` arrays to confirm the split.  
A:  

---

### ⚖️ 4. Imbalance Handling & Final Preprocessing

Q: Apply the SMOTE technique to address class imbalance. Importantly, apply it *only* to the training data. Show the class distribution of the training target variable before and after.  
A:  

Q: Normalize the numerical features using `StandardScaler`. Fit the scaler *only* on the training data, then transform both the training and testing data. Why must you not fit the scaler on the test data?  
A:  

Q: Display the shape of your final, preprocessed training features (`X_train_processed`) and testing features (`X_test_processed`).  
A: