# 🔴 GlucoTrack – Advanced Track

## ✅ Week 1: Exploratory Data Analysis (EDA)

---

### 📦 1. Data Integrity & Structure

Q: Are there any missing, duplicate, or incorrectly formatted entries in the dataset?  
A:  No missing values were found in the dataset, as 253680 instances are complete.
   24206 rows (corresponding to 9.5% of the dataset) were identified as duplicates and would be further assessed for removal to ensure each observation is          unique and does not bias the model.
   No formatting issues were observed; however, all features were stored as float64, while many of them should be integers or categories.

Q: Are all data types appropriate (e.g., numeric, categorical)?  
A:  No, not all data types are appropriate in the raw dataset. The appropriate format is given below
   Binary categorical Features: 15 Columns: Outcome, Sex, HighBP, HighChol, CholCheck, Smoker, Stroke, HeartDiseaseorAttack, PhysActivity, Fruits, Veggies, HvyAlcoholConsump, AnyHealthcare, NoDocbcCost, DiffWalk
   ordinal/categorical_features: 4 Columns: ['GenHlth', 'Age', 'Education', 'Income']
   numeric_features: 3 Columns ['BMI', 'MentHlth', 'PhysHlth']
   While some features like BMI, mental health, and physical health are naturally numeric and stored as float, other columns are binary or ordinal categorical and needed to be converted into the appropriate datatype (category or int) to reduce memory usage. 
   Binary categorical variables were misrepresented as continuous (float).
   Ordinal numeric/categorical variables lost their semantic meaning.
   Therefore, the data types were not initially appropriate, and needed explicit correction based on domain knowledge

Q: Did you detect any constant, near-constant, or irrelevant features?  
A:  No constant features (i.e., columns with only one unique value) were detected. Every feature had at least two unique values, indicating variability     in responses.
   There are a few features that showed near-constant behavior, where one category accounted for the vast majority of entries. For example
   CholCheck                   96.27%      
   Stroke                      95.94      
   AnyHealthcare               95.11      
These features are not removed immediately but flagged for possible exclusion or dimensionality reduction if they do not significantly contribute during model evaluation.
Initially, there are no clear indications of irrelevant features. All retained features may have potential predictive power for diabetes classification and should be evaluated during model training for importance.

---

### 🎯 2. Target Variable Assessment

Q: What is the distribution of `Diabetes_binary`?  
A:  The distribution of target_class is highly imbalanced, with class 0 (no diabetes) comprising approximately 86% of the data (218334 instances), class 1 (prediabetes or diabetes) accounting for 14.0% (35,346 instances). This suggests a strong class skew that may require addressing in modelling.

Q: Is there a class imbalance? If so, how significant is it?  
A:  Yes, there is a significant class imbalance. Class 0 dominates the dataset with approximately 86.07% of the instances (218334 out of 253,680), while class 1 accounts for 13.930% (35,346 instances), and class 1 comprises only 1.8% (4,631 instances). The disparity between the majority and minority classes is substantial and may adversely affect model performance (due to algorithmic bias) if not addressed

Q: How might this imbalance influence your choice of evaluation metrics or model strategy?  
A:  The imbalance makes it important to use balanced evaluation metrics (such as precision, recall, F1-score, and area under the precision-recall curve) and model strategies that ensure fair learning across all classes. Ignoring this can lead to a model that performs poorly on critical minority outcomes, which is unacceptable in sensitive domains like healthcare.
We may have to consider resampling methods such as SMOTE, ADASYN, oversampling, undersampling or class weighting.
We may have to consider threshold tuning to optimise recall or the F1 score, depending on the clinical priority

---

### 📊 3. Feature Distribution & Quality

Q: Which numerical features are skewed or contain outliers?  
A:  MentHlth has strong positive skew (~2.72) with many zeros: about 14% flagged as outliers.
    PhysHlth is positively skewed (~2.21) with about 16% flagged as outliers
    BMI: Right skewed (2.12) with a long tail: 3.9% flagged as outlier, and the maximum observed at 98.

Q: Did any features contain unrealistic or problematic values?  
A:  MentHlth and PhysHlth are within the 0-30 days. No negative or >30
    BMI: No negatives or <10; however, 279 observations > 80 were flagged. Such values can be physiologically rare and may warrant verification or          capping.

Q: What transformation methods (if any) might improve these feature distributions?  
A:  MentHlth and PhysHlth: Yeo-Johnson or Square-root transformation for variance stabilisation
    BMI: Winsoring or capping a high percentile or Use robustscaler or Yeo-Johnson 

---

### 📈 4. Feature Relationships & Patterns

Q: Which categorical features (e.g., `GenHealth`, `PhysicalActivity`, `Smoking`) show visible patterns in relation to `Diabetes_binary`?  
A:  All the categorical features 

Q: Are there any strong pairwise relationships or multicollinearity between features?  
A:  There is no multicollinearity; there is a weak correlation between the features

Q: What trends or correlations stood out during your analysis?  
A:  PhysHlth and MentHlth showed a highest but weak correlation (0.35)
    BMI and Outcome (0.22)

---

### 🧰 5. EDA Summary & Preprocessing Plan

Q: What are your 3–5 biggest takeaways from EDA?  
A:  The dataset class is imbalanced, so there is a need to carefully address it to avoid bias in the model.
    Near-constant value can be potentially dropped
    Outliers should be handled with appropriate scaling

Q: Which features will you scale, encode, or exclude in preprocessing?  
A:  Scale: BMI, MentHlth, PhysHlth
Encoding: Ordinal Categorical features (Income, Age, Education, GenHlth)

Q: What does your cleaned dataset look like (rows, columns, shape)?  
A:  Dropping the duplicates, we have (229474, 22)

---
