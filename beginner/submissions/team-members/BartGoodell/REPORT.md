# 🟢 GlucoTrack – Beginner Track

## ✅ Week 1: Exploratory Data Analysis (EDA)

---

### 📦 1. Data Integrity & Structure

Q: Are there any missing, duplicate, or incorrectly formatted entries in the dataset?  
A: No missing values were found in the dataset; all 253,680 instances were complete. A total of 24,206 rows (9.5% of the data) were flagged as duplicates and will be further assessed for removal to avoid bias from repeated observations. No formatting issues were present, but all features were stored as float64. Several of these variables should be represented as integers or categorical values. 

Q: Are all data types appropriate (e.g., numeric, categorical)?  
A:  Not all data types in the raw dataset were appropriately represented. While some measures such as BMI, mental health, and physical health are naturally numeric and correctly stored as floats, other features were misclassified. Binary categorical variables were stored as continuous floats, and ordinal variables lost their semantic meaning when represented in this way. To address this, explicit datatype corrections were required, informed by domain knowledge, to properly classify features as categorical, ordinal, or numeric.

Feature Type	Columns
Binary categorical	Outcome, Sex, HighBP, HighChol, CholCheck, Smoker, Stroke, HeartDiseaseorAttack, PhysActivity, Fruits, Veggies, HvyAlcoholConsump, AnyHealthcare, NoDocbcCost, DiffWalk
Ordinal / Categorical	GenHlth, Age, Education, Income
Numeric	BMI, MentHlth, PhysHlth

Q: Did you detect any constant, near-constant, or irrelevant features?  
A: No constant features were detected, as all variables had at least two unique values.  At this stage, no features appeared irrelevant, and all were retained for further assessment in training.

---

### 🎯 2. Target Variable Assessment

Q: What is the distribution of `Diabetes_binary`?  
A:  After removing 24,602 duplicate instances, the target variable Diabetes_binary remained imbalanced, with 84.7% of observations labeled “0” (no diabetes) and 15.3% labeled “1” (diabetes positive). This pronounced skew toward the negative class underscores the importance of applying class-balance strategies such as weighting or resampling during model development to avoid bias toward the majority class.


Q: Is there a class imbalance? If so, how significant is it?  
A: Yes, the dataset shows a significant class imbalance. After removing 24,602 duplicate instances, 84.7% of observations belong to the negative class (no diabetes), while only 15.3% belong to the positive class (diabetes). This imbalance is substantial and requires mitigation during model training to prevent the classifier from being biased toward the majority class. 

Q: How might this imbalance influence your choice of evaluation metrics or model strategy?  
A:  The pronounced class imbalance means that accuracy alone would be a misleading evaluation metric, as a model could achieve high accuracy by predicting the majority class only. Instead, metrics that account for the minority class, such as precision, recall, F1-score, ROC-AUC, and PR-AUC, are more appropriate. From a modeling perspective, strategies such as class weighting, resampling (oversampling the minority or undersampling the majority), or using algorithms robust to imbalance (e.g., tree-based ensembles with balanced class weights) will be considered to ensure the model does not ignore the minority class of interest.

---

### 📊 3. Feature Distribution & Quality

Q: Which numerical features are skewed or contain outliers?  
A:  Using a Z-score threshold of 3, we identified 15,328 potential outlier rows across the numerical features BMI, MentHlth, and PhysHlth. This confirms what was visible in the histograms: MentHlth and PhysHlth are highly skewed, with a large number of values far from the mean. BMI also contributed to the outlier count, though less dramatically.

For GenHlth, although it is an ordinal categorical variable (1–5), extreme values at the higher “poor health” end still appear far from the mean relative to the standard deviation, which is consistent with the skewed distribution observed.

Overall, the Z-score analysis highlights that outliers are most pronounced in the health-related features (MentHlth and PhysHlth). These findings underscore the importance of considering strategies such as feature transformation, outlier handling, or robust modeling approaches to mitigate their impact.

Q: Did any features contain unrealistic or problematic values?  
A:  While the dataset contained no missing values or incorrectly formatted entries, several features—particularly BMI, MentHlth, and PhysHlth—show a substantial number of statistically unusual values. These outliers, though still within the defined ranges, are atypical relative to the overall distributions and may impact model performance. No variables were found to contain values outside their expected ranges or categories.

Q: What transformation methods (if any) might improve these feature distributions?  
A:  For MentHlth and PhysHlth, which are heavily right-skewed and have a large number of zeros, the Log transformation with log(x+1) or Square Root transformation could be good starting points.
For BMI, which is moderately right-skewed, any of the power transformations (Log, Square Root, Box-Cox, or Yeo-Johnson) or Quantile transformation could be considered.
The best transformation method often depends on the specific distribution of the feature and requires experimentation to see which one yields the most desirable distribution and improves model performance.
---

### 📈 4. Feature Relationships & Patterns

Q: Which categorical features (e.g., `GenHealth`, `PhysicalActivity`, `Smoking`) show visible patterns in relation to `Diabetes_binary`?  
A:  To examine whether categorical features show patterns in relation to Diabetes_binary, cross-tabulations were generated for GenHlth (General Health), PhysActivity (Physical Activity), and Smoker (Smoking).

GenHlth: A strong pattern is evident—individuals reporting worse general health have a much higher prevalence of diabetes compared to those reporting excellent or very good health.

PhysActivity: Those reporting no physical activity show a markedly higher diabetes rate (21.14%) than those who are physically active (11.61%).

Smoker: Smokers have a slightly higher diabetes prevalence (16.29%) compared to non-smokers (12.06%).

In summary, all three features demonstrate visible patterns, with general health and physical activity showing the strongest associations with diabetes, and smoking showing a weaker but still notable relationship.

Q: Are there any strong pairwise relationships or multicollinearity between features?  
A: Pairwise relationships and potential multicollinearity were assessed using a correlation matrix. Several moderate correlations were observed, such as GenHlth with PhysHlth (≈0.42), DiffWalk (≈0.32), and HighBP (≈0.30). PhysHlth was also moderately correlated with MentHlth (≈0.28) and DiffWalk (≈0.32). As expected, Income correlated with Education (≈0.45), and HighBP correlated with HighChol (≈0.30).

While these relationships are intuitive, none exceeded the common thresholds (0.7–0.8) that typically signal multicollinearity concerns. In summary, the dataset contains several moderate, meaningful associations, but no evidence of severe multicollinearity that would compromise models such as logistic regression or tree-based methods. 

Q: What trends or correlations stood out during your analysis?  
A: The analysis revealed several important trends and correlations. First, the target variable Diabetes_binary shows a strong class imbalance, with 84.7% of individuals not having diabetes and 15.3% having diabetes, a factor that must be addressed in model training. Among numerical features, MentHlth and PhysHlth are highly skewed with many outliers, while BMI also shows right skewness and some extreme values.

Clear relationships emerged between general health and diabetes: individuals reporting poorer overall health had a much higher prevalence of diabetes. Physical activity was linked with lower diabetes rates, while smoking showed a weaker but still notable association with higher rates. Correlation analysis revealed expected moderate relationships, such as between GenHlth and PhysHlth, and between Income and Education, but no severe multicollinearity.

Finally, features such as GenHlth, BMI, HighBP, and HighChol showed stronger positive correlations with Diabetes_binary, suggesting they may play an important role in predicting diabetes status. These insights emphasize the need to address imbalance, skewness, and outliers during preprocessing, while focusing on the most predictive health and demographic factors. 

---

### 🧰 5. EDA Summary & Preprocessing Plan

Q: What are your 3–5 biggest takeaways from EDA?  
A:  Severe Class Imbalance
The target Diabetes_binary is heavily skewed (84.7% no diabetes vs. 15.3% diabetes), making class imbalance a central challenge that will affect model choice and evaluation metrics.

Skewness and Outliers in Health Features
MentHlth and PhysHlth are highly skewed with many extreme values, while BMI also shows skewness and outliers. These distributions may distort models unless handled through transformation, robust methods, or outlier strategies.

Strong Link Between General Health and Diabetes
Poorer self-reported general health is strongly associated with higher diabetes prevalence, making GenHlth a particularly important predictor.

Lifestyle Factors Show Patterns
Physical activity is linked with substantially lower diabetes rates, while smoking is associated with somewhat higher rates. These lifestyle features provide meaningful predictive signal.

Moderate but Manageable Correlations
Expected relationships appear (e.g., Income with Education, GenHlth with PhysHlth), but no correlations are high enough to indicate serious multicollinearity problems.

Q: Which features will you scale, encode, or exclude in preprocessing?  
A: For preprocessing, three main considerations were addressed: scaling, encoding, and exclusion. Several numerical features—BMI, GenHlth, MentHlth, PhysHlth, Age, and Income—were scaled to account for their different ranges and distributions, ensuring they are suitable for algorithms sensitive to feature magnitude. Encoding was not required beyond the dataset’s existing structure: binary variables such as HighBP, HighChol, and Smoker are already represented as 0s and 1s, while ordinal variables like GenHlth, Age, Education, and Income are encoded as integers. No features were excluded, as none contained excessive missing values or were judged irrelevant. Where needed, dimensionality reduction techniques such as PCA can capture the most important variance and mitigate less informative features.

In summary, the main preprocessing adjustment was scaling numerical features, while encoding and exclusion were not necessary at this stage. 

Q: What does your cleaned dataset look like (rows, columns, shape)?  
A: After removing the duplicate rows, our cleaned dataset, stored in the DataFrame df, has the following characteristics:

Shape: The shape of the DataFrame is (229474, 22).
Rows: This means the dataset contains 229,474 rows (which represent individual observations or patients after removing duplicates).
Columns: It has 22 columns (which represent the features and the target variable). 

---
