# 🔴 GlucoTrack – Advanced Track

## ✅ Week 1: Exploratory Data Analysis (EDA)

---

### 📦 1. Data Integrity & Structure

Q: Are there any missing, duplicate, or incorrectly formatted entries in the dataset?  
A:  

Q: Are all data types a### 📊 3. Visualization & Communication

Q: How did you visualize feature contributions and model explanations for stakeholders?  
A: I created a **horizontal bar chart** showing the top 15 features by gradient importance magnitude, providing clear visual ranking of feature influence. I also implemented **clinical grouping** of features (cardiovascular, metabolic, lifestyle factors) to make results more interpretable for healthcare stakeholders. However, the **very small importance values** (0.0001-0.0006) and **counterintuitive rankings** make these visualizations **difficult to defend** to clinical experts.

Q: What challenges did you encounter when interpreting or presenting model explanations?  
A: Major challenges included: 1) **SHAP incompatibility** forcing use of inferior gradient methods, 2) **Counterintuitive feature rankings** that contradict medical knowledge, 3) **Very small importance values** suggesting weak individual feature signals, 4) **Difficulty explaining** why traditional diabetes indicators ranked low. The **misalignment with domain expertise** makes it challenging to present these findings confidently to healthcare professionals.

Q: How would you summarize your model's interpretability and reliability to a non-technical audience?  
A: **Honest assessment**: "Our diabetes prediction model shows moderate ability to identify potential diabetes cases but has significant limitations. The model's explanations don't align well with medical knowledge - it emphasizes lifestyle factors like alcohol consumption over established risk factors like BMI and age. This suggests the model may be learning from data patterns rather than true medical relationships. **The model is not ready for clinical use** and requires substantial improvement, validation with medical experts, and better explainability tools before deployment consideration."e (e.g., numeric, categorical)?  
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
---

## ✅ Week 2: Feature Engineering & Deep Learning Prep

---

### 🏷️ 1. Categorical Feature Encoding

Q: Which categorical features in the dataset have more than two unique values?  
A:  

Q: Apply integer-encoding to these high-cardinality features. Why is this strategy suitable for a subsequent neural network with an embedding layer?  
A:  

Q: Display the first 5 rows of the transformed data to show the new integer labels.  
A:  

---

### ⚖️ 2. Numerical Feature Scaling

Q: Which numerical features did your EDA from Week 1 suggest would benefit from scaling?  
A:  

Q: Apply a scaling technique to these features. Justify your choice of `StandardScaler` vs. `MinMaxScaler` or another method.  
A:  

Q: Show the summary statistics of the scaled data to confirm the transformation was successful.  
A:  

---

### ✂️ 3. Stratified Data Splitting

Q: Split the data into training, validation, and testing sets (e.g., 70/15/15). What function and parameters did you use?  
A:  

Q: Why is it critical to use stratification for this specific dataset?  
A:  

Q: Verify the stratification by showing the class distribution of `Diabetes_binary` in each of the three resulting sets.  
A:  

---

### 📦 4. Deep Learning Dataset Preparation

Q: Convert your three data splits into PyTorch `DataLoader` or TensorFlow `tf.data.Dataset` objects. What batch size did you choose and why?  
A:  

Q: To confirm they are set up correctly, retrieve one batch from your training loader. What is the shape of the features (X) and labels (y) in this batch?  
A:  

Q: Explain the role of the `shuffle` parameter in your training loader. Why is this setting important for the training set but not for the validation or testing sets?  
A:

---

## ✅ Week 3: Neural Network Design & Baseline Training

---

### 🏗️ 1. Neural Network Architecture

Q: How did you design your baseline Feedforward Neural Network (FFNN) architecture?  
A:  

Q: What was your rationale for the number of layers, units per layer, and activation functions used?  
A:  

Q: How did you incorporate Dropout, Batch Normalization, and ReLU in your model, and why are these components important?  
A:  

---

### ⚙️ 2. Model Training & Optimization

Q: Which loss function and optimizer did you use for training, and why are they suitable for this binary classification task?  
A:  

Q: How did you monitor and control overfitting during training?  
A:  

Q: What challenges did you face during training (e.g., convergence, instability), and how did you address them?  
A:  

---

### 📈 3. Experiment Tracking

Q: How did you use MLflow (or another tool) to track your deep learning experiments?  
A:  

Q: What parameters, metrics, and artifacts did you log for each run?  
A:  

Q: How did experiment tracking help you compare different architectures and training strategies?  
A:  

---

### 🧮 4. Model Evaluation

Q: Which metrics did you use to evaluate your neural network, and why are they appropriate for this problem?  
A:  

Q: How did you interpret the Accuracy, Precision, Recall, F1-score, and AUC results?  
A:  

Q: Did you observe any trade-offs between metrics, and how did you decide which to prioritize?  
A:  

---

### 🕵️ 5. Error Analysis

Q: How did you use confusion matrices or ROC curves to analyze your model’s errors?  
A:  

Q: What types of misclassifications were most common, and what might explain them?  
A:  

Q: How did your error analysis inform your next steps in model improvement?  
A:  

---

### 📝 6. Model Selection & Insights

Q: Based on your experiments, which neural network configuration performed best and why?  
A:  

Q: What are your top 3–5 insights from neural network development and experimentation?  
A:  

Q: How would you communicate your model’s strengths and limitations to a non-technical stakeholder?  
A:

---

## ✅ Week 4: Model Tuning & Explainability

---

### 🛠️ 1. Model Tuning & Optimization

Q: Which hyperparameters did you tune for your neural network, and what strategies (e.g., grid search, random search) did you use?  
A: I implemented a **targeted hyperparameter search** focusing on two key parameters: **learning rate** (1e-4 vs 1e-3) and **dropout rate** (0.3 vs 0.5). Rather than grid/random search, I used a **structured approach** testing 3 configurations: LowLR_SMOTE, LowDropout_SMOTE, and Optimized_SMOTE (combining both optimizations). This was a **limited search space** - in retrospect, a more comprehensive approach including architecture parameters (layer sizes, batch size) would have been beneficial.

Q: How did you implement early stopping or learning rate scheduling, and what impact did these techniques have on your training process?  
A: I implemented **early stopping with patience=15 epochs**, monitoring validation loss with no improvement threshold. This prevented overfitting and reduced training time from 50 to ~25-35 epochs on average. However, I did **not implement learning rate scheduling** - this was a missed opportunity. The lower learning rate (1e-4) showed **modest improvements** in F1 score (0.428 vs 0.422), but the impact was **less dramatic than expected**, suggesting the models may have benefited from more sophisticated scheduling.

Q: What evidence did you use to determine your model was sufficiently optimized and not overfitting?  
A: I monitored **validation loss plateauing** and **training/validation metric convergence**. However, looking critically at the results, the models were **not sufficiently optimized** - all F1 scores remained below 0.5, and SMOTE models showed concerning precision drops (29.6% vs 38.3% for class weighting). The **Week 3 Balanced_FFNN remained champion**, suggesting my Week 4 tuning was **insufficient**. More evidence needed: learning curves, validation curves, and cross-validation.  

---

### 🧑‍🔬 2. Model Explainability

Q: Which explainability technique(s) (e.g., SHAP, LIME, Integrated Gradients) did you use, and why did you choose them?  
A: I initially planned to use **SHAP (SHapley Additive exPlanations)** for comprehensive model interpretability, but encountered **technical incompatibility** with BatchNorm layers in my neural network. As a fallback, I implemented **gradient-based feature importance** using backpropagation to calculate the absolute magnitude of gradients with respect to input features. This choice was **pragmatic rather than optimal** - gradient importance provides limited insight compared to SHAP's interaction-aware explanations.

Q: How did you apply these techniques to interpret your model's predictions?  
A: I calculated **average absolute gradients** across 500 test samples, ranking features by their gradient magnitude. This revealed individual feature sensitivity but **missed interaction effects** crucial for understanding neural network behavior. I grouped features into clinical categories (cardiovascular, metabolic, lifestyle) to provide **domain-relevant insights**. However, this approach is **fundamentally limited** - it shows what the model uses, not necessarily what drives real diabetes risk.

Q: What were the most influential features according to your explainability analysis, and how do these findings align with domain knowledge?  
A: The analysis revealed **concerning misalignment with medical knowledge**: Heavy alcohol consumption ranked as the top predictor (importance: 0.0006), followed by healthcare access (NoDocbcCost) and cholesterol checking. Traditional diabetes risk factors like **BMI and age ranked surprisingly low**. This suggests either **spurious correlations**, **dataset quality issues**, or **inadequate feature engineering**. The findings **contradict established diabetes risk factors**, raising serious questions about model validity.  

---

### 📊 3. Visualization & Communication

Q: How did you visualize feature contributions and model explanations for stakeholders?  
A:  

Q: What challenges did you encounter when interpreting or presenting model explanations?  
A:  

Q: How would you summarize your model’s interpretability and reliability to a non-technical audience?