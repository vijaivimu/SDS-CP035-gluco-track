# 🔴 GlucoTrack – Advanced Track

## ✅ Week 1: Exploratory Data Analysis (EDA)

---

### 📦 1. Data Integrity & Structure

**Q: Are there any missing, duplicate, or incorrectly formatted entries in the dataset?**  
**A:** No missing values were detected for any column (all columns show 0 missing). The notebook also reported **0 duplicate rows**. Column value ranges and sample previews matched the documented codebook expectations, so no incorrectly formatted entries were observed.

**Q: Are all data types appropriate (e.g., numeric, categorical)?**  
**A:** Yes. Binary indicators are encoded as 0/1 integers (e.g., `HighBP`, `HighChol`, `Smoker`, etc.). Ordinal categories appear as small integer buckets (e.g., `GenHlth` 1–5, `Education` 1–6, `Income` 1–8, `Age` 1–13). Count/continuous-like features (`BMI`, `MentHlth`, `PhysHlth`) are numeric. This aligns with the dataset’s schema and supports both classical ML and NN pipelines.

**Q: Did you detect any constant, near-constant, or irrelevant features?**  
**A:** **No constant features** were found. Unique-value counts show at least two levels for all predictors. `ID` is analytically irrelevant and will be excluded from modeling to prevent leakage.

---

### 🎯 2. Target Variable Assessment

**Q: What is the distribution of `Diabetes_binary`?**  
**A:** The distribution shows: **Non‑Diabetes: 86.0667%**, **Pre-diabetes/Diabetes: 13.9333%**.

**Q: Is there a class imbalance? If so, how significant is it?**  
**A:** Yes—substantial. The negative:positive ratio is approximately **6.18 : 1**, indicating a minority positive class (~14%).

**Q: How might this imbalance influence your choice of evaluation metrics or model strategy?**  
**A:** Accuracy alone would be misleading. We will prioritize **PR‑AUC** (baseline equals the positive prevalence), report **ROC‑AUC**, and track **recall** at clinically reasonable false‑positive rates. During training we’ll use **class weighting** (and consider focal loss / threshold tuning). Probability **calibration** will be checked before deployment.

---

### 📊 3. Feature Distribution & Quality

**Q: Which numerical features are skewed or contain outliers?**  
**A:** The histograms/boxplots indicate **right‑skew** and visible high-end outliers for **`BMI`**, **`PhysHlth`**, and **`MentHlth`**.

**Q: Did any features contain unrealistic or problematic values?**  
**A:** None observed. Ordinal features stayed within documented ranges (e.g., `GenHlth` 1–5; `Age` 1–13; `Education` 1–6; `Income` 1–8). `MentHlth`/`PhysHlth` counts ranged 0–30 as expected. `BMI` shows high values/outliers but nothing flagged as invalid in the notebook outputs.

**Q: What transformation methods (if any) might improve these feature distributions?**  
**A:** For linear/NN models we’ll **standardize** key numerics (`BMI`, `PhysHlth`, `MentHlth`). If tails remain influential, we’ll consider **winsorization** or a **Yeo‑Johnson/Quantile** transform on those features. Tree models generally handle skew without transformation.

---

### 📈 4. Feature Relationships & Patterns

**Q: Which categorical features (e.g., `GenHlth`, `PhysicalActivity`, `Smoking`) show visible patterns in relation to `Diabetes_binary`?**  
**A:** The “positive‑rate by category” plots show clear signal for:  
- **`GenHlth`**: prevalence increases monotonically from *Excellent* to *Poor*.  
- **`PhysActivity`**: active individuals show a lower positive rate than inactive.  
- **`HighBP` / `HighChol`**: groups with value 1 exhibit noticeably higher positive rates.  
- **Age buckets**: older age groups show higher prevalence.

**Q: Are there any strong pairwise relationships or multicollinearity between features?**  
**A:** The correlation heatmap highlighted a strong pair: **`GenHlth`–`PhysHlth` ≈ 0.524** (|ρ|>0.5). Several other health‑status variables showed moderate associations, so we’ll monitor multicollinearity (especially for linear baselines) and consider dropping/combining redundant features if needed.

**Q: What trends or correlations stood out during your analysis?**  
**A:** Worse self‑reported health, cardiometabolic indicators (`HighBP`, `HighChol`), and inactivity aligned with higher diabetes prevalence; age also showed an increasing trend. `BMI` displayed a positive association with the target in stratified views.

---

### 🧰 5. EDA Summary & Preprocessing Plan

**Q: What are your 3–5 biggest takeaways from EDA?**  
**A:**  
1) **Class imbalance** is significant (~14% positive), so metric choice and training strategy must account for it.  
2) **Health‑status and behavior variables** (`GenHlth`, `PhysActivity`, `HighBP`, `HighChol`, `BMI`, `Age`) carry strong signal.  
3) **Skewed numerics** (`BMI`, `PhysHlth`, `MentHlth`) warrant scaling and possibly robust transforms.  
4) **No missing data** and **no duplicates** detected; the table is analysis‑ready.  
5) **Some collinearity** among health status variables (e.g., `GenHlth` with `PhysHlth`) should be managed for linear models.

**Q: Which features will you scale, encode, or exclude in preprocessing?**  
**A:**  
- **Scale**: `BMI`, `PhysHlth`, `MentHlth` (standardization; consider robust transform if needed).  
- **Encode**: Treat `GenHlth`, `Age`, `Education`, `Income` as **ordinal** for classical models; for NNs, use **integer indices + embeddings**. Binary flags remain as 0/1.  
- **Exclude**: `ID` (non‑predictive identifier).

**Q: What does your cleaned dataset look like (rows, columns, shape)?**  
**A:** Notebook reports **(253,680 rows × 24 columns)** with 0 missing and 0 duplicates. For modeling, we will **drop `ID`**, yielding **23 features + 1 target** (24 columns total during training artifacts).

