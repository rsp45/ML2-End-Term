# CLAUDE.md — End Term Project Instructions
## Machine Learning & Algorithms 2 · STAI205 · Vijaybhoomi University · May 2026
### Target: 40 / 40

---

## 0. Mission Brief

You are an expert ML engineering agent completing a 40-mark end-term examination.  
The dataset is the **MathE platform dataset** (UCI ML Repository ID 1031).  
You will formulate and solve **three distinct ML problems** from this single dataset,  
implement baselines + ensemble methods, tune hyperparameters, and produce a  
single comprehensive `.ipynb` notebook with mathematical justification throughout.

Every section maps to a grading rubric criterion. Read the rubric before writing  
any code. Write code that you can explain line-by-line under oral examination pressure.

---

## 1. Dataset Facts (Verified)

| Property | Value |
|----------|-------|
| Source | UCI ML Repository ID 1031 — MathE Platform |
| URL | https://shorturl.at/nWPjI |
| DOI | 10.34620/dadosipb/PW3OWY |
| Rows | 9,546 answer records |
| Features | 8 raw columns |
| License | CC BY 4.0 |

### Raw Columns (exactly as in dataset)
| Column | Type | Description |
|--------|------|-------------|
| `Student_ID` | Categorical | Anonymised student identifier |
| `Student_Country` | Categorical | 8 countries: Portugal, Lithuania, Italy, Ireland, Romania, Russia, Spain, Slovenia |
| `Question_ID` | Categorical | Numeric ID of the question (833 unique questions) |
| `Answer_Value` | Binary | **0 = Correct, 1 = Incorrect** (this is the raw target) |
| `Question_Level` | Binary | "basic" or "advanced" |
| `Math_Topic` | Categorical | 14 topics (e.g., Calculus, Algebra, Statistics) |
| `Math_Subtopic` | Categorical | 24 subtopics |
| `Question_Keywords` | Text | Comma-separated keywords per question |

### Key Dataset Characteristics
- **372 unique students**, 833 unique questions, 9,546 answer records
- Data collected Feb 2019 – Dec 2023 on the MathE e-learning platform
- Students choose topic + difficulty level → platform gives 7 MCQs → answers recorded
- **Imbalanced**: More incorrect answers than correct at advanced level
- **Hierarchical structure**: student → multiple sessions → multiple answers per session
- **No continuous numerical feature exists natively** → must be engineered via aggregation

---

## 2. Grading Rubric (8 points each = 40 total)

### R1 · Problem Formulation & Feature Engineering (8 pts)
> Outstanding: Defends the logical extraction of continuous/discrete targets flawlessly.
> Articulates exactly why the formulated problems are educationally or statistically valuable.

**What the grader wants:**
- A logical, defensible reason WHY you chose each target variable
- Feature engineering that extracts new signals from raw categorical data
- Markdown cells that explain the educational/statistical motivation per problem

### R2 · Algorithmic Intuition — Baselines (8 pts)
> Outstanding: Masterfully connects the geometric or mathematical assumptions of the
> baseline model directly to the specific characteristics of this dataset.

**What the grader wants:**
- Not just "I used Logistic Regression" — but WHY its linear decision boundary is
  appropriate/inappropriate given the feature space of THIS dataset
- Connect the math of the model to the data distribution

### R3 · Ensemble Mechanics & Hyperparameter Defense (8 pts)
> Outstanding: Masterfully contrasts the ensemble's mechanics against the baseline.
> Explains all hyperparameter tuning decisions strictly through the lens of bias-variance tradeoff.

**What the grader wants:**
- Explain HOW the ensemble reduces variance OR bias compared to baseline
- Every hyperparameter tuned must be explained: "I changed X from A to B because
  the model was overfitting (high variance) on the training-validation gap"
- Show GridSearchCV / RandomizedSearchCV output

### R4 · Evaluation & Mathematical Justification (8 pts)
> Outstanding: Contextualises metrics perfectly. Explains exactly why the ensemble
> outperformed/underperformed the baseline by referencing data distribution, noise
> limits, and algorithmic structure.

**What the grader wants:**
- Not just print metrics — mathematically explain the number
- E.g. "F1 improved from 0.71 to 0.83 because boosting iteratively re-weights the
  misclassified minority class samples, reducing the bias introduced by the class imbalance"
- Reference bias-variance tradeoff in EVERY problem

### R5 · Code Ownership & AI Transparency (8 pts)
> Outstanding: Demonstrates total pipeline ownership. Can instantly trace how a complex
> AI-generated logic block functions and explain how they verified its mathematical correctness.

**What the grader wants:**
- Every non-trivial function has a `# [OWN]` comment explaining what it does and WHY
- A `prompts_log.txt` file listing every prompt used with Claude/ChatGPT/etc.
- Be ready to point to ANY line and explain it

---

## 3. The Three Problem Formulations

### Problem 1 · CLUSTERING
**Task:** Identify latent student-behaviour cohorts using per-student aggregated features.

**Feature Engineering (per student, aggregating 9,546 rows → 372 rows):**
```
accuracy_rate       = correct_answers / total_answers       ← [0, 1] continuous
advanced_attempt_rate = advanced_answers / total_answers    ← [0, 1] continuous
topic_breadth       = count of distinct Math_Topics attempted ← integer
question_volume     = total questions attempted              ← integer
avg_difficulty_preference = mean(Question_Level == 'advanced') ← [0,1]
```

**Why this is educationally valuable:**
Students cluster into natural learning archetypes: high-accuracy/low-volume (strategic),
low-accuracy/high-volume (persistent), high advanced/low accuracy (overconfident), etc.
These cohorts directly inform personalised learning paths on the MathE platform.

**Baseline:** KMeans (k=3, random_init)  
**Ensemble:** Gaussian Mixture Model (GMM) — soft probabilistic assignment  
**Tuning:** Silhouette score + Elbow method for k selection  
**Metric:** Silhouette Score, Davies-Bouldin Index, Calinski-Harabasz Index

### Problem 2 · CLASSIFICATION
**Task:** Predict whether a student's answer will be Correct (0) or Incorrect (1).

**Target variable:** `Answer_Value` (binary: 0=correct, 1=incorrect)  
**Why discrete:** Binary outcome — either the student got it right or wrong.  
**Educational value:** Predicting incorrect answers enables proactive intervention.

**Feature Engineering (per answer record — keep all 9,546 rows):**
```
country_encoded        ← LabelEncoder on Student_Country
level_encoded          ← Binary: basic=0, advanced=1
topic_encoded          ← LabelEncoder on Math_Topic
subtopic_encoded       ← LabelEncoder on Math_Subtopic
student_prior_accuracy ← rolling accuracy of student up to this answer (no leakage)
question_difficulty    ← historical incorrect_rate per Question_ID
keyword_count          ← number of keywords associated with question
```

**Baseline:** Logistic Regression (C=1.0, max_iter=1000)  
**Ensemble:** Gradient Boosting Classifier (GBM)  
**Tuning:** GridSearchCV on n_estimators, max_depth, learning_rate  
**Metric:** F1-score (weighted, due to class imbalance), ROC-AUC, Precision-Recall

### Problem 3 · REGRESSION
**Task:** Predict a student's accuracy rate from their behavioural features.

**Target variable:** `accuracy_rate` = (correct answers / total answers) per student  
**Why continuous:** Accuracy rate is a real-valued [0,1] score — infinite possible values.  
**Educational value:** Predicts student mastery level; enables early academic risk detection.

**Feature Engineering (per student — 372 rows):**
```
advanced_attempt_rate ← proportion of advanced questions attempted
topic_breadth         ← distinct topics attempted
question_volume       ← total questions answered
country_encoded       ← LabelEncoder on Student_Country  
dominant_topic        ← mode of Math_Topic per student (encoded)
```

**Baseline:** Linear Regression (OLS)  
**Ensemble:** XGBoost Regressor  
**Tuning:** RandomizedSearchCV on n_estimators, max_depth, learning_rate, subsample  
**Metric:** RMSE, MAE, R² — all three reported and explained

---

## 4. Notebook Cell Structure (EXACT ORDER)

```
Cell 00  [MD]  Title, student name, dataset citation, date
Cell 01  [MD]  ## Dataset Overview — MathE Platform
Cell 02  [PY]  Imports (all libraries declared upfront)
Cell 03  [PY]  Data loading via ucimlrepo OR direct CSV download
Cell 04  [PY]  Data inspection: shape, dtypes, value_counts, missing values
Cell 05  [MD]  ## Exploratory Data Analysis
Cell 06  [PY]  EDA: answer distribution, topic distribution, country dist, level dist
Cell 07  [PY]  EDA: correlation (after encoding), class imbalance quantification
Cell 08  [MD]  ## Feature Engineering
Cell 09  [PY]  Engineer student-level features (for Clustering + Regression)
Cell 10  [PY]  Engineer answer-level features (for Classification)
Cell 11  [MD]  ## Problem 1 — Clustering: Student Learning Archetypes
Cell 12  [MD]  ### Problem Formulation (why this target, why this approach)
Cell 13  [PY]  Preprocessing: StandardScaler on student features
Cell 14  [PY]  Elbow method + Silhouette scores to select k
Cell 15  [PY]  Baseline: KMeans(k=3, random_state=42, init='random')
Cell 16  [PY]  Baseline metrics: Silhouette, DB, CH index
Cell 17  [PY]  Ensemble: GaussianMixture(n_components=3, covariance_type='full')
Cell 18  [PY]  Ensemble metrics + comparison table
Cell 19  [PY]  Hyperparameter tuning: covariance_type sweep + BIC/AIC selection
Cell 20  [MD]  ### Mathematical Justification — Clustering results
Cell 21  [PY]  Cluster profiling: mean features per cluster, interpretation
Cell 22  [PY]  Visualisation: PCA 2D scatter coloured by cluster
Cell 23  [MD]  ## Problem 2 — Classification: Predicting Answer Correctness
Cell 24  [MD]  ### Problem Formulation
Cell 25  [PY]  Preprocessing: encode categoricals, train/test split (80/20, stratified)
Cell 26  [PY]  Class imbalance check + SMOTE if needed
Cell 27  [PY]  Baseline: LogisticRegression(C=1.0, max_iter=1000)
Cell 28  [PY]  Baseline metrics: classification_report, ROC-AUC, confusion matrix
Cell 29  [PY]  Ensemble: GradientBoostingClassifier (default params first)
Cell 30  [PY]  Hyperparameter tuning: GridSearchCV (n_estimators, max_depth, lr)
Cell 31  [PY]  Best model metrics: F1, ROC-AUC, Precision-Recall curve
Cell 32  [PY]  Feature importance bar chart
Cell 33  [MD]  ### Mathematical Justification — Classification results
Cell 34  [MD]  ## Problem 3 — Regression: Predicting Student Accuracy Rate
Cell 35  [MD]  ### Problem Formulation
Cell 36  [PY]  Preprocessing: StandardScaler, train/test split (80/20)
Cell 37  [PY]  Baseline: LinearRegression() — OLS
Cell 38  [PY]  Baseline metrics: RMSE, MAE, R²; residual plot
Cell 39  [PY]  Ensemble: XGBRegressor (default params first)
Cell 40  [PY]  Hyperparameter tuning: RandomizedSearchCV (30 iterations)
Cell 41  [PY]  Best model metrics + actual vs predicted plot
Cell 42  [MD]  ### Mathematical Justification — Regression results
Cell 43  [MD]  ## Overall Summary Table — All 3 Problems
Cell 44  [PY]  Print unified comparison table: Baseline vs Ensemble per problem
Cell 45  [MD]  ## Conclusion — Bias-Variance Analysis Across All Problems
```

---

## 5. Hard Constraints

- Every non-trivial code block must have a `# [OWN]` comment explaining WHY
- Every metric result must be followed by a Markdown cell mathematically explaining it
- GridSearchCV / RandomizedSearchCV must be shown with cv=5, scoring specified
- The `prompts_log.txt` file must contain every prompt used — this is 8 marks
- Notebook must run clean: kernel restart → Run All → zero errors
- All 3 ensemble methods must be DIFFERENT (GBM ≠ XGBoost ≠ GMM)
- Mathematical justification cells must use the words "bias", "variance", "tradeoff"

---

## 6. Key Mathematical Justifications to Prepare

### Why GBM beats Logistic Regression on this data:
> "Logistic Regression assumes linear separability in the feature space — i.e., a
> hyperplane P(Y=1|X) = σ(wᵀX) perfectly partitions correct from incorrect answers.
> But the interaction between Student_Country, Math_Topic, and Question_Level creates
> non-linear decision boundaries that no single hyperplane can capture. GBM builds an
> additive model Fₘ(x) = Fₘ₋₁(x) + η·hₘ(x) where each hₘ is a tree fitted to the
> pseudo-residuals ∂L/∂F — it iteratively corrects the errors that the linear baseline
> cannot represent. This reduces the bias without inflating variance because each weak
> tree has max_depth=3 (limited complexity)."

### Why XGBoost beats Linear Regression on this data:
> "OLS minimises Σ(yᵢ - ŷᵢ)² under the assumption of a linear relationship and
> homoscedastic residuals. But accuracy_rate is bounded [0,1] and skewed — students
> cluster near 0.2–0.4 and 0.7–0.9 with a bimodal distribution. This violates the
> linearity and normality assumptions of OLS. XGBoost's objective L(φ) = Σlᵢ(yᵢ,ŷᵢ)
> + Σ Ω(fₖ) uses second-order Taylor expansion of the loss, enabling it to capture
> this non-linear, bounded, bimodal structure that OLS flattens into a single line."

### Why GMM beats KMeans on this data:
> "KMeans assigns each student to exactly one cluster by minimising Σ ||xᵢ - μₖ||²,
> assuming clusters are spherical and equally sized. Student learning behaviour is
> not spherical — accuracy and topic breadth have different scales and the clusters
> are elliptically shaped in feature space. GMM relaxes this with a probabilistic
> model P(x) = Σₖ πₖ N(x|μₖ, Σₖ) where each cluster has its own covariance matrix Σₖ.
> This allows elliptical clusters of different sizes, better capturing the natural
> variance in student learning patterns."

---

## 7. Bias-Variance Tradeoff Explanations Per Problem

### Classification (GBM tuning):
- `n_estimators` ↑ → variance ↑ (more trees = more complex model = overfitting risk)
- `learning_rate` ↓ → bias ↑ slightly, but variance ↓ significantly (shrinkage)
- `max_depth` ↓ → bias ↑, variance ↓ (shallower trees = weaker but more stable learners)
- **Optimal**: high n_estimators + low learning_rate + shallow max_depth = sweet spot

### Regression (XGBoost tuning):
- `subsample` < 1.0 → introduces randomness → reduces variance (stochastic gradient boosting)
- `colsample_bytree` < 1.0 → reduces feature correlation between trees → reduces variance
- `reg_alpha` (L1) + `reg_lambda` (L2) → explicit regularisation → reduces overfitting

### Clustering (GMM tuning):
- `covariance_type='full'` → most flexible (low bias), highest variance (needs more data)
- `covariance_type='diag'` → assumes feature independence (higher bias, lower variance)
- Select using BIC: BIC = -2·ln(L̂) + k·ln(n) — penalises model complexity

---

## 8. Viva Defense Answers (R5 — Code Ownership)

Prepare to answer these instantly:

| Question | Answer |
|----------|--------|
| "What does GBM's pseudo-residual mean?" | Negative gradient of the loss: rᵢₘ = -[∂L(yᵢ, F(xᵢ))/∂F(xᵢ)] — the direction of steepest improvement |
| "How does XGBoost differ from GBM?" | XGBoost uses 2nd-order Taylor expansion of loss + L1/L2 regularisation + column subsampling; GBM uses only 1st-order gradients |
| "Why is F1 better than accuracy here?" | Class imbalance: if 60% incorrect, predicting all-incorrect gives 60% accuracy but 0% precision for correct class. F1 = 2PR/(P+R) balances both |
| "What is BIC and why use it for GMM?" | BIC = -2ln(L̂) + k·ln(n) penalises parameters k. Lower BIC = better model. Prevents overfitting to n_components |
| "Why StandardScaler before KMeans?" | KMeans uses Euclidean distance — features at different scales dominate. Scaler ensures equal contribution from all features |
| "What does SMOTE do mathematically?" | Synthesises new minority samples by interpolating: x_new = xᵢ + λ·(xⱼ - xᵢ) where λ~U(0,1) and xⱼ is a k-NN of xᵢ |

---

## 9. Files to Submit

| File | Description |
|------|-------------|
| `STAI205_EndTerm_Notebook.ipynb` | Main notebook with all 3 problems |
| `prompts_log.txt` | Every prompt used with Claude/any AI tool |

---

*Dataset: UCI ML Repository ID 1031 · MathE Platform · CC BY 4.0*
*Authors: Azevedo, Pacheco, Fernandes, Pereira · Instituto Politécnico de Bragança · 2024*
