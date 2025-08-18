# 🌟 Amazon ML Summer School 2025 – Module 3: Dimensionality Reduction

<div align="center">

![Dimensionality Reduction](https://img.shields.io/badge/Module%203-Dimensionality%20Reduction-blue?style=for-the-badge)
![Amazon ML](https://img.shields.io/badge/Learning-Amazon%20ML%20School-orange?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen?style=for-the-badge)

*Roadmap to simplifying complex, high-dimensional data – theory, intuition, math, and practical use-cases.*

</div>

---

## 📚 Step-by-Step Revision Guide

### 1️⃣ **Foundations: Why Dimensionality Reduction?**

- **Curse of Dimensionality:** 
  - As features (dimensions) increase, data becomes sparse, models overfit, computation slows, and interpretation suffers.
  - Example: Bag of words in text = 10,000+ features. Most are irrelevant!
- **Types of Problems:**
  - High latency (too many features in real-time).
  - Regulatory/Legal need for explainable models (e.g., loan approvals).
  - Model interpretability for trust and debug.
- **Solution:** Reduce the feature count via:
  - **Feature Selection:** Keep only the important features.
  - **Feature Extraction:** Re-combine features into informative lower-dimensional ones.

---

### 2️⃣ **Feature Selection Techniques (What to Keep?)**

#### A. **Wrapper Methods**
   - **_Sequential Forward Selection:_** Start with none, add one feature at a time that most improves accuracy. Repeat until adding features doesn’t help.
   - **_Backward Elimination:_** Start with all features, remove the least helpful one each time. Good after forward selection to remove redundancies.
   - **_Limitation:_** 2ⁿ possible subsets for n features. Greedy methods often used since exhaustive search isn’t feasible.

#### B. **Filter Methods**
   - **_Statistical scoring to rank features:_**
     - **Pearson Correlation Coefficient:** Linear relation for numerical features.
     - **Mutual Information:** Measures information gain; works for categorical features.
     - **Chi-squared Test:** For classification tasks.
   - **_Process:_** Score each feature → Sort → Select top-k.

   **_Limitation:_** Can’t capture feature interactions (e.g., two features only predictive when used together).

#### C. **Embedded Methods**
   - **Model-based selection built-in to training**.
     - **L1 Regularization (Lasso):** Shrinks some feature weights to zero. Great for sparse, interpretable models.
     - **L2 Regularization (Ridge):** Shrinks all weights but usually keeps most nonzero.
     - **Decision trees/ensemble methods:** Pick features by how useful they are for splitting.

---

### 3️⃣ **Feature Extraction (Combining/Reprojecting Features)**

#### A. **Principal Component Analysis (PCA)**
   - **Goal:** Find orthogonal axes (principal components) capturing the most variance.
   - **Procedure:**
     1. **(Optional):** Center data (zero-mean for each feature).
     2. Compute covariance matrix.
     3. Find eigenvectors and eigenvalues (directions of max variance and their strengths).
     4. Project data onto first k eigenvectors to reduce dimension.
   - **Properties:**
     - **Linear technique:** Components are weighted sums of original features.
     - Most info captured in first few components.
     - **Reconstruction:** Can approximately recover the original data from reduced representation.
   - **Applications:** Visualization, noise reduction, image compression.

#### B. **Singular Value Decomposition (SVD)**
   - **Goal:** Decompose large matrix (e.g., user x item, term x doc) into product of three matrices (U, S, V).
     - `X = U S Vᵗ`
     - U, V: orthogonal/unitary; S: singular values.
   - **Use-cases:**
     - Topic modeling (latent semantic analysis).
     - Recommender systems.
     - Handles missing data via matrix factorization.

#### C. **t-SNE (t-Distributed Stochastic Neighbor Embedding)**
   - **Goal:** Visualize high-dimensional data in 2D/3D by preserving local relationships.
   - **How:** 
     - Converts high-dim distances to probabilities.
     - Tries to ensure close points in high dim stay close in 2D/3D.
     - Uses Student-t dist in low-dim for better separation (clusters more visible).
   - **Limitation:** Nonlinear/non-parametric (can’t project new points without retraining), slow for very large datasets.

#### D. **Non-negative Matrix Factorization (NMF)**
   - **Goal:** Factorize data matrix into the product of two or more low-dimensional, non-negative matrices.
   - **Result:** Latent “parts”/factors are always positive, leading to more interpretable topics/components (e.g., in text, images).

---

### 4️⃣ **Intuition with Math: How the Techniques Work**

#### - **L1/L2 Regularization:**
   - Add penalty to loss:
     - L1: Makes sum of absolute weights small → sparsity.
     - L2: Makes sum of squared weights small → shrinkage, smoother weights.
   - L1 “punishes” large weights more for individual features, driving some to exact zero.

#### - **SVD/PCA Visualization:**
   - SVD identifies directions with maximal “energy” in your data; PCA uses leading eigenvectors.
   - If original data is a high-rank (fat) matrix, most signal lies in a few “big” directions (principal components).

#### - **Matrix Factorization (for Recommendations):**
   - Approximates user-item matrix as products of user-feature and item-feature matrices.
   - Learns user/item “embeddings” in smaller space.
   - Updates via alternating least squares or stochastic gradient descent.

---

### 5️⃣ **Practical Examples & Applications**

| Area                        | Dim Red Use           | Real-World Example             |
|-----------------------------|-----------------------|-------------------------------|
| Text Classification         | Feature Selection, NMF| Filter out irrelevant words, find topics & key phrases |
| Computer Vision             | PCA, SVD, NMF         | Image compression, face recognition, dimension reduction for clustering |
| Recommender Systems         | Matrix Factorization  | User & item embeddings, Netflix/Amazon product splines |
| Genomics                    | PCA/SVD               | Visualize ancestry, infer groups from genetic data      |
| Explainable AI              | L1/L2, Filter Methods | Reduce model to human-size number of features           |
| Real-time ML (Edge, Mobile) | All                   | Reduce computations for fast predictions                |

---

### 6️⃣ **Review Table: Choosing a Dimensionality Reduction Technique**

| Technique         | Preserves Interactions? | Linear/Nonlinear | Best For                               |
|-------------------|------------------------|------------------|----------------------------------------|
| Filter Methods    | No                     | Linear           | Quick, large datasets (feature ranking)|
| Wrapper Methods   | Yes (some)             | Linear/Nonlinear | Small dataset, high interpretability   |
| Embedded (L1/L2)  | Yes (model-driven)     | Linear           | Sparse, interpretable, compressed     |
| PCA/SVD           | No                     | Linear           | Visualization, structure discovery     |
| t-SNE/UMAP        | Partial                | Nonlinear        | Visualizing clusters from raw data     |
| NMF               | Yes                    | Linear           | Topic modeling (positive features)     |
| Matrix Factorization| Yes                  | Linear           | Recommendations, missing data          |

---

## 🟩 **Why This Matters for the Future**

- **Better Generalization:** Avoid overfitting, improve downstream model performance (focus on signal, not noise).
- **Speed:** Faster prediction & training—major edge for real-world and big data problems.
- **Explainability:** Smaller models and input sets are clearer (trust, regulatory compliance).
- **Visualization:** See big picture and patterns humanly (plots, clusters, anomaly detection).
- **Industrial Impact:** Used at Amazon and everywhere for recommender engines, vision, speech, and beyond.
- **Foundation for Deep Learning:** Modern neural networks (like autoencoders, BERT) take these concepts further!

---

## 🔁 **Quick Revision Checklist**

- [ ] Can you explain in one line why dimensionality reduction is needed?
- [ ] Do you know the PROs and CONs of wrapper, filter, and embedded feature selection methods?
- [ ] Can you perform PCA step-by-step on a small dataset?
- [ ] Do you know how SVD helps in practical applications?
- [ ] Can you sketch the difference between L1 and L2 regularization?
- [ ] Do you understand when to use t-SNE versus PCA?
- [ ] Can you explain matrix factorization in a movie recommendation setting?
- [ ] Do you know what makes NMF factors interpretable?

---

 
---

**#DimensionalityReduction #PCA #SVD #NMF #AmazonMLSummerSchool #DataScience #FeatureSelection #MachineLearning #RevisionGuide**
