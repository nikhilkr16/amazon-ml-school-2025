# 🤖 Amazon ML Summer School 2025 – Module 4: Unsupervised Learning Mastery

<div align="center">

![Unsupervised Learning](https://img.shields.io/badge/Module%204-Unsupervised%20Learning-blue?style=for-the-badge)
![Amazon ML](https://img.shields.io/badge/Learning-Amazon%20ML%20School-orange?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen?style=for-the-badge)

*From finding hidden patterns to learning new data representations – Unsupervised Learning at scale!*

</div>

---

## 📆 Summary: What I Learned Today

### 🌟 1️⃣ **Unsupervised Learning - The Big Picture**
- Goal: Extract meaningful patterns and structure from unlabeled data.
- Real-world motivation: Most data is unlabeled—unsupervised methods are essential in business, science, and AI!

---

### 🌟 2️⃣ **Core Techniques & Algorithms**

#### A. **Dimensionality Reduction**
- **Why:** Easier visualization, model simplicity, input compression.
- **Methods:**
  - **PCA (Principal Component Analysis):**
    - Projects data onto axes maximizing variance.
    - Great for data compression or noise reduction.
    - Steps:
      1. Center data.
      2. Compute covariance matrix.
      3. Find eigenvectors/values.
      4. Project onto top components.
  - **Matrix Factorization:** For collaborative filtering, topic models.

#### B. **Clustering**
- **Goal:** Group data points by similarity, discover hidden subgroups or patterns.
- **Main Types:**
  - **Centroid-based (K-Means):** Assigns each point to the nearest cluster center, iteratively optimizes.
  - **Connectivity-based (Hierarchical):** Builds clusters by merging or splitting, often visualized as dendrograms.
  - **Density/Distribution-based (Gaussian Mixture, DBSCAN):** Models data density, probabilistic cluster membership.
- **K-Means Step-by-Step:**
  1. Choose number of clusters (K), randomly initialize K centers.
  2. Assign each point to nearest center.
  3. Update centers to mean of assigned points.
  4. Repeat until convergence.
- **Applications:** Customer segmentation, image compression, anomaly detection.

#### C. **Generative Modeling**
- **Goal:** Learn data distribution for sampling, anomaly detection, or imputation.
- **Approaches:**
  - **Classical (Naive Bayes, Multivariate Gaussian):** Estimate explicit probability models.
  - **GANs (Generative Adversarial Networks):**
    - Two networks (Generator & Discriminator) compete.
    - Generator produces fake samples; Discriminator distinguishes real vs. fake.
    - Trained via adversarial loss; produces new samples or images.
    - Foundation of modern deep generative modeling.
  - **Conditional GANs:** Generate under specific conditions (e.g. turn dog sketch into dog photo).

---

### 🌟 3️⃣ **Representation Learning (for Text, Graphs, Images)**
- **Text:**
  - **Word2Vec/Skipgram:** Learns word embeddings based on context (“meaning is context”).
  - **ELMo, BERT:** Contextualized embeddings; word meaning changes by sentence.
- **Graphs:**
  - **Node2Vec, DeepWalk:** Embedding graph nodes using random walks & skipgram analogy.
  - **Applications:** Social graphs, recommendation, link prediction.
- **Images:**
  - **Self-Supervised Learning (SimCLR, Contrastive Learning):**
    - Augment images, train network to recognize similar (positive) pairs vs. different (negative) pairs.
    - SOTA for extracting features and transfer learning from unlabelled images.

---

## 🧩 Step-by-Step Revision Guide

### 1️⃣ When to Use Unsupervised Learning?
- Data is mostly unlabeled
- You want to discover structure (groups, patterns) without prior labels
- Feature engineering, input compression, data cleaning

### 2️⃣ Which Method?  
| Task                       | Go-to Algorithm   |
|----------------------------|------------------|
| Visualize or reduce input  | PCA, t-SNE       |
| Group customers, patterns  | K-Means, Hierarchical |
| Generate new samples       | GANs, VAEs       |
| Feature extraction         | Word2Vec, Node2Vec, SimCLR |

### 3️⃣ Practical Steps (Reference Checklist)
- [ ] For clustering: Normalize data, select K, run K-Means, interpret clusters
- [ ] For PCA: Center data, compute covariance, project onto top PCs, plot/interpret
- [ ] For GAN: Set up Generator & Discriminator, alternate training, check generator outputs
- [ ] For representation: Train embeddings (Word2Vec, Node2Vec), visualize using t-SNE/PCA, use downstream

### 4️⃣ Understanding the Math
- **PCA:** Eigenvalues/vectors of covariance, maximum variance
- **K-Means:** Minimize total within-cluster distance
- **GANs:** Minimax adversarial loss, competition dynamic
- **Word2Vec:** Negative sampling for speeding up training, softmax for word prediction

### 5️⃣ Business & Research Impact
- **Customer Segmentation:** Marketing, customer experience
- **Anomaly Detection:** Finance, cybersecurity, manufacturing fault detection
- **Personalized Recommendations:** E-commerce, streaming services
- **Image/text generation & feature extraction:** Deep learning foundation

---

## 🚀 Why This Helps In The Future

- **Unlocks Unlabelled Data Value:** Use 90%+ raw data for automatic insights.
- **Foundation for Generative AI:** Enhances creativity, boosts data augmentation, critical in synthetic media.
- **Better, Cheaper Models:** Reduced need for expensive annotation; unsupervised pretraining now SOTA in NLP/CV tasks.
- **Builds Intuition for Research:** Core concepts apply in deep learning architectures, anomaly detection, more.

---

## 📚 **Reference Links**

- 💻 **Session Video:** [AMAZON ML Summer School 2024 – Module 4: Unsupervised Learning](https://youtu.be/Dfc3xSHEbrk?si=rMbHyMvsT0RVU00v)
- 📖 **Recommended Reading:**
  - Deep Learning, Chapter 15: Unsupervised Feature Learning and Deep Learning (Goodfellow et al.)
  - “A Tutorial on Unsupervised Machine Learning” (ArXiv)
  - GANs in Action (Book)

---

**#UnsupervisedLearning #Clustering #GANs #RepresentationLearning #PCA #Word2Vec #Node2Vec #ContrastiveLearning #MachineLearning #AmazonMLSummerSchool #RevisionGuide #DataScience**

---

> _“Unsupervised learning is not just a technique; it's a mindset for discovery. The hidden patterns you reveal today become the insights of tomorrow!”_
