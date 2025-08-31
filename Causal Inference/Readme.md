# 🔍 Amazon ML Summer School 2025 – Module 8: Causal Inference

<div align="center">

![Causal Inference](https://img.shields.io/badge/Module%207-Causal%20Inference-green?style=for-the-badge)
![Amazon ML](https://img.shields.io/badge/Learning-Amazon%20ML%20School-orange?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen?style=for-the-badge)

*From correlation to causation – Understanding cause-effect relationships for robust AI decision making!*

</div>

---

## 📅 What I Learned Today

### 🎯 **The Fundamental Problem: Correlation ≠ Causation**

#### 🚨 **Why Causal Inference Matters**
Understanding the difference between correlation and causation is crucial for making reliable decisions in AI and data science. Traditional supervised learning focuses on prediction accuracy, but causal inference answers the "What if?" questions that drive real-world interventions.

##### **Classic Examples of Spurious Correlations**
```python
# Marriage & Income: Positive correlation but age confounds both
# Ice Cream Sales & Sunburn Cases: Hot weather causes both
# Exercise & Cholesterol: Age creates misleading global pattern
# Firefighters & Fire Incidents: Government policy confounds both
```

#### 💡 **Simpson's Paradox**
A phenomenon where a trend appears in different groups of data but disappears or reverses when the groups are combined. This highlights the critical importance of identifying and controlling for confounding variables.

**Key Example**: Medical treatments showing opposite effects at aggregate vs. stratified levels due to age confounding.

---

### 🎯 **Potential Outcomes Framework (Rubin Causal Model)**

#### 🔬 **Core Concepts**
The foundation of modern causal inference, defining causal effects through counterfactual reasoning.

##### **Fundamental Notation**
```python
# For individual i and binary treatment T:
Yi(1) = Potential outcome if treated (T=1)
Yi(0) = Potential outcome if not treated (T=0)

# Individual Treatment Effect (ITE)
τi = Yi(1) - Yi(0)

# The Fundamental Problem: We can only observe one potential outcome
# If Ti = 1, we observe Yi(1) but Yi(0) is counterfactual
# If Ti = 0, we observe Yi(0) but Yi(1) is counterfactual
```

#### ⚖️ **Critical Assumptions for Causal Identification**

**1. SUTVA (Stable Unit Treatment Value Assumption)**
```
No interference between units + One version of treatment
Example violation: Product discounts affecting similar products
```

**2. Ignorability/Unconfoundedness**
```python
(Yi(1), Yi(0)) ⊥ Ti | Xi
# Treatment assignment independent of potential outcomes given confounders
# All confounders are observed and measured
```

**3. Positivity**
```python
0 < P(T=1|X=x) < 1 for all x
# Non-zero probability of treatment assignment for all covariate values
# Ensures overlap between treatment and control groups
```

---

### 🎯 **Treatment Effects Hierarchy**

#### 📊 **Types of Treatment Effects**

**Average Treatment Effect (ATE) - Population Level**
```python
ATE = E[Yi(1)] - E[Yi(0)]
# Overall effect across entire population
# Used for policy decisions and general interventions
```

**Conditional Average Treatment Effect (CATE) - Subgroup Level**
```python
CATE = E[Yi(1)|X] - E[Yi(0)|X]
# Effect within specific subgroups
# Used for targeted interventions
```

**Individual Treatment Effect (ITE) - Individual Level**
```python
ITE = Yi(1) - Yi(0)
# Personalized effect for each individual
# Used for precision medicine and personalized recommendations
```

---

### 🎯 **Classical Causal Inference Methods**

#### 📊 **1. Propensity Score Methods**

**Propensity Score Definition**
```python
e(X) = P(T=1|X) = Probability of treatment given confounders
# Key insight: If (Y(1),Y(0)) ⊥ T|X, then (Y(1),Y(0)) ⊥ T|e(X)
# Reduces high-dimensional X to scalar e(X)
```

##### **Inverse Propensity Weighting (IPW)**
```python
# ATE Estimator using importance sampling
ATE_IPW = (1/N) * Σ[Ti * Yi / e(Xi)] - (1/N) * Σ[(1-Ti) * Yi / (1-e(Xi))]

# Intuition: Reweight observations to mimic RCT
# Rare treatment assignments get higher weights
# Creates pseudo-population with no confounding
```

**Mathematical Foundation: Importance Sampling**
```python
# Want to estimate E_H[f(Y)] but have samples from G
# Solution: E_H[f(Y)] = E_G[f(Y) * h(Y)/g(Y)]
# In causal context:
# - G: Observational distribution with propensity e(X)
# - H: RCT distribution with uniform treatment assignment (0.5)
# - Weight = 0.5 / e(X) for treated, 0.5 / (1-e(X)) for control
```

##### **Doubly Robust Estimation**
```python
# Combines IPW with outcome modeling
# Two regression models: μ₀(X) = E[Y|X,T=0], μ₁(X) = E[Y|X,T=1]

ATE_DR = ATE_IPW + (1/N) * Σ[(1-Ti) * (μ₁(Xi) - μ₀(Xi))]

# Robustness: Consistent if EITHER propensity OR outcome model is correct
# Lower bias than IPW, more robust than pure regression
```

##### **Subclassification (Stratification)**
```python
# Partition data into J strata based on propensity score quantiles
# Within each stratum j: ATE_j = mean(Y_treated) - mean(Y_control)
# Overall ATE = weighted average of stratum-specific effects

# Geometric intuition: Equal weighting gives rare groups higher influence
# Converges to IPW as number of strata → ∞
```

---

### 🎯 **Modern Representation Learning Approaches**

#### 🧠 **TARNet (Treatment-Agnostic Representation Networks)**

##### **Core Innovation: Balanced Representations**
```python
# Learn representation Φ(X) that is balanced across treatment groups
# Minimize: α * L_factual + β * IPM(Φ(X)|T=1, Φ(X)|T=0)

# L_factual: Standard supervised loss on observed outcomes
# IPM: Integral Probability Metric measuring distribution distance
```

##### **Architecture Design**
```python
# Shared representation network
Φ(X) = shared_network(X)  # Learn balanced features

# Separate outcome heads for each treatment
μ₀(X) = outcome_head_0(Φ(X))  # Control outcome predictor
μ₁(X) = outcome_head_1(Φ(X))  # Treatment outcome predictor

# Individual Treatment Effect
ITE = μ₁(X) - μ₀(X)
```

##### **Maximum Mean Discrepancy (MMD)**
```python
# Practical estimate of IPM using kernel methods
MMD²(P,Q) = E_P[k(X,X')] + E_Q[k(Y,Y')] - 2*E_{P,Q}[k(X,Y)]

# Intuition: High MMD when distributions are far apart
# Used to balance representations across treatment groups
```

#### 🌳 **Causal Forests**

##### **Key Innovation: Treatment Effect-Based Splits**
```python
# Standard Random Forest: Maximize information gain
# Causal Forest: Maximize treatment effect heterogeneity

# Split criterion:
heterogeneity = |ATE(left_child) - ATE(right_child)|
best_split = argmax(heterogeneity)

# Result: Leaves contain homogeneous treatment effects
# Enables estimation of heterogeneous treatment effects
```

---

### 🎯 **Data Generative Process Comparison**

#### 📊 **Causal Inference vs. Supervised Learning vs. Reinforcement Learning**

| **Aspect** | **Causal Inference** | **Supervised Learning** | **Reinforcement Learning** |
|------------|----------------------|--------------------------|----------------------------|
| **Data Structure** | (Xi, Ti, Yi) with Ti→Yi, Xi→Ti | (Xi, Ti, Yi) with no Xi→Ti | Sequential (St, At, Rt, St+1) |
| **Objective** | Estimate counterfactuals | Minimize prediction error | Maximize long-term reward |
| **Key Challenge** | Confounding, selection bias | Overfitting, generalization | Exploration vs exploitation |
| **Distribution** | Observational ≠ Interventional | Train = Test | Policy-dependent |

---

## 🔄 Step-by-Step Revision Guide

### 📖 **Phase 1: Conceptual Foundation (Weeks 1-2)**

#### **Week 1: Understanding the Problem**
- [ ] **Day 1-2**: Study correlation vs causation examples
  - Work through ice cream/sunburn, marriage/income examples
  - Understand confounding variables and Simpson's Paradox
  - Practice identifying confounders in real scenarios

- [ ] **Day 3-4**: Master the Potential Outcomes Framework
  - Understand Yi(1), Yi(0) notation
  - Grasp the fundamental problem of causal inference
  - Learn the difference between factual and counterfactual

- [ ] **Day 5-7**: Study the three core assumptions
  - SUTVA: When does it hold/fail?
  - Ignorability: Observed vs unobserved confounders
  - Positivity: Overlap and common support

#### **Week 2: Treatment Effects**
- [ ] **Day 1-3**: Understand ATE, CATE, ITE
  - When to use each level of analysis
  - Medical vs business applications
  - Practice calculating from synthetic data

- [ ] **Day 4-7**: Study identification strategies
  - Randomized experiments vs observational data
  - Selection bias and its consequences
  - Design of natural experiments

### 📖 **Phase 2: Classical Methods (Weeks 3-4)**

#### **Week 3: Propensity Score Methods**
- [ ] **Day 1-2**: Master propensity score theory
  - Balancing property of propensity scores
  - Dimension reduction from X to e(X)
  - Connection to randomized experiments

- [ ] **Day 3-4**: Implement IPW estimator
  - Code propensity score estimation
  - Handle extreme weights and trimming
  - Calculate standard errors

- [ ] **Day 5-6**: Understand importance sampling
  - Mathematical derivation of IPW
  - Connection to survey sampling
  - Bias-variance trade-offs

- [ ] **Day 7**: Study doubly robust methods
  - Combine propensity scores with outcome modeling
  - Understand robustness properties
  - Implementation best practices

#### **Week 4: Stratification and Matching**
- [ ] **Day 1-3**: Implement subclassification
  - Quantile-based binning strategies
  - Balance checking within strata
  - Sensitivity to number of bins

- [ ] **Day 4-7**: Study matching methods
  - Nearest neighbor matching
  - Caliper matching and common support
  - Assess match quality

### 📖 **Phase 3: Modern Methods (Weeks 5-6)**

#### **Week 5: Representation Learning**
- [ ] **Day 1-3**: Understand TARNet architecture
  - Shared representations vs separate heads
  - Factual vs counterfactual loss
  - Domain adaptation connections

- [ ] **Day 4-5**: Implement MMD calculation
  - Kernel choice and hyperparameters
  - Sample size considerations
  - Computational efficiency

- [ ] **Day 6-7**: Build end-to-end TARNet
  - PyTorch/TensorFlow implementation
  - Loss function balancing
  - Hyperparameter tuning

#### **Week 6: Tree-Based Methods**
- [ ] **Day 1-4**: Implement Causal Forests
  - Modify splitting criteria
  - Handle treatment effect heterogeneity
  - Cross-validation for hyperparameters

- [ ] **Day 5-7**: Compare all methods
  - Synthetic data experiments
  - Real-world applications
  - Method selection guidelines

### 📖 **Phase 4: Advanced Topics (Weeks 7-8)**

#### **Week 7: Evaluation and Diagnostics**
- [ ] **Day 1-3**: Model selection and validation
  - Cross-validation for causal inference
  - Sensitivity analysis techniques
  - Robustness checks

- [ ] **Day 4-7**: Advanced assumptions
  - Violations and their consequences
  - Instrumental variables
  - Regression discontinuity

#### **Week 8: Applications and Extensions**
- [ ] **Day 1-4**: Real-world case studies
  - A/B testing and experimentation
  - Personalized medicine
  - Economics and policy evaluation

- [ ] **Day 5-7**: Current research frontiers
  - Causal discovery algorithms
  - Deep learning approaches
  - Federated causal inference

---

## 🚀 Why This Helps In The Future

### 🎯 **Immediate Technical Benefits**

#### **Robust Decision Making**
- **Move beyond correlation**: Make decisions based on true causal relationships
- **Avoid Simpson's Paradox**: Properly account for confounding in analysis
- **Quantify interventions**: Estimate the actual impact of policy changes
- **Handle selection bias**: Work with observational data when RCTs are impossible

#### **Advanced Analytics Applications**
- **A/B Testing**: Design and analyze experiments properly
- **Personalization**: Estimate individual treatment effects for recommendations
- **Attribution**: Understand true contribution of marketing channels
- **Policy Evaluation**: Measure effectiveness of business/government interventions

### 🌟 **Career & Research Advantages**

#### **Industry Applications**
- **Tech Companies**: 
  - Recommendation systems with causal guarantees
  - Ad attribution and marketing mix modeling
  - Platform experiments and product launches
  - User behavior analysis and intervention design

- **Healthcare & Pharma**:
  - Personalized treatment recommendations
  - Real-world evidence studies
  - Clinical trial design and analysis
  - Health economics and outcomes research

- **Finance & Insurance**:
  - Risk modeling with causal interpretation
  - Policy intervention analysis
  - Customer lifetime value modeling
  - Regulatory compliance and fair lending

- **Government & Policy**:
  - Program evaluation and impact assessment
  - Evidence-based policy making
  - Resource allocation optimization
  - Social intervention design

#### **Emerging Research Areas**
- **Causal AI**: Integration of causal reasoning in ML systems
- **Explainable AI**: Understanding model decisions through causal lens
- **Fairness & Ethics**: Detecting and removing discriminatory patterns
- **Federated Learning**: Handling heterogeneity across institutions
- **AI Safety**: Ensuring robust decision-making under distribution shift

### 💼 **Real-World Impact**

#### **Business Intelligence & Strategy**
```python
# Instead of: "Sales increased after marketing campaign"
# Causal thinking: "Marketing campaign caused X% increase in sales"
# Enables: ROI calculation, budget optimization, strategic planning
```

#### **Scientific Research**
```python
# Medical Research: Does treatment A work better than treatment B?
# Social Science: Do education programs improve outcomes?
# Economics: What's the effect of minimum wage on employment?
```

#### **Ethical AI & Fairness**
```python
# Detect discrimination: Is outcome disparity due to legitimate factors?
# Ensure fairness: Remove causal paths that shouldn't influence decisions
# Build trust: Provide causal explanations for AI decisions
```

#### **Long-term Strategic Thinking**
- **Scenario Planning**: Model consequences of different interventions
- **Risk Management**: Understand causal chains leading to failures
- **Innovation**: Identify causal mechanisms for breakthrough improvements
- **Sustainability**: Design interventions with long-term positive effects

---

## 📚 **Implementation Roadmap**

### 🔧 **Beginner Projects (Months 1-2)**
1. **Simpson's Paradox Demonstration**
   - Recreate medical treatment example
   - Show reversal through stratification
   - Visualize confounding effects

2. **Basic IPW Implementation**
   - Generate synthetic data with known treatment effects
   - Implement propensity score estimation
   - Calculate ATE using IPW

3. **Propensity Score Diagnostics**
   - Check balance before/after weighting
   - Assess common support
   - Handle extreme weights

### 🔧 **Intermediate Projects (Months 3-4)**
4. **Doubly Robust Estimator**
   - Combine propensity scores with outcome models
   - Compare robustness properties
   - Handle model misspecification

5. **TARNet Implementation**
   - Build neural network architecture
   - Implement MMD loss
   - Train on semi-synthetic datasets

6. **Causal Forest**
   - Modify random forest splitting criteria
   - Estimate heterogeneous treatment effects
   - Validate on known ground truth

### 🔧 **Advanced Projects (Months 5-6)**
7. **A/B Test Analysis Platform**
   - Handle network effects and interference
   - Multiple testing corrections
   - Heterogeneous treatment effect analysis

8. **Causal Discovery Pipeline**
   - Implement PC algorithm
   - Structure learning from data
   - Validate discovered relationships

9. **Real-world Case Study**
   - Choose domain (healthcare, marketing, policy)
   - Apply multiple causal methods
   - Compare results and provide recommendations

---

## 📖 **Essential Resources**

### **📚 Foundational Textbooks**
- **"Causal Inference: The Mixtape"** by Scott Cunningham
  - Practical, code-heavy approach
  - Real-world examples and implementations
  - Excellent for practitioners

- **"Causality: Models, Reasoning and Inference"** by Judea Pearl
  - Theoretical foundation and causal diagrams
  - Mathematical rigor and formal proofs
  - Essential for researchers

- **"Mostly Harmless Econometrics"** by Angrist & Pischke
  - Applied econometric perspective
  - Natural experiments and identification strategies
  - Policy evaluation focus

### **💻 Implementation Libraries**

**Python Ecosystem**
```python
# Core libraries
import causalml          # Uber's causal ML library
import dowhy            # Microsoft's causal inference
import econml           # Microsoft's econometric ML
import causalinference  # Standard causal methods

# Deep learning extensions
import pytorch_lightning
import tensorflow_probability
```

**R Ecosystem**
```r
# Essential packages
library(grf)           # Generalized Random Forests
library(tmle)          # Targeted Maximum Likelihood
library(MatchIt)       # Matching methods
library(WeightIt)      # Propensity score weighting
```

### **🎯 Practical Datasets**
- **IHDP**: Infant Health and Development Program
- **Jobs**: LaLonde's job training program evaluation
- **Twins**: Natural experiment data
- **ACIC**: Atlantic Causal Inference Conference challenges

---

## 🎯 **Quick Reference Guide**

### **When to Use Each Method**
| **Scenario** | **Best Method** | **Rationale** |
|--------------|----------------|---------------|
| **High-dimensional confounders** | TARNet, DR methods | Handle complexity |
| **Strong propensity overlap** | IPW, Subclassification | Simple, interpretable |
| **Heterogeneous effects needed** | Causal Forest, TARNet | Individual-level estimates |
| **Model uncertainty** | Doubly Robust | Robustness guarantee |
| **Limited data** | Matching | Non-parametric |

### **Common Pitfalls & Solutions**
| **Problem** | **Symptom** | **Solution** |
|-------------|-------------|--------------|
| **Unmeasured confounding** | Implausible effect sizes | Sensitivity analysis, IV |
| **Positivity violations** | Extreme weights | Trimming, overlap checks |
| **Model misspecification** | Poor balance | Ensemble methods, DR |
| **Selection bias** | Strong T-Y correlation | Better confounder measurement |

### **Key Formulas**
```python
# Propensity Score
e(x) = P(T=1|X=x)

# IPW Estimator
ATE = E[T*Y/e(X)] - E[(1-T)*Y/(1-e(X))]

# Doubly Robust
ATE_DR = ATE_IPW + E[(1-T)*(μ₁(X) - μ₀(X))]

# MMD (Maximum Mean Discrepancy)
MMD²(P,Q) = E[k(X,X')] + E[k(Y,Y')] - 2E[k(X,Y)]
```

---

**#CausalInference #TreatmentEffects #PropensityScore #TARNet #CausalForests #DoublyRobust #AmazonMLSummerSchool #MachineLearning #Statistics #DataScience #RevisionGuide**

---

> *"The goal is to replace much of human intuition about causation with a more principled approach based on causal models. Causal inference is not just a statistical technique – it's a way of thinking about the world that leads to more reliable decisions and scientific discoveries."*
