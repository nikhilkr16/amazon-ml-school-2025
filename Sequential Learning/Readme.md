# 🔄 Amazon ML Summer School 2024 – Module 5: Sequential Learning

<div align="center">

![Sequential Learning](https://img.shields.io/badge/Module%205-Sequential%20Learning-blue?style=for-the-badge)
![Amazon ML](https://img.shields.io/badge/Learning-Amazon%20ML%20School-orange?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen?style=for-the-badge)

*From Hidden Markov Models to Transformers – Mastering sequence data for NLP, speech, and time series!*

</div>

---

## 📅 What I Learned Today

### 🌟 **Part 1: Hidden Markov Models (HMMs) by Arin**

#### 🔍 **Core Concept**
Hidden Markov Models are **generative probabilistic models** for sequential data where:
- **Observable sequence**: X₁, X₂, ..., Xₙ (words we see)
- **Hidden states**: Y₁, Y₂, ..., Yₙ (POS tags we want to predict)
- **Goal**: Model joint distribution P(X,Y) to find optimal Y given X

#### 🏗️ **HMM Components**
```
π_i = Initial state probabilities
a_{i,j} = Transition probabilities (state i → state j)  
b_{j,k} = Emission probabilities (state j emits symbol k)
```

#### 🎯 **Three Fundamental Problems**

##### **1. Inference Problem (Viterbi Algorithm)**
- **Task**: Find most probable hidden state sequence
- **Method**: Dynamic programming with Viterbi decoding
- **Formula**: 
  ```
  δ_s(t) = max probability of reaching state s at time t
  δ_s(t) = max[δ_{s'}(t-1) × a_{s',s}] × b_s(x_t)
  ```
- **Complexity**: O(T × N²) vs brute force O(N^T)

##### **2. Likelihood Problem (Forward-Backward Algorithm)**
- **Forward Algorithm**: α_s(t) = P(x₁...xₜ, state=s at t)
- **Backward Algorithm**: β_s(t) = P(x_{t+1}...x_T | state=s at t)
- **Total Likelihood**: Σ_s α_s(T) or Σ_s α_s(1) × β_s(1)

##### **3. Training Problem**
**Supervised Learning:**
```
a_{i,j} = Count(i→j) / Count(i)
b_{j,k} = Count(j emits k) / Count(j)
```

**Unsupervised Learning (EM Algorithm):**
- **E-step**: Compute posterior probabilities using current parameters
- **M-step**: Update parameters using expected counts
- **Iterate** until convergence

#### 🗣️ **Speech Applications**
- **Continuous observations**: Use Gaussian Mixture Models instead of discrete probabilities
- **Process**: Audio → Spectral features → HMM → Phone sequence

---

### 🌟 **Part 2: Attention & Transformers by Pram Verma**

#### 🎯 **Sequence-to-Sequence Models**
```
Encoder: Input → Hidden States → Final Context Vector
Decoder: Context + Previous Output → Next Output
```

#### ⚡ **Attention Mechanism Revolution**

##### **Why Attention?**
- **Problem**: Single context vector loses information in long sequences
- **Solution**: Use ALL encoder hidden states, not just the last one

##### **Bahdanau Attention (Additive)**
```
1. Linear transform: h_dec × W_dec, h_enc × W_enc  
2. Add & activate: tanh(W_dec × h_dec + W_enc × h_enc)
3. Score: W_align × tanh(...)
4. Softmax: attention weights
5. Context: weighted sum of encoder states
```

##### **Luong Attention (Multiplicative)**
- **Dot product**: h_dec · h_enc
- **General**: h_dec^T × W × h_enc  
- **Concat**: W × [h_dec; h_enc]

#### 🔄 **Problems with RNNs**
1. **Linear interaction distance**: Hard to capture long dependencies
2. **No parallelization**: Sequential processing bottleneck

---

#### 🚀 **Self-Attention & Transformers**

##### **Self-Attention Mechanism**
```
For each word, create:
- Query (Q) = X × W_Q
- Key (K) = X × W_K  
- Value (V) = X × W_V

Attention(Q,K,V) = softmax(QK^T/√d_k)V
```

##### **Step-by-Step Process**
1. **Linear projections**: Transform inputs to Q, K, V vectors
2. **Compute scores**: Q·K for all word pairs
3. **Scale**: Divide by √d_k to prevent large values
4. **Softmax**: Normalize to get attention weights
5. **Weighted sum**: Multiply weights with Values

##### **Multi-Head Attention**
- **Concept**: Run attention in parallel with different W_Q, W_K, W_V
- **Benefits**: Capture different types of relationships
- **Implementation**: 8 heads × 64 dims = 512 total dimensions

##### **Positional Encoding**
- **Problem**: Self-attention has no inherent position awareness
- **Solution**: Add position vectors to input embeddings
- **Types**: Sinusoidal (fixed) or Learnable embeddings

#### 🏗️ **Transformer Architecture**

##### **Encoder Stack**
```
Input → Positional Encoding → 
Multi-Head Self-Attention → 
Add & Norm → 
Feed Forward → 
Add & Norm → Next Layer
```

##### **Decoder Stack**
```
Output → Positional Encoding →
Masked Self-Attention →  
Add & Norm →
Encoder-Decoder Attention →
Add & Norm → 
Feed Forward →
Add & Norm → Next Layer
```

##### **Key Components**
- **Residual connections**: X + Attention(X) for gradient flow
- **Layer normalization**: Stabilize training
- **Feed-forward**: Position-wise fully connected layers

---

#### 🔤 **Subword Modeling**

##### **Why Subwords?**
- **Word-level problems**: OOV words, infinite vocabulary
- **Solution**: Break words into meaningful subunits

##### **Byte Pair Encoding (BPE)**
```
1. Start with character vocabulary
2. Find most frequent character pair
3. Merge pair, add to vocabulary  
4. Repeat until desired vocab size
5. "unfortunately" → "un" + "fort" + "un" + "ate" + "ly"
```

##### **Other Methods**
- **Unigram**: Likelihood-based tokenization
- **WordPiece**: Google's approach for BERT
- **SentencePiece**: Language-agnostic tokenization

---

#### 🏋️ **Pre-training Strategies**

##### **GPT (Decoder Pre-training)**
- **Objective**: Language modeling (predict next word)
- **Architecture**: Stack of Transformer decoders
- **Fine-tuning**: Add classification head for downstream tasks

##### **BERT (Encoder Pre-training)**
- **Objective**: Masked Language Modeling + Next Sentence Prediction
- **Masking**: 15% of tokens (80% mask, 10% random, 10% unchanged)
- **Architecture**: Stack of Transformer encoders
- **Bidirectional**: Can see full context (unlike GPT)

##### **T5 (Text-to-Text Transfer Transformer)**
- **Innovation**: Every task as text generation
- **Architecture**: Full encoder-decoder Transformer
- **Pre-training**: Span corruption (predict masked spans)
- **Versatility**: Same model for translation, summarization, classification

---

## 🔄 Step-by-Step Revision Guide

### **📖 Core Concept Review**

#### **1. HMM Fundamentals**
- [ ] Can you explain the three HMM parameters (π, A, B)?
- [ ] Do you understand Markov and observation independence assumptions?
- [ ] Can you trace through Viterbi algorithm step-by-step?

#### **2. Attention Mechanisms**  
- [ ] Why is attention better than fixed context vectors?
- [ ] Can you compute attention weights manually?
- [ ] What's the difference between Bahdanau and Luong attention?

#### **3. Self-Attention Deep Dive**
- [ ] How do Q, K, V matrices transform input representations?
- [ ] Why do we scale by √d_k in attention computation?
- [ ] What problems does multi-head attention solve?

#### **4. Transformer Architecture**
- [ ] Can you trace data flow through encoder and decoder stacks?
- [ ] Why are residual connections and layer normalization important?
- [ ] How does positional encoding solve the position problem?

### **🧠 Mathematical Understanding**

#### **HMM Calculations**
```python
# Viterbi Algorithm
δ[t][s] = max(δ[t-1][s'] * a[s'][s]) * b[s][obs[t]]

# Forward Algorithm  
α[t][s] = Σ(α[t-1][s'] * a[s'][s]) * b[s][obs[t]]
```

#### **Attention Computation**
```python
# Self-Attention
scores = Q @ K.T / sqrt(d_k)
weights = softmax(scores)
output = weights @ V
```

### **🔧 Implementation Checklist**

#### **For HMMs:**
- [ ] Implement Viterbi decoding
- [ ] Code Forward-Backward algorithms  
- [ ] Build EM training loop

#### **For Transformers:**
- [ ] Implement multi-head attention
- [ ] Add positional encoding
- [ ] Build encoder/decoder stacks

### **📊 Practical Applications**

#### **When to Use What:**
| **Task** | **Best Model** | **Why** |
|----------|----------------|---------|
| POS Tagging | HMM/BiLSTM-CRF | Sequential labeling with dependencies |
| Machine Translation | Transformer | Long sequences, parallel training |
| Text Classification | BERT | Bidirectional context understanding |
| Text Generation | GPT | Autoregressive language modeling |
| Multi-task Learning | T5 | Unified text-to-text framework |

---

## 🚀 Why This Helps In The Future

### **🎯 Immediate Technical Benefits**

#### **NLP Mastery**
- **Foundation Skills**: Understanding from classical (HMM) to modern (Transformers)
- **Architecture Selection**: Know when to use attention vs recurrence vs convolution
- **Pre-training Intuition**: Leverage massive models efficiently

#### **Production-Ready Knowledge**
- **Scalability**: Transformers parallelize better than RNNs
- **Transfer Learning**: Pre-trained models reduce time-to-market
- **Subword Handling**: Robust tokenization for real-world text

### **🌟 Career & Research Advantages**

#### **Industry Applications**
- **Search & Ranking**: Attention mechanisms in retrieval systems
- **Conversational AI**: Seq2seq models power chatbots and virtual assistants  
- **Content Generation**: GPT-style models for creative applications
- **Code Understanding**: Transformers excel at programming language tasks

#### **Research Directions**
- **Efficient Transformers**: Linear attention, sparse patterns
- **Multimodal Learning**: Vision + Language with cross-attention
- **Few-shot Learning**: In-context learning with large language models

### **💼 Real-World Impact**

#### **At Amazon Scale**
- **Alexa NLU**: Intent detection across multiple languages
- **Product Search**: Understanding customer queries with context
- **Review Analysis**: Sentiment and aspect extraction from text
- **Machine Translation**: Supporting global e-commerce

#### **Emerging Opportunities**
- **Large Language Models**: Contributing to GPT, PaLM, ChatGPT development
- **AI Safety**: Understanding model behavior and alignment
- **Efficient AI**: Making Transformers work on mobile and edge devices

---

## 📚 **Quick Reference Tables**

### **HMM vs Neural Approaches**
| **Aspect** | **HMM** | **Neural Models** |
|------------|---------|-------------------|
| **Interpretability** | High (explicit states) | Low (learned representations) |
| **Data Requirements** | Low | High |
| **Scalability** | Limited | Excellent |
| **Long Dependencies** | Poor | Excellent (Transformers) |

### **Attention Mechanisms Comparison**
| **Type** | **Computation** | **Use Case** |
|----------|----------------|--------------|
| **Bahdanau** | Additive (concat + MLP) | RNN-based seq2seq |
| **Luong** | Multiplicative (dot product) | Faster computation |
| **Self-Attention** | Q, K, V projections | Transformers |
| **Cross-Attention** | Q from decoder, K,V from encoder | Encoder-decoder models |

### **Pre-training Objectives**
| **Model** | **Objective** | **Architecture** | **Best For** |
|-----------|---------------|------------------|--------------|
| **GPT** | Next token prediction | Decoder-only | Text generation |
| **BERT** | Masked LM + NSP | Encoder-only | Text understanding |
| **T5** | Span corruption | Encoder-decoder | Multi-task learning |

---

## 🎯 **Revision Checklist**

### **✅ Conceptual Understanding**
- [ ] HMM three problems and their solutions
- [ ] Attention mechanism intuition and mathematics
- [ ] Transformer architecture components and data flow
- [ ] Pre-training vs fine-tuning strategies

### **✅ Mathematical Proficiency**
- [ ] Viterbi and Forward-Backward algorithms
- [ ] Attention score computation and normalization
- [ ] Backpropagation through self-attention layers

### **✅ Practical Skills**
- [ ] Implement basic HMM for POS tagging
- [ ] Code self-attention mechanism from scratch
- [ ] Use pre-trained models (BERT, GPT) for downstream tasks
- [ ] Apply subword tokenization techniques

### **✅ Application Knowledge**
- [ ] Know which model to use for different NLP tasks
- [ ] Understand computational trade-offs
- [ ] Recognize when to use transfer learning vs training from scratch

---

## 📖 **Essential References**

### **📚 Papers to Read**
- **Attention Is All You Need** (Vaswani et al., 2017)
- **BERT: Bidirectional Encoder Representations** (Devlin et al., 2018)
- **Language Models are Unsupervised Multitask Learners** (GPT-2)
- **Exploring the Limits of Transfer Learning** (T5)

### **💻 Implementation Resources**
- **Hugging Face Transformers**: Pre-trained model library
- **The Annotated Transformer**: Line-by-line implementation guide
- **PyTorch Tutorials**: Official sequence modeling tutorials

 
---

**#SequentialLearning #HiddenMarkovModels #Attention #Transformers #BERT #GPT #T5 #NLP #AmazonMLSummerSchool #MachineLearning #RevisionGuide**

---

> *"From the statistical elegance of HMMs to the parallel power of Transformers – sequential learning has evolved, but the core challenge remains: understanding and generating meaningful sequences from data."*
