# 🎮 Amazon ML Summer School 2025 – Module 6: Reinforcement Learning

<div align="center">

![Reinforcement Learning](https://img.shields.io/badge/Module%206-Reinforcement%20Learning-purple?style=for-the-badge)
![Amazon ML](https://img.shields.io/badge/Learning-Amazon%20ML%20School-orange?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen?style=for-the-badge)

*From Multi-Armed Bandits to Deep Q-Networks – Learning optimal decisions through trial and error!*

</div>

---

## 📅 What I Learned Today

### 🎯 **Part 1: Multi-Armed Bandits & Contextual Bandits**

#### 🎰 **Multi-Armed Bandits (MAB)**
Multi-Armed Bandits are a simplified version of reinforcement learning where:
- **No states**: Only actions and immediate rewards
- **Goal**: Maximize cumulative reward over time
- **Key challenge**: **Exploration vs Exploitation** dilemma

##### **Core MAB Algorithms**

**1. ε-Greedy Algorithm**
```python
# With probability ε: explore (random action)
# With probability (1-ε): exploit (best known action)
if random() < epsilon:
    action = random_action()
else:
    action = argmax(Q_values)
```

**2. Upper Confidence Bound (UCB)**
```
UCB(a) = Q̄(a) + √(2 ln(t) / N(a))
        ↑        ↑
   exploitation  exploration
```
- **First term**: Average reward (exploitation)
- **Second term**: Confidence interval (exploration)
- **Intuition**: Choose actions with high reward OR high uncertainty

**3. Thompson Sampling (Bayesian Approach)**
```python
# For each arm:
# 1. Sample θ from posterior P(θ|data)
# 2. Compute value using sampled θ
# 3. Choose arm with highest sampled value
```

##### **Thompson Sampling Example: Product Recommendation**
- **Rewards**: Binary (purchase/no purchase)  
- **Model**: Bernoulli distribution with Beta prior
- **Posterior**: Beta(α + purchases, β + impressions - purchases)
- **Conjugate prior advantage**: Closed-form posterior updates

#### 🎯 **Contextual Bandits**
Extension of MAB where:
- **Context**: User features, product features, environmental factors
- **Personalization**: Different optimal actions for different contexts
- **Applications**: Personalized recommendations, targeted advertising

##### **Contextual Algorithms**

**Linear UCB**
```
UCB(s,a) = θᵀφ(s,a) + α√(φ(s,a)ᵀA⁻¹φ(s,a))
           ↑              ↑
      predicted reward   uncertainty
```

**Key Advantages:**
- **ε-Greedy**: Simple, explicit exploration control
- **UCB**: Principled exploration, no hyperparameter tuning
- **Thompson Sampling**: Efficient, handles delayed feedback well

---

### 🎯 **Part 2: Reinforcement Learning Fundamentals**

#### 🔄 **Markov Decision Process (MDP)**
**5-tuple**: (S, A, P, R, γ)
- **S**: Set of states
- **A**: Set of actions  
- **P**: Transition probabilities P(s'|s,a)
- **R**: Reward function R(s,a)
- **γ**: Discount factor (0 ≤ γ ≤ 1)

##### **Key RL Concepts**

**1. Policy (π)**
- **Deterministic**: π(s) = a
- **Stochastic**: π(a|s) = probability of action a in state s

**2. Value Functions**
```
Action-Value Function:
Q^π(s,a) = E[Σ(γᵗRₜ₊₁) | Sₜ=s, Aₜ=a, π]

Optimal Action-Value Function:
Q*(s,a) = max_π Q^π(s,a)
```

**3. Bellman Equations**
```
Bellman Expectation:
Q^π(s,a) = Σ P(s'|s,a)[R(s,a) + γ Σ π(a'|s') Q^π(s',a')]

Bellman Optimality:
Q*(s,a) = Σ P(s'|s,a)[R(s,a) + γ max_{a'} Q*(s',a')]
```

#### 🎯 **Value-Based RL: Q-Learning**

##### **Q-Learning Algorithm**
```python
# Initialize Q(s,a) arbitrarily
for episode in episodes:
    s = initial_state
    while s != terminal:
        # ε-greedy action selection
        a = epsilon_greedy(Q, s, epsilon)
        # Take action, observe reward and next state
        r, s_prime = environment.step(s, a)
        # Q-learning update
        Q[s,a] = Q[s,a] + α(r + γ max_a' Q[s',a'] - Q[s,a])
        s = s_prime
```

##### **Key Features**
- **Model-free**: No need for transition probabilities
- **Off-policy**: Can learn optimal policy while following different policy
- **Temporal Difference**: One-step updates (between Monte Carlo and DP)
- **Convergence**: Proven to converge to Q* under certain conditions

#### 🎯 **Policy-Based RL: REINFORCE**

##### **Policy Gradient Theorem**
```
∇J(θ) = E[Σₜ ∇log π(aₜ|sₜ,θ) Gₜ]
        ↑     ↑                ↑
    expectation  gradient of    return
                log policy
```

##### **REINFORCE Algorithm**
```python
# Sample episode τ ~ π_θ
for each step t in episode:
    # Compute return from step t
    G_t = Σ(γ^k * r_{t+k+1})
    # Update policy parameters
    θ = θ + α * ∇log π(a_t|s_t,θ) * G_t
```

##### **Q-Learning vs REINFORCE Comparison**
| **Aspect** | **Q-Learning** | **REINFORCE** |
|------------|----------------|---------------|
| **Learning** | Value function → Policy | Direct policy learning |
| **Action Space** | Discrete (argmax needed) | Continuous friendly |
| **Data Usage** | Off-policy (can reuse data) | On-policy (need fresh data) |
| **Bias/Variance** | High bias, low variance | Low bias, high variance |
| **Convergence** | Faster | Slower |

---

### 🎯 **Part 3: Deep Reinforcement Learning**

#### 🧠 **Deep Q-Networks (DQN)**

##### **Why Deep Learning?**
- **High-dimensional states**: Raw pixels, continuous spaces
- **Function approximation**: Neural networks replace lookup tables
- **Generalization**: Learn from seen states to unseen states

##### **DQN Architecture**
```python
# Input: State (e.g., 84x84x4 stacked frames)
# Output: Q-values for each action

class DQN(nn.Module):
    def __init__(self, input_shape, n_actions):
        self.conv_layers = nn.Sequential(
            nn.Conv2d(4, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU()
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(64*7*7, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )
```

##### **DQN Innovations**

**1. Experience Replay**
```python
# Store experiences in buffer
replay_buffer.store(state, action, reward, next_state, done)

# Sample random batch for training
batch = replay_buffer.sample(batch_size)
# This breaks correlation between consecutive experiences
```

**2. Target Network**
```python
# Separate network for computing targets
target_q = reward + γ * max_a' Q_target(next_state, a')
loss = (Q_main(state, action) - target_q)²

# Periodically update target network
if step % update_frequency == 0:
    Q_target.load_state_dict(Q_main.state_dict())
```

##### **DQN Training Process**
```python
for episode in range(num_episodes):
    state = env.reset()
    for step in range(max_steps):
        # ε-greedy with neural network
        if random() < epsilon:
            action = random_action()
        else:
            action = argmax(Q_network(state))
        
        # Environment interaction
        next_state, reward, done = env.step(action)
        
        # Store experience
        replay_buffer.store(state, action, reward, next_state, done)
        
        # Train network on random batch
        if len(replay_buffer) > batch_size:
            batch = replay_buffer.sample(batch_size)
            train_network(batch)
        
        state = next_state
        if done: break
```

#### 🏆 **Amazing DQN Achievements**
- **Atari 2600**: Single architecture, 49 games, human-level performance
- **AlphaGo**: Defeated world champion Lee Sedol
- **AlphaZero**: Mastered Chess, Shogi, and Go from scratch
- **AlphaStar**: Reached Grandmaster level in StarCraft II

---

## 🔄 Step-by-Step Revision Guide

### 📖 **Conceptual Foundation**

#### **1. Multi-Armed Bandits Mastery**
- [ ] Can you explain the exploration-exploitation trade-off?
- [ ] Do you understand why UCB balances exploration and exploitation?
- [ ] Can you derive the Thompson Sampling update for Beta-Bernoulli case?
- [ ] What are the advantages of contextual bandits over regular MAB?

#### **2. MDP & Value Functions**
- [ ] Can you define the 5-tuple of an MDP?
- [ ] What's the difference between value function and action-value function?
- [ ] How do Bellman equations relate current and future values?
- [ ] Why do we need discount factor γ?

#### **3. RL Algorithms Deep Dive**
- [ ] Can you trace through one iteration of Q-learning update?
- [ ] What makes Q-learning "off-policy"?
- [ ] How does REINFORCE use policy gradient theorem?
- [ ] When would you choose value-based vs policy-based methods?

### 🧠 **Mathematical Understanding**

#### **MAB Regret Analysis**
```python
# Total regret definition
Regret = T * μ* - Σ(t=1 to T) μ(a_t)
        ↑         ↑
   optimal       actual
   performance   performance
```

#### **Q-Learning Update Rule**
```python
# Temporal Difference Error
δ = r + γ max_a' Q(s',a') - Q(s,a)
    ↑     ↑                 ↑
  reward  future value    current estimate

# Update rule
Q(s,a) ← Q(s,a) + α * δ
```

#### **Policy Gradient**
```python
# Log-derivative trick
∇J(θ) = E[∇log π(a|s,θ) * G_t]
# Where G_t is the return from time t
```

### 🔧 **Implementation Checklist**

#### **For Multi-Armed Bandits:**
- [ ] Implement ε-greedy algorithm
- [ ] Code UCB algorithm with confidence bounds
- [ ] Build Thompson Sampling for Beta-Bernoulli rewards
- [ ] Extend to contextual bandits with LinUCB

#### **For Q-Learning:**
- [ ] Implement tabular Q-learning
- [ ] Add ε-greedy exploration strategy
- [ ] Test on simple grid world environments
- [ ] Analyze convergence behavior

#### **For Deep RL:**
- [ ] Implement experience replay buffer
- [ ] Build target network mechanism
- [ ] Create DQN training loop
- [ ] Test on OpenAI Gym environments

### 📊 **Algorithm Selection Guide**

| **Problem Type** | **Best Algorithm** | **Why** |
|------------------|-------------------|---------|
| **Simple bandit** | Thompson Sampling | Efficient, Bayesian |
| **Contextual recommendation** | LinUCB | Handles context well |
| **Discrete state RL** | Q-Learning | Proven convergence |
| **Continuous action** | Policy Gradient | Direct policy optimization |
| **High-dimensional state** | DQN | Function approximation |
| **Real-time strategy** | Actor-Critic | Balance bias-variance |

---

## 🚀 Why This Helps In The Future

### 🎯 **Immediate Technical Benefits**

#### **Decision-Making Systems**
- **Recommendation engines**: Personalized content, products, ads
- **Resource allocation**: Cloud computing, network routing
- **Trading strategies**: Algorithmic trading, portfolio optimization
- **Clinical trials**: Adaptive treatment assignment

#### **Production-Ready Skills**
- **A/B testing**: Contextual bandits for experiment optimization
- **Real-time personalization**: Online learning algorithms
- **Cold start problem**: Exploration strategies for new users/items
- **Multi-objective optimization**: Reward engineering and trade-offs

### 🌟 **Career & Research Advantages**

#### **Industry Applications**
- **Autonomous systems**: Self-driving cars, robotics, drones
- **Game development**: AI opponents, procedural content generation
- **Finance**: Risk management, fraud detection, portfolio management  
- **Healthcare**: Treatment protocols, drug discovery, medical imaging

#### **Emerging Technologies**
- **Large Language Models**: RLHF (Reinforcement Learning from Human Feedback)
- **Robotics**: Learning manipulation, navigation, human-robot interaction
- **Multi-agent systems**: Distributed optimization, game theory
- **AI safety**: Alignment, interpretability, robustness

### 💼 **Real-World Impact**

#### **At Amazon Scale**
- **Product recommendations**: Millions of users, billions of items
- **Supply chain optimization**: Inventory, pricing, logistics
- **Alexa**: Natural language understanding, dialogue management
- **AWS**: Resource scheduling, auto-scaling, cost optimization

#### **Cutting-Edge Research**
- **Foundation models**: Training large-scale AI systems
- **Human-AI collaboration**: Interactive machine learning
- **Sim-to-real transfer**: Virtual training, real-world deployment
- **Ethical AI**: Fair, transparent, accountable decision-making

---

## 📚 **Quick Reference Tables**

### **Bandit Algorithms Comparison**
| **Algorithm** | **Exploration** | **Computational Cost** | **Best For** |
|---------------|----------------|------------------------|--------------|
| **ε-Greedy** | Fixed rate | O(1) | Simple, interpretable |
| **UCB** | Adaptive | O(K) | No hyperparameters |
| **Thompson Sampling** | Probabilistic | O(K) | Bayesian, efficient |

### **RL Algorithm Categories**
| **Category** | **Examples** | **Pros** | **Cons** |
|--------------|--------------|----------|----------|
| **Value-Based** | Q-Learning, DQN | Sample efficient | Limited to discrete actions |
| **Policy-Based** | REINFORCE, PPO | Continuous actions | High variance |
| **Actor-Critic** | A3C, SAC | Best of both | More complex |

### **Deep RL Techniques**
| **Technique** | **Purpose** | **Implementation** |
|---------------|-------------|-------------------|
| **Experience Replay** | Break correlation | Store and sample transitions |
| **Target Networks** | Stabilize training | Separate network for targets |
| **Double DQN** | Reduce overestimation | Decouple action selection/evaluation |
| **Dueling DQN** | Better value estimation | Separate value and advantage streams |

---

## 🎯 **Revision Checklist**

### **✅ Theoretical Understanding**
- [ ] Multi-armed bandit problem formulation and regret minimization
- [ ] MDP components and Bellman equations
- [ ] Difference between model-based and model-free methods
- [ ] Policy gradient theorem and REINFORCE derivation

### **✅ Algorithmic Proficiency**
- [ ] Implement ε-greedy, UCB, and Thompson Sampling
- [ ] Code Q-learning with function approximation
- [ ] Build experience replay and target networks
- [ ] Understand when to use each algorithm type

### **✅ Practical Applications**
- [ ] Design reward functions for real problems
- [ ] Handle exploration-exploitation in production systems
- [ ] Scale RL algorithms to high-dimensional problems
- [ ] Evaluate and debug RL systems

### **✅ Advanced Concepts**
- [ ] Understand why deep RL is challenging
- [ ] Know limitations and failure modes
- [ ] Recognize when RL is/isn't appropriate
- [ ] Connect RL to other ML paradigms

---

## 📖 **Essential Resources**

### **📚 Must-Read Papers**
- **Playing Atari with Deep Reinforcement Learning** (Mnih et al., 2013)
- **Human-level control through deep reinforcement learning** (Mnih et al., 2015)
- **Policy Gradient Methods** (Sutton et al., 2000)
- **Thompson Sampling for Contextual Bandits** (Agrawal & Goyal, 2013)

### **💻 Implementation Resources**
- **OpenAI Gym**: Standard RL environments
- **Stable Baselines3**: High-quality RL implementations
- **DeepMind Lab**: 3D learning environments
- **Ray RLlib**: Scalable RL library

### **📖 Textbooks**
- **Reinforcement Learning: An Introduction** (Sutton & Barto)
- **Algorithms for Reinforcement Learning** (Szepesvári)
- **Bandit Algorithms** (Lattimore & Szepesvári)

---

**#ReinforcementLearning #MultiArmedBandits #DeepRL #QLearning #PolicyGradient #DQN #AmazonMLSummerSchool #MachineLearning #RevisionGuide**

---

> *"Reinforcement Learning is not just about algorithms – it's about learning to make sequential decisions under uncertainty, which is the essence of intelligence itself."*
