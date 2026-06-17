# 🦯 Blind Navigator 

> *A deep reinforcement learning agent that learns to navigate safely through dynamic environments using only sensor signals — no camera, no map, and no hand-crafted navigation rules.*

[![Streamlit App](https://img.shields.io/badge/Live%20Demo-Streamlit-FF4B4B?logo=streamlit)](https://blindnavigator-w7xz2xhhus9rpehszdthne.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch)](https://pytorch.org)
[![Gymnasium](https://img.shields.io/badge/Gymnasium-0.29-black)](https://gymnasium.farama.org)

---

## 🏆 Results at a Glance

| Algorithm | Success Rate | Mean Reward | Collision Rate |
|-----------|:-----------:|:-----------:|:--------------:|
| **Dueling-PER v1** | ✅ **100%** | +545 | 23.0% |
| **Dueling-PER v2** *(Safety)* | ✅ **100%** | +526 | **7.7%** ⬇️ |
| Q-Learning | ❌ 0% | -168 | 73.7% |
| SARSA | ❌ 0% | -174 | 74.7% |

*Evaluated over 300 episodes · Phase 4 (maximum complexity) · 6 diverse maps*

> **Note on tabular baselines**: The poor performance of Q-Learning and SARSA should not be interpreted as a weakness of the algorithms themselves. Rather, it reflects the fundamental mismatch between tabular representations and a high-dimensional continuous observation space. This comparison is intentional — it demonstrates that *representation capacity* matters as much as the choice of algorithm.

---

## 🎯 What Is This?

Blind Navigator is a deep reinforcement learning project that explores how intelligent navigation systems can be developed for future assistive technologies.

The agent perceives the world through **20 sensor values** — like echolocation — and must navigate from start to goal across a **16×16 grid** while avoiding hazards, dynamic pedestrians, battery constraints, and movement uncertainty.

Rather than relying on a camera, a predefined map, or manually programmed navigation rules, the agent **learns entirely through interaction and reward-based learning**.

### Real-World Potential

| Environment | Challenge |
|-------------|-----------|
| 🏥 Hospital corridors | Crowded, complex layout |
| 🏙️ Urban crossings | Moving pedestrians and changing surroundings |
| 🏢 Office buildings | Dynamic obstacles and indoor navigation |
| 🚉 Transit hubs | High-density pedestrian traffic |

---

## 🗺️ The Environment

**Custom Gymnasium environment** — `BlindNavigatorEnv`

- **Grid**: 16×16 (256 cells) with 6 BFS-validated maps
- **Maps**: Office Corridors · Hospital · Street Crossing · Mall · Apartment · Outdoor Park
- **Agent starts** at top-left corner → must reach **Goal ★** at bottom-right
- **Obstacles**: Static hazards · Edge/curb zones · Up to 5 social-force pedestrians
- **Constraints**: Battery drain every step · Motor slip (up to 12%) · Loop detection

### 20-Dimensional Sensor State

```
[0–7]   8-directional LiDAR distances
[8]     Goal distance (Manhattan)
[9]     Goal angle (relative bearing)
[10–13] Nearest pedestrian: distance, angle, velocity x/y
[14]    Battery level (normalized)
[15–16] Heading sin/cos
[17]    Loop density (visited-cell counter)
[18–19] Last action + steps normalized
```

### Reward Structure

| Event | Reward |
|-------|--------|
| Goal reached | **+500** + time bonus |
| Progress toward goal | +3.0 |
| Pedestrian collision | **-100** (v1) · **-300** (v2) |
| Static hazard | -50 |
| Battery exhaustion | -30 |
| Near-miss *(v2 only)* | up to -15 |
| Loop penalty | -5 |
| Safe step | -0.5 |

---

## 🤖 Algorithms

### 1. Q-Learning & SARSA (Tabular Baselines)

Classic tabular RL methods that store state-action values in a Q-table.

**Limitation**: The continuous 20-dimensional state space causes severe discretization challenges. Both methods perform reasonably in simple environments but collapse to **0% success** as complexity increases — not because the algorithms are flawed, but because the environment has outgrown what a lookup table can represent.

### 2. Dueling Double-DQN + PER + LSTM ⭐

**Why LSTM?** Because the agent does not observe the full environment, navigation becomes a **Partially Observable MDP (POMDP)**. The LSTM provides memory of previous observations and actions, helping the agent infer hidden state over time — enabling loop detection and path memory that emerge purely from training.

```
                    ┌─────────────┐
  Sensor Input      │   Encoder   │   Linear(20→256)
   (20-dim)  ──────▶│ LayerNorm   │   + ReLU × 2
                    │   + ReLU    │
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │    LSTM     │   256 → 128
                    │  (memory)   │   temporal context
                    └──────┬──────┘
                           │
              ┌────────────┴────────────┐
              │                         │
       ┌──────▼──────┐           ┌──────▼──────┐
       │ Value Head  │           │  Adv. Head  │
       │  V(s) → 1   │           │ A(s,a) → 4  │
       └──────┬──────┘           └──────┬──────┘
              │                         │
              └────────────┬────────────┘
                           │
                    Q(s,a) = V(s) + A(s,a) − mean(A)
```

**Why this combination works:**
- **Dueling architecture** — separates state-value estimation from action selection, critical in corridors where most moves have similar value
- **LSTM** — captures temporal dependencies without hand-crafted features
- **PER** — focuses learning on near-collisions and surprising pedestrian interactions
- **Double DQN** — reduces value overestimation and improves training stability

---

## 📚 5-Phase Curriculum Training

| Phase | Episodes | Environment | Key Challenge |
|-------|----------|-------------|---------------|
| 0 | 3,000 | Empty maps | Basic navigation |
| 1 | 4,000 | Static hazards + 5% slip | Obstacle avoidance |
| 2 | 6,000 | 2 moving pedestrians | Dynamic avoidance |
| 3 | 8,000 | 4 pedestrians + 10% slip + all 6 maps | Full complexity |
| 4 | 8,000 | 5 pedestrians + 12% slip | Generalization |

**Total: 29,000 training episodes per algorithm**

Key technique: **per-phase epsilon restart** — the agent re-explores whenever environment complexity increases, preventing catastrophic forgetting and improving adaptation to new challenges.

---

## 🛡️ Safety Upgrade — v2

`core_v2.py` introduces a **near-miss penalty** — the agent is penalized when approaching a pedestrian too closely, *before* any collision occurs.

```python
PEDESTRIAN_COLLISION_REWARD = -300   # v1: -100
NEAR_MISS_PENALTY_MAX       = -15    # v2: new
NEAR_MISS_THRESHOLD         = 0.2    # normalized sensor distance
```

**Result**: Collision rate drops from **23.0% → 7.7%** while maintaining 100% success rate.

This upgrade encourages **proactive safety-aware navigation** rather than purely reactive behavior — the difference between an agent that *reacts* to danger and one that *anticipates* it.

---

## 📊 Dashboard

Interactive research dashboard built with **Streamlit** — 4 pages:

| Page | Content |
|------|---------|
| 🏠 Overview | Project summary, KPIs, real-world applications |
| 🗺️ Environment | Live map viewer, reward structure, curriculum breakdown |
| 📊 Training Results | Learning curves, per-phase/per-map analysis, head-to-head |
| 🤖 Agent Walkthrough | Side-by-side algorithm comparison with live animation |

### 🔗 [Live Demo → blindnavigator.streamlit.app](https://blindnavigator-w7xz2xhhus9rpehszdthne.streamlit.app/)

### Run Locally

```bash
git clone https://github.com/Ahd-Sayed/Blind_Navigator
cd Blind_Navigator
pip install -r requirements.txt
streamlit run app.py
```

---

## ⏱️ Training Cost

| | Details |
|---|---|
| **Hardware** | Local GPU (NVIDIA RTX 2050) |
| **v1 Training Time** | ~6 hours (29,000 episodes × 3 algorithms) |
| **v2 Retraining Time** | ~5 hours (22,000 episodes, phases 2–4) |
| **Total Episodes** | 87,000 (v1) + 22,000 (v2) |
| **Framework** | PyTorch — local training, no cloud required |

---

## ⚠️ Limitations
- **Research scope only** — this project is intended as a research-oriented simulation and not as a medical or deployment-ready assistive system.
- **Grid-world abstraction** — real environments have continuous spaces and richer sensor modalities
- **Simulated sensor noise** — motor slip and pedestrian behavior are simplified models
- **No real-world deployment** — results are simulation-based only
- **Limited pedestrian diversity** — social-force model does not capture all real human movement patterns
- **Single agent** — no coordination with other robots or infrastructure

---

## 📁 Project Structure

```
Blind_Navigator/
├── core.py                    ← v1 environment + all agents
├── core_v2.py                 ← v2 environment (near-miss penalty)
├── app.py                     ← Streamlit dashboard
├── training_notebook.ipynb    ← Full v1 training pipeline
├── retrain_v2.ipynb           ← v2 retraining pipeline
├── requirements.txt
├── .streamlit/                ← Streamlit configuration
├── outputs/                   ← v1 results
│   ├── dueling_per_eval.json
│   ├── q_learning_eval.json
│   ├── sarsa_eval.json
│   ├── learning_curves.png
│   └── *_curriculum_logs.json
└── outputs_v2/                ← Safety upgrade results
    ├── core_v2.py
    ├── retrain_v2.ipynb
    ├── dueling_per_v2_final.pt
    ├── v2_summary.json
    └── dueling_per_eval.json
```

---

## 🚀 Future Work

- Real-world sensor integration (ultrasonic, IR)
- Voice-guided navigation assistance
- Mobile robot deployment
- Multi-agent pedestrian interaction scenarios
- Vision-assisted hybrid navigation

---

## 🛠️ Tech Stack

![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch)
![Gymnasium](https://img.shields.io/badge/Gymnasium-0.29-black)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?logo=streamlit)
![NumPy](https://img.shields.io/badge/NumPy-1.26-013243?logo=numpy)
![Matplotlib](https://img.shields.io/badge/Matplotlib-3.x-11557C?logo=python)

---

## 🏫 Academic Context

Developed as part of **AI318 — Reinforcement Learning**
Faculty of Artificial Intelligence · Menoufia University

---

## 🙏 Acknowledgments

Special thanks to **Ahmed Heikal, Mohamed Medhat, Mohamed Alla, and Aya Adel** for their support and shared learning throughout this journey.

---

*Built with curiosity, persistence, and a lot of failed episodes 🦯*
