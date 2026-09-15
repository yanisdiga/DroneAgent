# 🚁 Autonomous Drone RL Environment (Gymnasium)

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Gymnasium](https://img.shields.io/badge/Gymnasium-1.0-green)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0-orange)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

> **2D Simulation environment for training autonomous agents to navigate, explore, and track moving targets in constrained environments without GPS.**

---

## 📸 Demo

![Simulation Preview](https://via.placeholder.com/800x400?text=Insert+Gameplay+GIF+Here)

*Figure 1: The drone (green) using LIDAR rays to avoid obstacles (red) and track the target.*

---

## 🚀 Key Features

This project implements a custom **Reinforcement Learning (RL)** environment compatible with the `Gymnasium` API. It was designed to study autonomous decision-making under uncertainty.

* **Custom Physics Engine:** Implements inertia, friction, and "wall sliding" logic to simulate realistic movement constraints.
* **Complex Sensory Inputs:** The agent does not rely on absolute coordinates (No GPS). It uses:
    * **LIDAR:** 16-ray distance sensors for obstacle detection.
    * **Vector Radar:** Relative direction and signal strength to the target.
    * **Spatial Memory (Heatmap):** Olfactory-like sensors to remember visited zones and encourage exploration.
* **Dynamic Scenarios:** Randomized maps, obstacle generation, and moving targets for robust training.
* **Anti-Stuck Mechanism:** "Shake" logic to free the agent if physically trapped in local minima.

---

## 🧠 Environment Architecture

### 1. Observation Space (25 Continuous Values)
The neural network receives a normalized vector `[-1, 1]`:

| Index | Feature | Description |
| :--- | :--- | :--- |
| `0-1` | **Proprioception** | Current Velocity ($v_x, v_y$). |
| `2-3` | **Radar Vector** | Normalized direction vector pointing to the target. |
| `4` | **Radar Signal** | Proximity intensity (0.0 = Far, 1.0 = On Target). |
| `5-20` | **LIDAR** | 16 rays measuring distance to walls (1.0 = Danger). |
| `21-24` | **Heatmap Sensors** | "Smell" of visited areas in 4 directions (N, S, E, W). |

### 2. Action Space (Continuous)
The agent controls the drone via a continuous vector:
* `[dx, dy]`: Thrust vector applied to the physics engine.

### 3. Reward Function (Hybrid)
To solve the *sparse reward* problem, the environment uses a composite function:
* **Exploration:** `+Reward` for discovering new "cold" cells in the heatmap.
* **Hunting:** `+Reward` for reducing distance to the target (Dense reward).
* **Capture:** `+500` for stabilizing on the target for >1 second.
* **Survival:** Small penalty per step to encourage speed; Penalty for collisions.

---

## 🛠️ Installation

```bash
# Clone the repository
git clone [https://github.com/yanisdiga/DroneAgent.git](https://github.com/yanisdiga/DroneAgent.git)
cd DroneAgent

# Install dependencies
gymnasium
numpy
pygame
stable-baselines3
torch
