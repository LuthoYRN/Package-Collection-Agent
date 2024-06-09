# 🕹️ Reinforcement Learning Agent for FourRooms Environment 🏠🏠🏠🏠

## Overview

This project implements a reinforcement learning agent to navigate and collect packages in the FourRooms environment. The agent learns through Q-learning, updating its Q-values based on rewards received from the environment. Three scenarios are considered: simple package collection, multiple package collection, and ordered multiple package collection.

## 🚀 Features

- 🤖 Q-learning based reinforcement learning agent
- 📦 Three different scenarios for package collection
- 📊 Visualization of agent's path in the environment

## 📂 Files

- **FourRooms.py**: Contains the implementation of the FourRooms environment class, including functions for taking actions, initializing the environment, and displaying the path.
- **Scenario1.py**: Agent implementation for the simple package collection scenario.
- **Scenario2.py**: Agent implementation for the multiple package collection scenario.
- **Scenario3.py**: Agent implementation for the ordered multiple package collection scenario.
- **requirements.txt**: Contains library names needed for code to work.
- **Makefile**: Contains commands for setting up a virtual environment and running the different scripts.
- **Images/**: Contains images generated for all scenarios.

## 📸 Screenshots
<div style="display:flex;flex-direction:row;justify-content:space-between;">
<img src="images/scenario1.png" width="33%">
<img src="images/scenario2.png" width="33%">
<img src="images/scenario3.png" width="33%">
</div>

## 🔧 Installation

Follow these steps to set up the project locally.

### Prerequisites

- 🐍 Python 3.x

### Installation Steps

1. **Clone the repository**:
   ```sh
   git clone https://github.com/LuthoYRN/package-collection-agent.git
   ```
2. **Navigate to the project directory**:
   ```sh
   cd package-collection-agent
   ```
3. **Set up the environment and install dependencies**:
   ```sh
   make setup
   ```

## 🏃 Usage

### Running Scenarios

1. **Simple Package Collection**:
   ```sh
   make run1
   ```
2. **Multiple Package Collection**:
   ```sh
   make run2
   ```
3. **Ordered Multiple Package Collection**:
   ```sh
   make run3
   ```

### Running Scenarios with Stochastic Flag

1. **Simple Package Collection (Stochastic)**:
   ```sh
   make run1 FLAG=-stochastic
   ```
2. **Multiple Package Collection (Stochastic)**:
   ```sh
   make run2 FLAG=-stochastic
   ```
3. **Ordered Multiple Package Collection (Stochastic)**:
   ```sh
   make run3 FLAG=-stochastic
   ```

### Training

The script will train the reinforcement learning agent in the specified scenario for a predefined number of epochs, updating the Q-values based on rewards received.

### Visualization

After training, the script will save the final path taken by the agent in the FourRooms environment under the `Images/` folder.

## 🛠️ Requirements

- 🐍 Python 3.x

## 📌 Note

- Ensure the FourRooms environment class (`FourRooms.py`) is accessible to the scenario files.
- Ensure you set up an environment using `make setup`.
