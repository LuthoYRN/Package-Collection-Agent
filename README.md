## Reinforcement Learning Agent for FourRooms Environment

### Overview
This project implements a reinforcement learning agent to navigate and collect packages in the FourRooms environment. The agent learns through Q-learning, updating its Q-values based on rewards received from the environment. Three scenarios are considered: simple package collection, multiple package collection, and ordered multiple package collection.

### Files
- **FourRooms.py**: Contains the implementation of the FourRooms environment class, including functions for taking actions, initializing the environment, and displaying the path.
- **Scenario1.py**: Implements the agent for the simple package collection scenario.
- **Scenario2.py**: Implements the agent for the multiple package collection scenario.
- **Scenario3.py**: Implements the agent for the ordered multiple package collection scenario.
- **requirements.txt**: Contains library names needed for code to work
- **Makefile**: Has commands for setting up a virtual environment and running the different scripts
- **Images/** : Has images generated for all scenarios

### Usage
1. **Navigate to the Project Directory**: Open a terminal and change the directory to where the project files are located.

2. **Setup**: Ensure you have Python 3 installed on your system.
              Create a virtual environment with installed dependancies using `make setup`

3. **Execute Scenario File**: Run the desired scenario file (Scenario1.py, Scenario2.py, or Scenario3.py) using the makefile command
 `make run1` for Scenario1.py
 `make run2` for Scenario2.py
 `make run3` for Scenario3.py

 **running scenarios with stochastic flag**
 `make run1 FLAG=-stochastic` for a stochastic Scenario1.py
 `make run2 FLAG=-stochastic` for a stochastic Scenario2.py
 `make run3 FLAG=-stochastic` for a stochastic Scenario3.py
 
4. **Training**: The script will train the reinforcement learning agent in the specified scenario for a predefined number of epochs, updating the Q-values based on rewards received.
5. **Visualization**: After training, the script will save the final path taken by the agent in the FourRooms environment under the images folder.

### Requirements
- Python 3.x

### Note
- Ensure the FourRooms environment class (FourRooms.py) is accessible to the scenario files.
- Ensure you setup an environment using `make setup`

## Developed on Nightmare