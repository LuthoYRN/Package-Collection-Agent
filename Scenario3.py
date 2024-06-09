import sys
import numpy as np
import random
from FourRooms import FourRooms

# Constants
EPOCHS = 5000
EPSILON_GREEDY = 0.6
DECAY_RATE = 0.8
DISCOUNT_FACTOR = 0.9
colors = {1: "Red", 2: "Green", 3: "Blue"}

def initialize_environment(stochastic = False):
    # Create a FourRooms object for multi-package collection scenario
    room = FourRooms('rgb',stochastic = stochastic)
    
    # Initialize Q and R tables
    Q_table = np.zeros((169, 4, 4))  # Q-values table for state-action pairs
    R_table = np.zeros((169, 4, 4))  # Rewards table for state-action pairs
    
    print('....Starting Simulation....')
    if stochastic:
        print("....stochastic....")
    print('....Agent starts at', room.getPosition())  # Print initial agent position
    return room, Q_table, R_table

def train_agent(room, Q_table, R_table):
    for epoch in range(EPOCHS):
        room.newEpoch()  # Start a new epoch
        prev_position = room.getPosition()  # Get initial agent position
        exploration_rate = EPSILON_GREEDY  # Initialize exploration rate
        remaining_packages = 3  # Initialize remaining packages to collect
        
        visited = initialize_visited_matrix()  # Initialize matrix to track visited states and actions
        print(f"Training... Epoch {epoch+1}/{EPOCHS}")

        while not room.isTerminal():
            visited, prev_position, remaining_packages, exploration_rate = update_environment(room, prev_position, remaining_packages, exploration_rate, visited, epoch, Q_table, R_table)

def initialize_visited_matrix():
    # Initialize a matrix to track visited states and actions
    return np.zeros((169, 4, 4), dtype=int)

def choose_action(prev_position, remaining_packages, exploration_rate, Q_table):
    # Choose an action using epsilon-greedy policy
    index = prev_position[0] + prev_position[1] * 13  # Convert position to index in Q-table
    if random.random() < exploration_rate:
        action = random.randint(0, 3)  # Choose a random action for exploration
    else:
        max_q_value = np.max(Q_table[index][remaining_packages])  # Get maximum Q-value for the current state
        choices = np.where(Q_table[index][remaining_packages] == max_q_value)[0]  # Get indices of maximum Q-values
        action = random.choice(choices)  # Choose one of the actions with maximum Q-value
    return action

def update_environment(room, prev_position, remaining_packages, exploration_rate, visited, epoch, Q_table, R_table):
    index = prev_position[0] + prev_position[1] * 13  # Convert current position to index in Q-table
    action = choose_action(prev_position, remaining_packages, exploration_rate, Q_table)  # Choose action using epsilon-greedy policy

    visited[index][remaining_packages][action] += 1  # Update visited state-action pair
    grid_type_new, new_position, _, is_terminal = room.takeAction(action)  # Take action in the environment

    if prev_position == new_position:
        R_table[index][remaining_packages][action] -= 3  # Penalize for staying in the same position
    elif grid_type_new in [1, 2, 3]:
        if grid_type_new==(4-remaining_packages):
            R_table[index][remaining_packages][action] = 2000  # Reward for collecting a package
            R_table[new_position[1]*13 + new_position[0]][remaining_packages] = [0,0,0,0]  # Reset rewards for the collected package position
            if (epoch==EPOCHS-1):
                print(f"{colors[grid_type_new]} package collected!")  # Print collected package color
        else:
            R_table[index][remaining_packages][action] = -300  # Penalize for collecting wrong package

    learning_rate = 1 / (1 + visited[index][remaining_packages][action])
    max_next_q_value = max(Q_table[new_position[0] + new_position[1] * 13][remaining_packages])
    Q_table[index][remaining_packages][action] += learning_rate * (
            R_table[index][remaining_packages][action] + (DISCOUNT_FACTOR * max_next_q_value) -
            Q_table[index][remaining_packages][action]) - visited[index][remaining_packages][action]

    prev_position = new_position  # Update previous position

    if grid_type_new == 0:
        exploration_rate *= DECAY_RATE  # Decay exploration rate
    else:
        remaining_packages -= 1  # Decrement remaining packages if a package is collected

    return visited, prev_position, remaining_packages, exploration_rate

def main():
    # Check if the -stochastic flag is present
    stochastic_flag = '-stochastic' in sys.argv
    # Initialize environment and train agent
    room, Q_table, R_table = initialize_environment(stochastic=stochastic_flag)
    train_agent(room, Q_table, R_table)
    
    # Display results
    print()
    print('....Training done.....')
    print('....Showing path......')
    room.showPath(-1)
    if stochastic_flag:
        room.showPath(-1, "images/Scenario3-stochastic.png")  # Save the path as an image
    else:
        room.showPath(-1, "images/Scenario3.png")  # Save the path as an image       


if __name__ == "__main__":
    main()