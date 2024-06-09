import sys
import numpy as np
import random
from FourRooms import FourRooms 
# Constants
NUM_EPOCHS = 5000  # Number of training epochs
EPSILON_G = 0.6  # Initial exploration rate
DECAY_R = 0.8  # Decay rate for exploration rate
DISCOUNT_FACTOR = 0.9  # Discount factor for future rewards

def initialize_environment(stochastic=False):
    # Initialize the environment (room), Q-values, and rewards matrix
    room = FourRooms('simple',stochastic=stochastic)  # Create a room object
    q_values = np.zeros((169, 4))  # Initialize Q-values matrix
    rewards = np.zeros((169, 4))  # Initialize rewards matrix
    print('....Starting Simulation....')
    if stochastic:
        print("....stochastic....")
    print('....Agent starts at', room.getPosition())  # Print agent's starting position
    return room, q_values, rewards

def select_action(q_values, rewards, index, exploration_rate):
    # Select an action based on epsilon-greedy policy
    if random.random() < exploration_rate:
        # Exploration: Choose a random action
        action = random.randint(0, 3)  # Choose a random action for exploration
    else:
        # Exploitation: Choose the action with the highest Q-value
        max_q_value = np.max(q_values[index])
        choices = np.where(q_values[index] == max_q_value)[0]
        action = random.choice(choices)
    return action

def update_q_values(q_values, rewards, prev_pos, action, new_pos, remaining_packages, visited):
    # Update Q-values based on the Bellman equation
    index = prev_pos[1] * 13 + prev_pos[0]  # Convert position to index in Q-values matrix
    learning_rate = 1 / (1 + 0.3 * visited[index][action])  # Learning rate with decay

    if prev_pos == new_pos:
        rewards[index][action] -= 1  # Penalize for staying in the same position
    elif remaining_packages == 0:
        rewards[index][action] = 100  # Reward for reaching the goal

    max_next_q_value = max(q_values[new_pos[1] * 13 + new_pos[0]])  # Max Q-value for next state
    # Update Q-value using the Bellman equation
    q_values[index][action] += learning_rate * (
        rewards[index][action] + DISCOUNT_FACTOR * max_next_q_value - q_values[index][action]
    )

def train_agent(room, q_values, rewards):
    # Train the agent using Q-learning
    for epoch in range(NUM_EPOCHS):
        room.newEpoch()  # Start a new epoch
        prev_pos = room.getPosition()  # Get initial position of the agent
        visited = np.zeros((169, 4), dtype=int)  # Track visited states and actions
        exploration_rate = EPSILON_G  # Initialize exploration rate
        print(f"Training... Epoch {epoch + 1}/{NUM_EPOCHS}")

        while not room.isTerminal():
            index = prev_pos[1] * 13 + prev_pos[0]  # Convert position to index
            action = select_action(q_values, rewards, index, exploration_rate)  # Select action
            visited[index][action] += 1  # Update visited state-action pair
            _, new_pos, remaining_packages, terminal = room.takeAction(action)  # Take action in the environment
            update_q_values(q_values, rewards, prev_pos, action, new_pos, remaining_packages, visited)  # Update Q-values
            prev_pos = new_pos  # Update previous position
            exploration_rate *= DECAY_R  # Decay exploration rate
def main():
    # Check if the -stochastic flag is present
    stochastic_flag = '-stochastic' in sys.argv
    # Main function
    room, q_values, rewards = initialize_environment(stochastic=stochastic_flag)  # Pass the stochastic flag
    train_agent(room, q_values, rewards)  # Train the agent
    print('\n....Training done.....')
    print('....Showing path......')
    room.showPath(-1)  # Show the path taken by the agent
    if stochastic_flag:
        room.showPath(-1, "images/Scenario1-stochastic.png")  # Save the path as an image
    else:
        room.showPath(-1, "images/Scenario1.png")  # Save the path as an image       

if __name__ == "__main__":
    main() 