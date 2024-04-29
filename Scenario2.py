import numpy as np
import random
from FourRooms import FourRooms

# Constants
EPOCHS = 5000
EPSILON_GREEDY = 0.6
DECAY_RATE = 0.8
DISCOUNT_FACTOR = 0.9
colors = {1: "Red", 2: "Green", 3: "Blue"}

def main():
    room,Q_table,R_table = initialize_environment()
 
    train_agent(room,Q_table,R_table)

    print()
    print('....Training done.....')
    print('....Showing path......')
    room.showPath(-1, "images/Scenario2.png")

def initialize_environment():
    # Create a FourRooms object for multi-package collection scenario
    room = FourRooms('multi')
    # Initialize Q and R tables
    Q_table = np.zeros((169, 4, 4))
    R_table = np.zeros((169, 4, 4))
    print('....Starting Simulation....')
    print('....Agent starts at', room.getPosition())
    return room,Q_table,R_table

def execute_action_sequence(room, action_sequence):
    for action in action_sequence:
        _, _, _, is_terminal = room.takeAction(action)
        if is_terminal:
            break

def train_agent(room,Q_table,R_table):
    for epoch in range(EPOCHS):
        room.newEpoch()
        prev_position = room.getPosition()
        exploration_rate = EPSILON_GREEDY
        remaining_packages = 3
        
        visited = initialize_visited_matrix()
        print(f"Training... Epoch {epoch+1}/{EPOCHS}")

        while not room.isTerminal():
            visited, prev_position, remaining_packages, exploration_rate = update_environment(room, prev_position,remaining_packages,exploration_rate,visited, epoch,Q_table,R_table)

def initialize_visited_matrix():
    return np.zeros((169, 4, 4), dtype=int)

def choose_action(prev_position, remaining_packages, exploration_rate,Q_table):
    index = prev_position[0] + prev_position[1] * 13
    if random.random() < exploration_rate:
        action = random.randint(0, 3)
    else:
        max_q_value = np.max(Q_table[index][remaining_packages])
        choices = np.where(Q_table[index][remaining_packages] == max_q_value)[0]
        action = random.choice(choices)
    return action

def update_environment(room, prev_position, remaining_packages, exploration_rate, visited, epoch,Q_table,R_table):
    index = prev_position[0] + prev_position[1] * 13
    action = choose_action(prev_position, remaining_packages, exploration_rate,Q_table)

    visited[index][remaining_packages][action] += 1
    grid_type_new, new_position, _, is_terminal = room.takeAction(action)

    if prev_position == new_position:
        R_table[index][remaining_packages][action] -= 3
    elif grid_type_new in [1, 2, 3]:
        R_table[index][remaining_packages][action] = 2000
        if epoch == EPOCHS - 1:
            print(f"{colors[grid_type_new]} package collected!")

    learning_rate = 1 / (1 + visited[index][remaining_packages][action])
    max_next_q_value = max(Q_table[new_position[0] + new_position[1] * 13][remaining_packages])
    Q_table[index][remaining_packages][action] += learning_rate * (
            R_table[index][remaining_packages][action] + (DISCOUNT_FACTOR * max_next_q_value) -
            Q_table[index][remaining_packages][action]) - visited[index][remaining_packages][action]

    prev_position = new_position

    if grid_type_new == 0:
        exploration_rate *= DECAY_RATE
    else:
        remaining_packages -= 1

    return visited, prev_position, remaining_packages, exploration_rate


if __name__ == "__main__":
    main()