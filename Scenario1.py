import numpy as np
import random
from FourRooms import FourRooms

# Constants
NUM_EPOCHS = 5000
EPSILON_G = 0.6
DECAY_R = 0.8
DISCOUNT_FACTOR = 0.9

def initialize_environment():
    room = FourRooms('simple')
    q_values = np.zeros((169, 4))
    rewards = np.zeros((169, 4))
    print('....Starting Simulation....')
    print('....Agent starts at', room.getPosition())
    return room, q_values, rewards

def select_action(q_values, rewards, index, exploration_rate):
    if random.random() < exploration_rate:
        action = random.choice(np.where(rewards[index] >= 0)[0])
    else:
        max_q_value = np.max(q_values[index])
        choices = np.where(q_values[index] == max_q_value)[0]
        action = random.choice(choices)
    return action

def update_q_values(q_values, rewards, prev_pos, action, new_pos, remaining_packages, visited):
    index = prev_pos[1] * 13 + prev_pos[0]
    learning_rate = 1 / (1 + 0.3 * visited[index][action])

    if prev_pos == new_pos:
        rewards[index][action] -= 1
    elif remaining_packages == 0:
        rewards[index][action] = 100

    max_next_q_value = max(q_values[new_pos[1] * 13 + new_pos[0]])
    q_values[index][action] += learning_rate * (
        rewards[index][action] + DISCOUNT_FACTOR * max_next_q_value - q_values[index][action]
    )

def train_agent(room, q_values, rewards):
    for epoch in range(NUM_EPOCHS):
        room.newEpoch()
        prev_pos = room.getPosition()
        visited = np.zeros((169, 4), dtype=int)
        exploration_rate = EPSILON_G
        print(f"Training... Epoch {epoch + 1}/{NUM_EPOCHS}")

        while not room.isTerminal():
            index = prev_pos[1] * 13 + prev_pos[0]
            action = select_action(q_values, rewards, index, exploration_rate)
            visited[index][action] += 1
            _, new_pos, remaining_packages, terminal = room.takeAction(action)
            update_q_values(q_values, rewards, prev_pos, action, new_pos, remaining_packages, visited)
            prev_pos = new_pos
            exploration_rate *= DECAY_R

def main():
    room, q_values, rewards = initialize_environment()
    train_agent(room, q_values, rewards)
    print('\n....Training done.....')
    print('....Showing path......')
    room.showPath(-1, "images/Scenario1.png")

if __name__ == "__main__":
    main()