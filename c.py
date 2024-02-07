# Author: hlsun (sun.har@northeastern.edu)
# Date: 06 February 2024
# EECE 5698 - ST: Reinforcement Learning
# 2-armed Bandit Problem

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def main():
    num_runs = 100
    num_steps = 1000
    epsilon = 0.1
    alpha = 0.1
    initial_preferences = [0, 0]

    average_rewards_gradient = np.zeros(num_steps)
    average_rewards_epsilon_greedy = np.zeros(num_steps)


    def reward_distribution(action):
        if action == 0:
            return np.random.normal(5, np.sqrt(10))
        else:
            if np.random.rand() < 0.5:
                return np.random.normal(10, np.sqrt(15))
            else:
                return np.random.normal(4, np.sqrt(10))

    def softmax(H):
        exp_H = np.exp(H - np.max(H))  # Subtract max(H) to prevent overflow
        return exp_H / np.sum(exp_H)

    for run in range(num_runs):
        # Gradient Bandit Initialization
        H = np.array(initial_preferences, dtype=np.float64)
        R_bar = 0
        
        # Epsilon-greedy Initialization
        Q = np.array([0, 0], dtype=np.float64)

        for step in range(num_steps):
            # Gradient Bandit Policy
            pi_t = softmax(H)
            action_gradient = np.random.choice([0, 1], p=pi_t)
            reward_gradient = reward_distribution(action_gradient)
            R_bar = R_bar + (reward_gradient - R_bar) / (step + 1)
            H[action_gradient] += alpha * (reward_gradient - R_bar) * (1 - pi_t[action_gradient])
            H[1-action_gradient] -= alpha * (reward_gradient - R_bar) * pi_t[1-action_gradient]
            average_rewards_gradient[step] += (reward_gradient - average_rewards_gradient[step]) / (run + 1)

            # Epsilon-greedy Policy
            if np.random.rand() < epsilon:
                action_epsilon = np.random.choice([0, 1])
            else:
                action_epsilon = np.argmax(Q)
            reward_epsilon = reward_distribution(action_epsilon)
            Q[action_epsilon] += alpha * (reward_epsilon - Q[action_epsilon])
            average_rewards_epsilon_greedy[step] += (reward_epsilon - average_rewards_epsilon_greedy[step]) / (run + 1)

    # Plot the average rewards over time for both Gradient Bandit and ε-greedy
    plt.figure(figsize=(12, 8))
    plt.plot(average_rewards_gradient, label='Gradient Bandit')
    plt.plot(average_rewards_epsilon_greedy, label='ε-greedy')
    plt.xlabel('Steps')
    plt.ylabel('Average Reward')
    plt.title('Average Rewards over Time')
    plt.legend()
    plt.ylim((0, 10))
    plt.show()



# call the main function
if __name__ == "__main__":
    print("Hello...")
    main()
    print("Goodbye.")
