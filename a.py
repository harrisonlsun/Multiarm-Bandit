# Author: hlsun (sun.har@northeastern.edu)
# Date: 06 February 2024
# EECE 5698 - ST: Reinforcement Learning
# 2-armed Bandit Problem

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def main():
    # Parameters
    num_runs = 100  # Number of independent runs
    num_steps = 1000  # Number of steps per run
    epsilons = [0, 0.1, 0.2, 0.5]  # Epsilon values for the epsilon-greedy policy
    learning_rates = [1, lambda k: 0.9**k, lambda k: 1/(1+np.log(1+k)), lambda k: 1/max(1, k)]

    # Initialize the arrays to hold final Q-values and average rewards
    final_q_values = np.zeros((len(learning_rates), len(epsilons), 2))
    average_rewards = np.zeros((len(learning_rates), len(epsilons), num_steps))

    # Define the reward distributions for both actions
    def reward_distribution(action):
        if action == 0:
            # Gaussian distribution for the first lever
            return np.random.normal(5, np.sqrt(10))
        else:
            # Mixture of two Gaussian distributions for the second lever
            if np.random.rand() < 0.5:
                return np.random.normal(10, np.sqrt(15))
            else:
                return np.random.normal(4, np.sqrt(10))

    # Simulate the multi-armed bandit problem
    for i, alpha in enumerate(learning_rates):
        for j, epsilon in enumerate(epsilons):
            all_rewards = []  # Collect all rewards for averaging
            for run in range(num_runs):
                Q = np.zeros(2)  # Initialize Q-values for both actions
                rewards = np.zeros(num_steps)  # Initialize rewards array
                for step in range(num_steps):
                    # Select action based on epsilon-greedy policy
                    if np.random.rand() < epsilon:
                        action = np.random.choice([0, 1])
                    else:
                        action = np.argmax(Q)
                    # Obtain reward
                    reward = reward_distribution(action)
                    # Update Q-value using an incremental implementation of the learning rate
                    alpha_value = alpha(step+1) if callable(alpha) else alpha  # Ensure alpha_value is calculated correctly
                    update = alpha_value * (reward - Q[action])
                    if np.isfinite(update):
                        Q[action] += update
                    # Store the reward
                    rewards[step] = reward
                all_rewards.append(rewards)
                # Print average rewards for this run
                # print(f"Run {run+1}, Learning Rate: {alpha if not callable(alpha) else alpha.__name__}, Epsilon: {epsilon}, Average Reward: {np.sum(rewards)/num_steps}")
            # Store the average of the rewards from all runs and the final Q-values
            average_rewards[i, j, :] = np.mean(all_rewards, axis=0).cumsum()
            final_q_values[i, j, :] = Q

    # Plot the average rewards over time for each combination of learning rate and epsilon
    for i, alpha in enumerate(learning_rates):
        plt.figure(figsize=(12, 8))
        for j, epsilon in enumerate(epsilons):
            # Calculate the average reward at each time step
            plt.plot(range(1, num_steps+1), average_rewards[i, j, :] / np.arange(1, num_steps+1), label=f'ε={epsilon}')
        plt.title(f'Average Rewards over Time')
        plt.xlabel('Time steps')
        plt.ylabel('Average Reward')
        plt.xlim((1, num_steps))
        plt.ylim((0, np.max(average_rewards[i, :, :] / np.arange(1, num_steps+1)) * 1.1)) 
        plt.legend()
        plt.show()

    # Save the final Q-values to CSV files
    for i, alpha in enumerate(learning_rates):
        # Create a DataFrame to store all epsilon values for the current alpha
        combined_df = pd.DataFrame()
        for j, epsilon in enumerate(epsilons):
            # Extract the Q-values for the current alpha and epsilon
            q_values = final_q_values[i, j, :].reshape(1, -1)
            # Create a DataFrame for the current epsilon
            df = pd.DataFrame(q_values, columns=['Q(a1)', 'Q(a2)'])
            # Add a column for epsilon
            df['epsilon'] = epsilon
            # Append the current epsilon DataFrame to the combined DataFrame
            combined_df = combined_df.append(df, ignore_index=True)
        # Save the combined DataFrame to a CSV file named with the alpha index
        filename = f'final_q_values_alpha_{i+1}.csv'
        combined_df.to_csv(filename, index=False)



# call the main function
if __name__ == "__main__":
    print("Hello...")
    main()
    print("Goodbye.")
