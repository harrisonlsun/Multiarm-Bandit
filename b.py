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
    epsilon = 0.1
    alpha = 0.1
    initial_Q = [[0, 0], [5, 7], [20, 20]]

    # Dataframe to store final Q values
    q_values_df = pd.DataFrame()

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

    # Iterate over different initial Q-values
    for Q_init in initial_Q:
        all_rewards = []  # Collect all rewards for averaging
        all_Q_values = []  # Collect all Q-values for reporting

        # Simulate the multi-armed bandit problem
        for run in range(num_runs):
            Q = np.array(Q_init, dtype=np.float64)  # Initialize Q-values for both actions
            rewards = np.zeros(num_steps)  # Initialize rewards array for this run
            Q_values_run = []  # Store Q-values for this run

            for step in range(num_steps):
                # Select action based on epsilon-greedy policy
                action = np.random.choice([0, 1]) if np.random.rand() < epsilon else np.argmax(Q)
                # Obtain reward
                reward = reward_distribution(action)
                # Update Q-value
                Q[action] += alpha * (reward - Q[action])
                # Store the reward
                rewards[step] = reward
                Q_values_run.append(Q.copy())

            all_rewards.append(rewards)
            all_Q_values.append(Q_values_run)

        # Compute average reward at each time step
        average_rewards = np.mean(all_rewards, axis=0)

        # Plot the average rewards over time for this initial Q-value set
        plt.plot(range(num_steps), average_rewards, label=f'Initial Q-values: {Q_init}')

        # Print the Q-values for debugging
        #print(f'Q-values for initial Q {Q_init}:')
        #for q_values in all_Q_values:
        #    print(q_values)

        # Add the final Q-values for this initial Q-values set to the dataframe
        final_Q = np.mean([q_values[-1] for q_values in all_Q_values], axis=0)
        q_values_df = q_values_df.append({
            'Initial Q(a1)': Q_init[0],
            'Initial Q(a2)': Q_init[1],
            'Final Q(a1)': final_Q[0],
            'Final Q(a2)': final_Q[1]
        }, ignore_index=True)

    # Finalize plot
    plt.title('Average Rewards over Time')
    plt.xlabel('Time steps')
    plt.ylabel('Average Reward')
    plt.legend()
    plt.ylim((0, 10))
    plt.show()

    # Print the final Q-values for each set of initial Q-values
    print(q_values_df)



# call the main function
if __name__ == "__main__":
    print("Hello...")
    main()
    print("Goodbye.")
