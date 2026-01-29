from TestEnv import HydroElectric_Test
import matplotlib.pyplot as plt
from BaselineAgent import BaselineAgent
from visualizer import MetricsVisualizer
from validate_tabular_agent import TabularQAgentValidator

import argparse
import os
import numpy as np
import pandas as pd

def get_excel_file_path():    
    parser = argparse.ArgumentParser()
    parser.add_argument('--excel_file', default='train', 
                        type=str, help='Excel file to be chosen, train or validate') 
    args = parser.parse_args()

    if args.excel_file == 'train' or args.excel_file == 'validate':
        current_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.normpath(os.path.join(current_dir, '..', 'data', args.excel_file + '.xlsx'))
    else:
        raise ValueError(f"Excel file '{args.excel_file}' not found. Please use 'train' or 'validate'.")

    return file_path

def validate_agent(env, RL_agent):
    n_steps = len(env.test_data) * 24 - 1
    history = {
        "t": np.arange(n_steps, dtype=np.int32),
        "action": np.empty(n_steps, dtype=np.float32),
        "reward": np.empty(n_steps, dtype=np.float32),
        "cum_reward": np.empty(n_steps, dtype=np.float32),
        "price": np.empty(n_steps, dtype=np.float32),
        "volume": np.empty(n_steps, dtype=np.float32),
        "hour": np.empty(n_steps, dtype=np.int8),
        "day_of_week": np.empty(n_steps, dtype=np.int8),
    }

    cumulative_reward = 0.0
    observation = env.observation()

    for i in range(int(len(env.test_data)) * 24 - 1): # Loop through full dataset
        # Act
        # print('Step:', i, 'Volume:', env.volume, 'Price:', env.price_values[env.day-1][env.hour-1])
        action = RL_agent.act(observation)
        
        next_observation, reward, terminated, truncated, info = env.step(action)
        cumulative_reward += reward
        print(f"Step {i}: Action {action:.2f}, Reward {reward:.2f}, Cumulative Reward {cumulative_reward:.2f}")
        # Gather metrics
        # Current observation [volume, price, hour_of_day, day_of_week, day_of_year, month_of_year, year]
        volume, price, hour_of_day, day_of_week = observation[:4]

        history['action'][i] = action * env.max_flow
        history['reward'][i] = reward
        history['cum_reward'][i] = cumulative_reward
        history['price'][i] = price
        history['volume'][i] = volume
        history['hour'][i] = hour_of_day
        history['day_of_week'][i] = day_of_week

        done = terminated or truncated
        observation = next_observation

    print('Cumulative reward: ', cumulative_reward)
     
    return history

def main():
    file_path = get_excel_file_path()

    # Init environment and agent
    env = HydroElectric_Test(file_path)
    # RL_agent = BaselineAgent(max_hours_history=24)
    run_folder = 'results/20260129_001924_Ep5_Gamma0.99_Zscore'
    RL_agent = TabularQAgentValidator(run_folder)  # Update this path
    # action_history, reward_history, cumulative_reward
    history = validate_agent(env, RL_agent)
    history_df = pd.DataFrame(history)

    visualizer = MetricsVisualizer(history_df)
    visualizer.plot_cumulative_reward()
    # visualizer.plot_volume(start_day=60, n_days_detail=14)
    # visualizer.plot_action_history(start_day=60, n_days_detail=7)
    # plot_action_history(metric_data[0])
    # plot_reward_history(metric_data[1])






if __name__ == "__main__":
    main()
