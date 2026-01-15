from TestEnv import HydroElectric_Test
import matplotlib.pyplot as plt
from BaselineAgent import BaselineAgent
from visualizer import MetricsVisualizer

import argparse
import os
import numpy as np
import pandas as pd

def get_excel_file_path():
    import argparse, os
    parser = argparse.ArgumentParser()
    parser.add_argument('--excel_file', default='validate.xlsx', type=str)
    parser.add_argument('--out', default='outputs/history.pkl', type=str,
                    help='Where to save the history dataframe (pkl or csv)')
    args = parser.parse_args()

    # If they passed an actual path or filename that exists, use it directly
    candidate = args.excel_file
    if os.path.exists(candidate):
        return candidate, args.out

    # Otherwise treat it like "train" or "validate" and look next to main.py
    here = os.path.dirname(__file__)
    candidate2 = os.path.join(here, f"{candidate}.xlsx")
    if os.path.exists(candidate2):
        return candidate2, args.out

    raise FileNotFoundError(f"Could not find excel file: {args.excel_file}")

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
        action = RL_agent.act(observation)
        next_observation, reward, terminated, truncated, info = env.step(action)
        cumulative_reward += reward
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
    file_path, out_path = get_excel_file_path()

    # Init environment and agent
    env = HydroElectric_Test(file_path)
    RL_agent = BaselineAgent(max_hours_history=24)

    # action_history, reward_history, cumulative_reward
    history = validate_agent(env, RL_agent)
    history_df = pd.DataFrame(history)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    # Force CSV if that's what you want
    if not out_path.endswith(".csv"):
        out_path = out_path.rsplit(".", 1)[0] + ".csv"

    history_df.to_csv(out_path, index=False)
    print(f"Saved history CSV to {out_path}")

    visualizer = MetricsVisualizer(history_df)
    visualizer.plot_cumulative_reward()
    # visualizer.plot_volume(start_day=60, n_days_detail=14)
    # visualizer.plot_action_history(start_day=60, n_days_detail=7)
    # plot_action_history(metric_data[0])
    # plot_reward_history(metric_data[1])






if __name__ == "__main__":
    main()
