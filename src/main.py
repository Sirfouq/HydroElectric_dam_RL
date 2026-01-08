from TestEnv import HydroElectric_Test
import argparse
import matplotlib.pyplot as plt
from BaselineAgent import BaselineAgent
import os

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
    reward_history = []
    action_history = []
    cumulative_reward = []

    observation = env.observation()
    done = False
    for i in range(len(env.test_data) * 24 -1): # Loop through full dataset
    # while not done:
        action = RL_agent.act(observation)
        action_history.append(action)
        # The observation is the tuple: [volume, price, hour_of_day, day_of_week, day_of_year, month_of_year, year]
        next_observation, reward, terminated, truncated, info = env.step(action)
        reward_history.append(reward)
        cumulative_reward.append(sum(reward_history))

        done = terminated or truncated
        observation = next_observation
        
        # if i % 1000 == 0:
        #     print(f'Step {i}, Cumulative reward: {cumulative_reward[-1]}')

    print('Cumulative reward: ', cumulative_reward[-1])
    return (action_history, reward_history, cumulative_reward)

def plot_cum_reward(metric_data):
        # Plot the cumulative reward over time
        action_history, reward_history, cumulative_reward = metric_data

        # Plot the cumulative reward over time
        plt.figure(figsize=(10, 6))
        plt.plot(cumulative_reward)
        plt.ylabel('Cumulative Reward')
        plt.xlabel('Time (Hours)')
        plt.grid(True)

        plt.show()

def main():
    file_path = get_excel_file_path()

    # Init environment and agent
    env = HydroElectric_Test(file_path)
    RL_agent = BaselineAgent(max_hours_history=24)

    # action_history, reward_history, cumulative_reward
    metric_data = validate_agent(env, RL_agent)
    plot_cum_reward(metric_data)







    # if (i+1)%1000 == 0:
    #     print(f'Step: {i+1}, Action: {action}, Reward: {reward}, Volume: {env.volume}, Hour: {env.hour}, Day: {env.day}, Price: {env.price_values[env.day-1][env.hour-1]}')

    






if __name__ == "__main__":
    main()
