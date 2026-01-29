import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from tqdm import tqdm
import argparse  # <--- NEW IMPORT
from TestEnv import HydroElectric_Test
from dqn_agent1_1 import DQN_agent 
from util import FeatureEngineering

parser = argparse.ArgumentParser()
parser.add_argument('--excel_file', type=str, default='./data/validate.xlsx')
args = parser.parse_args()


model_path = './model/dqn_model5.pth' 
num_of_years = 1.0 




env = HydroElectric_Test(path_to_test_data=args.excel_file)

fe = FeatureEngineering()

# Initialize Agent
agent = DQN_agent(
    input_size=10,          
    action_size=5,         
    memory_size=1,         
    epsilon_start=0.0,     # Zero randomness for validation
    epsilon_end=0.0,
    epsilon_decay=0.0
)

# Load Model
print(f"Loading model from {model_path}...")
try:
    agent.policy_net.load_state_dict(torch.load(model_path, map_location=agent.device))
    agent.policy_net.eval()
    print("Model loaded successfully.")
except FileNotFoundError:
    print(f"ERROR: Model file '{model_path}' not found.")
    print("Did you finish the training run? Make sure dqn_BEST_EVER.pth exists.")
    exit()

# --- 2. VALIDATION LOOP ---
print(f"Starting Validation...")

env.counter = 0
env.hour = 1
env.day = 1
env.volume = env.max_volume / 2
fe.reset()

raw_state = env.observation()
state = fe.process(raw_state)
done = False
total_reward = 0

history = {
    'timestamps': [],
    'prices': [],
    'volumes': [],
    'actions': [],
    'rewards': []
}

with torch.no_grad():
    # Fix for progress bar using public attribute
    pbar = tqdm(total=env.price_values.size) 
    
    while not done:
        action_index, action_value = agent.select_action(state=state)
        
        raw_next_state, reward, terminated, truncated, _ = env.step(action_value)
        done = terminated or truncated
        
        # Log Data
        vol = raw_next_state[0]
        price = raw_next_state[1]
        
        history['timestamps'].append(env.counter)
        history['prices'].append(price)
        history['volumes'].append(vol)
        history['actions'].append(action_value) 
        history['rewards'].append(reward)
        
        next_state = fe.process(raw_next_state)
        state = next_state
        total_reward += reward
        pbar.update(1)
    pbar.close()

# Calculate Results
annual_profit = total_reward / num_of_years
print(f"\nResults on UNSEEN Data:")
print(f"---------------------------")
print(f"Total Reward:   €{total_reward:,.2f}")
print(f"Annual Profit:  €{annual_profit:,.2f}")
print(f"---------------------------")

# --- 3. STRESS TEST PLOT ---
print("\nGenerating Stress Test Visualization...")

df = pd.DataFrame(history)
df['price'] = df['prices']
df['t'] = df['timestamps']
df['t_day'] = df['t'] / 24.0
df['action_mwh'] = df['actions']
max_vol = env.max_volume
df['vol_frac'] = df['volumes'] / max_vol

# Logic: Find Volatile vs Calm Windows (14 Days)
w = 14 * 24
roll = df["price"].rolling(w).std().dropna()

if len(roll) > 0:
    i_vol = int(roll.idxmax())
    i_cal = int(roll.idxmin())

    df_vol = df.iloc[i_vol - w + 1 : i_vol + 1]
    df_cal = df.iloc[i_cal - w + 1 : i_cal + 1]

    # Setup Colormap
    cmap = mcolors.LinearSegmentedColormap.from_list("ActionMap", ["green", "gray", "red"])
    norm = mcolors.Normalize(vmin=-1, vmax=1)
    ymax = float(max(df_vol["price"].quantile(0.995), df_cal["price"].quantile(0.995)))

    fig, ax = plt.subplots(
        2, 2, figsize=(14, 8),
        sharex="col",
        constrained_layout=True,
        gridspec_kw={"height_ratios": [3, 1]}
    )

    ax_p_vol, ax_p_cal = ax[0, 0], ax[0, 1]
    ax_v_vol, ax_v_cal = ax[1, 0], ax[1, 1]

    for axp, axv, dwin, title in [
        (ax_p_vol, ax_v_vol, df_vol, f"High Volatility Period"),
        (ax_p_cal, ax_v_cal, df_cal, f"Low Volatility Period"),
    ]:
        # Top Panel
        axp.plot(dwin["t_day"], dwin["price"], linewidth=0.8, alpha=0.35, color="0.25", label="Spot Price")
        sc = axp.scatter(
            dwin["t_day"], dwin["price"],
            c=dwin["action_mwh"], cmap=cmap, norm=norm,
            s=15, alpha=0.9, linewidth=0, rasterized=True, zorder=10
        )
        axp.set_title(title, pad=10, fontweight='bold')
        axp.set_ylim(0, ymax)
        axp.set_ylabel("Price (€/MWh)")
        axp.grid(True, alpha=0.2)

        # Bottom Panel
        axv.plot(dwin["t_day"], dwin["vol_frac"], linewidth=1.5, color='#1f77b4', label="Level")
        axv.fill_between(dwin["t_day"], dwin["vol_frac"], 0, color='#1f77b4', alpha=0.1)
        axv.set_ylim(0, 1.05)
        axv.set_ylabel("Reservoir %")
        axv.set_xlabel("Day of Test")
        axv.grid(True, alpha=0.2)
        axv.axhline(1, color='red', linestyle=':', alpha=0.5)
        axv.axhline(0, color='red', linestyle=':', alpha=0.5)

    cbar = fig.colorbar(sc, ax=[ax_p_vol, ax_p_cal], pad=0.01, aspect=40)
    cbar.set_label("Action")
    cbar.set_ticks([-1, 0, 1])
    cbar.set_ticklabels(['Sell', 'Hold', 'Pump'])

    plt.suptitle("Agent Performance on UNSEEN Test Data", fontsize=16, y=1.02)
    plt.savefig('./plots/validation_stress_test.png', bbox_inches='tight')
    print("Plot saved to ./plots/validation_stress_test.png")
    plt.show()
else:
    print("Dataset too short for 14-day rolling window analysis.")

# CURRENT MAIN
 
# from TestEnv import HydroElectric_Test
# import argparse
# import torch
# import numpy as np
# import os
# import matplotlib.pyplot as plt
# from dqn_agent1_1 import DQN_agent
# from util import FeatureEngineering

# parser = argparse.ArgumentParser()
# parser.add_argument('--excel_file', type=str, default='validate.xlsx') # Path to the excel file with the test data
# args = parser.parse_args()

# env = HydroElectric_Test(path_to_test_data=args.excel_file)

# fe = FeatureEngineering()

# num_of_years_in_set = 2.0
# total_reward = []
# cumulative_reward = []

# agent = DQN_agent(
#     input_size=9, 
#     action_size=5, 
#     memory_size=1  
# )

# model_path = './model/100dqn_model5_optimized.pth'

# agent.policy_net.load_state_dict(torch.load(model_path,map_location=agent.device))
# agent.policy_net.eval()
# print('Model loaded')

# observation_history = []
# action_history = []
# total_reward = []

# fe.reset()
# observation = env.observation()
# engineered_observation = fe.process(observation)

# for i in range(730*24 -1): # Loop through 2 years -> 730 days * 24 hours
#     # Choose a random action between -1 (full capacity sell) and 1 (full capacity pump)
#     # action = env.continuous_action_space.sample()
#     action = agent.act(engineered_observation)

#     observation_history.append(observation) 
#     action_history.append(action)
#     # Or choose an action based on the observation using your RL agent!:
#     # action = RL_agent.act(observation)
#     # The observation is the tuple: [volume, price, hour_of_day, day_of_week, day_of_year, month_of_year, year]
#     next_observation, reward, terminated, truncated, info = env.step(action)
#     total_reward.append(reward)
#     # cumulative_reward.append(sum(total_reward))

#     done = terminated or truncated
#     engineered_observation = fe.process(next_observation)
#     observation = next_observation


#     if done:
#         break
        
# total_profit = sum(total_reward)
# cumulative_reward = np.cumsum(total_reward)
# annual_profit = total_profit / num_of_years_in_set

# print('-----------------------------------')
# print(f' FINAL TOTAL REWARD: {total_profit:,.0f} €')
# print(f'ANNUAL REWARD:      {annual_profit:,.0f} €/year')
# print('-----------------------------------')


# prices = np.array([x[1] for x in observation_history])   # Index 1 is Price
# volumes = np.array([x[0] for x in observation_history])  # Index 0 is Volume
# actions = np.array(action_history)

# fig, (ax1, ax2, ax3,ax4) = plt.subplots(4, 1, figsize=(16, 16), sharex=True)

# # PANEL 1: Market Price
# ax1.plot(prices, color='black', alpha=0.6, linewidth=1, label='Market Price')
# ax1.set_ylabel('Price (€/MWh)', fontsize=12)
# ax1.set_title(f'Validation Results | Total: €{total_profit:,.0f} | Annual: €{annual_profit:,.0f}/yr', fontweight='bold')
# ax1.grid(True, alpha=0.3)
# ax1.legend(loc='upper right')

# # PANEL 2: Discrete Actions

# colors = np.where(actions > 0.01, 'red', np.where(actions < -0.01, 'green', 'gray'))

# ax2.scatter(range(len(actions)), actions, c=colors, s=10, alpha=0.6, marker='o')
# ax2.set_yticks([-1.0, -0.5, 0.0, 0.5, 1.0]) # Set fixed y-ticks for clarity
# # ax2.set_yticks([-1.0, -0.66, 0.33, 0.0, 0.33, 0.66, 1.0]) #7 actions
# ax2.set_ylabel('Action Selected', fontsize=12)
# ax2.set_title('Agent Decisions (Green=Pump, Red=Turbine)', fontsize=10)
# ax2.grid(True, alpha=0.3, axis='y')

# # PANEL 3 : Cumulative reward
# ax3.plot(cumulative_reward, color='green', linewidth=2, label='Cumulative Profit')
# ax3.fill_between(range(len(cumulative_reward)), cumulative_reward, 0, color='green', alpha=0.1)
# ax3.set_ylabel('Profit (€)', fontsize=12)
# ax3.set_title('Financial Performance (Equity Curve)', fontsize=10)
# ax3.grid(True, alpha=0.3)
# ax3.legend(loc='upper left')

# # PANEL 4 : Dam Water Level
# ax4.plot(volumes, color='#1f77b4', linewidth=1.5, label='Water Level')
# ax4.axhline(y=0, color='red', linestyle='--', alpha=0.5, label='Empty')
# ax4.axhline(y=100000, color='red', linestyle='--', alpha=0.5, label='Full')
# ax4.set_ylabel('Volume ($m^3$)', fontsize=12)
# ax4.set_xlabel('Time (Hours)', fontsize=12)
# ax4.legend(loc='upper right')
# ax4.grid(True, alpha=0.3)

# plt.tight_layout()


# save_path = os.path.join('plots', 'validation_dashboard.png')
# plt.savefig(save_path)
# print(f"📊 Dashboard saved to: {save_path}")
# plt.show()


#---BEST MODEL'S MAIN VALIDATION SO FAR SAVED FOR SAFETY----- 

# from TestEnv import HydroElectric_Test
# import argparse
# import torch
# import numpy as np
# import os
# import matplotlib.pyplot as plt
# from dqn_agent1_1 import DQN_agent
# from util import FeatureEngineering

# parser = argparse.ArgumentParser()
# parser.add_argument('--excel_file', type=str, default='validate.xlsx') # Path to the excel file with the test data
# args = parser.parse_args()

# env = HydroElectric_Test(path_to_test_data=args.excel_file)

# fe = FeatureEngineering()

# num_of_years_in_set = 2.0
# total_reward = []
# cumulative_reward = []

# agent = DQN_agent(
#     input_size=10, 
#     action_size=5, 
#     memory_size=1  
# )

# model_path = './model/dqn_model5.pth'

# agent.policy_net.load_state_dict(torch.load(model_path,map_location=agent.device))
# agent.policy_net.eval()
# print('Model loaded')

# observation_history = []
# action_history = []
# total_reward = []

# fe.reset()
# observation = env.observation()
# engineered_observation = fe.process(observation)

# for i in range(730*24 -1): # Loop through 2 years -> 730 days * 24 hours
#     # Choose a random action between -1 (full capacity sell) and 1 (full capacity pump)
#     # action = env.continuous_action_space.sample()
#     action = agent.act(engineered_observation)

#     observation_history.append(observation) 
#     action_history.append(action)
#     # Or choose an action based on the observation using your RL agent!:
#     # action = RL_agent.act(observation)
#     # The observation is the tuple: [volume, price, hour_of_day, day_of_week, day_of_year, month_of_year, year]
#     next_observation, reward, terminated, truncated, info = env.step(action)
#     total_reward.append(reward)
#     # cumulative_reward.append(sum(total_reward))

#     done = terminated or truncated
#     engineered_observation = fe.process(next_observation)
#     observation = next_observation


#     if done:
#         break
        
# total_profit = sum(total_reward)
# cumulative_reward = np.cumsum(total_reward)
# annual_profit = total_profit / num_of_years_in_set

# print('-----------------------------------')
# print(f'💰 FINAL TOTAL REWARD: {total_profit:,.0f} €')
# print(f'📅 ANNUAL REWARD:      {annual_profit:,.0f} €/year')
# print('-----------------------------------')

# # --- 5. Generate 3-Panel Dashboard ---
# # Prepare data arrays
# prices = np.array([x[1] for x in observation_history])   # Index 1 is Price
# volumes = np.array([x[0] for x in observation_history])  # Index 0 is Volume
# actions = np.array(action_history)

# fig, (ax1, ax2, ax3,ax4) = plt.subplots(4, 1, figsize=(16, 16), sharex=True)

# # PANEL 1: Market Price
# ax1.plot(prices, color='black', alpha=0.6, linewidth=1, label='Market Price')
# ax1.set_ylabel('Price (€/MWh)', fontsize=12)
# ax1.set_title(f'Validation Results | Total: €{total_profit:,.0f} | Annual: €{annual_profit:,.0f}/yr', fontweight='bold')
# ax1.grid(True, alpha=0.3)
# ax1.legend(loc='upper right')

# # PANEL 2: Discrete Actions (The "Ladder" Plot)
# # Colors: Red = Sell (>0), Green = Pump (<0), Gray = Hold (0)
# colors = np.where(actions > 0.01, 'orange', np.where(actions < -0.01, 'purple', 'gray'))

# ax2.scatter(range(len(actions)), actions, c=colors, s=10, alpha=0.6, marker='o')
# ax2.set_yticks([-1.0, -0.5, 0.0, 0.5, 1.0]) # Set fixed y-ticks for clarity
# ax2.set_ylabel('Action Selected', fontsize=12)
# ax2.set_title('Agent Decisions (Green=Pump, Red=Turbine)', fontsize=10)
# ax2.grid(True, alpha=0.3, axis='y')


# ax3.plot(cumulative_reward, color='purple', linewidth=2, label='Cumulative Profit')
# ax3.fill_between(range(len(cumulative_reward)), cumulative_reward, 0, color='purple', alpha=0.1)
# ax3.set_ylabel('Profit (€)', fontsize=12)
# ax3.set_title('Financial Performance (Equity Curve)', fontsize=10)
# ax3.grid(True, alpha=0.3)
# ax3.legend(loc='upper left')

# ax4.plot(volumes, color='#1f77b4', linewidth=1.5, label='Water Level')
# ax4.axhline(y=0, color='red', linestyle='--', alpha=0.5, label='Empty')
# ax4.axhline(y=100000, color='red', linestyle='--', alpha=0.5, label='Full')
# ax4.set_ylabel('Volume ($m^3$)', fontsize=12)
# ax4.set_xlabel('Time (Hours)', fontsize=12)
# ax4.legend(loc='upper right')
# ax4.grid(True, alpha=0.3)

# plt.tight_layout()

# # Save and Show
# save_path = os.path.join('plots', 'validation_dashboard.png')
# plt.savefig(save_path)
# print(f"📊 Dashboard saved to: {save_path}")
# plt.show()

