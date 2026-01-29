import numpy as np
import torch
import matplotlib.pyplot as plt
from TestEnv import HydroElectric_Test
from dqn_agent1_1 import DQN_agent 
from tqdm import tqdm
from util import FeatureEngineering




train_file = './data/train.xlsx'
model_path = './model/100dqn_model5_optimized.pth'
scores = []
episodes = 100
num_of_years_in_dataset= 3.0

env = HydroElectric_Test(path_to_test_data=train_file)
fe = FeatureEngineering()

agent = DQN_agent(
        input_size=9,         
        action_size=5,
        memory_size=100000,
        target_update_freq=1000,
        lr=0.0005,
        gamma_start=0.99,
        gamma_end=0.999,
        epsilon_start=1.0,
        epsilon_end=0.02,
        epsilon_decay=0.96
    )

print(f"Starting training on {agent.device}...")


for episode  in tqdm(range(episodes)):
        
    env.counter = 0
    env.hour = 1
    env.day = 1
    env.volume = env.max_volume / 2

    fe.reset()
    raw_state = env.observation()
    state = fe.process(raw_state)
    score =0
    done = False

    while not done:
        action_index ,action_value = agent.select_action(state=state)
        raw_next_state, reward, terminated, truncated, _ = env.step(action_value)
        done = terminated or truncated
        
        scaled_reward = reward / 100.0

        next_state = fe.process(raw_next_state)

        agent.memory.push(state, action_index, scaled_reward, next_state, done)
        
        agent.learn(batch_size=128)
        state = next_state
        score += reward

    agent.update_hyperparameters(episode, episodes)
    annual_score = score/num_of_years_in_dataset 
    scores.append(annual_score)

    if episode % 5 == 0:
        avg_score = np.mean(scores[-5:])
        tqdm.write(f"Ep {episode} | Annual: {annual_score:,.0f} | Avg (Ann): {avg_score:,.0f} | Eps: {agent.epsilon:.3f}")

torch.save(agent.policy_net.state_dict(), model_path)
print(f"Training Complete. Model saved to {model_path}")
plt.plot(scores)
plt.xlabel('Episode')
plt.ylabel('Total Profit')
plt.title('Training Performance')
plt.show()

#---BEST MODEL'S TRAIN LOOP SO FAR SAVED FOR SAFETY----- 

# import numpy as np
# import torch
# import matplotlib.pyplot as plt
# from TestEnv import HydroElectric_Test
# from dqn_agent1_1 import DQN_agent 
# from tqdm import tqdm
# from util import FeatureEngineering




# train_file = './data/train.xlsx'
# model_path = './model/dqn_model5.pth'
# scores = []
# episodes = 50
# num_of_years_in_dataset= 3.0

# env = HydroElectric_Test(path_to_test_data=train_file)
# fe = FeatureEngineering()

# agent = DQN_agent(
#         input_size=10,         
#         action_size=5,
#         memory_size=100000,
#         target_update_freq=1000,
#         lr=0.0005,
#         gamma_start=0.90,
#         gamma_end=0.99,
#         epsilon_start=1.0,
#         epsilon_end=0.02,
#         epsilon_decay=0.90
#     )

# print(f"Starting training on {agent.device}...")


# for episode  in tqdm(range(episodes)):
        
#     env.counter = 0
#     env.hour = 1
#     env.day = 1
#     env.volume = env.max_volume / 2

#     fe.reset()
#     raw_state = env.observation()
#     state = fe.process(raw_state)
#     score =0
#     done = False

#     while not done:
#         action_index ,action_value = agent.select_action(state=state)
#         raw_next_state, reward, terminated, truncated, _ = env.step(action_value)
#         done = terminated or truncated

#         next_state = fe.process(raw_next_state)

#         agent.memory.push(state, action_index, reward, next_state, done)
        
#         agent.learn(batch_size=128)
#         state = next_state
#         score += reward

#     agent.update_hyperparameters(episode, episodes)
#     annual_score = score/num_of_years_in_dataset 
#     scores.append(annual_score)

#     if episode % 5 == 0:
#         avg_score = np.mean(scores[-5:])
#         tqdm.write(f"Ep {episode} | Annual: {annual_score:,.0f} | Avg (Ann): {avg_score:,.0f} | Eps: {agent.epsilon:.3f}")

# torch.save(agent.policy_net.state_dict(), model_path)
# print(f"Training Complete. Model saved to {model_path}")
# plt.plot(scores)
# plt.xlabel('Episode')
# plt.ylabel('Total Profit')
# plt.title('Training Performance')
# plt.show()
