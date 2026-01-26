import numpy as np
import torch
import matplotlib.pyplot as plt
from TestEnv import HydroElectric_Test
from dqn_agent1_1 import DQN_agent 
from tqdm import tqdm


class Normalizer:
    def __init__(self, mins, maxs):
        self.mins = mins
        self.maxs = maxs
        
    def normalize(self, state):
        # Clip values within bounds
        state = np.clip(state, self.mins, self.maxs)
        # 0-1 range scaling 
        return (state - self.mins) / (self.maxs - self.mins)

train_file = './data/train.xlsx'
model_path = './model/dqn_model.pth'
scores = []
episodes = 50

env = HydroElectric_Test(path_to_test_data=train_file)
min_bounds = np.array([0.0,    -50.0,  0.0, 0.0, 0.0,  0.0, 2000.0])
max_bounds = np.array([100000.0, 200.0, 24.0, 7.0, 366.0, 12.0, 2020.0])
normalizer = Normalizer(min_bounds, max_bounds)

agent = DQN_agent(
        input_size=7,         
        action_size=5,
        memory_size=100000,
        target_update_freq=1000,
        lr=0.0005,
        gamma_start=0.90,
        gamma_end=0.99,
        epsilon_start=1.0,
        epsilon_end=0.02,
        epsilon_decay=0.90
    )

print(f"Starting training on {agent.device}...")


for episode  in tqdm(range(episodes)):
        
    env.counter = 0
    env.hour = 1
    env.day = 1
    env.volume = env.max_volume / 2
    raw_state = env.observation()
    state = normalizer.normalize(raw_state)
    score =0
    done = False

    while not done:
        action_idx ,action_value = agent.select_action(state=state)
        raw_next_state, reward, terminated, truncated, _ = env.step(action_value)
        done = terminated or truncated

        next_state = normalizer.normalize(raw_next_state)

        agent.memory.push(state, action_idx, reward, next_state, done)
        
        agent.learn(batch_size=128)
        state = next_state
        score += reward

    agent.update_hyperparameters(episode, episodes)
    scores.append(score)

    if episode % 5 == 0:
        avg_score = np.mean(scores[-5:])
        tqdm.write(f"Ep {episode} | Score: {score:.0f} | Avg: {avg_score:.0f} | Eps: {agent.epsilon:.3f}")

torch.save(agent.policy_net.state_dict(), model_path)
print(f"Training Complete. Model saved to {model_path}")
plt.plot(scores)
plt.xlabel('Episode')
plt.ylabel('Total Profit')
plt.title('Training Performance')
plt.show()



