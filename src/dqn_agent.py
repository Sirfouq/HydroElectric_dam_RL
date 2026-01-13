import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque

class DQN(nn.Module):
    def __init__(self, input_space, action_space ):
        
        
        super(DQN,self).__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features=input_space,out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128,out_features=64),
            nn.ReLU(),
            nn.Linear(64,action_space)
        )

    def forward(self,observation):
        fw = self.net(observation)
        return fw
    


class ReplayMemory:
    # def __init__(self,buffer_size,min_replay_size = 1000):
        # self.min_replay_size = min_replay_size
        # self.replay_buffer = deque(maxlen=buffer_size)
        # self.reward_buffer = deque([-200.0], maxlen = 100)
        # if torch.cuda.is_available:
        #     self.device = 'cuda'
        # elif torch.mps.is_available:
        #     self.device = 'cuda'
        # else:
        #     self.device = 'cpu'
    def __init__(self,buffer_size):
        self.buffer_size = deque(maxlen=buffer_size)

    def add_data(self, data : tuple, done):   #data tuple should be of shape : (state, action, reward, next_state)
      
        state, action, reward, next_state = data
        self.buffer_size.append((state,action,reward,next_state, done))
    
    def sample(self,batch_size):
        return random.sample(self.buffer_size,batch_size)
        
    
    
class DQN_agent:
        def __init__(self, input_size, action_size, 
                gamma=0.99, lr=0.001, 
                epsilon_start=1.0, 
                epsilon_decay=0.995,
                epsilon_end=0.01, 
                batch_size=64, min_replay_size=1000, update_target_step=1000):
            
            self.input_size = input_size
            self.action_size = action_size
            self.gamma = gamma
            self.lr = lr
            self.epsilon = epsilon_start 
            self.epsilon_decay = epsilon_decay
            self.epsilon_end = epsilon_end
            self.batch_size = batch_size
            self.min_replay_size = min_replay_size
            self.update_target_step = update_target_step
            self.learn_step_counter = 0

            

            if torch.cuda.is_available():
                self.device = torch.device('cuda')
            elif torch.backends.mps.is_available():
                self.device = torch.device('mps')
            else:
                self.device = torch.device('cpu')

            self.policy_net = DQN(input_size, action_size).to(self.device) # This is the network that we use to learn
            self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)


            self.target_net = DQN(input_size, action_size).to(self.device) # This netwoek provides the label so that policy_net can learn
            self.target_net.load_state_dict(self.policy_net.state_dict())

            self.memory = ReplayMemory(buffer_size=10000)
            self.loss = nn.MSELoss()
        
        def select_action(self,state):
            if random.random() <= self.epsilon:
                action_index = random.randrange(self.action_size)
                return action_index
            else:
                state_tensor = torch.tensor(state,dtype=torch.float32).unsqueeze(0).to(self.device)
                q_values = self.policy_net(state_tensor)
                return q_values.argmax().item()
            
        def learn(self):

            if len(self.memory.buffer_size) < self.min_replay_size: #we dont have enough data yet 
                return
            
            transitions = self.memory.sample(self.batch_size)
            
            state = np.array([t[0] for t in transitions]) 
            action = np.array([t[1] for t in transitions]) 
            reward = np.array([t[2] for t in transitions]) 
            next_state = np.array([t[3] for t in transitions]) 
            done = np.array([t[4] for t in transitions])

            
            state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device)
            action_t = torch.as_tensor(action, dtype=torch.int64, device=self.device).unsqueeze(-1)
            reward_t = torch.as_tensor(reward, dtype=torch.float32, device=self.device).unsqueeze(-1)
            done_t   = torch.as_tensor(done, dtype=torch.float32, device=self.device).unsqueeze(-1)
        
            next_state_t = torch.as_tensor(next_state, dtype=torch.float32, device=self.device)
            
            current_q = self.policy_net(state_t).gather(1, action_t)
            

            #target q 
            target_q_values = self.target_net(next_state_t).max(1)[0].unsqueeze(-1)
        
            #Bellman Equation
            expected_q = reward_t + (self.gamma * target_q_values * (1 - done_t))

        
            loss = self.loss(current_q, expected_q)
        
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # check for step to update target network (every 1000)
            self.learn_step_counter += 1
            if self.learn_step_counter % self.update_target_step == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())

            # adjust epsilon 
            if self.epsilon > self.epsilon_end:
                self.epsilon *= self.epsilon_decay





            


            

            


