import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np



class DQN(nn.Module):
    def __init__(self,input_space,action_space):
        super().__init__()

        self.net = nn.Sequential(
    nn.Linear(input_space, 64),  
    nn.Tanh(),                    
    nn.Linear(64, 128),          
    nn.Tanh(),
    nn.Linear(128, action_space)
)

    def forward(self,observation):
        return self.net(observation)
    


# Transition = namedtuple('Transition',
#                         ('state', 'action', 'next_state', 'reward'))

# class ExperienceReplay():

#     def __init__(self,capacity):
#         self.memory = deque(maxlen=capacity)

#     def push(self,*args):
#         self.memory.append(Transition(*args))

#     def sample(self,batch_size):
#         return random.sample(self.memory,batch_size)

class PrioritizedExperienceReplay:
    def __init__(self, capacity, alpha=0.6, beta=0.4, beta_increment=0.001, epsilon=0.0001):
        
        self.capacity = capacity
        self.memory = [None for _ in range(capacity)]
        self.position = 0
        self.size = 0  # tracks how many items are currently in memory 
        
        
        self.priorities = np.zeros(capacity, dtype=np.float32)
        
        
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.epsilon = epsilon

    def push(self, state, action, reward, next_state, done):
        
        current_experience = (state, action, reward, next_state, done)
        self.memory[self.position] = current_experience
        
        # Get the max priority so new experiences  could be sampled at least once 
        max_priority = self.priorities.max() if self.size > 0 else 1.0
        self.priorities[self.position] = max_priority
        
        # Move the circular pointer
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        # rest of the memory is zeros so we just slice until the size point.
        real_priorities = self.priorities[:self.size]
        
        
        # P(i) = p_i^alpha / sum(p^alpha)
        probs = real_priorities ** self.alpha
        probs /= probs.sum()
        
        # Randomly select indices
        indices = np.random.choice(self.size, batch_size, p=probs)
        
        
        # w_i = (1/N * 1/P(i))^ beta
        weights = (self.size * probs[indices]) ** (-self.beta)
        weights /= weights.max() # Normalize
        
        
        batch = [self.memory[idx] for idx in indices]
        states, actions, rewards, next_states, dones = zip(*batch)
        
        
        states = torch.tensor(np.array(states), dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.long).unsqueeze(1)
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1)
        weights = torch.tensor(weights, dtype=torch.float32).unsqueeze(1)

        return states, actions, rewards, next_states, dones, indices, weights

    def update_priorities(self, indices, td_errors):
        
        if isinstance(td_errors, torch.Tensor):
            td_errors = td_errors.detach().cpu().numpy()
            
        # We add  epsilon to ensure non-zero probability
        new_priorities = np.abs(td_errors) + self.epsilon
        
        
        self.priorities[indices] = new_priorities.flatten()

    def increase_beta(self):
        self.beta = min(1.0, self.beta + self.beta_increment)

          
          

class DQN_agent():

    def __init__(self, 
                 input_size, 
                 action_size, 
                 memory_size,
                 target_update_freq = 1000, 
                 lr= 0.0005,
                 gamma_start= 0.9,
                 gamma_end = 0.99,
                 epsilon_start = 1.0,
                 epsilon_end = 0.01,
                 epsilon_decay = 0.9995):
        
       
        self.input_size = input_size
        self.action_size = action_size
        self.memory_size = memory_size
        self.target_update_freq = target_update_freq
        self.learn_step_counter = 0
        self.lr = lr
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        
        # Dynamic discount rate
        self.gamma_start = gamma_start
        self.gamma_end = gamma_end
        self.gamma = gamma_start 
        
        
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            self.device = torch.device('mps')
        else:
            self.device = torch.device('cpu') 
        
        
        self.memory = PrioritizedExperienceReplay(capacity=memory_size)
        self.action_map = np.linspace(-1, 1, self.action_size)

        # Policy network will output the predicted best action and target network will represent the 'true' best action 
        self.policy_net = DQN(self.input_size, self.action_size).to(self.device)
        self.target_net = DQN(self.input_size, self.action_size).to(self.device)
        
        
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval() # Target net is never trained directly
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)
            
        


    def select_action(self,state):
        if random.random() < self.epsilon :
             action_index = random.randrange(self.action_size)
        else :
            state_tensor = torch.tensor(state,dtype=torch.float32).unsqueeze(dim=0).to(self.device)
            with torch.no_grad():
                q_values = self.policy_net(state_tensor)
                action_index = q_values.argmax().item()
        
        continuous_value = self.action_map[action_index]

        return action_index,continuous_value
    
    def act(self,state):  #THIS IS ONLY FOR VALIDATING NO TRAINING !!!!!
        
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(dim=0).to(self.device)
        with torch.no_grad():
            q_values = self.policy_net(state_tensor)
            action_index = q_values.argmax().item()

        
        continuous_value = self.action_map[action_index]
        
        return continuous_value
             

    def learn(self,batch_size =128):
        if self.memory.size < batch_size:
            return
        
        states,actions,rewards,next_states,dones,indices,weights = self.memory.sample(batch_size=batch_size)

        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        weights = weights.to(self.device)

        current_q = self.policy_net(states).gather(1, actions) #We gather the q-values of the  actions we actually took based on memory

        with torch.no_grad():
             # Use Policy Net to select the best action for the next state
            best_next_actions = self.policy_net(next_states).argmax(1).unsqueeze(1)
            
            # Use Target Net to calculate the value of that specific action
            next_q_values = self.target_net(next_states).gather(1, best_next_actions)
            
            # Bellman Equation: R + Gamma * Future_Value
            # If done is 1, (1-done) becomes 0, removing the future value.
            target_q = rewards + (self.gamma * next_q_values * (1 - dones))

        
        criterion = nn.SmoothL1Loss(reduction='none')
        element_wise_loss = criterion(current_q, target_q)
        
        # Multiply each loss by its Importance Sampling Weight (PER Logic)
        weighted_loss = (element_wise_loss * weights).mean()

       
        self.optimizer.zero_grad()
        weighted_loss.backward()
        #  Prevent exploding gradients by clipping
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100.0)
        self.optimizer.step()

        
        # Calculate TD errors (absolute difference) .This will decide the priority they get 
        td_errors = torch.abs(current_q - target_q).detach()
        self.memory.update_priorities(indices, td_errors)
        
        
        self.memory.increase_beta()

        #Update the target network after target_update_freq .
        self.learn_step_counter+=1
        if self.learn_step_counter % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
            # print('Target Net updated')

    def update_hyperparameters(self, episode, total_episodes):
        
        progress = episode / total_episodes
        self.gamma = self.gamma_start + (self.gamma_end - self.gamma_start) * progress
        self.gamma = min(self.gamma, self.gamma_end)
        
       
        if self.epsilon > self.epsilon_end:
            self.epsilon *= self.epsilon_decay

        
#---BEST ARCHITECTURE SO FAR SAVED FOR SAFETY----- 
         
        
# import math
# import random
# import matplotlib
# import matplotlib.pyplot as plt
# from collections import namedtuple, deque
# import torch
# import torch.nn as nn
# import torch.optim as optim
# import torch.nn.functional as F
# import numpy as np



# class DQN(nn.Module):
#     def __init__(self,input_space,action_space):
#         super().__init__()

#         self.net = nn.Sequential(
#             nn.Linear(input_space,64),
#             nn.Tanh(),
#             nn.Linear(64,128),
#             nn.Tanh(),
#             nn.Linear(128,action_space)
#         )

#     def forward(self,observation):
#         return self.net(observation)
    


# # Transition = namedtuple('Transition',
# #                         ('state', 'action', 'next_state', 'reward'))

# # class ExperienceReplay():

# #     def __init__(self,capacity):
# #         self.memory = deque(maxlen=capacity)

# #     def push(self,*args):
# #         self.memory.append(Transition(*args))

# #     def sample(self,batch_size):
# #         return random.sample(self.memory,batch_size)

# class PrioritizedExperienceReplay:
#     def __init__(self, capacity, alpha=0.6, beta=0.4, beta_increment=0.001, epsilon=0.0001):
        
#         self.capacity = capacity
#         self.memory = [None for _ in range(capacity)]
#         self.position = 0
#         self.size = 0  # tracks how many items are currently in memory 
        
        
#         self.priorities = np.zeros(capacity, dtype=np.float32)
        
        
#         self.alpha = alpha
#         self.beta = beta
#         self.beta_increment = beta_increment
#         self.epsilon = epsilon

#     def push(self, state, action, reward, next_state, done):
        
#         current_experience = (state, action, reward, next_state, done)
#         self.memory[self.position] = current_experience
        
#         # Get the max priority so new experiences  could be sampled at least once 
#         max_priority = self.priorities.max() if self.size > 0 else 1.0
#         self.priorities[self.position] = max_priority
        
#         # Move the circular pointer
#         self.position = (self.position + 1) % self.capacity
#         self.size = min(self.size + 1, self.capacity)

#     def sample(self, batch_size):
#         # rest of the memory is zeros so we just slice until the size point.
#         real_priorities = self.priorities[:self.size]
        
        
#         # P(i) = p_i^alpha / sum(p^alpha)
#         probs = real_priorities ** self.alpha
#         probs /= probs.sum()
        
#         # Randomly select indices
#         indices = np.random.choice(self.size, batch_size, p=probs)
        
        
#         # w_i = (1/N * 1/P(i))^ beta
#         weights = (self.size * probs[indices]) ** (-self.beta)
#         weights /= weights.max() # Normalize
        
        
#         batch = [self.memory[idx] for idx in indices]
#         states, actions, rewards, next_states, dones = zip(*batch)
        
        
#         states = torch.tensor(np.array(states), dtype=torch.float32)
#         actions = torch.tensor(actions, dtype=torch.long).unsqueeze(1)
#         rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1)
#         next_states = torch.tensor(np.array(next_states), dtype=torch.float32)
#         dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1)
#         weights = torch.tensor(weights, dtype=torch.float32).unsqueeze(1)

#         return states, actions, rewards, next_states, dones, indices, weights

#     def update_priorities(self, indices, td_errors):
        
#         if isinstance(td_errors, torch.Tensor):
#             td_errors = td_errors.detach().cpu().numpy()
            
#         # We add  epsilon to ensure non-zero probability
#         new_priorities = np.abs(td_errors) + self.epsilon
        
        
#         self.priorities[indices] = new_priorities.flatten()

#     def increase_beta(self):
#         self.beta = min(1.0, self.beta + self.beta_increment)

          
          

# class DQN_agent():

#     def __init__(self, 
#                  input_size, 
#                  action_size, 
#                  memory_size,
#                  target_update_freq = 1000, 
#                  lr= 0.0005,
#                  gamma_start= 0.9,
#                  gamma_end = 0.99,
#                  epsilon_start = 1.0,
#                  epsilon_end = 0.01,
#                  epsilon_decay = 0.9995):
        
       
#         self.input_size = input_size
#         self.action_size = action_size
#         self.memory_size = memory_size
#         self.target_update_freq = target_update_freq
#         self.learn_step_counter = 0
#         self.lr = lr
#         self.epsilon = epsilon_start
#         self.epsilon_end = epsilon_end
#         self.epsilon_decay = epsilon_decay
        
#         # Dynamic discount rate
#         self.gamma_start = gamma_start
#         self.gamma_end = gamma_end
#         self.gamma = gamma_start 
        
        
#         if torch.cuda.is_available():
#             self.device = torch.device('cuda')
#         elif torch.backends.mps.is_available():
#             self.device = torch.device('mps')
#         else:
#             self.device = torch.device('cpu') 
        
        
#         self.memory = PrioritizedExperienceReplay(capacity=memory_size)
#         self.action_map = np.linspace(-1, 1, self.action_size)

#         # Policy network will output the predicted best action and target network will represent the 'true' best action 
#         self.policy_net = DQN(self.input_size, self.action_size).to(self.device)
#         self.target_net = DQN(self.input_size, self.action_size).to(self.device)
        
        
#         self.target_net.load_state_dict(self.policy_net.state_dict())
#         self.target_net.eval() # Target net is never trained directly
        
#         self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)
            
        


#     def select_action(self,state):
#         if random.random() < self.epsilon :
#              action_index = random.randrange(self.action_size)
#         else :
#             state_tensor = torch.tensor(state,dtype=torch.float32).unsqueeze(dim=0).to(self.device)
#             with torch.no_grad():
#                 q_values = self.policy_net(state_tensor)
#                 action_index = q_values.argmax().item()
        
#         continuous_value = self.action_map[action_index]

#         return action_index,continuous_value
    
#     def act(self,state):  #THIS IS ONLY FOR VALIDATING NO TRAINING !!!!!
        
#         state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(dim=0).to(self.device)
#         with torch.no_grad():
#             q_values = self.policy_net(state_tensor)
#             action_index = q_values.argmax().item()

        
#         continuous_value = self.action_map[action_index]
        
#         return continuous_value
             

#     def learn(self,batch_size =128):
#         if self.memory.size < batch_size:
#             return
        
#         states,actions,rewards,next_states,dones,indices,weights = self.memory.sample(batch_size=batch_size)

#         states = states.to(self.device)
#         actions = actions.to(self.device)
#         rewards = rewards.to(self.device)
#         next_states = next_states.to(self.device)
#         dones = dones.to(self.device)
#         weights = weights.to(self.device)

#         current_q = self.policy_net(states).gather(1, actions) #We gather the q-values of the  actions we actually took based on memory

#         with torch.no_grad():
#              # Use Policy Net to select the best action for the next state
#             best_next_actions = self.policy_net(next_states).argmax(1).unsqueeze(1)
            
#             # Use Target Net to calculate the value of that specific action
#             next_q_values = self.target_net(next_states).gather(1, best_next_actions)
            
#             # Bellman Equation: R + Gamma * Future_Value
#             # If done is 1, (1-done) becomes 0, removing the future value.
#             target_q = rewards + (self.gamma * next_q_values * (1 - dones))

        
#         criterion = nn.SmoothL1Loss(reduction='none')
#         element_wise_loss = criterion(current_q, target_q)
        
#         # Multiply each loss by its Importance Sampling Weight (PER Logic)
#         weighted_loss = (element_wise_loss * weights).mean()

       
#         self.optimizer.zero_grad()
#         weighted_loss.backward()
#         #  Prevent exploding gradients by clipping
#         torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
#         self.optimizer.step()

        
#         # Calculate TD errors (absolute difference) .This will decide the priority they get 
#         td_errors = torch.abs(current_q - target_q).detach()
#         self.memory.update_priorities(indices, td_errors)
        
        
#         self.memory.increase_beta()

#         #Update the target network after target_update_freq .
#         self.learn_step_counter+=1
#         if self.learn_step_counter % self.target_update_freq == 0:
#             self.target_net.load_state_dict(self.policy_net.state_dict())
#             # print('Target Net updated')

#     def update_hyperparameters(self, episode, total_episodes):
        
#         progress = episode / total_episodes
#         self.gamma = self.gamma_start + (self.gamma_end - self.gamma_start) * progress
#         self.gamma = min(self.gamma, self.gamma_end)
        
       
#         if self.epsilon > self.epsilon_end:
#             self.epsilon *= self.epsilon_decay



    


