from TestEnv import HydroElectric_Test
import os
import numpy as np
import time
import matplotlib.pyplot as plt

#[volume, price, hour_of_day, day_of_week, day_of_year, month_of_year, year]
#Observation indices lookup
OBS_IDX = {
    'volume': 0,
    'price': 1,
    'hour_of_day': 2,
    'day_of_week': 3,
    'day_of_year': 4,
    'month_of_year': 5,
    'year': 6
}

class tabularQagent:    
    def __init__(self, state_dict, action_list):
        '''
        Initialize the Tabular Q-learning agent
        state_dict: List of dictionaries how to discretize its states in format per method
                    Methods are
                        - digitize based on bins for continuous variables, volume, price
                            {'feature_idx': 0, 'method': 'digitize', 'bins': [...], 'size': n_bins, 'name': 'Volume'}
                        - discrete, based on a function, for day/time based features
                            {'feature_idx': 1, 'method': function, 'function': function_name, 'size': n_bins, 'name': part_of_day}
                            The function should map each time observations to indices
        action_list: List of possible discrete actions the agent can take
        '''
        # Define paths to training and validation data
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.train_path = os.path.normpath(os.path.join(current_dir, '..', 'data\\train.xlsx'))
        self.validate_path = os.path.normpath(os.path.join(current_dir, '..', 'data\\validate.xlsx'))
        
        self.state_dict = state_dict
        self.action_list = action_list

        self._init_Qtable()

    def _init_Qtable(self):
        '''
        Initialize the Q-table based on state_dict and action_list, indices corresponding to order in OBS_IDX
        '''
        dimensions = [feature['size'] for feature in self.state_dict]
        dimensions.append(len(self.action_list))
        dimensions = np.array(dimensions, dtype=int)
        
        self.Q_table = np.zeros(dimensions)
        print(f"Initialized Q-table with shape: {self.Q_table.shape}")

    def get_discrete_state(self, observation):
        '''
        Convert observation to discrete state corresponding to Q-table indices
        '''
        discrete_state = np.empty(len(self.state_dict), dtype=int)

        for i, state_feature in enumerate(self.state_dict):
            obs_value = observation[state_feature['obs_idx']]
            method = state_feature['method']
   
            if method == 'digitize':
                bins = state_feature['bins']
                discrete_state[i] = np.digitize(obs_value, bins)
            elif method == 'function':
                func = state_feature['function']
                discrete_state[i] = func(obs_value)
  
        return tuple(discrete_state)
    
    def reset_environment(self, env):
        '''
        Reset the environment manually, as test env does not have reset() method
        '''
        # env = HydroElectric_Test(self.train_path)
        env.counter = 0
        env.hour = 1
        env.day = 1

        env.state = np.empty(7)
        env.volume = np.random.uniform(0, env.max_volume)  # Random initial volume      
        observation = env.observation()
        state_idx = self.get_discrete_state(observation)
        return env, observation, state_idx
    
    def get_greedy_action(self, state_idx):
        '''
        Get action using epsilon-greedy policy, based on current state index. 
        returns action and action index.
        '''
        # Implement here possible different exploration strategies
        if np.random.uniform(0,1) < self.epsilon:
            action_idx = np.random.randint(0, len(self.action_list))
        else:
            # action as index of max Q-value for current state
            action_idx = np.argmax(self.Q_table[state_idx])

        # actual action to give to environment
        action = self.action_list[action_idx]
        return action, action_idx
    
    def get_Qreward(self, reward, current_volume, next_volume, current_price):
        # ###########################
        # # Define Logic for reward shaping
        # # Use Qreward in to update Q-values
        # ###########################
        pass
    
    def update_Qtable(self, current_state_action_idx, next_state_idx, reward):
        '''
        Update Q-table using Q-learning update rule, via TD(0)
        '''
        # Additional methods as TD(lambda) can be implemented later
        Qtarget = reward + self.gamma*np.max(self.Q_table[next_state_idx])
        delta = Qtarget - self.Q_table[current_state_action_idx]
        self.Q_table[current_state_action_idx] += self.lr*delta
        
    def adapt_hyperparameters(self, episode, adaptive_epsilon, epsilon_decay, adaptive_lr):
        '''
        Adapt epsilon and learning rate based on episode number
        1. Epsilon decays linearly from 1.0 to 0.01 over epsilon_decay episodes
        2. Learning rate decays as 1/sqrt(episode+1)
        '''
        # Implement here different adaptation strategies if needed
        if adaptive_epsilon:
            epsilon_start = 1.0
            epsilon_end = 0.01
            self.epsilon = np.interp(episode, [0, epsilon_decay], [epsilon_start, epsilon_end])
        
        if adaptive_lr:
            self.lr = 1/np.sqrt(episode+1)

    def train(self, n_episodes=100, 
              lr=0.1, epsilon = 0.05, epsilon_decay = 20, 
              gamma = 0.9,
              adaptive_epsilon = True, 
              adaptive_lr = True):
        
        # Initialize environment once
        env = HydroElectric_Test(self.train_path)

        self.gamma = gamma
        self.lr = lr
        self.epsilon = epsilon

        rewards = []
        average_rewards = []

        for episode in range(n_episodes):
            total_reward = 0
            env, observation, state_idx = self.reset_environment(env)
            done = False

            self.adapt_hyperparameters(episode, adaptive_epsilon, adaptive_lr, epsilon_decay)
            start_time = time.time()

            while not done:

                # Get action - Epsilon-greedy
                action, action_idx = self.get_greedy_action(state_idx)
                
                # State-action index for Q-table update
                current_state_action_idx = state_idx + (action_idx,)            
                next_observation, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

                # Reward shaping can be added here
                # Use observation to get current and next volume, price, for custom reward shaping

                # Q-value update
                next_state_idx = self.get_discrete_state(next_observation)
                self.update_Qtable(current_state_action_idx, next_state_idx, reward)

                # Update reward, state, hyperparameters
                total_reward += reward
                state_idx = next_state_idx
                observation = next_observation

            rewards.append(total_reward)


            ######################
            # Print statements for debugging and monitoring
            ######################

            print(f"Episode {episode} finished with total reward: {total_reward}, in {time.time() - start_time:.2f} seconds.")

            #Calculate the average score over 100 episodes
            if (episode) % 10 == 0:
                # Check if Q-table is changing
                non_zero = np.count_nonzero(self.Q_table)
                q_mean = np.mean(self.Q_table)
                q_std = np.std(self.Q_table)
                q_max = np.max(self.Q_table)
                q_min = np.min(self.Q_table)
                print("\n")
                print(f"Episode {episode}: Q-table stats: "
                  f"non-zero={non_zero}/75, mean={q_mean:.2f}, "
                  f"std={q_std:.2f}, range=[{q_min:.2f}, {q_max:.2f}], "
                  f"Total Reward: {total_reward},"
                  f"Epsilon: {self.epsilon:.3f}, Learning Rate: {self.lr:.3f}")
            
            if episode % 100 == 0:
                mean_last_100 = np.mean(rewards[-100:])
                average_rewards.append(mean_last_100)
                print(f"Episode {episode}, Total Reward: {total_reward}, "
                      f"Average Reward (last 100): {mean_last_100:.2f}, "
                      f"Epsilon: {self.epsilon:.3f}, Learning Rate: {self.lr:.3f}")

        # Investigate final Q-table
        print('The simulation is done!')
        for action_idx in range(len(self.action_list)):
            print(f"\nQ-values for action {self.action_list[action_idx]}:")
            print(self.Q_table[:,:,action_idx])  
        

    def train_rollout(self, max_steps=None, verbose=True):
        # Perform a rollout using the learned Q-table policy on train data
        # Investigate the trajectory and print diagnostics

        env = HydroElectric_Test(self.train_path)
        observation = env.observation()
        state_idx = self.get_discrete_state(observation)
        done = False
        t = 0

        trajectory = []

        while not done:
            q_values = self.Q_table[state_idx]
            action_idx = np.argmax(q_values)
            action = self.action_list[action_idx]

            next_obs, reward, terminated, truncated, info = env.step(action)
            next_state_idx = self.get_discrete_state(next_obs)

            trajectory.append({
                "t": t,
                "price": observation[OBS_IDX["price"]],
                "volume": observation[OBS_IDX["volume"]],
                "state_idx": state_idx,
                "action": action,
                "action_idx": action_idx,
                "reward": reward,
                "Q_sell": q_values[0],
                "Q_hold": q_values[1],
                "Q_pump": q_values[2],
            })

            done = terminated or truncated
            observation = next_obs
            state_idx = next_state_idx
            t += 1

            if max_steps is not None and t >= max_steps:
                break

        # =========================
        # Aggregate diagnostics
        # =========================
        total_reward = sum(step["reward"] for step in trajectory)
        actions = [step["action"] for step in trajectory]

        action_counts = {
            a: actions.count(a) for a in set(self.action_list)
        }

        avg_reward = np.mean([step["reward"] for step in trajectory])
        reward_std = np.std([step["reward"] for step in trajectory])

        if verbose:
            print("\n=== ROLLOUT SUMMARY ===")
            print(f"Steps: {len(trajectory)}")
            print(f"Total reward: {total_reward:.2f}")
            print(f"Avg reward per step: {avg_reward:.2f} ± {reward_std:.2f}")
            print("Action counts:", action_counts)

            print("\n=== FIRST 20 STEPS ===")
            for step in trajectory[:20]:
                print(
                    f"t={step['t']:3d} | price={step['price']:.2f} | vol={step['volume']:.0f} | "
                    f"a={step['action']} | r={step['reward']:.2f} | "
                    f"Q=[{step['Q_sell']:.1f}, {step['Q_hold']:.1f}, {step['Q_pump']:.1f}]"
                )

            print("\n=== LAST 20 STEPS ===")
            for step in trajectory[-20:]:
                print(
                    f"t={step['t']:3d} | price={step['price']:.2f} | vol={step['volume']:.0f} | "
                    f"a={step['action']} | r={step['reward']:.2f}"
                )
        
        policy = np.argmax(self.Q_table, axis=2)
        print("\nPolicy (0=sell, 1=hold, 2=pump):")
        print(policy)


        return trajectory


    #########################
    # To do:
    # - Implement different exploration strategies
    # - Implement different Q-learning update methods (e.g., SARSA, TD(lambda))
    # - Implement rollout on validation data
    # - Implement saving/loading of Q-table
    # - Implement plotting of learning curves
    #########################


def get_price_bins(n_bins_prices):
    '''
    Return price bins based on training data percentiles
    5 bins -> 20th, 40th, 60th, 80th percentiles
    '''
    current_dir = os.path.dirname(os.path.abspath(__file__))
    train_path = os.path.normpath(os.path.join(current_dir, '..', 'data\\train.xlsx'))
    env = HydroElectric_Test(train_path)
    train_prices = env.price_values.flatten()
    percentile = np.linspace(100/n_bins_prices, 100-100/n_bins_prices, n_bins_prices-1)
    price_bins = np.percentile(train_prices, percentile)
    return price_bins

# Definition of helper functions for state discretization 
def get_day_period(hour):
    '''
    Map hour of day to part of day index 0,1,2,3
    '''
    if hour == 24:
        hour = 0

    if 0 <= hour < 6:
        return 0 # Night
    elif 6 <= hour < 12:
        return 1 # Morning
    elif 12 <= hour < 18:
        return 2 # Midday
    else:
        return 3 # Evening
    
def price_above_rolling_avg(current_price, agent):
    '''
    Determine if current price is above rolling average price
    0: below average
    1: above average
    '''
    return 1 if current_price >= agent.rolling_avg_price else 0

def get_weekday(day_of_week):
    '''
    Map day of week to weekday/weekend index
    0: weekday (Mon-Fri)
    1: weekend (Sat-Sun)
    '''
    if day_of_week in [5,6]:
        return 1
    else:
        return 0


def main():
    # Experiment with different discretization schemes
    n_price_bins = 5
    n_volume_bins = 5

    price_bins = get_price_bins(n_price_bins)
    volume_bins = np.linspace(0, 100_000, n_volume_bins+1)[1:-1]

    # Experiment with different action spaces
    # action_list = [-1, -0.5, 0, 0.5, 1]  # Sell all, sell half, hold, pump half, pump all
    action_list = [-1.0, 0.0, 1.0]  # Sell all, hold, pump all  


    state_dict = [
        {'obs_idx': OBS_IDX['price'], 'method': 'digitize', 'bins': price_bins, 'size': n_price_bins, 'name': 'price'},
        {'obs_idx': OBS_IDX['volume'], 'method': 'digitize', 'bins': volume_bins, 'size': n_volume_bins, 'name': 'volume'},
        # {'obs_idx': OBS_IDX['hour_of_day'], 'method': 'function', 'function': get_day_period,  'size': 4, 'name': 'day_period'},
        # {'obs_idx': OBS_IDX['day_of_week'], 'method': get_weekday, 'size': 2, 'name': 'weekday_weekend'}
    ]
    
    print("Price bins: ", price_bins)
    print("Volume bins: ", volume_bins)
    
    agent = tabularQagent(state_dict=state_dict, action_list=action_list)
    
    # Train the agent
    # try different hyperparameters here
    agent.train(
        n_episodes=100,
        epsilon_decay=100,  # Match n_episodes
        gamma=0.999,        # High for long episodes
        lr=0.1,
        adaptive_epsilon=True,
        adaptive_lr=True
    )

    agent.train_rollout()

if __name__ == "__main__":
    main()