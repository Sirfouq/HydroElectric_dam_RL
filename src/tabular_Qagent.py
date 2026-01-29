from TestEnv import HydroElectric_Test
from ExperimentLogger import ExperimentLogger
import os
import numpy as np
import time
import matplotlib.pyplot as plt
from collections import deque
from visualizer import *
from numpy.lib.stride_tricks import sliding_window_view

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

        self.init_Qtable()
        self.init_Ntable()

    def init_Qtable(self):
        '''
        Initialize the Q-table based on state_dict and action_list, indices corresponding to order in OBS_IDX
        '''
        dimensions = [feature['size'] for feature in self.state_dict]
        dimensions.append(len(self.action_list))
        dimensions = np.array(dimensions, dtype=int)
        
        self.Q_table = np.zeros(dimensions)
        print(f"Initialized Q-table with shape: {self.Q_table.shape}")

    def init_Ntable(self):
        '''
        Initialize the N-table for counting state-action visits
        '''
        dimensions = [feature['size'] for feature in self.state_dict]
        dimensions.append(len(self.action_list))
        dimensions = np.array(dimensions, dtype=int)
        
        self.N_table = np.zeros(dimensions, dtype=int)
        print(f"Initialized N-table with shape: {self.N_table.shape}")

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
            elif method == 'digitize_zscore':
                bins = state_feature['bins']
                discrete_state[i] = np.digitize((obs_value-self.avg_price)/self.std_price, bins)
            elif method == 'function':
                func = state_feature['function']
                discrete_state[i] = func(obs_value)
  
        return tuple(discrete_state)
    
    # def reset_environment(self, env):
    #     '''
    #     Reset the environment manually, as test env does not have reset() method
    #     '''
    #     total_days = len(env.price_values)
    #     self.lookback_prices = deque(maxlen=24)
    #     if self.days_per_episode >= total_days -50:
    #         start_day = 1
    #         start_hour = 1
    #         env.counter = ((start_day - 1) * 24) + (start_hour - 1)  
    #         self.avg_price = env.price_values.mean()
    #         self.std_price = env.price_values.std() + 1e-6
    #     else:
    #         start_day = np.random.randint(2, total_days - self.days_per_episode)
    #         start_hour = np.random.randint(1, 25)
    #         env.counter = ((start_day - 1) * 24) + (start_hour - 1) 
    #         # Get initial lookback prices for rolling average            
    #         for price in env.price_values.flatten()[ env.counter - 24:env.counter]:
    #             self.lookback_prices.append(price)
            
    #         self.avg_price = np.mean(self.lookback_prices)
    #         self.std_price = np.std(self.lookback_prices) + 1e-6

              
    #     env.hour = start_hour
    #     env.day = start_day      
    #     env.state = np.empty(7)
    #     env.volume = np.random.uniform(0, env.max_volume)  # Random initial volume

    #     observation = env.observation()
    #     state_idx = self.get_discrete_state(observation)
    #     return env, observation, state_idx

    def reset_environment(self, env):
        '''
        Reset the environment manually, as test env does not have reset() method
        '''
        total_days = len(env.price_values)
        self.lookback_prices = deque(maxlen=24)
        
        # 1. Determine the NEW start time first
        if self.days_per_episode >= total_days - 50:
            start_day = 1
            start_hour = 1
            # For full dataset runs, you might not have 24h history at index 0.
            # Usually safe to initialize with the first available price or 0.
            current_flat_idx = 0 
        else:
            # Random start (ensure we have at least 24h buffer at the start)
            start_day = np.random.randint(2, total_days - self.days_per_episode)
            start_hour = np.random.randint(1, 25)
            
            # Calculate the flat index for the NEW start time
            current_flat_idx = ((start_day - 1) * 24) + (start_hour - 1)

        # 2. Populate History based on the NEW start time
        # Check if we have enough history (index >= 24)
        flat_prices = env.price_values.flatten()
        
        if current_flat_idx >= 24:
            past_prices = flat_prices[current_flat_idx - 24 : current_flat_idx]
            self.lookback_prices.extend(past_prices)
            
            self.avg_price = np.mean(self.lookback_prices)
            self.std_price = np.std(self.lookback_prices) + 1e-6
        else:
            # Fallback if starting at the very beginning of data (Day 1)
            # Use the global mean or just the current price
            self.avg_price = flat_prices[current_flat_idx]
            self.std_price = 1e-6
            self.lookback_prices.append(flat_prices[current_flat_idx])

        # 3. Apply the Reset to the Environment
        env.counter = current_flat_idx    
        env.hour = start_hour
        env.day = start_day      
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
        
        if np.random.uniform(0,1) < self.epsilon:
            action_idx = np.random.randint(0, len(self.action_list))
        else:
            # action as index of max Q-value for current state
            action_idx = np.argmax(self.Q_table[state_idx])

        # actual action to give to environment
        action = self.action_list[action_idx]
        return action, action_idx
    
    def get_Qreward(self, env, action, 
                    reward, 
                    current_volume, 
                    next_volume, 
                    avg_price = 50, 
                    penalize=False):
    
        # Value the potential based on the rolling average of 24 hours
        current_potential = current_volume * env.volume_to_MWh * avg_price * env.flow_efficiency
        next_potential = next_volume * env.volume_to_MWh * avg_price * env.flow_efficiency

        potential_delta = self.gamma*next_potential - current_potential
        penalty = 0

        if penalize:
            penalty = self.get_reward_penalty(env, action, current_volume, next_volume)

        return reward + potential_delta + penalty
        
    def get_reward_penalty(self, env, action, current_volume, next_volume):
        """
        Penalizes when the actual volume change differs from intended flow.
        Action is in [-1, 1] range, intended_flow = action * max_flow
        """
        volume_delta = next_volume - current_volume
        intended_flow = action * env.max_flow
        
        penalty = 0
        
        if action > 0:  # Trying to pump
            # Penalize if couldn't pump as much as intended
            if volume_delta < intended_flow:
                flow_pct = volume_delta / intended_flow
                penalty = -10 * (1 - flow_pct)
        
        elif action < 0:  # Trying to sell/release
            # Penalize if couldn't sell as much as intended
            if abs(volume_delta) < abs(intended_flow):
                flow_pct = abs(volume_delta) / abs(intended_flow)
                penalty = -10 * (1 - flow_pct)
            
        return penalty
    
    def update_Qtable(self, current_state_action_idx, next_state_idx, reward):
        '''
        Update Q-table using Q-learning update rule.
        Returns: The absolute magnitude of the update (for convergence tracking).
        '''
        
        max_next_q = np.max(self.Q_table[next_state_idx])
        target = reward + self.gamma * max_next_q
        current_q = self.Q_table[current_state_action_idx]
        delta = target - current_q
        change = self.lr * delta
        self.Q_table[current_state_action_idx] += change
        self.N_table[current_state_action_idx] += 1
        return abs(change)

    def adapt_hyperparameters(self, episode, adaptive_epsilon, epsilon_decay, adaptive_lr):
        '''
        Adapt epsilon and learning rate based on episode number
        1. Epsilon decays linearly from 1.0 to 0.01 over epsilon_decay episodes
        2. Learning rate decays as 1/sqrt(episode+1)
        '''
        
        if adaptive_epsilon:
            epsilon_start = 1.0
            epsilon_end = 0.05
            self.epsilon = np.interp(episode, [0, epsilon_decay], [epsilon_start, epsilon_end])
        
        if adaptive_lr:
            self.lr = 1/np.sqrt(episode+1)

    def train(self, n_episodes=100, 
              lr=0.1, epsilon = 0.05, epsilon_decay = 100, 
              gamma = 0.9,
              adaptive_epsilon = True, 
              adaptive_lr = True,
              days_per_episode = 30):
        
        
        # Initialize environment once
        env = HydroElectric_Test(self.train_path)

        self.gamma = gamma
        self.lr = lr
        self.epsilon = epsilon
        self.days_per_episode = days_per_episode
        self.steps_per_episode = days_per_episode*24

        train_history = {
            'episode': [],
            'epsilon': [],
            'lr': [],
            'raw_rewards': [],
            'q_rewards': [],
            'train_rollout': [],
            'validate_rollout': [],
            'action_counts': [], #[action_list]
            'n_table_stats': [], # [pct non zero, min, max, mean, median]
            'q_delta': [],       # NEW: Tracks stability
            'q_mean': []         # NEW: Tracks value growth
        }
        start_time = time.time()
        for episode in range(n_episodes):
            
            total_reward = 0
            env, observation, state_idx = self.reset_environment(env)
           
            cum_raw_reward = 0
            cum_q_reward = 0
            train_rollout_reward = 0
            validate_rollout_reward = 0
            action_counts = np.zeros(len(self.action_list), dtype = int)
            episode_q_delta = 0  # Track total updates this episode
            episode_q_mean = 0  # Track total updates this episode
                      
            self.adapt_hyperparameters(
                episode=episode,
                adaptive_epsilon=adaptive_epsilon,
                epsilon_decay=epsilon_decay,
                adaptive_lr=adaptive_lr
            )

            train_history['episode'].append(episode)
            train_history['epsilon'].append(self.epsilon)
            train_history['lr'].append(self.lr)
            
            for _ in range(self.steps_per_episode):
                # Get action - Epsilon-greedy
                action, action_idx = self.get_greedy_action(state_idx)

                # State-action index for Q-table update             
                current_state_action_idx = state_idx + (action_idx,)            
                next_observation, reward, terminated, truncated, info = env.step(action)

                # Reward shaping
                # Use observation to get current and next volume, price, for reward shaping
                current_volume = observation[OBS_IDX['volume']]
                next_volume = next_observation[OBS_IDX['volume']]
                current_price = observation[OBS_IDX['price']]

                # 24 hour rolling average price
                self.lookback_prices.append(current_price)
                self.avg_price = np.mean(self.lookback_prices)
                self.std_price = np.std(self.lookback_prices) + 1e-6

                Qreward = self.get_Qreward(env, action, reward, 
                                           current_volume, next_volume,
                                           avg_price=self.avg_price, penalize = True)

                # Q-value update
                next_state_idx = self.get_discrete_state(next_observation)          
                step_q_delta = self.update_Qtable(current_state_action_idx, next_state_idx, Qreward)
                
                # Log metrics
                cum_raw_reward += reward
                cum_q_reward += Qreward
                action_counts[action_idx] += 1



                episode_q_delta += step_q_delta
                episode_q_mean += np.mean(self.Q_table)

                # Move to next state
                state_idx = next_state_idx
                observation = next_observation

            n_table_nonzero = np.count_nonzero(self.N_table)
            n_table_min = np.min(self.N_table[self.N_table>0])
            n_table_max = np.max(self.N_table)
            n_table_mean = np.mean(self.N_table[self.N_table>0])
            n_table_median = np.median(self.N_table[self.N_table>0])
            
            n_table_stats = (n_table_nonzero / self.N_table.size * 100,
                                n_table_min,
                                n_table_max,
                                n_table_mean,
                                n_table_median)

            train_rollout_reward = self.rollout(data='train')
            validate_rollout_reward = self.rollout(data='validate')

            # Store metrics at end of episode
            train_history['raw_rewards'].append(cum_raw_reward)
            train_history['q_rewards'].append(cum_q_reward)
            train_history['train_rollout'].append(train_rollout_reward)
            train_history['validate_rollout'].append(validate_rollout_reward)
            train_history['action_counts'].append(action_counts / self.steps_per_episode * 100)
            train_history['n_table_stats'].append(n_table_stats)
            train_history['q_delta'].append(episode_q_delta / self.steps_per_episode)
            train_history['q_mean'].append(episode_q_mean / self.steps_per_episode)
            
            # Print less frequently to keep console clean
            if episode % 10 == 0:
                elapsed_time = time.time() - start_time
                print(f"Finished episode {episode}/{n_episodes} in {elapsed_time:.2f} seconds")
                start_time = time.time()
                avg_train_10 = np.array(train_history['train_rollout'][-10:]).mean()
                avg_val_10 = np.array(train_history['validate_rollout'][-10:]).mean()
                # print(f"Train Rollout {train_rollout_reward:.2f} €, Validate Rollout {validate_rollout_reward:.2f} €")
                # print(f"  Avg Train Rollout (last 10): {avg_train_rollout_10:.2f} €, Avg Validate Rollout (last 10): {avg_val_rollout_10:.2f} €")
                print(f"  Train: {train_rollout_reward:10.2f} € | Val: {validate_rollout_reward:10.2f} €")
                print(f"  Avg (last 10) - Train: {avg_train_10:10.2f} € | Val: {avg_val_10:10.2f} €")
                print(f"  Epsilon: {self.epsilon:.3f} | LR: {self.lr:.4f}")
                print(f"  N-table coverage: {n_table_stats[0]:.1f}%")
        return train_history

    def rollout(self, data = 'train'):
        if data == 'train':
            env = HydroElectric_Test(self.train_path)
        elif data == 'validate':
            env = HydroElectric_Test(self.validate_path)

        # Reset history
        self.price_history_rollout = deque(maxlen=24)

        # Get initial observation        
        observation = env.observation()

        # Set avg and std price based on intial price
        current_price = observation[OBS_IDX['price']]
        self.avg_price = current_price
        self.std_price = 1e-6
        # Uses avg and std price
        state_idx = self.get_discrete_state(observation)
        done = False

        cum_reward = 0.0      
        while not done:
            current_price = observation[OBS_IDX['price']]
            self.price_history_rollout.append(current_price)
            self.avg_price = np.mean(self.price_history_rollout)
            self.std_price = np.std(self.price_history_rollout) + 1e-6

            state_idx = self.get_discrete_state(observation) 
            action_idx = np.argmax(self.Q_table[state_idx])
            action = self.action_list[action_idx]

            next_obs, reward, terminated, truncated, info = env.step(action)
            next_state_idx = self.get_discrete_state(next_obs)

            done = terminated or truncated
            observation = next_obs
            state_idx = next_state_idx

            cum_reward += reward
        
        return cum_reward

    #########################
    # To do:
    # - Implement different exploration strategies
    # - Implement different Q-learning update methods (e.g., SARSA, TD(lambda))
    # - Implement rollout on validation data
    # - Implement saving/loading of Q-table
    # - Implement plotting of learning curves
    #########################

def get_price_bins(n_bins_prices, relative = True):
    '''
    Return price bins based on training data percentiles
    5 bins -> 20th, 40th, 60th, 80th percentiles
    '''
    current_dir = os.path.dirname(os.path.abspath(__file__))
    train_path = os.path.normpath(os.path.join(current_dir, '..', 'data\\train.xlsx'))
    env = HydroElectric_Test(train_path)
    train_prices = env.price_values.flatten()

    # Get Z-scores for prices based on rolling 24-hour mean and std
    
    window = 24
    windows = sliding_window_view(train_prices, window_shape=window)
    means = windows.mean(axis=1)
    stds  = windows.std(axis=1)
    current_prices = train_prices[window-1:]

    eps = 1e-6
    z_scores = (current_prices - means) / np.maximum(stds, eps)


    percentile = np.linspace(100/n_bins_prices, 100-100/n_bins_prices, n_bins_prices-1)
    
    if relative:
        price_bins = np.percentile(z_scores, percentile)
    else:
        price_bins = np.percentile(train_prices, percentile)
    return price_bins

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
    n_price_bins = 5
    n_volume_bins = 5

    price_bins = get_price_bins(n_price_bins, relative=True)
    # price_bins = np.linspace(-2.5, 2.5, n_price_bins+1)[1:-1]
    volume_bins = np.linspace(0, 100_000, n_volume_bins+1)[1:-1]
    
    # Experiment with different action spaces
    # action_list = [-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1]  # Sell all, sell half, hold, pump half, pump all
    action_list = [-1.0, -0.5, 0.0, 0.5, 1.0]  # Sell all, hold, pump all  
    # action_list = [-1.0, 0.0, 1.0]  # Sell all, hold, pump all  

    n_episodes = 200
    epsilon_decay = 180
    days_per_episode = int(365/2)
    gamma = 0.99
  
    state_dict = [
        # {'obs_idx': OBS_IDX['price'], 'method': 'digitize', 'bins': price_bins, 'size': n_price_bins, 'name': 'price'},
        {'obs_idx': OBS_IDX['price'], 'method': 'digitize_zscore', 'bins': price_bins, 'size': n_price_bins, 'name': 'z_score_price'},
        {'obs_idx': OBS_IDX['volume'], 'method': 'digitize', 'bins': volume_bins, 'size': n_volume_bins, 'name': 'volume'},
        # {'obs_idx': OBS_IDX['hour_of_day'], 'method': 'function', 'function': get_day_period,  'size': 4, 'name': 'day_period'},
        {'obs_idx': OBS_IDX['day_of_week'], 'method': 'function', 'function': get_weekday,  'size': 2, 'name': 'weekday_weekend'},
        # {'obs_idx': OBS_IDX['day_of_week'], 'method': 'function', 'function': price_above_rolling_avg, 'method': get_weekday, 'size': 2, 'name': 'weekday_weekend'}
    ]
    
    print("Price bins: ", price_bins)
    print("Volume bins: ", volume_bins)
    
    run_name = f"Ep{n_episodes}_Gamma{gamma}_Zscore"
    logger = ExperimentLogger(base_dir='results', run_name=run_name)

    config = {
        'n_episodes': n_episodes,
        'epsilon_decay': epsilon_decay,
        'days_per_episode': days_per_episode,
        'gamma': gamma,
        'action_list': action_list,
        'state_dict_summary': [s['name'] for s in state_dict], # Simplify complex dicts for JSON
        'state_dict': state_dict,
        'price_bins': price_bins,
        'volume_bins': volume_bins
    }
  

    agent = tabularQagent(state_dict=state_dict, action_list=action_list)
    
    print("Starting Training...")
    train_history = agent.train(
        n_episodes=n_episodes,
        epsilon_decay=epsilon_decay,
        gamma=gamma,
        lr=0.1,
        adaptive_epsilon=True,
        adaptive_lr=False,
        days_per_episode=days_per_episode
    )

    config['final_train_reward'] = train_history['train_rollout'][-1]
    config['final_validate_reward'] = train_history['validate_rollout'][-1]

    # --- 4. SAVE RESULTS ---
    print("Saving Results...")
    
    logger.save_config(config)
    logger.save_agent(agent)       
    logger.save_history(train_history)
    
    print(f"Done! Final Train: {config['final_train_reward']:.0f}, Final Val: {config['final_validate_reward']:.0f}")
    
    print("Done!")

    # plot_learning_metrics(history)
    # plot_policy_heatmap(agent)
    # print("Train Rollout")
    # agent.rollout(data='train', )
    # print("Validate Rollout")
    # agent.rollout(data='validate')

if __name__ == "__main__":
    main()