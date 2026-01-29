import numpy as np
import os
import json
import pandas as pd
from collections import deque

# Import the necessary functions and OBS_IDX from tabular_Qagent
from tabular_Qagent import OBS_IDX, get_day_period, get_weekday


def load_agent(run_folder):
    """Load saved agent from disk"""
    
    # Load Tables
    data = np.load(os.path.join(run_folder, 'model.npz'))
    q_table = data['Q_table']
    n_table = data['N_table']
    
    # Load Config (to reconstruct the agent structure)
    with open(os.path.join(run_folder, 'config.json'), 'r') as f:
        config = json.load(f)
        
    print(f"Loaded agent from {run_folder}")
    print(f"  Q-table shape: {q_table.shape}")
    print(f"  Action list: {config['action_list']}")
    print(f"  State features: {config['state_dict_summary']}")
    
    return q_table, n_table, config


def load_metrics(run_folder):
    """Load the training history CSV for plotting"""
    csv_path = os.path.join(run_folder, 'training_log.csv')
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)
    else:
        print(f"Warning: No training_log.csv found in {run_folder}")
        return None


class TabularQAgentValidator:
    """
    Lightweight validator for tabular Q-agent.
    Only implements act() method for use with main.py validation loop.
    """
    
    def __init__(self, run_folder: str):
        """
        Load a trained agent from disk.
        
        Args:
            run_folder: Path to results directory (e.g., 'results/Ep20_Gamma0.99_Zscore_20250129_143022')
        """
        self.q_table, self.n_table, config = load_agent(run_folder)
        
        # Reconstruct state_dict with function references
        # (JSON can't store functions, so we need to re-attach them)
        self.state_dict = self._reconstruct_state_dict(config['state_dict'])
        self.action_list = config['action_list']
        
        # Initialize price history for Z-score calculation
        self.price_history = deque(maxlen=24)
        self.avg_price = 50.0  # Default fallback
        self.std_price = 10.0  # Default fallback
        
        print("TabularQAgentValidator initialized and ready to act()")
    
    def _reconstruct_state_dict(self, state_dict_from_json):
        """
        Reconstruct state_dict with actual function references.
        JSON serialization loses function pointers, so we map them back.
        """
        reconstructed = []
        
        for feature in state_dict_from_json:
            feature_copy = feature.copy()
            
            # If this feature uses a function, map the name back to the actual function
            if feature['method'] == 'function':
                func_name = feature.get('function', None)
                
                # Map function names to actual functions
                function_map = {
                    'get_day_period': get_day_period,
                    'get_weekday': get_weekday,
                }
                
                if func_name in function_map:
                    feature_copy['function'] = function_map[func_name]
                else:
                    raise ValueError(f"Unknown function: {func_name}")
            
            reconstructed.append(feature_copy)
        
        return reconstructed
    
    def get_discrete_state(self, observation):
        """
        Convert observation to discrete state tuple.
        Must match the training agent's discretization logic exactly.
        """
        discrete_state = np.empty(len(self.state_dict), dtype=int)

        for i, state_feature in enumerate(self.state_dict):
            obs_value = observation[state_feature['obs_idx']]
            method = state_feature['method']
   
            if method == 'digitize':
                bins = state_feature['bins']
                bin_idx = np.digitize(obs_value, bins)
                discrete_state[i] = np.clip(bin_idx, 0, state_feature['size'] - 1)
                
            elif method == 'digitize_zscore':
                bins = state_feature['bins']
                # Compute Z-score using rolling statistics
                z_score = (obs_value - self.avg_price) / self.std_price
                bin_idx = np.digitize(z_score, bins)
                discrete_state[i] = np.clip(bin_idx, 0, state_feature['size'] - 1)
                
            elif method == 'function':
                func = state_feature['function']
                discrete_state[i] = func(obs_value)
  
        return tuple(discrete_state)
    
    def act(self, observation):
        """
        Select action based on learned Q-table (greedy policy).
        This is the only method called by main.py's validate_agent() loop.
        
        Args:
            observation: numpy array [volume, price, hour, day_of_week, ...]
        
        Returns:
            action: float in [-1, 1] range (e.g., -1.0, -0.5, 0.0, 0.5, 1.0)
        """
        # Update rolling price statistics for Z-score features
        current_price = observation[OBS_IDX['price']]
        self.price_history.append(current_price)
        
        if len(self.price_history) > 0:
            self.avg_price = np.mean(self.price_history)
            self.std_price = np.std(self.price_history) + 1e-6
        
        # Get discrete state
        discrete_state = self.get_discrete_state(observation)
        
        # Greedy action selection (no exploration during validation)
        action_idx = np.argmax(self.q_table[discrete_state])
        action = self.action_list[action_idx]
        
        return action


# Example usage / test
if __name__ == "__main__":
    # Test loading and acting
    run_folder = 'results/20260129_001924_Ep5_Gamma0.99_Zscore'  # Update this path
    
    if os.path.exists(run_folder):
        agent = TabularQAgentValidator(run_folder)
        
        # Test with dummy observation
        dummy_obs = np.array([50000, 75.5, 14, 3, 100, 4, 2024])
        action = agent.act(dummy_obs)
        print(f"Test action: {action}")
        
        # Load and display metrics
        metrics = load_metrics(run_folder)
        if metrics is not None:
            print(f"\nTraining summary:")
            print(f"  Episodes: {len(metrics)}")
            print(f"  Final train reward: {metrics['train_rollout'].iloc[-1]:.0f}")
            print(f"  Final validate reward: {metrics['validate_rollout'].iloc[-1]:.0f}")
    else:
        print(f"Run folder not found: {run_folder}")
        print("Please update the path to your actual results directory")
