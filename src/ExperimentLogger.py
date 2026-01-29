import os
import json
import pandas as pd
import numpy as np
import pickle
from datetime import datetime

class ExperimentLogger:
    def __init__(self, base_dir='experiments', run_name=None):
        # Create a unique timestamped folder
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if run_name:
            self.run_dir = os.path.join(base_dir, f"{timestamp}_{run_name}")
        else:
            self.run_dir = os.path.join(base_dir, timestamp)
            
        os.makedirs(self.run_dir, exist_ok=True)
        print(f"📁 Logging results to: {self.run_dir}")

    def save_config(self, config_dict):
        """
        Saves hyperparameters with proper serialization of numpy types and functions.
        """
        # Convert config to JSON-serializable format
        serialized_config = self._make_json_serializable(config_dict)
        
        path = os.path.join(self.run_dir, 'config.json')
        with open(path, 'w') as f:
            json.dump(serialized_config, f, indent=4)
    
    def _make_json_serializable(self, obj):
        """
        Recursively convert objects to JSON-serializable format.
        Handles: numpy types, functions, nested dicts/lists, and state_dict.
        """
        # Handle None
        if obj is None:
            return None
        
        # Handle numpy types
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        
        # Handle functions (store their name)
        if callable(obj):
            return obj.__name__
        
        # Handle dictionaries (recurse on values)
        if isinstance(obj, dict):
            return {key: self._make_json_serializable(value) 
                    for key, value in obj.items()}
        
        # Handle lists/tuples (recurse on elements)
        if isinstance(obj, (list, tuple)):
            return [self._make_json_serializable(item) for item in obj]
        
        # Handle other types (strings, ints, floats, bools)
        if isinstance(obj, (str, int, float, bool)):
            return obj
        
        # Fallback: convert to string
        return str(obj)

    def save_agent(self, agent):
        """Saves the Q-table and N-table efficiently"""
        path = os.path.join(self.run_dir, 'model.npz')
        np.savez_compressed(
            path, 
            Q_table=agent.Q_table, 
            N_table=agent.N_table,
            action_list=agent.action_list
        )

    def save_history(self, history):
        """
        Saves metrics in both CSV (readable) and Pickle (complete) formats.
        """
        # 1. Save Complete Object (Pickle) - useful for nested structures
        with open(os.path.join(self.run_dir, 'history_full.pkl'), 'wb') as f:
            pickle.dump(history, f)

        # 2. Save Flattened CSV - check if n_table_stats exists and is not empty
        if 'n_table_stats' in history and len(history['n_table_stats']) > 0:
            # Transpose the list of tuples: rows → columns
            n_stats_columns = list(zip(*history['n_table_stats']))
            
            flat_data = {
                'episode': history['episode'],
                'epsilon': history['epsilon'],
                'lr': history['lr'],
                'raw_reward': history['raw_rewards'],
                'q_reward': history['q_rewards'],
                'train_rollout': history['train_rollout'],
                'val_rollout': history['validate_rollout'],
                'q_delta': history['q_delta'],
                'q_mean': history['q_mean'],
                
                # Unpack transposed columns
                'n_table_pct_non_zero': n_stats_columns[0],
                'n_table_min': n_stats_columns[1],
                'n_table_max': n_stats_columns[2],
                'n_table_mean': n_stats_columns[3],
                'n_table_median': n_stats_columns[4]
            }
        else:
            # If n_table_stats is missing or empty, save without those columns
            flat_data = {
                'episode': history['episode'],
                'epsilon': history['epsilon'],
                'lr': history['lr'],
                'raw_reward': history['raw_rewards'],
                'q_reward': history['q_rewards'],
                'train_rollout': history['train_rollout'],
                'val_rollout': history['validate_rollout'],
                'q_delta': history['q_delta'],
                'q_mean': history['q_mean']
            }
        
        # Create DataFrame and save
        df = pd.DataFrame(flat_data)
        df.to_csv(os.path.join(self.run_dir, 'training_log.csv'), index=False)
        
        print(f"✅ Saved training log: {len(df)} episodes")


# Example usage test
if __name__ == "__main__":
    # Test the logger with various data types
    from tabular_Qagent import get_weekday, get_day_period
    
    logger = ExperimentLogger(base_dir='test_results', run_name='logger_test')
    
    # Test config with various types
    test_config = {
        'n_episodes': 100,
        'gamma': 0.99,
        'epsilon_decay': np.int64(500),  # numpy type
        'learning_rate': np.float64(0.1),  # numpy type
        'action_list': np.array([-1.0, 0.0, 1.0]),  # numpy array
        'price_bins': np.array([20, 40, 60, 80]),  # numpy array
        'state_dict': [
            {
                'obs_idx': 0,
                'method': 'digitize',
                'bins': np.array([25000, 50000, 75000]),
                'size': 4,
                'name': 'volume'
            },
            {
                'obs_idx': 1,
                'method': 'function',
                'function': get_weekday,  # Actual function object
                'size': 2,
                'name': 'weekday'
            },
            {
                'obs_idx': 2,
                'method': 'function',
                'function': get_day_period,  # Another function
                'size': 4,
                'name': 'day_period'
            }
        ]
    }
    
    print("\nTesting config save with functions and numpy types...")
    logger.save_config(test_config)
    
    # Load it back and verify
    with open(os.path.join(logger.run_dir, 'config.json'), 'r') as f:
        loaded = json.load(f)
    
    print("\n✅ Saved config:")
    print(json.dumps(loaded, indent=2))
    
    # Verify function names were saved correctly
    assert loaded['state_dict'][1]['function'] == 'get_weekday'
    assert loaded['state_dict'][2]['function'] == 'get_day_period'
    print("\n✅ Functions serialized correctly as names!")
    
    # Verify numpy types were converted
    assert isinstance(loaded['epsilon_decay'], int)
    assert isinstance(loaded['learning_rate'], float)
    assert isinstance(loaded['action_list'], list)
    print("✅ Numpy types converted correctly!")