
#######################3
Steps

1. Set parameters, features, and config in tabular_Qagent
2. Train tabular agent running tabular_Qagent.py
3. Log results are stored in folder ../results/..
    Containing
    - config.json - config used in training
    - history_full.pkl - intermediate results of training in pkl format
    - model.npz - containing the Qtable, Ntable, action_list
    - training_log.csv - intermediate results of training in csv format
4. Use functionality in analyze_results.ipynb
    - read the log files
    - can be used for plotting learning curves and intermediate results
5. Use main.py for final rollout on validation data
    - Here make sure to import the tabular agent, using for example
        run_folder = 'results/20260129_001924_Ep5_Gamma0.99_Zscore'
        RL_agent = TabularQAgentValidator(run_folder)  # Update this path
