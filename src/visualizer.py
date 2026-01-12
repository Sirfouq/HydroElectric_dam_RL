import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np


class MetricsVisualizer:
    def __init__(self, history_df):
        """
        Parameters:
            history_df: DataFrame with columns [t, action, reward, cum_reward, 
                        price, volume, hour, day_of_week]
        """
        self.df = history_df
        self.max_volume = 100_000  # Should be passed in or read from config
        self.max_flow = 18_000

    def plot_cumulative_reward(self):
        """Plot cumulative reward over time"""
        plt.figure(figsize=(12, 6))
        plt.plot(self.df['t'], self.df['cum_reward'], linewidth=1.5)
        plt.ylabel('Cumulative Reward (€)', fontsize=12)
        plt.xlabel('Time (Hours)', fontsize=12)
        plt.title('Cumulative Reward Over Time', fontsize=14)

        plt.grid(True, alpha=0.3)
        plt.ylim(0, 1.2 * self.df['cum_reward'].max())
        plt.show()

    def plot_volume(self, start_day, n_days_detail):
        fig = plt.figure(figsize=(12, 6))
       
        volume_pct = (self.df['volume'] / self.max_volume) * 100
           
        start_hour = start_day*24
        end_hour = (start_day+n_days_detail)*24
        df_subset = self.df.iloc[start_hour:end_hour]

        plt.plot(df_subset['t'], (df_subset['volume']/self.max_volume)*100, 
                linewidth=1.5, color='steelblue')
        plt.axhline(15, color='red', linestyle='--', linewidth=2, alpha=0.7)
        plt.axhline(50, color='gray', linestyle=':', linewidth=1.5, alpha=0.5)
        plt.ylabel('Capacity (%)', fontsize=11)
        plt.xlabel('Time (Hours)', fontsize=11)
        plt.title(f'Volume Detail - First {n_days_detail} Days', fontsize=12, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 105)

        plt.show()

    # def plot_action_history(self, start_day, n_days_detail):
    #     fig = plt.figure(figsize=(12, 6))
       
    #     action_pct = (self.df['action'] / self.max_volume) * 100
           
    #     start_hour = start_day*24
    #     end_hour = (start_day+n_days_detail)*24
    #     df_subset = self.df.iloc[start_hour:end_hour]

    #     plt.plot(df_subset['t'], (df_subset['action']/self.max_flow)*100, 
    #             linewidth=1.5, color='steelblue')
    #     plt.axhline(15, color='red', linestyle='--', linewidth=2, alpha=0.7)
    #     plt.axhline(50, color='gray', linestyle=':', linewidth=1.5, alpha=0.5)
    #     plt.ylabel('Action flow (%)', fontsize=11)
    #     plt.xlabel('Time (Hours)', fontsize=11)
    #     plt.title(f'Action Detail - First {n_days_detail} Days', fontsize=12, fontweight='bold')
    #     plt.grid(True, alpha=0.3)
    #     plt.ylim(-105, 105)

    #     plt.show()

   
    