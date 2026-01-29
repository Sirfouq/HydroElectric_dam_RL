import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm, LinearSegmentedColormap

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

   
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_learning_metrics(history):
    """
    Visualizes the learning process: Reward, Convergence, and Value Growth.
    """
    plt.figure(figsize=(15, 5))
    
    # Plot 1: Rewards (Moving Average)
    plt.subplot(1, 3, 1)
    rewards_series = pd.Series(history['raw_rewards'])
    plt.plot(rewards_series, alpha=0.3, color='gray', label='Episode Reward')
    plt.plot(rewards_series.rolling(window=50).mean(), color='blue', linewidth=2, label='50-Ep Avg')
    plt.title("Learning Curve (Total Reward)")
    plt.xlabel("Episode")
    plt.ylabel("Reward (€)")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot 2: Q-Delta (Convergence)
    # If this line goes down and flattens, the agent has converged!
    plt.subplot(1, 3, 2)
    delta_series = pd.Series(history['q_delta'])
    plt.plot(delta_series.rolling(window=50).mean(), color='green')
    plt.title("Convergence (Magnitude of Q-Updates)")
    plt.xlabel("Episode")
    plt.ylabel("Sum of |ΔQ|")
    plt.yscale('log') # Log scale helps see fine convergence
    plt.grid(True, alpha=0.3)

    # Plot 3: Q-Mean (Value Estimation)
    plt.subplot(1, 3, 3)
    plt.plot(history['q_mean'], color='purple')
    plt.title("Average Value Estimate (Q-Mean)")
    plt.xlabel("Episode")
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def plot_policy_heatmap(agent):
    """
    Visualizes the policy: Price vs Volume -> Best Action.
    Only works effectively for 2D states (Price, Volume).
    """
    # Assuming State 0 is Price, State 1 is Volume
    # We take the argmax action for every combination
    
    # Extract dimensions
    n_prices = agent.Q_table.shape[0]
    n_volumes = agent.Q_table.shape[1]
    
    # Create Policy Grid
    policy_grid = np.argmax(agent.Q_table, axis=3)
    
    # Plot
    plt.figure(figsize=(8, 6))
    sns.heatmap(policy_grid.T, annot=True, fmt='d', cmap='coolwarm', 
                cbar_kws={'label': 'Action Index (0=Sell, 1=Hold, 2=Pump)'})
    
    plt.title("Learned Policy: Best Action by State")
    plt.xlabel("Price Bin (Low -> High)")
    plt.ylabel("Volume Bin (Low -> High)")
    plt.gca().invert_yaxis() # Put High Volume at top
    plt.show()


def plot_rollout_analysis(df):

    
    # --- CONFIGURATION ---
    cmap = LinearSegmentedColormap.from_list("bgr", ["#2166ac", "#bdbdbd", "#b2182b"], N=256)
    max_abs = float(np.abs(df["action_mwh"]).max())
    if max_abs == 0: max_abs = 1.0 # Prevent crash if no action taken
    norm = TwoSlopeNorm(vmin=-max_abs, vcenter=0.0, vmax=max_abs)
    
    # --- PLOT 1: Volatile vs Calm Windows ---
    # Find windows
    w = 14 * 24 # 14 days
    if len(df) > w:
        roll_std = df["price"].rolling(w).std().dropna()
        i_vol = int(roll_std.idxmax())
        i_cal = int(roll_std.idxmin())
        
        df_vol = df.iloc[max(0, i_vol - w + 1) : i_vol + 1]
        df_cal = df.iloc[max(0, i_cal - w + 1) : i_cal + 1]
        
        ymax = float(max(df_vol["price"].quantile(0.995), df_cal["price"].quantile(0.995)))
        
        fig, ax = plt.subplots(2, 2, figsize=(14, 7), sharex="col", 
                                gridspec_kw={"height_ratios": [3, 1]}, constrained_layout=True)
        
        sc = None # Placeholder for colorbar
        
        # Loop for Left (Volatile) and Right (Calm) columns
        scenarios = [
            (ax[0,0], ax[1,0], df_vol, f"Most Volatile 14 Days (Day {int(df_vol['t_day'].min())}-{int(df_vol['t_day'].max())})"),
            (ax[0,1], ax[1,1], df_cal, f"Most Calm 14 Days (Day {int(df_cal['t_day'].min())}-{int(df_cal['t_day'].max())})")
        ]
        
        for ax_p, ax_v, dwin, title in scenarios:
            # Top Row: Price & Actions
            ax_p.plot(dwin["t_day"], dwin["price"], linewidth=0.8, alpha=0.35, color="0.25", label='Price')
            sc = ax_p.scatter(dwin["t_day"], dwin["price"], c=dwin["action_mwh"], 
                                cmap=cmap, norm=norm, s=15, alpha=0.9, zorder=3)
            ax_p.set_title(title, fontweight='bold')
            ax_p.set_ylim(0, 200)
            ax_p.set_ylabel("Price (€/MWh)")
            ax_p.grid(True, alpha=0.2)
            
            # Bottom Row: Volume
            ax_v.plot(dwin["t_day"], dwin["vol_frac"], linewidth=1.5, color='black', label='Volume')
            ax_v.fill_between(dwin["t_day"], 0, dwin["vol_frac"], color='black', alpha=0.1)
            ax_v.set_ylim(0, 1.05)
            ax_v.set_ylabel("Fill %")
            ax_v.set_xlabel("Day")
            ax_v.grid(True, alpha=0.2)
            
        fig.colorbar(sc, ax=ax[:,:], shrink=0.6, label="Action (Red=Pump, Blue=Gen)")
        plt.show()
    else:
        print("Not enough data for 14-day window analysis.")

    # --- PLOT 2: Policy Map (Price Dev vs Volume) ---
    # This confirms if the agent learned "Buy Low, Sell High"
    
    # Data prep
    d = df.dropna(subset=["price_dev"]).copy()
    
    # Binning
    try:
        # Price Deviation Bins (Quantiles)
        price_bin, price_edges = pd.qcut(d["price_dev"], q=20, retbins=True, duplicates="drop")
        # Volume Bins (Uniform)
        vol_edges = np.linspace(0, 1, 10)
        vol_bin = pd.cut(d["vol_frac"], bins=vol_edges, include_lowest=True)
        
        # Pivot Table (The Brain Scan)
        policy = d.pivot_table(index=vol_bin, columns=price_bin, values="action_mwh", aggfunc="mean")
        
        # Plotting
        fig, ax = plt.subplots(figsize=(10, 6))
        im = ax.imshow(policy.values, aspect="auto", origin="lower", cmap=cmap, norm=norm)
        
        ax.set_title("Learned Policy Map: Action vs (Price Deviation & Volume)")
        ax.set_xlabel("Price Deviation from 24h Avg (€)")
        ax.set_ylabel("Reservoir Fill Level (0-1)")
        
        # Custom Ticks
        nx = policy.shape[1]
        ny = policy.shape[0]
        
        # X-Axis Ticks (Price Dev)
        x_indices = np.linspace(0, nx-1, 6).astype(int)
        x_labels = [f"{0.5*(price_edges[i]+price_edges[i+1]):.0f}" for i in x_indices]
        ax.set_xticks(x_indices)
        ax.set_xticklabels(x_labels)
        
        # Y-Axis Ticks (Volume)
        y_indices = np.linspace(0, ny-1, 6).astype(int)
        y_labels = [f"{0.5*(vol_edges[i]+vol_edges[i+1]):.1f}" for i in y_indices]
        ax.set_yticks(y_indices)
        ax.set_yticklabels(y_labels)
        
        plt.colorbar(im, label="Mean Action (Red=Pump, Blue=Sell)")
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        print(f"Could not generate policy map: {e}")