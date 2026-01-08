import numpy as np
import pandas as pd
from collections import deque
from scipy.stats import norm

from TestEnv import HydroElectric_Test

class BaselineAgent:    
    def __init__(self, 
                 safe_volume_ratio_low = 0.15, 
                 safe_volume_ratio_high = 1, 
                 max_hours_history=24):

        self.max_hours_history = max_hours_history
        self.price_history = deque(maxlen=self.max_hours_history)
        self.max_volume = 100_000

        self.safe_volume_low = safe_volume_ratio_low * self.max_volume
        self.safe_volume_high = safe_volume_ratio_high * self.max_volume

    def get_observation(self, observation):  # Returns the current state
        # [volume, price, hour_of_day, day_of_week, day_of_year, month_of_year, year]
        self.volume = observation[0]
        self.current_price = observation[1]
        
        self.price_history.append(self.current_price)

        self.avg_price = sum(self.price_history)/len(self.price_history)
        self.std_price = np.std(self.price_history)+1e-10
        
        # How far is the current price above or below the average?
        # act MORE aggressively when prices are MORE extreme
        # Assume a normal distribution of prices in the current history
        # Thus if P(-a < Z < a) = P(Z < a) - P(Z < -a)
        # Symmetry of normal distribution
        # P(Z < -a) = P(Z > a) = 1-P(Z < a)
        # P(-a < Z < a) = 2*P(Z < a) - 1
        # If this probability is high, the current price is an extreme value

        self.Z_score = (self.current_price - self.avg_price)/self.std_price
        self.price_factor = 2 * norm.cdf(abs(self.Z_score)) - 1
        self.price_factor = max(0, self.price_factor)  

        # If the volume is low, remain a buffer to be able to sell more when price spikes
        # If the volume is high, pump less to safe space for when future prices are better
        if self.volume < self.safe_volume_low:
            # Low volume: scale from 0 to 1 as volume goes from 0 to safe_volume_low
            self.volume_factor = self.volume / self.safe_volume_low
            
        elif self.volume > self.safe_volume_high:
            # High volume: scale from 1 to 0 as volume goes from safe_volume_high to max_volume
            self.volume_factor = (self.max_volume - self.volume) / (self.max_volume - self.safe_volume_high)
            
        else:
            # Safe middle range: no restrictions
            self.volume_factor = 1

    def act(self, observation):
        '''
        Baseline logic

        Keep track of average price over the past 7 days.
        If current price is less than the average, pump.
        If current price is greater than the average, sell.

        Sell smaller amounts if volume is below safe percentage of max volume.
        Pump smaller amounts if volume is above safe percentage of max volume.
        This leaves space if price increases/decreases in next timesteps

        '''
        self.get_observation(observation)

        if self.current_price < self.avg_price:
            action = 1*self.price_factor            # pump  
            if self.volume > self.safe_volume_high:
                action = action*self.volume_factor   
        elif self.current_price > self.avg_price:   # sell
            action = -1*self.price_factor            
            if self.volume < self.safe_volume_low:
                action = action*self.volume_factor    
        else:
            action = 0  # hold
        
        return action

    
