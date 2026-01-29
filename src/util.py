# import numpy as np
# from collections import deque

# class FeatureEngineering():
#     def __init__(self):
#         # MEMORY: 168 hours (1 Week) for the Long-Term Z-Score
#         self.price_history = deque(maxlen=168) 
        
#         # Volume bounds (Fixed)
#         self.vol_min = 0.0
#         self.vol_max = 100000.0

#     def process(self, state):
#         """
#         Input: Raw State [volume, price, hour, day_week, day_year, month, year]
#         Output: Engineered State [9 features]  <-- UPDATED COUNT
#         """
#         # Unpack raw state
#         vol, price, hour, day_week, day_year, month, year = state
        
#         # --- 1. Volume (Normalized 0-1) ---
#         norm_vol = (vol - self.vol_min) / (self.vol_max - self.vol_min)
        
#         # --- 2. Update Price History ---
#         self.price_history.append(price)
#         history_array = np.array(self.price_history)
        
#         # --- 3. Feature A: Long-Term Z-Score (168h / 1 Week) ---
#         if len(history_array) < 168:
#             z_168 = 0.0
#         else:
#             mean_168 = np.mean(history_array)
#             std_168 = np.std(history_array) + 1e-5
#             z_168 = np.clip((price - mean_168) / std_168, -4.0, 4.0) / 4.0

#         # --- 4. Feature B: Short-Term Z-Score (24h / 1 Day) ---
#         if len(history_array) < 24:
#             z_24 = 0.0
#         else:
#             short_history = history_array[-24:] 
#             mean_24 = np.mean(short_history)
#             std_24 = np.std(short_history) + 1e-5
#             z_24 = np.clip((price - mean_24) / std_24, -4.0, 4.0) / 4.0

#         # --- REMOVED FEATURE C (Peak/Trend) --- 

#         # --- 5. Cyclical Time (Sin/Cos) ---
#         # 24 Hour Cycle
#         sin_hr = np.sin(2 * np.pi * hour / 24.0)
#         cos_hr = np.cos(2 * np.pi * hour / 24.0)
        
#         # 7 Day Cycle
#         sin_dw = np.sin(2 * np.pi * day_week / 7.0)
#         cos_dw = np.cos(2 * np.pi * day_week / 7.0)
        
#         # 365 Day Cycle
#         sin_dy = np.sin(2 * np.pi * day_year / 366.0)
#         cos_dy = np.cos(2 * np.pi * day_year / 366.0)
        
#         # 9 final features (Removed peak_ratio)
#         return np.array([
#             norm_vol,    # 1. Volume
#             z_168,       # 2. Long Term Value
#             z_24,        # 3. Short Term Value
#             sin_hr, cos_hr, # 4-5. Hour
#             sin_dw, cos_dw, # 6-7. Day
#             sin_dy, cos_dy  # 8-9. Season
#         ], dtype=np.float32)

#     def reset(self):
#         self.price_history.clear()

#---BEST MODEL'S FEAT.ENGINEERING SO FAR SAVED FOR SAFETY----- 

import numpy as np
from collections import deque

# class Normalizer:
#     def __init__(self, mins, maxs):
#         self.mins = mins
#         self.maxs = maxs
        
#     def normalize(self, state):
#         # Clip values within bounds
#         state = np.clip(state, self.mins, self.maxs)
#         # 0-1 range scaling 
#         return (state - self.mins) / (self.maxs - self.mins)
    
class FeatureEngineering():
    def __init__(self):
        # MEMORY: 168 hours (1 Week) for the Long-Term Z-Score
        self.price_history = deque(maxlen=168) 
        self.last_price = 0
        
        # Volume bounds (Fixed)
        self.vol_min = 0.0
        self.vol_max = 100000.0

    def process(self, state):
        """
        Input: Raw State [volume, price, hour, day_week, day_year, month, year]
        Output: Engineered State [10 features]
        """
        # Unpack raw state
        vol, price, hour, day_week, day_year, month, year = state
        
        # Volume (Normalized 0-1) 
        norm_vol = (vol - self.vol_min) / (self.vol_max - self.vol_min)
        
        #  Update Price History 
        self.price_history.append(price)
        history_array = np.array(self.price_history)
        
        #  Feature A: Long-Term Z-Score (168h / 1 Week) 
        if len(history_array) < 168:
            z_168 = 0.0
        else:
            mean_168 = np.mean(history_array)
            std_168 = np.std(history_array) + 1e-5
            # Clip huge spikes (e.g. 100-sigma) to 4.0 so gradients don't explode
            z_168 = np.clip((price - mean_168) / std_168, -4.0, 4.0) / 4.0

        # Short-Term Z-Score (24h / 1 Day) 
        
        if len(history_array) < 24:
            z_24 = 0.0
        else:
            short_history = history_array[-24:] 
            mean_24 = np.mean(short_history)
            std_24 = np.std(short_history) + 1e-5
            z_24 = np.clip((price - mean_24) / std_24, -4.0, 4.0) / 4.0

        # Feature C: Price Trend 
        price_trend = np.sign(price - self.last_price)
        self.last_price = price

        
        # 24 Hour Cycle
        sin_hr = np.sin(2 * np.pi * hour / 24.0)
        cos_hr = np.cos(2 * np.pi * hour / 24.0)
        
        #day of week
        sin_dw = np.sin(2 * np.pi * day_week / 7.0)
        cos_dw = np.cos(2 * np.pi * day_week / 7.0)
        
        #day of year
        sin_dy = np.sin(2 * np.pi * day_year / 366.0)
        cos_dy = np.cos(2 * np.pi * day_year / 366.0)
        

        return np.array([
            norm_vol,   
            z_168,      
            z_24,       
            price_trend,
            sin_hr, cos_hr, 
            sin_dw, cos_dw, 
            sin_dy, cos_dy  
        ], dtype=np.float32)

    def reset(self):
        self.price_history.clear()
        self.last_price = 0