import pandas as pd
import numpy as np
from scipy.interpolate import PchipInterpolator

def fill_missing_values(data):
    data = data.copy()
    for column in data.columns:
        max_value = data[column].max()
        min_value = data[column].min()
        
        n = data[column].isna().sum()
        if n > 0:
            x = np.arange(len(data))
            y = data[column].values
            mask = ~np.isnan(y)
            
            interpolator = PchipInterpolator(x[mask], y[mask], extrapolate=False)
            
            y_interp = interpolator(x)
            y_interp = np.clip(y_interp, min_value, max_value)  
            
            data[column] = y_interp
        
        if data[column].isna().sum() > 0:
            data[column] = data[column].interpolate(method='linear', limit_direction='both')
        
        if data[column].isna().sum() > 0:
            for i in range(len(data)):
                if pd.isna(data[column].iloc[i]):
                    if i > 0 and i < len(data) - 1 and not pd.isna(data[column].iloc[i - 1]) and not pd.isna(data[column].iloc[i + 1]):
                        interpolated_value = (data[column].iloc[i - 1] + data[column].iloc[i + 1]) / 2
                        data[column].iloc[i] = min(max(interpolated_value, min_value), max_value)
    
    return data

if __name__ == "__main__":
    fake_processed_data = pd.read_csv('data/fake_processed_data.csv')
    preprocessed_data = fill_missing_values(fake_processed_data)
    preprocessed_data.to_csv('data/preprocessed_data.csv', index=False)
