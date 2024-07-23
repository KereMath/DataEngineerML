import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def generate_planetary_motion_data(num_points=100):
    t = np.linspace(0, 2 * np.pi, num_points)
    x = np.cos(t)
    y = np.sin(t)
    data = {'time': t, 'x': x, 'y': y}
    return pd.DataFrame(data)

def save_data(data, filename):
    data.to_csv(filename, index=False)

def plot_data(data):
    plt.plot(data['x'], data['y'])
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.title('Planetary Motion Simulation')
    plt.show()

if __name__ == "__main__":
    real_data = generate_planetary_motion_data()
    save_data(real_data, 'data/real_data.csv')
    plot_data(real_data)
