import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import yfinance as yf
from scipy.spatial.distance import cdist

# --- 1. FETCH LIVE GOLD DATA AS PROXY ---
print("Fetching live Gold data from Yahoo Finance...")
try:
    # Fetch 1-minute data for Gold Futures (GC=F) for the last day
    gold_data = yf.download("GC=F", period="1d", interval="1m")
    # Calculate rolling localized price volatility to act as herding pressure K(t)
    prices = gold_data['Close'].values.flatten()
    returns = np.diff(np.log(prices))
    live_volatility = np.std(returns) if len(returns) > 0 else 0.02
    print(f"Live Gold Volatility proxy calculated: {live_volatility:.6f}")
except Exception as e:
    print(f"Failed to fetch live data ({e}). Falling back to baseline simulation numbers.")
    live_volatility = 0.015

# --- 2. CONFIGURATION & NEURAL GRAPH SETUP ---
N = 180
np.random.seed(42)

# Create a brain-like point cloud structure
phi = np.random.uniform(0, 2 * np.pi, N)
costheta = np.random.uniform(-1, 1, N)
u = np.random.uniform(0, 1, N)
theta_coord = np.arccos(costheta)
r = 4.0 * np.cbrt(u)

x = r * np.sin(theta_coord) * np.cos(phi)
y = r * np.sin(theta_coord) * np.sin(phi)
z = 5.0 * r * np.cos(theta_coord) * (1.0 - 0.3 * np.sin(phi)**2)
pos = np.vstack((x, y, z)).T

# Adjacency based on physical distance
dist_matrix = cdist(pos, pos)
threshold = 3.5
A = (dist_matrix < threshold).astype(float)
np.fill_diagonal(A, 0)

# Kuramoto parameters scaled by real-world market input
phases = np.random.uniform(-np.pi, np.pi, N)
omega = np.random.normal(0.5, 0.05, N)
dt = 0.05

fig = plt.figure(figsize=(10, 8), facecolor='black')
ax = fig.add_subplot(111, projection='3d', facecolor='black')
ax.set_axis_off()

scat = ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2], c='cyan', s=30, edgecolors='none', alpha=0.8)

def update(frame):
    global phases
    # Scaling factor K driven directly by the asset's real-world volatility profile
    K = 0.5 + (live_volatility * 1500) * (1.0 + np.sin(frame * 0.03))
    
    # Vectorized Kuramoto phase progression
    phase_diffs = phases[:, None] - phases[None, :]
    coupling = np.sum(A * np.sin(phase_diffs), axis=1)
    phases += dt * (omega - (K / N) * coupling)
    
    # Derive colors and sizes using phase mapping (simulating sync tracking)
    R = np.abs(np.mean(np.exp(1j * phases)))
    intensity = 0.5 + 0.5 * np.sin(phases)
    
    colors = plt.cm.plasma(intensity)
    # Brighten node parameters if the collective network reaches phase-lock thresholds
    if R > 0.65:
        colors[:, :3] += (R - 0.65) * 1.2
        colors = np.clip(colors, 0, 1)
        
    scat.set_color(colors)
    scat.set_sizes(15 + 40 * intensity * (1.0 + R))
    
    ax.view_init(elev=20 + 5 * np.sin(frame * 0.01), azim=frame * 0.4)
    return scat,

ani = animation.FuncAnimation(fig, update, frames=600, interval=30, blit=False)
plt.show()