import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import yfinance as yf
from scipy.spatial.distance import cdist
import imageio_ffmpeg

# --- EXTRA: AUTO-ROUTE VIA SELF-CONTAINED EMBEDDED MP4 WRITER ---
import matplotlib as mpl
mpl.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()

# --- 1. FETCH LIVE GOLD DATA AS PROXY ---
print("Fetching live Gold data from Yahoo Finance...")
try:
    gold_data = yf.download("GC=F", period="1d", interval="1m")
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

phi = np.random.uniform(0, 2 * np.pi, N)
costheta = np.random.uniform(-1, 1, N)
u = np.random.uniform(0, 1, N)
theta_coord = np.arccos(costheta)
r = 4.0 * np.cbrt(u)

x = r * np.sin(theta_coord) * np.cos(phi)
y = r * np.sin(theta_coord) * np.sin(phi)
z = 5.0 * r * np.cos(theta_coord) * (1.0 - 0.3 * np.sin(phi)**2)
pos = np.vstack((x, y, z)).T

dist_matrix = cdist(pos, pos)
threshold = 3.5
A = (dist_matrix < threshold).astype(float)
np.fill_diagonal(A, 0)

phases = np.random.uniform(-np.pi, np.pi, N)
omega = np.random.normal(0.5, 0.05, N)
dt = 0.05

fig = plt.figure(figsize=(10, 8), facecolor='black')
ax = fig.add_subplot(111, projection='3d', facecolor='black')
ax.set_axis_off()

scat = ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2], c='cyan', s=30, edgecolors='none', alpha=0.8)

def update(frame):
    global phases
    K = 0.5 + (live_volatility * 1500) * (1.0 + np.sin(frame * 0.03))
    
    phase_diffs = phases[:, None] - phases[None, :]
    coupling = np.sum(A * np.sin(phase_diffs), axis=1)
    phases += dt * (omega - (K / N) * coupling)
    
    R = np.abs(np.mean(np.exp(1j * phases)))
    intensity = 0.5 + 0.5 * np.sin(phases)
    
    colors = plt.cm.plasma(intensity)
    if R > 0.65:
        colors[:, :3] += (R - 0.65) * 1.2
        colors = np.clip(colors, 0, 1)
        
    scat.set_color(colors)
    scat.set_sizes(15 + 40 * intensity * (1.0 + R))
    
    ax.view_init(elev=20 + 5 * np.sin(frame * 0.01), azim=frame * 0.4)
    return scat,

# Frame total capped for an efficient compile loop
ani = animation.FuncAnimation(fig, update, frames=300, interval=30, blit=False)

# --- 3. EXECUTE EXPLICIT VIDEO EXPORT ON THE LOCAL DISK ---
output_filename = "live_gold_brain.mp4"
print(f"Compiling your MP4 video file... Writing directly to {output_filename}")

writervideo = animation.FFMpegWriter(fps=30)
ani.save(output_filename, writer=writervideo, dpi=150, facecolor="black")

print(f"✓ Video successfully compiled! Check your current folder for '{output_filename}'")
plt.close()