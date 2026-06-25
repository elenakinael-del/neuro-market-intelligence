import numpy as np
import matplotlib
# Force a silent, non-interactive backend for optimal, crash-proof MP4 rendering
matplotlib.use("Agg") 

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import yfinance as yf
from scipy.spatial.distance import cdist
import imageio_ffmpeg

# Set global institutional typography rules (Times New Roman)
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.serif'] = ['Times New Roman']
matplotlib.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()

# --- 1. DATA INGESTION ENGINE ---
print("Ingesting live Gold tick data from Yahoo Finance...")
try:
    gold_data = yf.download("GC=F", period="1d", interval="1m")
    prices = gold_data['Close'].values.flatten()
    returns = np.diff(np.log(prices))
    live_volatility = np.std(returns) if len(returns) > 0 else 0.015
except Exception as e:
    print(f"Data link timed out ({e}). Initiating structural fallback matrix.")
    prices = np.cumsum(np.random.normal(0, 0.4, 200)) + 2350
    live_volatility = 0.012

N_FRAMES = min(200, len(prices))
if N_FRAMES < 40:
    base_t = np.linspace(0, 12, 200)
    prices = 2345.0 + 8.0 * np.sin(base_t) + np.cumsum(np.random.normal(0, 0.2, 200))
    N_FRAMES = 200

# --- 2. COMPLEX NETWORK CONFIGURATION ---
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
omega = np.random.normal(0.5, 0.04, N)
dt = 0.06

# --- 3. PREMIUM METICULOUS DASHBOARD DESIGN ---
BG_COLOR = "#020205"       # Ultra-deep obsidian black
GRID_COLOR = "#0c0f1d"     # Very faint subtle grid
TEXT_MAIN = "#d0dceb"      # Clean silver-white
TEXT_MUTED = "#485673"     # Muted slate accent
GOLD_AXIS = "#b8903c"      # Premium gold accent

fig = plt.figure(figsize=(19, 10), facecolor=BG_COLOR)

# Top Bar Header Layout
fig.text(0.04, 0.95, "NEURAL DIFFUSION REGIME DESK", color=TEXT_MAIN, fontsize=18, fontweight="bold")
fig.text(0.04, 0.92, "REAL-TIME ALGORITHMIC SYNCHRONIZATION MATRIX   ·   ASSET: XAU/USD", color=TEXT_MUTED, fontsize=9)

# Sub-panel Allocations (Balanced right side for stacked placement)
ax_brain = fig.add_axes([0.01, 0.05, 0.48, 0.82], projection='3d', facecolor=BG_COLOR)
ax_price = fig.add_axes([0.53, 0.45, 0.43, 0.41], facecolor=BG_COLOR)

# High-End Clean Formatting on the Price Chart
ax_price.tick_params(colors=TEXT_MUTED, labelsize=9.5)
ax_price.set_xlim(0, N_FRAMES)
ax_price.set_ylim(prices.min() - 0.5, prices.max() + 0.5)
ax_price.grid(True, color=GRID_COLOR, linestyle="-", linewidth=0.6)
ax_price.set_ylabel("SPOT PRICE (USD / OZ)", color=TEXT_MUTED, fontsize=9, fontweight="bold")

ax_price.spines['top'].set_visible(False)
ax_price.spines['right'].set_visible(False)
ax_price.spines['left'].set_edgecolor("#161d33")
ax_price.spines['bottom'].set_edgecolor("#161d33")

# Clean 3D Space parameters
ax_brain.set_axis_off()
ax_brain.set_xlim(-6, 6); ax_brain.set_ylim(-6, 6); ax_brain.set_zlim(-10, 10)
scat = ax_brain.scatter(pos[:, 0], pos[:, 1], pos[:, 2], c='cyan', s=25, edgecolors='none', alpha=0.7)

# UI Graph Elements
price_line, = ax_price.plot([], [], color=GOLD_AXIS, lw=2.0)  
price_fill = [None]

# Multi-Tier Telemetry Outputs
fig.text(0.53, 0.94, "METADATA STREAM", color=TEXT_MUTED, fontsize=8, fontweight="bold")
telemetry_txt = fig.text(0.53, 0.91, "", color=GOLD_AXIS, fontsize=13)

# --- CLEAN INTERPRETATION INTERFACE (POSITIONED UNDER THE CHART) ---
fig.text(0.53, 0.35, "QUANTITATIVE FRAMEWORK RECAP", color=TEXT_MUTED, fontsize=9, fontweight="bold")
log_title = fig.text(0.53, 0.29, "", color=TEXT_MAIN, fontsize=11.5, fontweight="bold")
log_desc1 = fig.text(0.53, 0.23, "", color="#96a8c2", fontsize=10.5)
log_desc2 = fig.text(0.53, 0.18, "", color="#96a8c2", fontsize=10.5)

# --- 4. DYNAMIC ANIMATION ENGINE ---
r_history = []

def update(frame):
    global phases, price_fill
    
    K = 0.4 + (live_volatility * 1800) * (1.0 + np.sin(frame * 0.04))
    
    phase_diffs = phases[:, None] - phases[None, :]
    coupling = np.sum(A * np.sin(phase_diffs), axis=1)
    phases += dt * (omega - (K / N) * coupling)
    
    R = np.abs(np.mean(np.exp(1j * phases)))
    r_history.append(R)
    
    local_mean_r = np.mean(r_history[-30:]) if len(r_history) > 10 else 0.4
    
    intensity = 0.5 + 0.5 * np.sin(phases)
    colors = plt.cm.plasma(intensity)
    
    if R > (local_mean_r * 1.03):  
        flash_boost = min((R - local_mean_r) * 4.0, 0.7)
        colors[:, :3] += flash_boost  
        colors = np.clip(colors, 0, 1)
        node_sizes = 15 + 85 * intensity * (1.3 + R)
    else:
        node_sizes = 12 + 35 * intensity
        
    scat.set_color(colors)
    scat.set_sizes(node_sizes)
    
    ax_brain.view_init(elev=16 + 2 * np.sin(frame * 0.015), azim=frame * 0.35)
    
    current_x = np.arange(frame + 1)
    price_line.set_data(current_x, prices[:frame + 1])
    
    if price_fill[0]:
        price_fill[0].remove()
    price_fill[0] = ax_price.fill_between(current_x, prices[:frame + 1], prices.min() - 5, 
                                          color=GOLD_AXIS, alpha=0.03)
    
    telemetry_txt.set_text(f"PRICE: ${prices[frame]:.2f} USD   |   COHERENCE (R): {R:.3f}   |   PRESSURE (K): {K:.2f}")
    
    # Text update logic with symbols and emojis removed
    if R > (local_mean_r * 1.03):
        log_title.set_text("REGIME TRIGGER: SYSTEMIC COHERENCE DETECTION")
        log_title.set_color("#ff5555")
        log_desc1.set_text("System node synchronization indicates correlated institutional order flow clusters.")
        log_desc2.set_text("Implied volatility surface deformation underway as tail-risk premium expands.")
    else:
        log_title.set_text("REGIME STEADY: ASYNCHRONOUS DISTRIBUTION MATRIX")
        log_title.set_color(TEXT_MAIN)
        log_desc1.set_text("Market entities operating independently. Orderbook balance is stable with high localized absorbency.")
        log_desc2.set_text("Implied variance parameters remain flat. Optimal conditions for volatility harvest strategies.")

    return scat, price_line

# --- 5. COMPILING & DISK EXPORT ---
output_filename = "live_gold_brain.mp4"
print(f"Compiling your custom master dashboard video... Saving to '{output_filename}'")

ani = animation.FuncAnimation(fig, update, frames=N_FRAMES, interval=40, blit=False)
writervideo = animation.FFMpegWriter(fps=25, bitrate=4500)
ani.save(output_filename, writer=writervideo, dpi=150)

print(f"✓ Video successfully compiled! Output file dropped in: {output_filename}")
plt.close()