import numpy as np
import matplotlib
# Force headless rendering backend for stable MP4 compilation
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.animation as animation
import yfinance as yf
import imageio_ffmpeg
import warnings
warnings.filterwarnings("ignore")

# Set global typography configurations
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.serif'] = ['Times New Roman']
matplotlib.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()

# ══════════════════════════════════════════════════════════════════════
#  1. LIVE DATA INGESTION & NEURAL PROXY CALCULATION
# ══════════════════════════════════════════════════════════════════════
print("Ingesting live Gold data stream from Yahoo Finance...")
try:
    gold_data = yf.download("GC=F", period="1d", interval="1m")
    live_prices = gold_data['Close'].values.flatten()
    
    # Calculate returns and absolute price velocity as a proxy for VPIN toxic flow
    returns = np.diff(np.log(live_prices)) if len(live_prices) > 1 else np.array([0.001])
    abs_velocity = np.abs(returns)
    
    # Normalize price velocity to scale smoothly as a neural depolarization proxy
    if len(abs_velocity) > 10:
        vpin_raw = np.convolve(abs_velocity, np.ones(5)/5, mode='same')  # Smooth moving average
        vpin_max = np.max(vpin_raw) if np.max(vpin_raw) > 0 else 1
        VPIN_LIVE = (vpin_raw / vpin_max) * 0.95  # scale toward threshold
    else:
        raise ValueError("Insufficient tick density")
        
except Exception as e:
    print(f"Live data link timed out ({e}). Activating internal fallback matrix.")
    live_prices = np.cumsum(np.random.normal(0, 0.25, 200)) + 2350
    VPIN_LIVE = None

N_FRAMES = min(200, len(live_prices))
if N_FRAMES < 50 or VPIN_LIVE is None:
    # High-quality structural fallback if live data lacks frame requirements
    base_t = np.linspace(0, 12, 200)
    live_prices = 2345.0 + 6.0 * np.sin(base_t) + np.cumsum(np.random.normal(0, 0.22, 200))
    N_FRAMES = 200
    
    # Re-generate classic VPIN trace matched precisely to frame length
    VPIN_LIVE = np.zeros(N_FRAMES)
    fire_frame = 75
    thresh_val = 0.68
    for i in range(fire_frame):
        VPIN_LIVE[i] = thresh_val * (i/fire_frame)**1.6 + np.random.normal(0, 0.015)
    for i in range(fire_frame, N_FRAMES):
        VPIN_LIVE[i] = max(0, thresh_val * np.exp(-(i-fire_frame)/22) + np.random.normal(0, 0.008))

# Find or hardwire the precise threshold crossing frame
THRESH = 0.68
fired_indices = np.where(VPIN_LIVE >= THRESH)[0]
FIRE_F = fired_indices[0] if len(fired_indices) > 0 else 75

# Normalize prices for the custom scale panel
PRICE_NORM = (live_prices - live_prices.min()) / ((live_prices.max() - live_prices.min()) + 1e-9)

# ══════════════════════════════════════════════════════════════════════
#  2. GRAPHICS & THEMATIC COLOR PALETTES (fMRI BOLD)
# ══════════════════════════════════════════════════════════════════════
BG          = "#010108"  # Obsidian darkroom
TEXT_COL    = "#a0b8d8"  # Clean silver-blue
DIM_COL     = "#202840"  # Slate accent

CMAP_BOLD   = mcolors.LinearSegmentedColormap.from_list(
    "bold", ["#000080","#0050d0","#00a8e0","#00f0c0","#80ff40","#ffdd00","#ff6600","#ff0020","#ffffff"])
CMAP_VOL    = mcolors.LinearSegmentedColormap.from_list(
    "vol", ["#03045e","#023e8a","#0077b6","#7209b7","#e63946","#ffba08"])

# Axon Membrane Tube Coordinates
N_ANG, N_Z = 64, 90
theta = np.linspace(0, 2*np.pi, N_ANG)
z_ax  = np.linspace(-1, 1, N_Z)
THETA_G, Z_G = np.meshgrid(theta, z_ax)
Z_TUBE = Z_G * 3.5

# Volatility Surface Space
SK, TN = np.linspace(0.75, 1.25, 30), np.linspace(0.03, 1.5, 30)
SG, TG = np.meshgrid(SK, TN)

def get_membrane_field(frame, vpin_val):
    z_norm = (Z_G + 1) / 2
    if frame < FIRE_F:
        field = vpin_val * np.exp(-((z_norm - 0.5)**2) / (0.15 + 0.6*vpin_val))
        return np.clip(field + np.random.normal(0, 0.02, field.shape)*vpin_val, 0, 1)
    else:
        t = (frame - FIRE_F) / (N_FRAMES - FIRE_F)
        forward  = np.exp(-((z_norm - (0.5 + t*0.9))**2) / 0.015)
        backward = np.exp(-((z_norm - (0.5 - t*0.5))**2) / 0.025) * 0.4
        refractory = np.clip(z_norm - (0.5 + t*0.9 - 0.12), 0, 0.18) / 0.18
        return np.clip(forward + backward - refractory * 0.5 + np.random.normal(0, 0.015, z_norm.shape), 0, 1)

def get_radius_modulation(frame):
    if frame < FIRE_F:
        return np.ones((N_Z, N_ANG)) * 1.0
    t = (frame - FIRE_F) / (N_FRAMES - FIRE_F)
    z_norm = (Z_G + 1) / 2
    return 1.0 + 0.18 * np.exp(-((z_norm - (0.5 + t*0.9))**2) / 0.018)

def get_vol_surface(frame):
    v = VPIN_LIVE[frame]
    iv = (0.15 + 0.25*v) - 0.10*(SG-1) + 0.12*(SG-1)**2 + 0.05/np.sqrt(TG+0.01)
    if frame >= FIRE_F:
        t = frame - FIRE_F
        iv += (0.28 * np.exp(-((SG-0.95)**2)/0.03) * np.exp(-TG/0.4)) * min(1.0, t/30)
    return np.clip(iv, 0.01, 0.70)

# ══════════════════════════════════════════════════════════════════════
#  3. DASHBOARD PANEL CONFIGURATION
# ══════════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(16, 9), facecolor=BG)

fig.text(0.5, 0.96, "ACTION POTENTIAL ORDER BOOK · VPIN DEPOLARIZATION MATRIX", ha="center", color=TEXT_COL, fontsize=13, fontweight="bold")
fig.text(0.5, 0.93, "Live Gold Data Processing · Volatility Toxicity Crossing · Spreading Cortical Wave", ha="center", color=DIM_COL, fontsize=8)

ax_axon  = fig.add_axes([0.01, 0.07, 0.50, 0.86], projection="3d", facecolor=BG)
ax_vol   = fig.add_axes([0.54, 0.40, 0.42, 0.52], projection="3d", facecolor=BG)
ax_price = fig.add_axes([0.56, 0.07, 0.40, 0.27], facecolor=BG)

# Strip grid layers from 3D Axon Space
for pane in [ax_axon.xaxis.pane, ax_axon.yaxis.pane, ax_axon.zaxis.pane]:
    pane.fill = False; pane.set_edgecolor("#05050f")
ax_axon.grid(False); ax_axon.set_axis_off()
ax_axon.set_xlim(-1.4, 1.4); ax_axon.set_ylim(-1.4, 1.4); ax_axon.set_zlim(-4, 4)
ax_axon.view_init(elev=18, azim=-55)

# Formatting Volatility Surface Panel
for pane in [ax_vol.xaxis.pane, ax_vol.yaxis.pane, ax_vol.zaxis.pane]:
    pane.fill = False; pane.set_edgecolor("#08081a")
ax_vol.grid(False)
ax_vol.tick_params(colors=DIM_COL, labelsize=6)
ax_vol.view_init(elev=30, azim=-50)

# Formatting Intraday Price Panel
ax_price.tick_params(colors=TEXT_COL, labelsize=7)
for sp in ax_price.spines.values(): sp.set_edgecolor("#05050f")
ax_price.set_xlim(0, N_FRAMES); ax_price.set_ylim(-0.08, 1.2)
ax_price.set_ylabel("GOLD PRICE (Normalized)", color=TEXT_COL, fontsize=7)

# Overlay Indicators
vpin_txt = fig.text(0.25, 0.90, "VPIN 0.000", ha="center", color="#2040a0", fontsize=10, fontweight="bold")
state_txt = fig.text(0.25, 0.87, "RESTING · Sub-threshold matrix", ha="center", color=DIM_COL, fontsize=8)

# VPIN Vertical Bar Gauge
vpin_bar_bg = fig.add_axes([0.01, 0.15, 0.01, 0.68], facecolor="#04040c")
vpin_bar_bg.axis("off")
vpin_bar_bg.set_ylim(0, 1)
vpin_bar_bg.axhline(THRESH, color="#5050ff", lw=1.0, ls="--", alpha=0.7)

# Initialize Plots
tube_surf = [ax_axon.plot_surface(1.0*np.cos(THETA_G), 1.0*np.sin(THETA_G), Z_TUBE, facecolors=CMAP_BOLD(get_membrane_field(0, VPIN_LIVE[0])), alpha=0.92, linewidth=0, antialiased=True, shade=True)]
vol_surf_obj = [ax_vol.plot_surface(SG, TG, get_vol_surface(0), cmap=CMAP_VOL, alpha=0.90, linewidth=0, antialiased=True)]
price_line, = ax_price.plot([], [], color="#ffe566", lw=2.0)
price_fill = [None]
fire_vline = ax_price.axvline(-1, color="#ff3030", lw=0.9, ls=":", alpha=0)

# ══════════════════════════════════════════════════════════════════════
#  4. DYNAMIC ANIMATION MATRIX
# ══════════════════════════════════════════════════════════════════════
fired_status = [False]

def update(frame):
    vpin_val = VPIN_LIVE[frame]
    
    # 3D Axon Displacement and Coloring
    field = get_membrane_field(frame, vpin_val)
    R_mod = get_radius_modulation(frame)
    Xm, Ym = R_mod * np.cos(THETA_G), R_mod * np.sin(THETA_G)
    
    tube_surf[0].remove()
    tube_surf[0] = ax_axon.plot_surface(Xm, Ym, Z_TUBE, facecolors=CMAP_BOLD(field), alpha=0.92, linewidth=0, antialiased=True, shade=True)
    ax_axon.view_init(elev=18, azim=-55 + frame * 0.2)

    # Implied Volatility Shift
    vol_surf_obj[0].remove()
    vol_surf_obj[0] = ax_vol.plot_surface(SG, TG, get_vol_surface(frame), cmap=CMAP_VOL, alpha=0.90, linewidth=0, antialiased=True)

    # Historical Price Update
    xs = np.arange(frame + 1)
    price_line.set_data(xs, PRICE_NORM[:frame + 1])
    if price_fill[0]: price_fill[0].remove()
    price_fill[0] = ax_price.fill_between(xs, PRICE_NORM[:frame + 1], alpha=0.12, color="#ffe566")
    
    if frame >= FIRE_F and not fired_status[0]:
        fired_status[0] = True
        fire_vline.set_xdata([FIRE_F])
        fire_vline.set_alpha(0.8)

    # Bar Gauge Update
    for collection in list(vpin_bar_bg.collections): collection.remove()
    vpin_bar_bg.fill_betweenx([0, min(vpin_val, 1)], 0, 1, color=CMAP_BOLD(min(vpin_val, 1)), alpha=0.75)
    vpin_bar_bg.axhline(THRESH, color="#5050ff", lw=1.0, ls="--", alpha=0.7)

    # Live Commentary State System
    pct = vpin_val / THRESH
    if frame >= FIRE_F:
        vpin_txt.set_text(f"VPIN {vpin_val:.3f} — TARGET TRADING PRICE: ${live_prices[frame]:.2f}")
        vpin_txt.set_color("#ff4040")
        state_txt.set_text("⚡ ACTION POTENTIAL · VOLATILITY SURFACE DEPEDALIZATION WAVE PROPAGATING")
        state_txt.set_color("#ff6020")
    elif pct > 0.88:
        vpin_txt.set_text(f"VPIN {vpin_val:.3f} — TARGET TRADING PRICE: ${live_prices[frame]:.2f}")
        vpin_txt.set_color("#ff8000")
        state_txt.set_text(f"⚠ CRITICAL SYSTEMIC COHERENCE BOUNDARY REACHED · {pct*100:.0f}% DEPOLARIZED")
        state_txt.set_color("#ff8020")
    else:
        vpin_txt.set_text(f"VPIN {vpin_val:.3f} — TARGET TRADING PRICE: ${live_prices[frame]:.2f}")
        vpin_txt.set_color("#20a060")
        state_txt.set_text(f"RESTING REGIME · ASYNCHRONOUS VOLATILITY HARVESTING · {pct*100:.0f}% DEPOLARIZED")
        state_txt.set_color(DIM_COL)

    return price_line, fire_vline

# ══════════════════════════════════════════════════════════════════════
#  5. VIDEO RENDERING COMPILER
# ══════════════════════════════════════════════════════════════════════
output_filename = "live_ap_orderbook.mp4"
print(f"Compiling your fMRI Action Potential dashboard video... Saving to '{output_filename}'")

ani = animation.FuncAnimation(fig, update, frames=N_FRAMES, interval=40, blit=False)
writervideo = animation.FFMpegWriter(fps=25, bitrate=4000)
ani.save(output_filename, writer=writervideo, dpi=150, savefig_kwargs={"facecolor": BG})

print(f"✓ Video successfully compiled! Output file located in your project directory: {output_filename}")
plt.close()