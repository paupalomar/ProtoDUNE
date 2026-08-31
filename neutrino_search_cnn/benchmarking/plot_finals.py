import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path

# ============================================================================================
#                                      \\ Initial Configuration //
# ============================================================================================

script_name = "model_21"  
model_output_dir = Path(f'/afs/cern.ch/work/p/ppalomar/neutrino_search_cnn/models_storage/{script_name}')
plots_dir = model_output_dir / 'final_plots'
plots_dir.mkdir(parents=True, exist_ok=True)

json_path = model_output_dir / f'{script_name}_results.json'
test_npz_path = plots_dir / f'{script_name}_test_inference.npz'
spillON_npz_path = plots_dir / f'{script_name}_spillON_inference.npz'

# ============================================================================================
#                                      \\ Data Loading //
# ============================================================================================

with open(json_path, 'r') as f:
    results_data = json.load(f)

n_exp_nu = results_data["scaling_factors"]["expected_pure_neutrinos"]
n_exp_bkg = results_data["scaling_factors"]["expected_pure_bkg"]
working_points = results_data["threshold_methods"]

test_data = np.load(test_npz_path)
scores_test = test_data['scores_test']
labels_test = test_data['labels_test']
energy_test = test_data['energy_test']
vertexX_test = test_data['vertexX_test']  # <-- Nova variable carregada
vertexZ_test = test_data['vertexZ_test']  # <-- Nova variable carregada

on_data = np.load(spillON_npz_path)
scores_on = on_data['scores_spill_on']
energy_on = on_data['energy_spill_on']

mask_mc = (labels_test == 1)
mask_off = (labels_test == 0)

scores_mc = scores_test[mask_mc]
scores_off = scores_test[mask_off]
energy_mc = energy_test[mask_mc]
energy_off = energy_test[mask_off]

weight_mc = np.full(len(scores_mc), n_exp_nu / len(scores_mc)) if len(scores_mc) > 0 else []
weight_off = np.full(len(scores_off), n_exp_bkg / len(scores_off)) if len(scores_off) > 0 else []

# ============================================================================================
#                                      \\ Plotting Parameters //
# ============================================================================================

plt.rcParams.update({
    'font.size': 14,
    'axes.labelsize': 16,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 12,
    'font.family': 'serif',  
    'figure.dpi': 300,       
    'savefig.bbox': 'tight'  
})

colors = {'mc': '#e41a1c', 'off': '#377eb8', 'on': 'black'}

# ============================================================================================
#                             PLOT 1: Score Distribution (Data vs MC)
# ============================================================================================
print("Generating Plot 1: Score Distribution...")

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
bins_score = np.linspace(0, 1.0, 51)
bin_width = bins_score[1] - bins_score[0]

counts_on, bins_on = np.histogram(scores_on, bins=bins_score)
bin_centers = (bins_on[:-1] + bins_on[1:]) / 2

counts_off, _, _ = ax1.hist(scores_off, bins=bins_score, weights=weight_off, 
         histtype='step', color=colors['off'], label='Test BKG (Scaled Spill OFF)', 
         linewidth=2)

counts_mc, _, _ = ax1.hist(scores_mc, bins=bins_score, weights=weight_mc, 
         histtype='step', color=colors['mc'], label='Test Signal (Scaled MC)', 
         linewidth=2)

counts_stack = counts_mc + counts_off

ax1.hist(bin_centers, bins=bins_score, weights=counts_stack, 
         histtype='step', color='black', linestyle='--', linewidth=1.5, 
         label='Total Predicted (MC + OFF)')

ax1.errorbar(bin_centers, counts_on, yerr=np.sqrt(counts_on), fmt='ko', 
             label='Data (Spill ON)', markersize=4, capsize=2)

line_styles = ['--', '-.', ':']
for i, (wp_name, wp_info) in enumerate(working_points.items()):
    th = wp_info['threshold']
    #ax1.axvline(th, color='gray', linestyle=line_styles[i%3], linewidth=1.5, label=f'{wp_name} ({th:.2f})')

ax1.set_ylabel('Events')
ax1.set_yscale('log')
ax1.set_ylim(bottom=0.5)
ax1.grid(True, linestyle=':', alpha=0.5)
#ax1.legend(loc='upper center', bbox_to_anchor=(0.5, 1.3), ncol=2)
ax1.legend(loc='best')

ratio = np.divide(counts_on, counts_stack, out=np.zeros_like(counts_on, dtype=float), where=counts_stack!=0)
ratio_err = np.divide(np.sqrt(counts_on), counts_stack, out=np.zeros_like(counts_on, dtype=float), where=counts_stack!=0)

ax2.errorbar(bin_centers, ratio, yerr=ratio_err, fmt='ko', markersize=4)
ax2.axhline(1.0, color='gray', linestyle='--')
ax2.set_xlabel('CNN Score')
ax2.set_ylabel('Data / Expected')
ax2.set_ylim(0, 2)

plt.tight_layout()
plt.savefig(plots_dir / 'score_distribution.png')
plt.close()

# ============================================================================================
#                             PLOT 1B: Normalized Score Distribution (Area = 1)
# ============================================================================================
print("Generating Plot 1B: Normalized Score Distribution...")

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), gridspec_kw={'height_ratios': [3, 1]}, sharex=True)

# Calculate normalization factors (Integration sum = Area)
norm_off = np.sum(counts_off) * bin_width if np.sum(counts_off) > 0 else 1
norm_mc = np.sum(counts_mc) * bin_width if np.sum(counts_mc) > 0 else 1
norm_stack = np.sum(counts_stack) * bin_width if np.sum(counts_stack) > 0 else 1
norm_on = np.sum(counts_on) * bin_width if np.sum(counts_on) > 0 else 1

ax1.hist(bins_score[:-1], bins=bins_score, weights=counts_stack / norm_stack, 
         histtype='step', color='black', linestyle='--', label='Total Predicted (Shape)', linewidth=1.5)

ax1.errorbar(bin_centers, counts_on / norm_on, yerr=np.sqrt(counts_on) / norm_on, fmt='ko', 
             label='Data (Shape)', markersize=4, capsize=2)

ax1.set_ylabel('Probability Density (Area = 1)')
ax1.set_yscale('log')
ax1.grid(True, linestyle=':', alpha=0.5)
ax1.legend(loc='best')

# Hide X-axis ticks of the top panel
plt.setp(ax1.get_xticklabels(), visible=False)

# --- Secondary Panel (ax2): Shape Ratio ---
# Calculate the PDFs
pdf_on = counts_on / norm_on
pdf_stack = counts_stack / norm_stack

# Calculate the shape ratio
ratio_shape = np.divide(pdf_on, pdf_stack, out=np.zeros_like(pdf_on, dtype=float), where=pdf_stack!=0)

# Error propagation for the normalized ratio
ratio_shape_err = np.divide(np.sqrt(counts_on) / norm_on, pdf_stack, out=np.zeros_like(pdf_on, dtype=float), where=pdf_stack!=0)

ax2.errorbar(bin_centers, ratio_shape, yerr=ratio_shape_err, fmt='ko', markersize=4)
ax2.axhline(1.0, color='gray', linestyle='--')
ax2.set_xlabel('CNN Score')
ax2.set_ylabel('Data / Expectation')
ax2.set_ylim(0, 2) # Tighter bounds for shape comparison
ax2.grid(True, linestyle=':', alpha=0.5)

plt.tight_layout()
plt.savefig(plots_dir / 'score_distribution_normalized.png')
plt.close()

# ============================================================================================
#                             PLOT 2: Reconstructed Energy (Data vs Prediction + Ratio)
# ============================================================================================
print("Generating Plot 2: Energy Distribution for all thresholds...")

import matplotlib.gridspec as gridspec

num_wps = len(working_points)
fig = plt.figure(figsize=(6 * num_wps, 8))
gs = gridspec.GridSpec(2, num_wps, height_ratios=[3, 1], hspace=0.1)

bins_energy = np.linspace(0, 115, 40)
bin_centers_en = (bins_energy[:-1] + bins_energy[1:]) / 2

for i, (wp_name, wp_info) in enumerate(working_points.items()):
    ax1 = fig.add_subplot(gs[0, i])
    ax2 = fig.add_subplot(gs[1, i], sharex=ax1)
    
    threshold = wp_info["threshold"]

    sel_energy_mc = energy_mc[scores_mc > threshold]
    sel_energy_off = energy_off[scores_off > threshold]
    sel_energy_on = energy_on[scores_on > threshold]

    w_sel_mc = np.full(len(sel_energy_mc), n_exp_nu / len(scores_mc)) if len(scores_mc) > 0 else []
    w_sel_off = np.full(len(sel_energy_off), n_exp_bkg / len(scores_off)) if len(scores_off) > 0 else []

    counts_on_en, _ = np.histogram(sel_energy_on, bins=bins_energy)
    counts_mc_en, _ = np.histogram(sel_energy_mc, bins=bins_energy, weights=w_sel_mc)
    counts_off_en, _ = np.histogram(sel_energy_off, bins=bins_energy, weights=w_sel_off)
    counts_stack_en = counts_mc_en + counts_off_en 

    ax1.hist(bins_energy[:-1], bins=bins_energy, weights=counts_off_en,
             histtype='step', color=colors['off'], linewidth=1.5,
             label='Rescaled BKG in Test', alpha = 0.5)

    ax1.hist(bins_energy[:-1], bins=bins_energy, weights=counts_mc_en,
             histtype='step', color=colors['mc'], linewidth=1.5,
             label='Rescaled Signal (MC) in Test', alpha = 0.5)

    ax1.hist(bins_energy[:-1], bins=bins_energy, weights=counts_stack_en,
             histtype='step', color='purple', linestyle='--', linewidth=1.5,
             label='Rescaled Total in Test')
    
    ax1.errorbar(bin_centers_en, counts_on_en, yerr=np.sqrt(counts_on_en), fmt='ko', 
                label='Data (Spill ON)', markersize=5)

    clean_wp_name = wp_name.replace("TH_", "").replace("_", " ")
    ax1.set_title(f'Selected events with {clean_wp_name}\n(th: {threshold:.3f})', pad=10)
    ax1.set_ylabel('Events')
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(True, linestyle=':', alpha=0.5)
    
    plt.setp(ax1.get_xticklabels(), visible=False)

    ratio_en = np.divide(counts_on_en, counts_stack_en, out=np.zeros_like(counts_on_en, dtype=float), where=counts_stack_en!=0)
    ratio_err_en = np.divide(np.sqrt(counts_on_en), counts_stack_en, out=np.zeros_like(counts_on_en, dtype=float), where=counts_stack_en!=0)

    ax2.errorbar(bin_centers_en, ratio_en, yerr=ratio_err_en, fmt='ko', markersize=4)
    ax2.axhline(1.0, color='gray', linestyle='--')
    ax2.set_xlabel('Reconstructed Energy [GeV]')
    
    if i == 0:
        ax2.set_ylabel('Data / Expected')
    
    ax2.set_ylim(0, 2)
    ax2.grid(True, linestyle=':', alpha=0.5)

plt.tight_layout()
plt.savefig(plots_dir / 'energy_spectrum_all_thresholds.png')
plt.close()

# ============================================================================================
#                             PLOT 3: Extracted Signal Rate vs Threshold (With Zoom Inset)
# ============================================================================================
print("Generating Plot 3: Signal Extraction Ratio with Inset...")

thresholds = np.linspace(0.1, 0.99, 50)

extracted_signals_on = []
extracted_signals_on_err = []

extracted_signals_test = []
extracted_signals_test_err = []

alpha = n_exp_bkg / len(scores_off) if len(scores_off) > 0 else 0 
beta = n_exp_nu / len(scores_mc) if len(scores_mc) > 0 else 0

for th in thresholds:
    n_on_sel = np.sum(scores_on > th)
    n_off_sel = np.sum(scores_off > th)
    n_mc_sel = np.sum(scores_mc > th)
    
    s_hat_on = n_on_sel - (alpha * n_off_sel)
    err_on = np.sqrt(n_on_sel + (alpha**2 * n_off_sel))
    
    extracted_signals_on.append(s_hat_on)
    extracted_signals_on_err.append(err_on)

    pseudo_on_sel = (n_mc_sel * beta) + (n_off_sel * alpha)
    s_hat_test = pseudo_on_sel - (alpha * n_off_sel)
    err_test = np.sqrt(pseudo_on_sel + (alpha**2 * n_off_sel))
    
    extracted_signals_test.append(s_hat_test)
    extracted_signals_test_err.append(err_test)

fig, ax = plt.subplots(figsize=(8, 6))

ax.errorbar(thresholds, extracted_signals_on, yerr=extracted_signals_on_err, fmt='o-', color='black', 
             markersize=4, capsize=3, label=r'Data ($\hat{S}_{ON}$)')

ax.errorbar(thresholds, extracted_signals_test, yerr=extracted_signals_test_err, fmt='s--', color='#e41a1c', 
             markersize=4, capsize=3, label=r'Test ($\hat{S}_{MC}$)', alpha=0.8)

ax.axhline(n_exp_nu, color='gray', linestyle=':', label=f'POT Expectation ({n_exp_nu:.1f})')

ax.set_xlabel('CNN Score Threshold')
ax.set_ylabel(r'Extracted Neutrino Signal ($\hat{S}$)')
ax.set_title('Background Subtraction Stability')
ax.grid(True, linestyle=':', alpha=0.7)
ax.legend(loc='upper right')

# --- INSET ZOOM CONFIGURATION ---
# Positioned at bottom-left [x0, y0, width, height] as fractions of the parent axes
axins = ax.inset_axes([0.70, 0.30, 0.4, 0.4])

axins.errorbar(thresholds, extracted_signals_on, yerr=extracted_signals_on_err, fmt='o-', color='black', markersize=3, capsize=2)
axins.errorbar(thresholds, extracted_signals_test, yerr=extracted_signals_test_err, fmt='s--', color='#e41a1c', markersize=3, capsize=2, alpha=0.8)
axins.axhline(n_exp_nu, color='gray', linestyle=':')

# Set limits for the high-purity zoom region
axins.set_xlim(0.8, 0.99)
valid_idx = np.where(thresholds >= 0.8)[0]
max_zoom_y = max(np.max(np.array(extracted_signals_on)[valid_idx]), np.max(np.array(extracted_signals_test)[valid_idx]))

# Dynamically set Y limit for the inset to allow clear visualization of the convergence
axins.set_ylim(0, max_zoom_y * 1.5)
axins.grid(True, linestyle=':', alpha=0.5)

plt.tight_layout()
plt.savefig(plots_dir / 'extracted_signal_vs_threshold.png')
plt.close()

# ============================================================================================
#                             PLOT 4: True Positives Vertex Z & X Distribution
# ============================================================================================
print("Generating Plot 3: True positives vertex in TEST...")

# Obtenim la distribució total de vèrtexs per al senyal (MC)
mc_mask = (labels_test == 1)
vertexZ_mc_total = vertexZ_test[mc_mask]
vertexX_mc_total = vertexX_test[mc_mask]

# Determinem els límits geomètrics per als bins
z_min, z_max = -5, 465
bins_z = np.linspace(z_min, z_max, 50)

# Límits per a X (Ajusta'ls a l'amplada de deriva física del teu detector)
x_min, x_max = -365, 365 
bins_x = np.linspace(x_min, x_max, 50)

# Creem una figura amb 2 files i tants subplots com Working Points
num_wps = len(working_points)
fig, axes = plt.subplots(2, num_wps, figsize=(6 * num_wps, 10))

# Ens assegurem que 'axes' sigui una matriu 2D bidimensional, fins i tot si num_wps == 1
if num_wps == 1:
    axes = np.array(axes).reshape(2, 1)

for i, (wp_name, wp_info) in enumerate(working_points.items()):
    
    threshold = wp_info["threshold"] 
    
    # Màscara d'esdeveniments True Positives (És MC i supera el tall de la CNN)
    tp_mask = (labels_test == 1) & (scores_test > threshold)
    vertexZ_tp = vertexZ_test[tp_mask]
    vertexX_tp = vertexX_test[tp_mask]
    
    clean_wp_name = wp_name.replace("TH_", "").replace("_", " ")
    
    # --- FILA 1: DISTRIBUCIÓ Z ---
    ax_z = axes[0, i]
    ax_z.hist(vertexZ_mc_total, bins=bins_z, color='gray', histtype='step', 
            linestyle='--', linewidth=1.5, label=f'Total MC (N={np.sum(mc_mask)})')
            
    ax_z.hist(vertexZ_tp, bins=bins_z, color='#e41a1c', histtype='step', 
            linewidth=2, label=f'True Positives (N={np.sum(tp_mask)})')
    
    ax_z.set_title(f'{clean_wp_name} (th: {threshold:.3f})\nSpatial Distribution (Z)', pad=10)
    ax_z.set_xlabel('True Vertex Z [cm]')
    if i == 0:
        ax_z.set_ylabel('Events')
    ax_z.grid(True, linestyle=':', alpha=0.6)
    ax_z.legend(loc='best', frameon=True, edgecolor='black', fontsize=11)
    
    # --- FILA 2: DISTRIBUCIÓ X ---
    ax_x = axes[1, i]
    ax_x.hist(vertexX_mc_total, bins=bins_x, color='gray', histtype='step', 
            linestyle='--', linewidth=1.5, label=f'Total MC (N={np.sum(mc_mask)})')
            
    ax_x.hist(vertexX_tp, bins=bins_x, color='#377eb8', histtype='step', 
            linewidth=2, label=f'True Positives (N={np.sum(tp_mask)})')
    
    ax_x.set_title(f'Spatial Distribution (X)', pad=10)
    ax_x.set_xlabel('True Vertex X [cm]')
    if i == 0:
        ax_x.set_ylabel('Events')
    ax_x.grid(True, linestyle=':', alpha=0.6)
    ax_x.legend(loc='best', frameon=True, edgecolor='black', fontsize=11)

plt.tight_layout()
plt.savefig(plots_dir / 'true_positives_vertex_distribution.png')
plt.close()

print(f"---> Vertex plot saved to: {plots_dir / 'true_positives_vertex_distribution.png'}")

print("All plots generated at:", plots_dir)