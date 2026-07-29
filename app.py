# Standard library and third-party imports needed for numerics, plotting, the
# web UI, and the Claude API.
import os
import json
import html
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.transforms as mtransforms
import streamlit as st
from streamlit_image_coordinates import streamlit_image_coordinates
import anthropic
from scipy.ndimage import gaussian_filter
from io import BytesIO
from PIL import Image

# ---------------------------------------------------------------------------
# Page config — must be first Streamlit call
# ---------------------------------------------------------------------------
# Sets the browser tab title and switches the layout to full-width. The CSS
# below tightens up spacing so more content fits on screen without scrolling.
st.set_page_config(page_title="MRI Simulator", layout="wide")

st.markdown("""
<style>
    .block-container { padding-top: 1.50rem; padding-bottom: 0rem; }
    h1 { margin-bottom: 0.1rem; font-size: 2.0rem; }
    [data-testid="stSidebar"] { padding-top: 0.2rem; }
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 { font-size: 1.2rem; margin-bottom: 0; }
    div[data-testid="stVerticalBlock"] > div { padding-bottom: 0rem; }
    [data-testid="stMetricValue"] { font-size: 1.5rem; }
    [data-testid="stMetricLabel"] p { font-size: 1.25rem; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Google Analytics 4
# ---------------------------------------------------------------------------
GA4_ID = "G-BH1W66432Q"
st.markdown(f"""
<!-- Google Analytics 4 -->
<script async src="https://www.googletagmanager.com/gtag/js?id={GA4_ID}"></script>
<script>
  window.dataLayer = window.dataLayer || [];
  function gtag(){{dataLayer.push(arguments);}}
  gtag('js', new Date());
  gtag('config', '{GA4_ID}');
</script>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Tissue parameters at 3T (ADC values in mm²/s for DWI)
# ---------------------------------------------------------------------------
# Reference values for the four tissue types simulated. Each entry holds the
# T1/T2/T2* relaxation times (ms), proton density (PD, 0–1), display colour,
# and apparent diffusion coefficient (ADC) used in DWI sequences.
TISSUES = {
    "WM":  {"T1": 1084,  "T2": 69,   "T2s": 26,  "PD": 0.70, "color": "#4472C4", "ADC": 0.00070},
    "GM":  {"T1": 1820,  "T2": 99,   "T2s": 33,  "PD": 0.82, "color": "#ED7D31", "ADC": 0.00080},
    "CSF": {"T1": 4163,  "T2": 2000, "T2s": 500, "PD": 1.00, "color": "#A9D18E", "ADC": 0.00300},
    "Fat": {"T1": 371,   "T2": 133,  "T2s": 17,  "PD": 1.00, "color": "#FFD966", "ADC": 0.00050},
}

# ---------------------------------------------------------------------------
# Field-strength-dependent tissue parameters
# ---------------------------------------------------------------------------
# T1 and T2 values are drawn from the following sources:
#   Bottomley et al., Med. Phys. 11(4), 1984  — empirical power-law scaling
#     T1 ∝ B0^α  (α ≈ 0.38 for WM/GM, ~0 for CSF, ~0.05 for fat)
#   O'Reilly & Webb, Magn. Reson. Med. 87(1), 2022 — in-vivo 0.064 T measurements
#   Stanisz et al., Magn. Reson. Med. 54(3), 2005  — reference values at 3 T
#
# T2* values are not tabulated at low field in those references.  They are
# estimated here as min(T2, T2s_3T × (3.0 / B0)), reflecting that susceptibility
# broadening decreases at lower field (T2* → T2 as B0 → 0).
#
# All other tissue properties (PD, colour, ADC) are field-strength-independent
# and are inherited from the base TISSUES dict.
FIELD_STRENGTH_TISSUES = {
    "0.064T": {
        # O'Reilly & Webb MRM 2022 — in-vivo brain at 0.064 T (Halcyon-class ULF)
        "WM":  {"T1": 275,  "T2": 92,   "T2s": 92},    # T2* ≈ T2 (negligible susceptibility)
        "GM":  {"T1": 327,  "T2": 101,  "T2s": 101},
        "CSF": {"T1": 4000, "T2": 1584, "T2s": 1584},
        "Fat": {"T1": 100,  "T2": 90,   "T2s": 90},
    },
    "0.5T": {
        # Bottomley et al. Med Phys 1984 empirical scaling from 3 T reference
        "WM":  {"T1": 400,  "T2": 90,   "T2s": 90},    # T2* capped at T2 (26 × 6 = 156 > 90)
        "GM":  {"T1": 570,  "T2": 100,  "T2s": 100},   # T2* capped at T2 (33 × 6 = 198 > 100)
        "CSF": {"T1": 4000, "T2": 1800, "T2s": 1800},  # T2* capped at T2
        "Fat": {"T1": 180,  "T2": 85,   "T2s": 85},    # T2* capped at T2
    },
    "1.0T": {
        # Bottomley et al. Med Phys 1984
        "WM":  {"T1": 530,  "T2": 88,   "T2s": 78},    # min(88, 26×3=78)
        "GM":  {"T1": 800,  "T2": 95,   "T2s": 95},    # min(95, 33×3=99) → 95
        "CSF": {"T1": 4000, "T2": 2000, "T2s": 1500},  # min(2000, 500×3=1500)
        "Fat": {"T1": 220,  "T2": 80,   "T2s": 51},    # min(80, 17×3=51)
    },
    "1.5T": {
        # Stanisz et al. MRM 2005 / Bottomley et al. Med Phys 1984
        "WM":  {"T1": 650,  "T2": 80,   "T2s": 52},    # min(80, 26×2=52)
        "GM":  {"T1": 1200, "T2": 90,   "T2s": 66},    # min(90, 33×2=66)
        "CSF": {"T1": 4000, "T2": 2000, "T2s": 1000},  # min(2000, 500×2=1000)
        "Fat": {"T1": 250,  "T2": 70,   "T2s": 34},    # min(70, 17×2=34)
    },
    "3.0T": {
        # Stanisz et al. MRM 2005 (reference field strength)
        "WM":  {"T1": 832,  "T2": 80,   "T2s": 26},
        "GM":  {"T1": 1331, "T2": 80,   "T2s": 33},
        "CSF": {"T1": 4000, "T2": 2000, "T2s": 500},
        "Fat": {"T1": 365,  "T2": 60,   "T2s": 17},
    },
    # ── Ultra-high-field entries ──────────────────────────────────────────
    # T1: Rooney et al., Magn. Reson. Med. 57(2), 2007 — empirical power-law fits
    #   T1_WM = 0.583 × B0^0.382,  T1_GM = 0.857 × B0^0.376  (B0 in T, T1 in s)
    # T2: de Graaf et al., Magn. Reson. Med. 56(2), 2006 — field-dependent T2 trend
    # T2* is not tabulated in those references. As at low field, it is estimated
    # as min(T2, T2s_3T × (3.0 / B0)), reflecting increasing susceptibility
    # broadening (T2* shrinking faster than T2) as B0 rises above 3 T.
    "4.7T": {
        "WM":  {"T1": 1050, "T2": 55,  "T2s": 17},   # min(55,  26×3/4.7=17)
        "GM":  {"T1": 1750, "T2": 60,  "T2s": 21},   # min(60,  33×3/4.7=21)
        "CSF": {"T1": 4500, "T2": 400, "T2s": 319},  # min(400, 500×3/4.7=319)
        "Fat": {"T1": 400,  "T2": 40,  "T2s": 11},   # min(40,  17×3/4.7=11)
    },
    "7.0T": {
        "WM":  {"T1": 1126, "T2": 45,  "T2s": 11},   # min(45,  26×3/7.0=11)
        "GM":  {"T1": 1939, "T2": 50,  "T2s": 14},   # min(50,  33×3/7.0=14)
        "CSF": {"T1": 4500, "T2": 350, "T2s": 214},  # min(350, 500×3/7.0=214)
        "Fat": {"T1": 500,  "T2": 35,  "T2s": 7},    # min(35,  17×3/7.0=7)
    },
    "9.4T": {
        "WM":  {"T1": 1280, "T2": 35,  "T2s": 8},    # min(35,  26×3/9.4=8)
        "GM":  {"T1": 2200, "T2": 40,  "T2s": 11},   # min(40,  33×3/9.4=11)
        "CSF": {"T1": 4500, "T2": 300, "T2s": 160},  # min(300, 500×3/9.4=160)
        "Fat": {"T1": 550,  "T2": 28,  "T2s": 5},    # min(28,  17×3/9.4=5)
    },
    "11.7T": {
        "WM":  {"T1": 1420, "T2": 28,  "T2s": 7},    # min(28,  26×3/11.7=7)
        "GM":  {"T1": 2450, "T2": 32,  "T2s": 8},    # min(32,  33×3/11.7=8)
        "CSF": {"T1": 4500, "T2": 250, "T2s": 128},  # min(250, 500×3/11.7=128)
        "Fat": {"T1": 600,  "T2": 22,  "T2s": 4},    # min(22,  17×3/11.7=4)
    },
}

# SNR scales approximately linearly with B0 (for a given coil / bandwidth).
# Factors below represent SNR relative to 3 T (= 1.000).
FIELD_SNR_SCALE = {
    "0.064T": 0.021,
    "0.5T":   0.167,
    "1.0T":   0.333,
    "1.5T":   0.500,
    "3.0T":   1.000,
    "4.7T":   1.567,   # 4.7 / 3.0
    "7.0T":   2.333,   # 7.0 / 3.0
    "9.4T":   3.133,   # 9.4 / 3.0
    "11.7T":  3.900,   # 11.7 / 3.0
}

# Maximum clinically useful TR per field strength.
# T1 values shorten at lower B0, so very long TRs add no contrast benefit.
# T1 values lengthen again at ultra-high field (Rooney et al. MRM 2007), so TR
# ceilings rise accordingly for 4.7 T and above.
# These values set both the TR slider ceiling and the T1 recovery curve x-axis limit.
TR_MAX_BY_FIELD = {
    "0.064T": 1000,
    "0.5T":   2000,
    "1.0T":   3000,
    "1.5T":   4000,
    "3.0T":   6000,
    "4.7T":   8000,
    "7.0T":   10000,
    "9.4T":   12000,
    "11.7T":  15000,
}

SNR_SCALE = 200.0

# ---------------------------------------------------------------------------
# Teaching protocol presets
# ---------------------------------------------------------------------------
# These are realistic starting points for common educational contrasts. The
# user can still adjust every control after choosing a preset.
PROTOCOL_PRESETS = {
    "Custom": {},
    "T1 FSE brain": {
        "seq": "FSE", "TR": 600, "TE": 15, "ETL": 4,
        "FOV_read": 400, "FOV_phase": 400, "freq_matrix": 256, "phase_matrix": 256,
        "slice_mm": 5, "NEX": 1, "BW": 200, "pf_label": "8/8 (Full)",
    },
    "T2 FSE brain": {
        "seq": "FSE", "TR": 4000, "TE": 100, "ETL": 16,
        "FOV_read": 400, "FOV_phase": 400, "freq_matrix": 320, "phase_matrix": 256,
        "slice_mm": 5, "NEX": 1, "BW": 200, "pf_label": "8/8 (Full)",
    },
    "FLAIR brain": {
        "seq": "FLAIR", "TR": 9000, "TE": 90, "TI": 2500, "ETL": 16,
        "FOV_read": 400, "FOV_phase": 400, "freq_matrix": 256, "phase_matrix": 256,
        "slice_mm": 5, "NEX": 1, "BW": 200,
    },
    "STIR fat suppression": {
        "seq": "STIR", "TR": 3000, "TE": 60, "TI": 210, "ETL": 8,
        "FOV_read": 400, "FOV_phase": 400, "freq_matrix": 256, "phase_matrix": 224,
        "slice_mm": 5, "NEX": 1, "BW": 200, "fat_sat": True,
    },
    "GRE T2*": {
        "seq": "GRE", "TR": 700, "TE": 25, "FA": 25,
        "FOV_read": 400, "FOV_phase": 400, "freq_matrix": 256, "phase_matrix": 192,
        "slice_mm": 5, "NEX": 1, "BW": 200, "pf_label": "8/8 (Full)",
    },
    "DWI b1000": {
        "seq": "DWI", "TR": 5000, "TE": 80, "b": 1000,
        "FOV_read": 400, "FOV_phase": 400, "freq_matrix": 128, "phase_matrix": 128,
        "slice_mm": 5, "NEX": 2, "BW": 250,
    },
    "MPRAGE 3D T1": {
        "seq": "MPRAGE", "TR": 2300, "TE": 3, "TI": 900, "FA": 9,
        "FOV_read": 400, "FOV_phase": 400, "freq_matrix": 256, "phase_matrix": 256,
        "Npartitions": 176, "NEX": 1, "BW": 200,
    },
    "EPI BOLD": {
        "seq": "EPI (single-shot)", "TR": 2000, "TE": 30,
        "FOV_read": 400, "FOV_phase": 400, "freq_matrix": 128, "phase_matrix": 128,
        "slice_mm": 4, "NEX": 1, "BW": 250, "n2_ghost_pct": 10, "pf_label": "8/8 (Full)",
    },
}

# ---------------------------------------------------------------------------
# Reference constants for the three-step Signal / Noise / SNR architecture
# ---------------------------------------------------------------------------
# Step 1 — Signal = physics_signal(tissue, seq) × voxel_vol / ref_voxel
#   ref_voxel is acquisition-type-specific (1 mm for 3D, 5 mm for 2D) so that
#   signals sit at baseline at default settings for both modes.
_ref_px        = 240.0 / 256.0          # in-plane pixel size at defaults
REF_VOXEL_VOL  = _ref_px ** 2 * 5.0    # 2-D reference voxel ≈ 4.394 mm³

# Step 2 — Noise = NOISE_REF × sqrt(BW / BW_REF) / sqrt(NEX)
#   Noise depends ONLY on BW and NEX — never on FOV, matrix, slice, Npartitions,
#   TR, TE, TI, or flip angle.
#   NOISE_REF is calibrated so that at 2-D default settings SNR magnitudes
#   match the pre-refactor output: NOISE_REF = sqrt(BW_REF)/(REF_VOXEL_VOL×SNR_SCALE)
BW_REF    = 200.0                                       # default bandwidth Hz/px
NEX_REF   = 1.0                                         # default NEX
NOISE_REF = np.sqrt(BW_REF) / (REF_VOXEL_VOL * SNR_SCALE)  # ≈ 0.01609

# For 3D MPRAGE, Npartitions is a voxel-volume parameter (equivalent to slice
# thickness in 2D), not a signal-averaging parameter.  A fixed slab thickness
# is divided by Npartitions to give the partition thickness used in voxel_vol.
# Increasing Npartitions → thinner partitions → smaller voxel → lower Signal
# and SNR, but Noise is completely unaffected.
MPRAGE_SLAB_MM       = 176.0   # fixed slab coverage (mm) — default 176 × 1 mm
MPRAGE_NPART_DEFAULT = 176     # partition count at default slider position
MPRAGE_TR_READOUT    = 7       # ms — fixed gradient echo spacing within the partition-encode readout train

# Step 3 — SNR = Signal / Noise  (computed directly, no circular dependencies)

# ---------------------------------------------------------------------------
# Physics functions
# ---------------------------------------------------------------------------
# One signal equation per MRI sequence, plus helper functions for SNR, scan
# time, Ernst angle, and inversion-null TI calculations. Each function takes
# tissue properties and sequence parameters and returns a signal intensity
# between 0 and 1.

def fse_signal(TR, TE, T1, T2, PD, fat_sat=False, is_fat=False):
    S = PD * (1 - np.exp(-TR / T1)) * np.exp(-TE / T2)
    if fat_sat and is_fat:
        S *= 0.05
    return S

def gre_signal(TR, TE, FA_deg, T1, T2s, PD, fat_sat=False, is_fat=False):
    FA = np.radians(FA_deg)
    E1 = np.exp(-TR / T1)
    S  = PD * np.sin(FA) * (1 - E1) / (1 - np.cos(FA) * E1) * np.exp(-TE / T2s)
    if fat_sat and is_fat:
        S *= 0.05
    return S

def flair_signal(TR, TI, TE, T1, T2, PD, fat_sat=False, is_fat=False):
    S = PD * abs(1 - 2 * np.exp(-TI / T1) + np.exp(-TR / T1)) * np.exp(-TE / T2)
    if fat_sat and is_fat:
        S *= 0.05
    return S

def bssfp_signal(TR, FA_deg, T1, T2, PD):
    FA = np.radians(FA_deg)
    E1 = np.exp(-TR / T1)
    E2 = np.exp(-TR / T2)
    S  = PD * np.sin(FA) * (1 - E1) / (1 - (E1 - E2) * np.cos(FA) - E1 * E2) * np.sqrt(E2)
    return S

def bssfp_optimal_fa(TR, T1, T2):
    E1 = np.exp(-TR / T1)
    E2 = np.exp(-TR / T2)
    return np.degrees(np.arccos((E1 - E2) / (1 - E1 * E2)))

def dir_signal(TR, TI1, TI2, TE, T1, T2, PD):
    S = PD * abs(1 - 2*np.exp(-TI2/T1) + 2*np.exp(-TI1/T1) - np.exp(-TR/T1)) * np.exp(-TE/T2)
    return S

def dir_null_ti2(TR, TI1, T1_tissue):
    ti2_arr = np.linspace(10, TI1 - 10, 5000)
    s = 1 - 2*np.exp(-ti2_arr/T1_tissue) + 2*np.exp(-TI1/T1_tissue) - np.exp(-TR/T1_tissue)
    idx = np.where(np.diff(np.sign(s)))[0]
    return float(ti2_arr[idx[0]]) if len(idx) > 0 else None

def dwi_signal(TR, TE, b, T1, T2, PD, ADC):
    S0 = PD * (1 - np.exp(-TR / T1)) * np.exp(-TE / T2)
    return S0 * np.exp(-b * ADC)

def mprage_signal(TR, TI, TE, FA_deg, T1, T2s, PD):
    FA = np.radians(FA_deg)
    S  = PD * abs(1 - 2*np.exp(-TI/T1) + np.exp(-TR/T1)) * np.sin(FA) * np.exp(-TE/T2s)
    return S

def epi_signal(TR, TE, T1, T2s, PD):
    return PD * (1 - np.exp(-TR / T1)) * np.exp(-TE / T2s)

def csf_null_ti(TR, T1):
    # TI that nulls a tissue with given T1 in an IR sequence.
    # Derived from 1 - 2*exp(-TI/T1) + exp(-TR/T1) = 0  →  TI = T1*ln(2/(1+exp(-TR/T1)))
    # Approaches T1*ln(2) ≈ 0.693*T1 when TR >> T1.
    return T1 * np.log(2.0 / (1.0 + np.exp(-TR / T1)))

def ernst_angle(TR, T1):
    return np.degrees(np.arccos(np.exp(-TR / T1)))

def calc_scan_time(TR, phase_matrix, ETL, NEX, seq, Npartitions=1):
    if seq == "GRE":
        return TR * phase_matrix * NEX / 1000.0
    elif seq == "MPRAGE":
        # Scan time = TR_total × matrix_phase × NEX.
        # Npartitions constrains minimum TR via the readout train duration
        # but does NOT appear in the scan time formula.
        return TR * phase_matrix * NEX / 1000.0
    else:
        return TR * (phase_matrix / ETL) * Npartitions * NEX / 1000.0


# ---------------------------------------------------------------------------
# Pulse sequence diagram
# ---------------------------------------------------------------------------
# Tuning constants — change values here to adjust all sequence diagrams
# without hunting through the drawing code below.

# Figure layout
PSD_FIG_SIZE        = (8.0, 3.0)   # (width, height) in inches
PSD_HSPACE          = 0.06         # vertical spacing between subplot rows
PSD_MARGIN_TOP      = 0.88
PSD_MARGIN_BOT      = 0.09
PSD_MARGIN_LEFT     = 0.09
PSD_MARGIN_RIGHT    = 0.97
PSD_Y_MIN           = -1.85        # lower y-limit for every row
PSD_Y_MAX           =  2.00        # upper y-limit for every row
PSD_YLABEL_PAD      = 30           # labelpad for row name text

# Line widths
WAVEFORM_LINEWIDTH  = 0.9   # RF pulses and gradient trapezoids
SIGNAL_LINEWIDTH    = 0.8   # echo / FID waveforms
RECOVERY_LINEWIDTH  = 0.6   # dashed Mz recovery guide
BASELINE_LINEWIDTH  = 0.5   # zero-line under each row
VMARK_LINEWIDTH     = 0.4   # vertical timing-marker lines
ARROW_LINEWIDTH     = 0.7   # annotation double-arrow width

# RF pulse half-widths (schematic time units)
RF_HW_EXCITE        = 0.32  # 90° excitation sinc — GRE, FSE
RF_HW_FLAIR         = 0.30  # 90° excitation sinc — FLAIR, STIR
RF_HW_BSSFP         = 0.27  # sinc in bSSFP (shorter TR periods)
RF_HW_INV           = 0.22  # 180° inversion rect — FLAIR, STIR
RF_HW_180_FSE       = 0.19  # 180° refocusing rect — FSE
RF_HW_180_REF       = 0.18  # 180° refocusing rect — FLAIR, STIR
RF_SINC_FREQ        = 2.5   # sinc lobe density (oscillations per half-width)
RF_SINC_N           = 80    # sample points per sinc waveform
RF_LABEL_OFFSET     = 0.22  # gap from pulse peak to label text

# Gradient trapezoid geometry (schematic time units)
GRAD_RISE           = 0.10  # standard rise / fall time
GRAD_RISE_MD        = 0.08  # medium rise — FLAIR/STIR lobes
GRAD_RISE_SM        = 0.10  # small rise  — FSE 180° Gss, PE blips.
GRAD_RISE_XS        = 0.05  # extra-small — bSSFP balanced gradients
GRAD_SS_FLAT        = 0.52  # flat-top: slice-select main lobe
GRAD_SS_AMP         =  1.00 # amplitude:  slice-select main lobe
GRAD_SS_REP_RISE    = 0.06  # rise: Gss rephaser (GRE)
GRAD_SS_REP_FLAT    = 0.28  # flat-top: Gss rephaser
GRAD_SS_REP_AMP     = -0.50 # amplitude: Gss rephaser
GRAD_SS_INV_AMP     =  0.90 # amplitude: Gss under inversion / refocus
GRAD_SS_INV_FLAT    =  0.56 # flat-top:  Gss under inversion / refocus
GRAD_PE_AMP         =  0.70 # amplitude: phase-encode main lobe
GRAD_PE_FLAT        =  0.40 # flat-top:  phase-encode main lobe
GRAD_FE_DEP_AMP     = -0.60 # amplitude: Gfe dephase lobe
GRAD_FE_DEP_FLAT    =  0.38 # flat-top:  Gfe dephase lobe
GRAD_FE_READ_AMP    =  1.00 # amplitude: Gfe readout lobe
GRAD_FE_READ_FLAT   =  1.15 # flat-top:  Gfe readout lobe

# Signal / echo waveform geometry
SIGNAL_N            = 100   # sample points for echo waveforms
SIGNAL_RANGE        = 1.8   # time extent each side = hw × SIGNAL_RANGE
SIGNAL_FREQ_GRE     = 18    # oscillation frequency: gradient echo
SIGNAL_FREQ_SE      = 15    # oscillation frequency: spin echo
SIGNAL_ENV_GRE      = 0.75  # Gaussian sigma factor: gradient echo
SIGNAL_ENV_SE       = 0.65  # Gaussian sigma factor: spin echo
SIGNAL_HW_GRE       = 0.36  # envelope half-width: gradient echo
SIGNAL_HW_SE        = 0.27  # envelope half-width: spin echo
SIGNAL_HW_BSSFP     = 0.22  # envelope half-width: bSSFP echo
RECOVERY_ALPHA      = 0.35  # opacity of dashed Mz recovery guide

# Annotation geometry
ANN_TE_Y            = -1.45 # y-position: TE double-arrow
ANN_TI_Y            = -1.45 # y-position: TI double-arrow
ANN_TR_Y            =  1.72 # y-position: TR double-arrow
ANN_LABEL_OFFSET    =  0.28 # gap from arrow to annotation text (below)
ANN_TR_LABEL_OFF    =  0.15 # gap from TR arrow to TR label (above)

# Font sizes (pt)
FONT_ROW_LABEL      = 6     # row labels (RF, Gss, Gpe …)
FONT_RF_LABEL       = 5.5   # flip-angle / pulse-type labels on RF row
FONT_ANN            = 5     # TE / TI / TR annotation text
FONT_TITLE          = 7     # diagram title
FONT_FOOTER         = 6     # "Time →" footer

# Schematic time axis
T_TOTAL             = 10.0  # full span in schematic units

# ── Section 1: FSE constants ──────────────────────────────────────────────
FSE_MAX_ESP             = 1.75  # maximum schematic echo spacing (caps crowding at high ETL)
FSE_MAX_ECHOES_TO_SHOW  = 6     # maximum echoes to show at once (caps crowding at high ETL)
FSE_T_OFFSET            = 2.6   # time reserved before first 180° pulse (90° excite + dephase lobes)
FSE_RF_TC               = 0.6   # time-centre of 90° excitation RF pulse on PSD time axis
FSE_GSS_T0              = 0.18  # start time of Gss slice-select main lobe on PSD time axis
FSE_GSS_REP_T0          = 0.92  # start time of Gss rephaser lobe on PSD time axis
FSE_GSS_REP_FLAT        = 0.26  # flat-top duration of Gss rephaser (FSE, slightly narrower than GRE)
FSE_GPE_T0              = 1.12  # start time of initial Gpe dephase lobe
FSE_GPE_FLAT            = 0.30  # flat-top of initial Gpe dephase lobe
FSE_GPE_AMP             = 0.80  # amplitude of initial Gpe dephase lobe
FSE_GFE_T0              = 1.12  # start time of initial Gfe dephase lobe
FSE_GFE_FLAT            = 0.30  # flat-top of initial Gfe dephase lobe
FSE_GFE_AMP             = -0.50 # amplitude of initial Gfe dephase lobe
FSE_ECHO_OFFSET         = 0.50  # echo centre as fraction of esp after 180° pulse centre
FSE_CRUSH_RISE          = 0.05  # outer rise/fall time (0→crusher and crusher→0)
FSE_CRUSH_SS_RISE       = 0.05  # inner rise time between crusher and SS plateau levels
FSE_CRUSH_PRE_FLAT      = 0.10  # leading crusher plateau duration (before SS lobe)
FSE_CRUSH_NEG_FLAT      = 0.10  # trailing negative crusher plateau duration (after SS lobe)
FSE_CRUSH_AMP           = 0.85  # amplitude of crusher lobes (above SS plateau level)
FSE_SS_AMP_180          = 0.50  # amplitude of SS plateau under each 180° sinc pulse (below crushers)
FSE_PE_AMP_MIN          = 0.10  # Gpe blip amplitude at eff_idx (k-space centre echo)
FSE_PE_AMP_MAX          = 0.75  # Gpe blip amplitude at outermost echoes (max k-space)
FSE_GPE_BLIP_FLAT       = 0.12  # flat-top of Gpe phase-encode blip per echo
FSE_GFE_ECHO_OFFSET     = 0.42  # (legacy — kept for reference; readout timing now computed from echo centre)
FSE_GFE_READ_FLAT       = 0.30  # flat-top of Gfe readout lobe per echo
FSE_GFE_ECHO_AMP        = 0.90  # amplitude of Gfe readout lobe (positive for all echoes)
FSE_SIG_AMP_EFF         = 1.00  # peak signal amplitude at the eff_idx echo
FSE_SIG_DECAY           = 0.40  # per-echo T2-decay exponent away from eff_idx
FSE_NOMINAL_ESP_MS      = 20.0  # nominal echo spacing (ms) used to map TE slider → eff_idx
FSE_XLIM_PAD            = 0.30  # x-axis padding beyond T_TOTAL

# ── Section 2: FLAIR / STIR constants ────────────────────────────────────
FLAIR_TI_MIN_S          = 1.4   # minimum schematic TI span — must exceed Gspoiler end time (0.45+0.36+0.56=1.37)
FLAIR_TI_RANGE          = 3.8   # T_TOTAL minus this sets the usable TI span
FLAIR_TI_CLAMP_MIN      = 1.0   # lower clamp on schematic ti_s
FLAIR_TI_CLAMP_MAX      = 6.8   # upper clamp on schematic ti_s
FLAIR_TE_S              = 0.85  # schematic TE span (inversion excite → echo)
FLAIR_T_INV             = 0.45  # time-centre of 180° inversion pulse
FLAIR_REC_T0_OFFSET     = 0.28  # recovery curve start offset after inversion centre
FLAIR_REC_T1_OFFSET     = 0.12  # recovery curve end offset before 90° excitation
FLAIR_REC_N             = 90    # sample points for Mz recovery guide curve
FLAIR_MZ_SCALE          = 0.68  # vertical scale of Mz recovery guide
FLAIR_T1_SCALE          = 0.48  # T1 time-constant scale factor for recovery curve
FLAIR_GSS_INV_OFFSET    = 0.36  # how far before inversion centre Gss lobe starts
FLAIR_GSS_EXC_OFFSET    = 0.46  # how far before 90° centre Gss lobe starts
FLAIR_GSS_REW_OFFSET    = 0.32  # time after 90° centre that Gss rephaser starts
FLAIR_GSS_REW_FLAT      = 0.22  # flat-top of Gss rephaser after 90°
FLAIR_GSS_REW_AMP       = -0.45 # amplitude of Gss rephaser after 90°
FLAIR_GPE_EXC_OFFSET    = 0.58  # time after 90° centre that Gpe dephase starts
FLAIR_GFE_EXC_OFFSET    = 0.58  # time after 90° centre that Gfe dephase starts
FLAIR_GFE_DEP_AMP       = -0.50 # amplitude of Gfe dephase lobe after 90°
FLAIR_GSS_REF_OFFSET    = 0.30  # how far before refocus centre Gss lobe starts
FLAIR_GSS_REF_FLAT      = 0.46  # flat-top of Gss lobe under refocus pulse
FLAIR_GSS_REF_AMP       = 0.85  # amplitude of Gss lobe under refocus pulse
FLAIR_GFE_READ_OFFSET   = 0.46  # how far before echo centre Gfe readout starts
FLAIR_GFE_READ_FLAT     = 0.72  # flat-top of Gfe readout lobe
FLAIR_GFE_READ_AMP      = 0.95  # amplitude of Gfe readout lobe
FLAIR_GPE_REW_OFFSET    = 0.22  # time after echo centre that Gpe rewind starts
FLAIR_ANN_TE_Y          = -1.05 # y-position of TE annotation arrow (STIR)
FLAIR_XLIM_PAD          = 0.30  # x-axis padding beyond last echo
# ── FLAIR FSE-host design: inversion Gspoiler ─────────────────────────────
FLAIR_GSPOILER_AMP      = 1.30  # Gspoiler amplitude — exceeds GRAD_SS_INV_AMP (0.90)
FLAIR_GSPOILER_FLAT     = 0.40  # Gspoiler flat-top — area (1.30×0.56=0.728) > Gss inv area (0.90×0.72=0.648)
FLAIR_NOMINAL_ESP_MS    = 25.0  # nominal ESP (ms) used to map TEeff slider → eff_idx (FLAIR ESP ≈ 25 ms)

# ── Section 2b: DIR (Double Inversion Recovery) constants ────────────────
DIR_T_INV1          = 0.45   # time-centre of first (non-selective) 180° hard pulse
DIR_NS_GSS_AMP      = 0.40   # non-selective Gss lobe amplitude (lower than slice-select)
DIR_NS_GSS_RISE     = 0.10   # non-selective Gss rise/fall (same slew as GRAD_RISE)
DIR_NS_GSS_FLAT     = 0.22   # non-selective Gss flat-top (spans ≈ RF_HW_INV × 2)
DIR_TI_TOTAL_RANGE  = 3.8    # T_TOTAL minus this sets the usable schematic (TI1+TI2) span
DIR_TI_TOTAL_MIN_S  = 1.7    # minimum total schematic TI span
DIR_TI1_MIN_S       = 0.7    # minimum schematic TI1 span (second inv Gss must not overlap first)
DIR_TI2_MIN_S       = 0.20   # minimum schematic TI2 span — numeric floor only; whether the
                              # Gspoiler itself fits (full size, compressed, or not at all) is
                              # now decided dynamically by the Gspoiler fit check in the DIR branch
DIR_GSPOILER_MIN_WIDTH_S = 0.12  # minimum visually-legible compressed Gspoiler width — below
                                  # this the fit is too tight to draw and a warning is shown instead
DIR_GSPOILER_MAX_AMP     = 1.90  # amplitude cap when compressing the Gspoiler (stays under PSD_Y_MAX)
DIR_NOMINAL_ESP_MS  = 25.0   # nominal ESP (ms) for TEeff→eff_idx mapping (same as FLAIR)
DIR_XLIM_PAD        = 0.30   # x-axis padding beyond last echo

# ── Section 3: bSSFP constants ────────────────────────────────────────────
N_BSSFP_REPS            = 3     # number of TR repetitions shown
BSSFP_RF_OFF            = 0.20  # RF centre as fraction of tr_w from each TR start
BSSFP_SIG_OFF           = 0.70  # echo centre as fraction of tr_w = RF_OFF + 0.50 (TE = TR/2)
BSSFP_SS_AMP            = 0.70  # Gss lobe amplitude (all three lobes share this amplitude)
BSSFP_PE_AMP            = 0.55  # Gpe encode / rewind lobe amplitude
BSSFP_FE_AMP            = 0.65  # Gfe prephaser / rewinder lobe amplitude
BSSFP_TR1_AMP           = 0.65  # RF amplitude for first (transient) TR
BSSFP_SIG_SCALE         = 0.78  # signal amplitude scale relative to RF amplitude
BSSFP_VMARK_COLOR       = "#444444"  # colour of TR-boundary vertical marker
BSSFP_FONT_REPS         = FONT_ANN - 0.5  # font size for "×N TRs shown" label
# Gss balanced 3-lobe geometry (pre-phaser → slice-select → re-phaser, net moment = 0)
BSSFP_GSS_MAIN_FLAT     = 0.20  # flat-top of SS main lobe — lobe is centred on RF pulse
BSSFP_GSS_HALFPRE_FLAT  = 0.075 # flat-top of pre/rephaser — area = SS_area/2 at same amplitude → net Gss = 0
# Gfe balanced 3-lobe geometry (neg before RF, neg after RF, readout at TE=TR/2, net moment = 0)
#   neg lobe amp = BSSFP_FE_AMP, flat = BSSFP_GSS_HALFPRE_FLAT (same timing as Gss pre/rephaser)
#   balance: BSSFP_GFE_RO_FLAT = 2×BSSFP_FE_AMP×(BSSFP_GSS_HALFPRE_FLAT+GRAD_RISE_XS)/GRAD_FE_READ_AMP − GRAD_RISE_XS
BSSFP_GFE_RO_FLAT       = 0.1125 # flat-top of Gfe readout lobe — derived so 2×neg_area = readout_area → net Gfe = 0
BSSFP_GFE_PERIPH_FLAT   = 0.22  # (retained for reference; superseded by the neg-lobe/readout balance above)
# Gpe balanced bipolar geometry (encode → rewind, net moment = 0)
# 'Repeating unit' dashed rectangle styling
BSSFP_REPEAT_COLOR      = "#6688AA"  # border colour of 'Repeating unit' rectangle
BSSFP_RECT_LINEWIDTH    = 0.8   # linewidth of 'Repeating unit' rectangle

# ── Section 4: GRE constants ──────────────────────────────────────────────
GRE_RF_HW               = 0.12  # GRE sinc RF pulse half-width (modern fast gradients)
GRE_GSS_RISE            = 0.04  # GRE Gss slice-select rise/fall time (modern fast gradients)
GRE_GSS_FLAT            = 0.24  # GRE Gss slice-select flat-top = 2×GRE_RF_HW so Gss centre ≡ RF centre
GRE_RF_TC               = 0.6   # time-centre of excitation RF pulse
GRE_TE_MIN              = 1.8   # minimum schematic TE position
GRE_TE_MAX_MS           = 100.0 # TE value (ms) that maps to maximum schematic TE
GRE_TE_RANGE            = 4.0   # schematic TE range (added to GRE_TE_MIN)
GRE_GSS_T0              = 0.44  # start time of Gss slice-select main lobe (flat-top starts at GRE_RF_TC − GRE_RF_HW)
GRE_GSS_REP_T0          = 0.92  # start time of Gss rephaser lobe (legacy — now computed from gss_ss_end)
GRE_GPE_T0              = 1.35  # start time of Gpe phase-encode lobe (legacy — now computed from gss_rep_end)
GRE_GPE_REWIND_OFFSET   = 0.62  # time after echo centre that Gpe rewind starts (legacy — now computed)
GRE_GFE_T0              = 1.35  # start time of Gfe dephase lobe (legacy — now computed)
GRE_GFE_READ_OFFSET     = 0.68  # how far before echo centre Gfe readout starts (legacy — now computed)
GRE_SIG_AMP             = 1.10  # gradient echo signal amplitude
GRE_GHOST_AMP           = 0.65  # RF amplitude for ghost next-TR pulse
GRE_GHOST_GSS_OFFSET    = 0.21  # how far before ghost RF centre the ghost Gss lobe starts (= GRE_RF_HW + GRE_GSS_RISE)
GRE_XLIM_SCALE          = 0.75  # fraction of (T_TOTAL − GRE_RF_TC) used to place the ghost TR position
GRE_GSS_DEFAULT_SLICE   = 5.0   # reference slice thickness (mm) at which Gss amplitude = GRE_GSS_REF_AMP
GRE_GSS_REF_AMP         = 1.00  # Gss plateau amplitude at GRE_GSS_DEFAULT_SLICE (normalisation reference)
GRE_GSS_MAX_AMP         = 1.90  # maximum Gss display amplitude at minimum slice thickness (< PSD_Y_MAX)
GRE_GSS_REP_AMP         = -1.00 # Gss rephaser amplitude = Gss amplitude (same slew rate, shortest duration)
GRE_GSS_REP_FLAT        = 0.06  # Gss rephaser flat-top: area = GRE_GSS_FLAT/2 + GRE_GSS_RISE/2 = 0.14
GRE_GFE_REF_BW          = 200.0 # reference readout bandwidth (Hz/px) for Gfe amplitude normalisation
GRE_GFE_REF_FOV         = 240.0 # reference FOV_read (mm) for Gfe amplitude normalisation

# ── Section 5: EPI (single-shot) constants ────────────────────────────────
EPI_RF_TC           = 0.6    # time-centre of 90° excitation
EPI_GSS_T0          = 0.34   # start of Gss main lobe: RF_TC − EPI_GSS_RISE − EPI_GSS_FLAT/2 = 0.34
EPI_GSS_RISE        = 0.06   # EPI-specific Gss rise/fall (shorter than GRAD_RISE — fast EPI grads)
EPI_GSS_FLAT        = 0.40   # EPI-specific Gss flat-top; centre = EPI_GSS_T0+rise+flat/2 = EPI_RF_TC
EPI_GSS_REP_AMP     = -1.00  # Gss rephaser amp (same as SS amp; shorter flat → area = SS_area/2)
EPI_GSS_REP_FLAT    = 0.17   # Gss rephaser flat: area = (EPI_GSS_FLAT+EPI_GSS_RISE)/2 = 0.23
EPI_GSS_REP_RISE    = 0.06   # Gss rephaser rise/fall (= EPI_GSS_RISE — same slew rate)
EPI_PE_PREP_RISE    = 0.05   # Gpe prephaser rise/fall
EPI_PE_PREP_FLAT    = 0.17   # Gpe prephaser flat-top
EPI_PE_PREP_AMP     = -0.80  # Gpe prephaser amplitude (negative → ky = −ky_max)
# Gfe prephaser rise = |prep_amp|/ro_amp × ro_rise = 0.80/1.00 × 0.04 = 0.032
# → identical slew rate to first readout lobe (both on Gfe axis)
EPI_FE_PREP_RISE    = 0.032  # Gfe prephaser rise/fall (matched slew rate to readout)
EPI_FE_PREP_FLAT    = 0.17   # Gfe prephaser flat-top
EPI_FE_PREP_AMP     = -0.80  # Gfe prephaser amplitude (negative → kx = −kx_max)
EPI_RO_RISE         = 0.04   # Gfe readout lobe rise/fall (~33% shorter; = half inter-lobe ramp)
EPI_RO_FLAT         = 0.21   # Gfe readout lobe flat-top (~40% shorter)
EPI_RO_AMP          = 1.00   # Gfe readout amplitude
EPI_BLIP_RISE       = 0.02   # Gpe blip rise/fall (fits in 2×EPI_RO_RISE = 0.08 window)
EPI_BLIP_FLAT       = 0.04   # Gpe blip flat-top
EPI_BLIP_AMP        = 0.30   # Gpe blip amplitude (small — one ky line per readout)
EPI_N_SHOW          = 6      # explicit readout lobes to draw before "..."
EPI_DOTS_X          = 3.26   # x-position of "..." (old + 0.16 timing shift from GSS_T0 change)
EPI_FINAL_RO_T0     = 3.41   # x-start of final (Nth) readout lobe
EPI_XLIM_RIGHT      = 3.96   # right x-axis limit
EPI_NOMINAL_ESP_MS  = 0.8    # nominal ms between EPI readout centres (typical 3T EPI)
EPI_SIG_PEAK        = 2.00   # signal amplitude at first (echo 0) readout: DAVE changed from 1 to 2
EPI_SIG_DECAY       = 0.35   # per-echo T2* decay exponent (steeper — fast T2* decay)
EPI_SIG_HW_SCALE    = 0.16   # echo hw multiplier: full_width=0.207 < ESP_s=0.29 → 28% gap

# ── Section 6: MPRAGE constants ───────────────────────────────────────────
MPRAGE_T_INV1           = 0.45   # time-centre of non-selective 180° hard inversion pulse (= DIR_T_INV1)
MPRAGE_TI_MIN_S         = 1.0    # minimum schematic TI span (clears the non-selective Gss lobe)
MPRAGE_TI_RANGE         = 4.0    # T_TOTAL minus this sets the usable schematic TI span
MPRAGE_TI_CLAMP_MIN     = 1.0    # lower clamp on schematic TI span
MPRAGE_TI_CLAMP_MAX     = 6.0    # upper clamp on schematic TI span
MPRAGE_RF_AMP_MIN       = 0.30   # small-flip sinc display amplitude at FA = 5°
MPRAGE_RF_AMP_PER_DEG   = 0.02   # display amplitude added per degree of FA above 5°
# One GRE-train repetition — same dephase→readout / encode→rewind shape as
# standalone GRE, sized compact so a multi-repetition train fits on the axis.
MPRAGE_REP_RF_HW        = 0.10   # small-flip sinc RF half-width
MPRAGE_REP_GSS_RISE     = 0.04   # Gss slice-select rise/fall
MPRAGE_REP_GSS_FLAT     = 0.20   # Gss slice-select flat-top (= 2 × RF half-width)
MPRAGE_REP_GSS_REP_RISE = 0.03   # Gss rephaser rise/fall
MPRAGE_REP_GSS_REP_FLAT = 0.05   # Gss rephaser flat-top
MPRAGE_REP_GSS_REP_AMP  = -0.45  # Gss rephaser amplitude
MPRAGE_REP_GPE_RISE     = 0.04   # Gpe encode/rewind rise/fall
MPRAGE_REP_GPE_FLAT     = 0.10   # Gpe encode/rewind flat-top
MPRAGE_REP_GFE_DEP_RISE = 0.03   # Gfe dephase (prephase) rise/fall
MPRAGE_REP_GFE_DEP_FLAT = 0.10   # Gfe dephase flat-top
MPRAGE_REP_GFE_DEP_AMP  = -0.55  # Gfe dephase amplitude
MPRAGE_REP_GFE_RO_RISE  = 0.04   # Gfe readout rise/fall
MPRAGE_REP_GFE_RO_FLAT  = 0.18   # Gfe readout flat-top
MPRAGE_REP_GFE_RO_AMP   = 1.00   # Gfe readout amplitude
MPRAGE_REP_SPACING      = 1.15   # schematic spacing between successive RF centres (TR_readout)
MPRAGE_PE_AMP_MAX       = 0.70   # Gpe amplitude at the k-space edge (sequential fill sweeps ±this)
MPRAGE_N_SHOW           = 3      # explicit GRE-train repetitions drawn before "..."
MPRAGE_DOTS_GAP         = 0.9    # extra schematic gap reserved for "..." before the final repetition
MPRAGE_SIG_MIN          = 0.12   # signal amplitude just after inversion (near the T1-null point)
MPRAGE_SIG_MAX          = 1.00   # signal amplitude once Mz has recovered toward equilibrium
MPRAGE_SIG_HW           = 0.13   # echo envelope half-width (matches Gfe readout half-width)
MPRAGE_RECOVERY_GAP_S   = 1.2    # schematic gap after the final repetition before the next inversion
MPRAGE_XLIM_PAD         = 0.30   # x-axis padding beyond the next inversion pulse


def draw_pulse_sequence(seq, TR, TE, TI, FA, ETL, slice_mm=5, BW=200, FOV_read=240,
                        epi_eff_idx=1, epi_n_readouts=64, TI2=0, Npartitions=176):
    """Return a matplotlib Figure with an oscilloscope-style pulse sequence
    diagram for FSE, GRE, FLAIR, STIR, bSSFP, DIR, MPRAGE, EPI.  Returns None for others."""
    if seq not in ("FSE", "GRE", "FLAIR", "STIR", "bSSFP", "DIR", "MPRAGE", "EPI (single-shot)"):
        return None

    ROW_LABELS = ["RF", "Gss", "Gpe", "Gfe", "Signal"]
    ROW_COLORS = ["#FFD700", "#FF6666", "#66DD66", "#6699FF", "#FFFFFF"]

    fig, axes = plt.subplots(
        5, 1, figsize=PSD_FIG_SIZE, facecolor="#1a1a1a", sharex=True,
        gridspec_kw={"hspace": PSD_HSPACE,
                     "top":    PSD_MARGIN_TOP,  "bottom": PSD_MARGIN_BOT,
                     "left":   PSD_MARGIN_LEFT, "right":  PSD_MARGIN_RIGHT})
    c_rf, c_ss, c_pe, c_fe, c_sig = ROW_COLORS
    ax_rf, ax_ss, ax_pe, ax_fe, ax_sig = axes

    for ax, label, col in zip(axes, ROW_LABELS, ROW_COLORS):
        ax.set_facecolor("#1a1a1a")
        ax.set_ylim(PSD_Y_MIN, PSD_Y_MAX)
        ax.axhline(0, color="#3a3a3a", linewidth=BASELINE_LINEWIDTH, zorder=0)
        ax.set_yticks([])
        ax.set_ylabel(label, color=col, fontsize=FONT_ROW_LABEL, rotation=0,
                      labelpad=PSD_YLABEL_PAD, va="center", ha="right")
        for sp in ax.spines.values():
            sp.set_visible(False)

    # ---- drawing helpers ------------------------------------------------
    def trap(ax, t0, rise, flat, fall, amp, col):
        t = [t0, t0+rise, t0+rise+flat, t0+rise+flat+fall]
        ax.plot(t, [0, amp, amp, 0], color=col, linewidth=WAVEFORM_LINEWIDTH)

    def sinc_rf(ax, tc, amp, hw, label=None, col="#FFD700"):
        tt = np.linspace(tc - hw, tc + hw, RF_SINC_N)
        yy = amp * np.sinc(RF_SINC_FREQ * (tt - tc) / hw)
        ax.plot(tt, yy, color=col, linewidth=WAVEFORM_LINEWIDTH)
        if label:
            ax.text(tc, abs(amp) + RF_LABEL_OFFSET, label, color=col,
                    fontsize=FONT_RF_LABEL, ha="center", va="bottom",
                    fontweight="bold")

    def rect_rf(ax, tc, amp, hw, label=None, col="#FFD700"):
        ax.plot([tc-hw, tc-hw, tc+hw, tc+hw], [0, amp, amp, 0],
                color=col, linewidth=WAVEFORM_LINEWIDTH)
        if label:
            ax.text(tc, amp + RF_LABEL_OFFSET, label, color=col,
                    fontsize=FONT_RF_LABEL, ha="center", va="bottom",
                    fontweight="bold")

    def grad_echo(ax, tc, amp, hw, col="#FFFFFF"):
        tt  = np.linspace(tc - hw*SIGNAL_RANGE, tc + hw*SIGNAL_RANGE, SIGNAL_N)
        env = amp * np.exp(-((tt-tc)**2) / (2*(hw*SIGNAL_ENV_GRE)**2))
        ax.plot(tt, env * np.cos(SIGNAL_FREQ_GRE*(tt-tc)),
                color=col, linewidth=SIGNAL_LINEWIDTH)

    def spin_echo(ax, tc, amp, hw, col="#FFFFFF"):
        tt  = np.linspace(tc - hw*SIGNAL_RANGE, tc + hw*SIGNAL_RANGE, SIGNAL_N)
        env = amp * np.exp(-((tt-tc)**2) / (2*(hw*SIGNAL_ENV_SE)**2))
        ax.plot(tt, env * np.cos(SIGNAL_FREQ_SE*(tt-tc)),
                color=col, linewidth=SIGNAL_LINEWIDTH)

    def ann_te(t_rf, t_echo, y=ANN_TE_Y, label=None, draw_axes=None):
        _draw = draw_axes if draw_axes is not None else (ax_fe, ax_sig)
        for ax in _draw:
            ax.annotate("", xy=(t_echo, y), xytext=(t_rf, y),
                        arrowprops=dict(arrowstyle="<->", color="#AAAAAA",
                                        lw=ARROW_LINEWIDTH))
        _lbl = label if label is not None else f"TE={TE} ms"
        if ax_sig in _draw:
            ax_sig.text((t_rf+t_echo)/2, y - ANN_LABEL_OFFSET, _lbl,
                        color="#AAAAAA", fontsize=FONT_ANN, ha="center", va="top")

    def ann_ti(t_inv, t_exc, y=ANN_TI_Y):
        for ax in (ax_ss, ax_sig):
            ax.annotate("", xy=(t_exc, y), xytext=(t_inv, y),
                        arrowprops=dict(arrowstyle="<->", color="#FFAA44",
                                        lw=ARROW_LINEWIDTH))
        ax_sig.text((t_inv+t_exc)/2, y - ANN_LABEL_OFFSET, f"TI={TI} ms",
                    color="#FFAA44", fontsize=FONT_ANN, ha="center", va="top")

    def ann_tr(t0, t1, y=ANN_TR_Y, ax=None):
        _ax = ax if ax is not None else ax_rf
        _ax.annotate("", xy=(t1, y), xytext=(t0, y),
                     arrowprops=dict(arrowstyle="<->", color="#888888",
                                     lw=ARROW_LINEWIDTH))
        _ax.text((t0+t1)/2, y + ANN_TR_LABEL_OFF, f"TR={TR} ms",
                 color="#888888", fontsize=FONT_ANN, ha="center", va="bottom")

    def vmark(t, col="#555555"):
        for ax in axes:
            ax.axvline(t, color=col, linewidth=VMARK_LINEWIDTH,
                       linestyle=":", alpha=0.6)

    # ================================================================
    # GRE
    # ================================================================
    if seq == "GRE":
        # ── Constant-only derived quantities (all independent of TE slider) ──────────
        gfe_ro_hw    = GRAD_RISE + GRAD_FE_READ_FLAT / 2                        # readout half-width (rise + half flat-top)
        gpe_hw       = GRAD_RISE + GRAD_PE_FLAT / 2                             # PE lobe half-width
        gfe_dep_rise = abs(GRAD_FE_DEP_AMP) / GRAD_FE_READ_AMP * GRAD_RISE     # prephaser ramp — same slew rate as readout
        gss_ss_end   = GRE_GSS_T0 + GRE_GSS_RISE + GRE_GSS_FLAT + GRE_GSS_RISE
        gss_rep_rise = abs(GRE_GSS_REP_AMP) / GRE_GSS_REF_AMP * GRE_GSS_RISE   # rephaser ramp — same slew rate as SS lobe
        gss_rep_end  = gss_ss_end + gss_rep_rise + GRE_GSS_REP_FLAT + gss_rep_rise
        _prep_w      = gfe_dep_rise + GRAD_FE_DEP_FLAT + gfe_dep_rise
        _te_s_min    = gss_rep_end + _prep_w + gfe_ro_hw
        _te_min_ms   = int(np.ceil((_te_s_min - GRE_TE_MIN) * GRE_TE_MAX_MS / GRE_TE_RANGE))
        # ── TE-dependent quantities — clamped to _te_s_min so diagram freezes below minimum ──
        te_s = max(_te_s_min, GRE_TE_MIN + (min(TE, GRE_TE_MAX_MS) / GRE_TE_MAX_MS) * GRE_TE_RANGE)
        gfe_ro_start = te_s - gfe_ro_hw
        gfe_prep_t0  = max(gss_rep_end, gfe_ro_start - gfe_dep_rise - GRAD_FE_DEP_FLAT - gfe_dep_rise)
        gpe_enc_t0   = gss_ss_end   # Gpe encode overlaps Gss rephaser (different axes) to minimise TE
        gpe_enc_c    = gpe_enc_t0 + gpe_hw
        gpe_rew_c    = 2 * te_s - gpe_enc_c
        gpe_rew_t0   = gpe_rew_c - gpe_hw
        # ── Slice-thickness-scaled Gss amplitudes (normalized: max at slice=1mm = GRE_GSS_MAX_AMP) ──
        _gss_scale  = GRE_GSS_MAX_AMP / (GRE_GSS_REF_AMP * slice_mm)
        gss_amp     = GRE_GSS_REF_AMP  * _gss_scale   # = GRE_GSS_MAX_AMP / slice_mm
        gss_rep_amp = GRE_GSS_REP_AMP  * _gss_scale   # rephaser scales identically; area ratio preserved
        # ── BW/FOV_read-scaled Gfe amplitudes (Gfe ∝ BW/FOV_read) ───────────────
        _gfe_scale  = (BW / GRE_GFE_REF_BW) * (GRE_GFE_REF_FOV / FOV_read)
        gfe_amp     = GRAD_FE_READ_AMP * _gfe_scale
        gfe_dep_amp = GRAD_FE_DEP_AMP  * _gfe_scale   # prephaser scales identically
        # ── Ghost position and x-axis limits ─────────────────────────────────────
        gre_ghost_x = GRE_RF_TC + (T_TOTAL - GRE_RF_TC) * GRE_XLIM_SCALE
        gpe_rew_end = gpe_rew_t0 + GRAD_RISE + GRAD_PE_FLAT + GRAD_RISE
        ghost_end   = gre_ghost_x - GRE_GHOST_GSS_OFFSET + GRE_GSS_RISE + GRE_GSS_FLAT + GRE_GSS_RISE
        xlim_right  = max(ghost_end, gpe_rew_end) + 0.3
        sinc_rf(ax_rf, GRE_RF_TC, 1.0, GRE_RF_HW, label=f"{FA}°")
        trap(ax_ss, GRE_GSS_T0,  GRE_GSS_RISE,  GRE_GSS_FLAT,      GRE_GSS_RISE,  gss_amp,         c_ss)
        trap(ax_ss, gss_ss_end,  gss_rep_rise,  GRE_GSS_REP_FLAT,  gss_rep_rise,  gss_rep_amp,     c_ss)
        trap(ax_pe, gpe_enc_t0,  GRAD_RISE,     GRAD_PE_FLAT,      GRAD_RISE,     GRAD_PE_AMP,     c_pe)
        trap(ax_pe, gpe_rew_t0,  GRAD_RISE,     GRAD_PE_FLAT,      GRAD_RISE,    -GRAD_PE_AMP,     c_pe)
        trap(ax_fe, gfe_prep_t0, gfe_dep_rise,  GRAD_FE_DEP_FLAT,  gfe_dep_rise,  gfe_dep_amp, c_fe)
        trap(ax_fe, gfe_ro_start,GRAD_RISE,     GRAD_FE_READ_FLAT, GRAD_RISE,     gfe_amp,     c_fe)
        grad_echo(ax_sig, te_s, GRE_SIG_AMP, SIGNAL_HW_GRE)
        sinc_rf(ax_rf, gre_ghost_x, GRE_GHOST_AMP, GRE_RF_HW, col=c_rf + "55")
        trap(ax_ss, gre_ghost_x - GRE_GHOST_GSS_OFFSET, GRE_GSS_RISE, GRE_GSS_FLAT, GRE_GSS_RISE,
             gss_amp, c_ss + "44")
        if TE < _te_min_ms:
            fig.text(0.5, 0.5,
                     f"⚠  TE too short: minimum achievable TE is {_te_min_ms} ms",
                     color="#FF4444", fontsize=6.5, ha="center", va="center",
                     bbox=dict(facecolor="#1a1a1a", alpha=0.9, edgecolor="#FF4444", boxstyle="round,pad=0.6"))
        vmark(GRE_RF_TC, c_rf); vmark(te_s, "#AAAAAA")
        ann_te(GRE_RF_TC, te_s, draw_axes=(ax_sig,))
        ann_tr(GRE_RF_TC, gre_ghost_x)
        axes[-1].set_xlim(0, xlim_right)

    # ================================================================
    # FSE
    # ================================================================
    elif seq == "FSE":
        n_echoes_to_show = min(ETL, FSE_MAX_ECHOES_TO_SHOW)
        # τ_s is fixed (depends only on TE/eff_idx via constants, not on ETL).
        # 180° pulses sit at (2n−1)τ_s, echoes at 2nτ_s, measured from FSE_RF_TC.
        tau_s = FSE_MAX_ESP - FSE_RF_TC   # schematic τ (half-ESP), = 1.15 units
        esp   = 2.0 * tau_s               # schematic ESP = 2τ_s, fixed regardless of ETL
        # eff_idx: which echo acquires k-space centre — derived from TE slider and nominal ESP
        eff_idx = max(0, min(n_echoes_to_show - 1,
                             round(TE / FSE_NOMINAL_ESP_MS) - 1))
        _max_dist = max(eff_idx, n_echoes_to_show - 1 - eff_idx, 1)

        # ── Constraint checks — render warning figure instead of sequence ────
        _esp_ms  = TE / (eff_idx + 1)
        _min_tr  = ETL * _esp_ms
        _psd_warn = []
        if TR < _min_tr:
            _psd_warn.append(
                f"TR too short: minimum TR for current ETL and TEeff is {int(_min_tr)} ms")
        if _esp_ms < 10.0:
            _psd_warn.append(
                "Echo spacing too short to be physically realistic\n"
                "— minimum ESP is approximately 10 ms")
        if _psd_warn:
            for _ax in axes:
                _ax.set_visible(False)
            fig.text(0.5, 0.5, "\n\n".join(f"⚠  {w}" for w in _psd_warn),
                     color="#FF4444", fontsize=6.5, ha="center", va="center",
                     multialignment="center",
                     bbox=dict(facecolor="#1a1a1a", alpha=0.9,
                               edgecolor="#FF4444", boxstyle="round,pad=0.6"))
            fig.text(0.5, 0.93, f"Pulse Sequence — {seq}", color="#CCCCCC",
                     fontsize=FONT_TITLE, ha="center", va="top", fontweight="bold")
            return fig

        # ── 90° excitation ──────────────────────────────────────────────────
        sinc_rf(ax_rf, FSE_RF_TC, 1.0, RF_HW_EXCITE, label="90°")
        # Faint dotted vline from sinc peak downward through all rows
        _exc_peak_ymax = (1.0 - PSD_Y_MIN) / (PSD_Y_MAX - PSD_Y_MIN)
        ax_rf.axvline(FSE_RF_TC, ymin=0, ymax=_exc_peak_ymax,
                      color="#555555", linewidth=VMARK_LINEWIDTH, linestyle=":", alpha=0.5)
        for _vax in (ax_ss, ax_pe, ax_fe, ax_sig):
            _vax.axvline(FSE_RF_TC, color="#555555", linewidth=VMARK_LINEWIDTH,
                         linestyle=":", alpha=0.5)

        # ── Gss excitation: main slice-select lobe + rephaser ───────────────
        trap(ax_ss, FSE_GSS_T0, GRAD_RISE, GRAD_SS_FLAT, GRAD_RISE, GRAD_SS_AMP, c_ss)
        _fse_gss_rep_start = FSE_GSS_T0 + GRAD_RISE + GRAD_SS_FLAT + GRAD_RISE
        trap(ax_ss, _fse_gss_rep_start, GRAD_SS_REP_RISE, FSE_GSS_REP_FLAT,
             GRAD_SS_REP_RISE, GRAD_SS_REP_AMP, c_ss)

        # ── Gfe readout prephase (before echo train) ─────────────────────────
        trap(ax_fe, FSE_GFE_T0, GRAD_RISE_MD, FSE_GFE_FLAT, GRAD_RISE_MD, FSE_GFE_AMP, c_fe)

        # ── Echo train ───────────────────────────────────────────────────────
        te_pos = None
        for i in range(n_echoes_to_show):
            t180  = FSE_MAX_ESP + i * esp
            techo = t180 + FSE_ECHO_OFFSET * esp

            # 180° refocusing: sinc pulse (not rect) matching excitation shape
            sinc_rf(ax_rf, t180, 1.0, RF_HW_180_FSE, label="180°")

            # Composite Gss: leading crusher(+) → fall to SS plateau → SS during 180° sinc
            # → rise to trailing crusher(+, equal area to leading) → fall to zero
            t_c0 = t180 - RF_HW_180_FSE - FSE_CRUSH_SS_RISE - FSE_CRUSH_PRE_FLAT - FSE_CRUSH_RISE
            _cx = [t_c0,
                   t_c0 + FSE_CRUSH_RISE,
                   t180 - RF_HW_180_FSE - FSE_CRUSH_SS_RISE,
                   t180 - RF_HW_180_FSE,
                   t180 + RF_HW_180_FSE,
                   t180 + RF_HW_180_FSE + FSE_CRUSH_SS_RISE,
                   t180 + RF_HW_180_FSE + FSE_CRUSH_SS_RISE + FSE_CRUSH_NEG_FLAT,
                   t180 + RF_HW_180_FSE + FSE_CRUSH_SS_RISE + FSE_CRUSH_NEG_FLAT + FSE_CRUSH_RISE]
            _cy = [0, FSE_CRUSH_AMP, FSE_CRUSH_AMP, FSE_SS_AMP_180,
                   FSE_SS_AMP_180, FSE_CRUSH_AMP, FSE_CRUSH_AMP, 0]
            ax_ss.plot(_cx, _cy, color=c_ss, linewidth=WAVEFORM_LINEWIDTH)

            # PE blip amplitude: smallest at eff_idx (k-space centre), increasing outward
            _dist  = abs(i - eff_idx)
            pe_amp = (FSE_PE_AMP_MIN + (FSE_PE_AMP_MAX - FSE_PE_AMP_MIN) * _dist / _max_dist) * ((-1)**i)

            # ── Timing anchors ──────────────────────────────────────────────
            # techo = t180 + esp/2 = 2nτ (FSE_ECHO_OFFSET = 0.5 enforces this)
            # t_crush_end: trailing crusher of THIS 180° ends here
            t_crush_end = (t180 + RF_HW_180_FSE + FSE_CRUSH_SS_RISE
                           + FSE_CRUSH_NEG_FLAT + FSE_CRUSH_RISE)
            # Gfe readout window, centred exactly on techo
            t_gfe_start = techo - GRAD_RISE_SM - FSE_GFE_READ_FLAT / 2
            t_gfe_end   = techo + FSE_GFE_READ_FLAT / 2 + GRAD_RISE_SM
            # t_next_crush_start: leading crusher of NEXT 180° starts here
            t_next_crush_start = ((t180 + esp) - RF_HW_180_FSE
                                  - FSE_CRUSH_SS_RISE - FSE_CRUSH_PRE_FLAT
                                  - FSE_CRUSH_RISE)
            # half-width of a Gpe blip (trap-start to flat-centre)
            _blip_hw = GRAD_RISE_XS + FSE_GPE_BLIP_FLAT / 2

            # ── Gpe encode blip: centred in [t_crush_end, t_gfe_start] ─────
            _enc_win = t_gfe_start - t_crush_end
            if _enc_win >= 2 * _blip_hw:
                _enc_c = (t_crush_end + t_gfe_start) / 2
            else:
                _enc_c = t_crush_end + _blip_hw   # butt against crusher if window tight
            trap(ax_pe, _enc_c - _blip_hw, GRAD_RISE_XS, FSE_GPE_BLIP_FLAT,
                 GRAD_RISE_XS, pe_amp, c_pe)

            # ── Gfe readout: positive, centred exactly at techo = t180 + esp/2
            trap(ax_fe, t_gfe_start, GRAD_RISE_SM, FSE_GFE_READ_FLAT,
                 GRAD_RISE_SM, FSE_GFE_ECHO_AMP, c_fe)

            # ── Gpe rewind: centred in [t_gfe_end, t_next_crush_start] ─────
            _rew_win = t_next_crush_start - t_gfe_end
            if _rew_win >= 2 * _blip_hw:
                _rew_c = (t_gfe_end + t_next_crush_start) / 2
            else:
                _rew_c = t_gfe_end + _blip_hw     # butt against readout if window tight
            trap(ax_pe, _rew_c - _blip_hw, GRAD_RISE_XS, FSE_GPE_BLIP_FLAT,
                 GRAD_RISE_XS, -pe_amp, c_pe)

            # Signal: T2-decay envelope centred on eff_idx echo
            eamp = max(FSE_SIG_AMP_EFF * np.exp(-_dist * FSE_SIG_DECAY), 0.15)
            spin_echo(ax_sig, techo, eamp, SIGNAL_HW_SE)
            if i == eff_idx:
                te_pos = techo

        # Last echo sits at FSE_RF_TC + 2*n_echoes_to_show*tau_s
        t_diagram_end = FSE_RF_TC + 2 * n_echoes_to_show * tau_s
        if te_pos:
            ann_te(FSE_RF_TC, te_pos, label=f"TEeff = {TE} ms", draw_axes=(ax_sig,))
        if ETL > FSE_MAX_ECHOES_TO_SHOW:
            fig.text(0.67, 0.95,
                     f"(Showing {FSE_MAX_ECHOES_TO_SHOW} of {ETL} echoes)",
                     color="white", fontsize=FONT_TITLE, fontweight="bold",
                     ha="left", va="top")
        ann_tr(FSE_RF_TC, t_diagram_end, ax=ax_ss)
        axes[-1].set_xlim(0, t_diagram_end + FSE_XLIM_PAD)

    # ================================================================
    # ================================================================
    # FLAIR / STIR — 180° sinc inversion + Gspoiler + FSE host sequence
    # ================================================================
    elif seq in ("FLAIR", "STIR"):
        # ── Schematic TI → position of 90° excitation ────────────────────────
        ti_s  = FLAIR_TI_MIN_S + (TI / TR) * (T_TOTAL - FLAIR_TI_RANGE)
        ti_s  = max(FLAIR_TI_CLAMP_MIN, min(ti_s, FLAIR_TI_CLAMP_MAX))
        t_inv = FLAIR_T_INV
        t_exc = t_inv + ti_s
        # ── FSE host geometry (shared FSE constants, offset from t_exc) ───────
        n_echoes_to_show = min(ETL, FSE_MAX_ECHOES_TO_SHOW)
        tau_s    = FSE_MAX_ESP - FSE_RF_TC       # half-ESP = 1.15 schematic units
        esp      = 2.0 * tau_s
        eff_idx  = max(0, min(n_echoes_to_show - 1,
                              round(TE / FLAIR_NOMINAL_ESP_MS) - 1))
        _max_dist = max(eff_idx, n_echoes_to_show - 1 - eff_idx, 1)
        _fse_off  = t_exc - FSE_RF_TC            # offset applied to all FSE-origin positions
        # ── Constraint checks ─────────────────────────────────────────────────
        _esp_ms  = TE / (eff_idx + 1)
        _min_tr  = TI + ETL * _esp_ms
        _psd_warn = []
        if TR < _min_tr:
            _psd_warn.append(
                f"TR too short: minimum TR for current TI, ETL and TEeff is {int(_min_tr)} ms")
        if _esp_ms < 10.0:
            _psd_warn.append(
                "Echo spacing too short to be physically realistic\n"
                "— minimum ESP is approximately 10 ms")
        if _psd_warn:
            for _ax in axes:
                _ax.set_visible(False)
            fig.text(0.5, 0.5, "\n\n".join(f"⚠  {w}" for w in _psd_warn),
                     color="#FF4444", fontsize=6.5, ha="center", va="center",
                     multialignment="center",
                     bbox=dict(facecolor="#1a1a1a", alpha=0.9,
                               edgecolor="#FF4444", boxstyle="round,pad=0.6"))
            fig.text(0.5, 0.93, f"Pulse Sequence — {seq}", color="#CCCCCC",
                     fontsize=FONT_TITLE, ha="center", va="top", fontweight="bold")
            return fig
        # ── 180° sinc inversion pulse + Gss slice-select lobe ─────────────────
        sinc_rf(ax_rf, t_inv, 1.0, RF_HW_INV, label="180° inv")
        trap(ax_ss, t_inv - FLAIR_GSS_INV_OFFSET, GRAD_RISE_MD, GRAD_SS_INV_FLAT,
             GRAD_RISE_MD, GRAD_SS_INV_AMP, c_ss)
        # Gspoiler: positive, immediately after Gss inv lobe; larger amp and area
        # Gss inv lobe ends at t_inv + FLAIR_GSS_INV_OFFSET (symmetric about t_inv)
        _gspoil_t0 = t_inv + FLAIR_GSS_INV_OFFSET
        trap(ax_ss, _gspoil_t0, GRAD_RISE_MD, FLAIR_GSPOILER_FLAT,
             GRAD_RISE_MD, FLAIR_GSPOILER_AMP, c_ss)
        # ── Faint dotted vlines at t_inv (all rows) and t_exc ─────────────────
        for _vax in axes:
            _vax.axvline(t_inv, color="#555555", linewidth=VMARK_LINEWIDTH,
                         linestyle=":", alpha=0.5)
        _exc_peak_ymax = (1.0 - PSD_Y_MIN) / (PSD_Y_MAX - PSD_Y_MIN)
        ax_rf.axvline(t_exc, ymin=0, ymax=_exc_peak_ymax,
                      color="#555555", linewidth=VMARK_LINEWIDTH, linestyle=":", alpha=0.5)
        for _vax in (ax_ss, ax_pe, ax_fe, ax_sig):
            _vax.axvline(t_exc, color="#555555", linewidth=VMARK_LINEWIDTH,
                         linestyle=":", alpha=0.5)
        # ── FSE host: 90° sinc excitation + Gss main lobe + rephaser ──────────
        sinc_rf(ax_rf, t_exc, 1.0, RF_HW_EXCITE, label="90°")
        _flair_gss_t0        = FSE_GSS_T0 + _fse_off
        trap(ax_ss, _flair_gss_t0, GRAD_RISE, GRAD_SS_FLAT, GRAD_RISE, GRAD_SS_AMP, c_ss)
        _flair_gss_rep_start = _flair_gss_t0 + GRAD_RISE + GRAD_SS_FLAT + GRAD_RISE
        trap(ax_ss, _flair_gss_rep_start, GRAD_SS_REP_RISE, FSE_GSS_REP_FLAT,
             GRAD_SS_REP_RISE, GRAD_SS_REP_AMP, c_ss)
        # ── Gfe readout prephase ───────────────────────────────────────────────
        trap(ax_fe, FSE_GFE_T0 + _fse_off, GRAD_RISE_MD, FSE_GFE_FLAT,
             GRAD_RISE_MD, FSE_GFE_AMP, c_fe)
        # ── Echo train ────────────────────────────────────────────────────────
        te_pos = None
        for i in range(n_echoes_to_show):
            t180  = (FSE_MAX_ESP + _fse_off) + i * esp
            techo = t180 + FSE_ECHO_OFFSET * esp
            # 180° refocusing sinc pulse
            sinc_rf(ax_rf, t180, 1.0, RF_HW_180_FSE, label="180°")
            # Composite Gss crusher waveform (identical to FSE)
            t_c0 = t180 - RF_HW_180_FSE - FSE_CRUSH_SS_RISE - FSE_CRUSH_PRE_FLAT - FSE_CRUSH_RISE
            _cx = [t_c0,
                   t_c0 + FSE_CRUSH_RISE,
                   t180 - RF_HW_180_FSE - FSE_CRUSH_SS_RISE,
                   t180 - RF_HW_180_FSE,
                   t180 + RF_HW_180_FSE,
                   t180 + RF_HW_180_FSE + FSE_CRUSH_SS_RISE,
                   t180 + RF_HW_180_FSE + FSE_CRUSH_SS_RISE + FSE_CRUSH_NEG_FLAT,
                   t180 + RF_HW_180_FSE + FSE_CRUSH_SS_RISE + FSE_CRUSH_NEG_FLAT + FSE_CRUSH_RISE]
            _cy = [0, FSE_CRUSH_AMP, FSE_CRUSH_AMP, FSE_SS_AMP_180,
                   FSE_SS_AMP_180, FSE_CRUSH_AMP, FSE_CRUSH_AMP, 0]
            ax_ss.plot(_cx, _cy, color=c_ss, linewidth=WAVEFORM_LINEWIDTH)
            # Gpe blip amplitude: smallest at eff_idx (k-space centre), increasing outward
            _dist  = abs(i - eff_idx)
            pe_amp = (FSE_PE_AMP_MIN + (FSE_PE_AMP_MAX - FSE_PE_AMP_MIN) * _dist / _max_dist) * ((-1)**i)
            # Timing anchors
            t_crush_end        = (t180 + RF_HW_180_FSE + FSE_CRUSH_SS_RISE
                                  + FSE_CRUSH_NEG_FLAT + FSE_CRUSH_RISE)
            t_gfe_start        = techo - GRAD_RISE_SM - FSE_GFE_READ_FLAT / 2
            t_gfe_end          = techo + FSE_GFE_READ_FLAT / 2 + GRAD_RISE_SM
            t_next_crush_start = ((t180 + esp) - RF_HW_180_FSE
                                  - FSE_CRUSH_SS_RISE - FSE_CRUSH_PRE_FLAT
                                  - FSE_CRUSH_RISE)
            _blip_hw = GRAD_RISE_XS + FSE_GPE_BLIP_FLAT / 2
            # Gpe encode blip: centred between trailing crusher and Gfe readout
            _enc_win = t_gfe_start - t_crush_end
            if _enc_win >= 2 * _blip_hw:
                _enc_c = (t_crush_end + t_gfe_start) / 2
            else:
                _enc_c = t_crush_end + _blip_hw
            trap(ax_pe, _enc_c - _blip_hw, GRAD_RISE_XS, FSE_GPE_BLIP_FLAT,
                 GRAD_RISE_XS, pe_amp, c_pe)
            # Gfe readout: positive, centred exactly on techo
            trap(ax_fe, t_gfe_start, GRAD_RISE_SM, FSE_GFE_READ_FLAT,
                 GRAD_RISE_SM, FSE_GFE_ECHO_AMP, c_fe)
            # Gpe rewind: centred between Gfe readout end and next leading crusher
            _rew_win = t_next_crush_start - t_gfe_end
            if _rew_win >= 2 * _blip_hw:
                _rew_c = (t_gfe_end + t_next_crush_start) / 2
            else:
                _rew_c = t_gfe_end + _blip_hw
            trap(ax_pe, _rew_c - _blip_hw, GRAD_RISE_XS, FSE_GPE_BLIP_FLAT,
                 GRAD_RISE_XS, -pe_amp, c_pe)
            # Signal: T2-decay envelope, largest amplitude at eff_idx
            eamp = max(FSE_SIG_AMP_EFF * np.exp(-_dist * FSE_SIG_DECAY), 0.15)
            spin_echo(ax_sig, techo, eamp, SIGNAL_HW_SE)
            if i == eff_idx:
                te_pos = techo
        # ── Annotations and x-axis ────────────────────────────────────────────
        t_diagram_end = t_exc + 2 * n_echoes_to_show * tau_s
        if te_pos:
            ann_te(t_exc, te_pos, label=f"TEeff = {TE} ms", draw_axes=(ax_sig,))
        if ETL > FSE_MAX_ECHOES_TO_SHOW:
            fig.text(0.67, 0.95,
                     f"(Showing {FSE_MAX_ECHOES_TO_SHOW} of {ETL} echoes)",
                     color="white", fontsize=FONT_TITLE, fontweight="bold",
                     ha="left", va="top")
        ann_ti(t_inv, t_exc)
        ann_tr(t_inv, t_diagram_end, ax=ax_ss)
        axes[-1].set_xlim(0, t_diagram_end + FSE_XLIM_PAD)

    # ================================================================
    # DIR — Double Inversion Recovery: two 180° inversions + FSE host
    # ================================================================
    elif seq == "DIR":
        _ti1 = TI    # TI1 is passed as TI (sidebar sets TI = TI1 for DIR)
        _ti2 = TI2

        # ── Schematic TI1/TI2 spans, scaled proportional to real timing ──────
        _total_ti_ms = max(1, _ti1 + _ti2)
        _ti_total_s  = (DIR_TI_TOTAL_MIN_S
                        + (_total_ti_ms / TR) * (T_TOTAL - DIR_TI_TOTAL_RANGE))
        _ti_total_s  = max(DIR_TI1_MIN_S + DIR_TI2_MIN_S,
                           min(_ti_total_s, 6.8))
        _ti1_s = max(DIR_TI1_MIN_S, _ti_total_s * _ti1 / _total_ti_ms)
        _ti2_s = max(DIR_TI2_MIN_S, _ti_total_s - _ti1_s)
        t_inv1 = DIR_T_INV1
        t_inv2 = t_inv1 + _ti1_s
        t_exc  = t_inv2 + _ti2_s

        # ── Gspoiler fit: available schematic width between the fixed offset
        # after the Gss-inv lobe (FLAIR_GSS_INV_OFFSET) and the fixed offset
        # before the 90° excitation's Gss lobe (FSE_RF_TC − FSE_GSS_T0). The
        # Gspoiler is drawn at nominal (FLAIR-matching) size when it fits;
        # otherwise its width is scaled down and its amplitude scaled up to
        # preserve area (capped at DIR_GSPOILER_MAX_AMP) — so it always
        # completes and returns to zero before the 90° Gss lobe begins and
        # never overlaps it.
        _gspoil_gap_fixed = FLAIR_GSS_INV_OFFSET + (FSE_RF_TC - FSE_GSS_T0)
        _gspoil_nom_width = 2 * GRAD_RISE_MD + FLAIR_GSPOILER_FLAT
        _gspoil_avail     = _ti2_s - _gspoil_gap_fixed

        # ── FSE host geometry (same as FLAIR, offset from t_exc) ─────────────
        n_echoes_to_show = min(ETL, FSE_MAX_ECHOES_TO_SHOW)
        tau_s    = FSE_MAX_ESP - FSE_RF_TC
        esp      = 2.0 * tau_s
        eff_idx  = max(0, min(n_echoes_to_show - 1,
                              round(TE / DIR_NOMINAL_ESP_MS) - 1))
        _max_dist = max(eff_idx, n_echoes_to_show - 1 - eff_idx, 1)
        _fse_off  = t_exc - FSE_RF_TC

        # ── Constraint checks ─────────────────────────────────────────────────
        _esp_ms  = TE / (eff_idx + 1)
        _min_tr  = _ti1 + _ti2 + ETL * _esp_ms
        _psd_warn = []
        if TR < _min_tr:
            _psd_warn.append(
                f"TR too short: minimum TR for current TI1, TI2, ETL and TEeff is {int(_min_tr)} ms")
        if _esp_ms < 10.0:
            _psd_warn.append(
                "Echo spacing too short to be physically realistic\n"
                "— minimum ESP is approximately 10 ms")
        if _gspoil_avail < DIR_GSPOILER_MIN_WIDTH_S:
            _psd_warn.append(
                f"TI2 too short: the inversion spoiler gradient cannot fit between "
                f"the second inversion and the 90° excitation at TI1={_ti1} ms, "
                f"TI2={_ti2} ms — increase TI2 (or decrease TI1)")
        if _psd_warn:
            for _ax in axes:
                _ax.set_visible(False)
            fig.text(0.5, 0.5, "\n\n".join(f"⚠  {w}" for w in _psd_warn),
                     color="#FF4444", fontsize=6.5, ha="center", va="center",
                     multialignment="center",
                     bbox=dict(facecolor="#1a1a1a", alpha=0.9,
                               edgecolor="#FF4444", boxstyle="round,pad=0.6"))
            fig.text(0.5, 0.93, f"Pulse Sequence — {seq}", color="#CCCCCC",
                     fontsize=FONT_TITLE, ha="center", va="top", fontweight="bold")
            return fig

        # ── First 180°: non-selective hard rect pulse + simple non-selective Gss ──
        rect_rf(ax_rf, t_inv1, 1.0, RF_HW_INV, label="180°")
        _ns_gss_t0 = t_inv1 - DIR_NS_GSS_RISE - DIR_NS_GSS_FLAT / 2
        trap(ax_ss, _ns_gss_t0, DIR_NS_GSS_RISE, DIR_NS_GSS_FLAT,
             DIR_NS_GSS_RISE, DIR_NS_GSS_AMP, c_ss)

        # ── Second 180°: slice-selective sinc + Gss SS lobe + Gspoiler ───────
        # Gss lobe is identical to the FLAIR inversion block. The Gspoiler
        # matches FLAIR's size when it fits in the available TI2 gap;
        # otherwise it is compressed (area-preserving, amplitude-capped) per
        # the fit check above so it never overlaps the 90° Gss lobe.
        sinc_rf(ax_rf, t_inv2, 1.0, RF_HW_INV, label="180°")
        trap(ax_ss, t_inv2 - FLAIR_GSS_INV_OFFSET, GRAD_RISE_MD, GRAD_SS_INV_FLAT,
             GRAD_RISE_MD, GRAD_SS_INV_AMP, c_ss)
        _gspoil_t0 = t_inv2 + FLAIR_GSS_INV_OFFSET
        if _gspoil_avail < _gspoil_nom_width:
            _gspoil_scale = _gspoil_avail / _gspoil_nom_width
            _gspoil_rise  = GRAD_RISE_MD * _gspoil_scale
            _gspoil_flat  = FLAIR_GSPOILER_FLAT * _gspoil_scale
            _gspoil_amp   = min(FLAIR_GSPOILER_AMP / _gspoil_scale, DIR_GSPOILER_MAX_AMP)
        else:
            _gspoil_rise, _gspoil_flat, _gspoil_amp = (
                GRAD_RISE_MD, FLAIR_GSPOILER_FLAT, FLAIR_GSPOILER_AMP)
        trap(ax_ss, _gspoil_t0, _gspoil_rise, _gspoil_flat,
             _gspoil_rise, _gspoil_amp, c_ss)

        # ── Faint dotted vlines at t_inv1, t_inv2 (all rows) and t_exc ───────
        for _tv in (t_inv1, t_inv2):
            for _vax in axes:
                _vax.axvline(_tv, color="#555555", linewidth=VMARK_LINEWIDTH,
                             linestyle=":", alpha=0.5)
        _exc_peak_ymax = (1.0 - PSD_Y_MIN) / (PSD_Y_MAX - PSD_Y_MIN)
        ax_rf.axvline(t_exc, ymin=0, ymax=_exc_peak_ymax,
                      color="#555555", linewidth=VMARK_LINEWIDTH, linestyle=":", alpha=0.5)
        for _vax in (ax_ss, ax_pe, ax_fe, ax_sig):
            _vax.axvline(t_exc, color="#555555", linewidth=VMARK_LINEWIDTH,
                         linestyle=":", alpha=0.5)

        # ── FSE host: 90° sinc + Gss main lobe + rephaser (identical to FLAIR) ──
        sinc_rf(ax_rf, t_exc, 1.0, RF_HW_EXCITE, label="90°")
        _dir_gss_t0 = FSE_GSS_T0 + _fse_off
        trap(ax_ss, _dir_gss_t0, GRAD_RISE, GRAD_SS_FLAT, GRAD_RISE, GRAD_SS_AMP, c_ss)
        _dir_gss_rep_start = _dir_gss_t0 + GRAD_RISE + GRAD_SS_FLAT + GRAD_RISE
        trap(ax_ss, _dir_gss_rep_start, GRAD_SS_REP_RISE, FSE_GSS_REP_FLAT,
             GRAD_SS_REP_RISE, GRAD_SS_REP_AMP, c_ss)

        # ── Gfe readout prephase ───────────────────────────────────────────────
        trap(ax_fe, FSE_GFE_T0 + _fse_off, GRAD_RISE_MD, FSE_GFE_FLAT,
             GRAD_RISE_MD, FSE_GFE_AMP, c_fe)

        # ── Echo train (identical to FLAIR) ───────────────────────────────────
        te_pos = None
        for i in range(n_echoes_to_show):
            t180  = (FSE_MAX_ESP + _fse_off) + i * esp
            techo = t180 + FSE_ECHO_OFFSET * esp
            sinc_rf(ax_rf, t180, 1.0, RF_HW_180_FSE, label="180°")
            # Composite Gss crusher (identical to FSE / FLAIR)
            t_c0 = t180 - RF_HW_180_FSE - FSE_CRUSH_SS_RISE - FSE_CRUSH_PRE_FLAT - FSE_CRUSH_RISE
            _cx = [t_c0,
                   t_c0 + FSE_CRUSH_RISE,
                   t180 - RF_HW_180_FSE - FSE_CRUSH_SS_RISE,
                   t180 - RF_HW_180_FSE,
                   t180 + RF_HW_180_FSE,
                   t180 + RF_HW_180_FSE + FSE_CRUSH_SS_RISE,
                   t180 + RF_HW_180_FSE + FSE_CRUSH_SS_RISE + FSE_CRUSH_NEG_FLAT,
                   t180 + RF_HW_180_FSE + FSE_CRUSH_SS_RISE + FSE_CRUSH_NEG_FLAT + FSE_CRUSH_RISE]
            _cy = [0, FSE_CRUSH_AMP, FSE_CRUSH_AMP, FSE_SS_AMP_180,
                   FSE_SS_AMP_180, FSE_CRUSH_AMP, FSE_CRUSH_AMP, 0]
            ax_ss.plot(_cx, _cy, color=c_ss, linewidth=WAVEFORM_LINEWIDTH)
            # Gpe blip: smallest at eff_idx, increasing outward
            _dist  = abs(i - eff_idx)
            pe_amp = (FSE_PE_AMP_MIN + (FSE_PE_AMP_MAX - FSE_PE_AMP_MIN)
                      * _dist / _max_dist) * ((-1)**i)
            # Timing anchors
            t_crush_end        = (t180 + RF_HW_180_FSE + FSE_CRUSH_SS_RISE
                                  + FSE_CRUSH_NEG_FLAT + FSE_CRUSH_RISE)
            t_gfe_start        = techo - GRAD_RISE_SM - FSE_GFE_READ_FLAT / 2
            t_gfe_end          = techo + FSE_GFE_READ_FLAT / 2 + GRAD_RISE_SM
            t_next_crush_start = ((t180 + esp) - RF_HW_180_FSE
                                  - FSE_CRUSH_SS_RISE - FSE_CRUSH_PRE_FLAT
                                  - FSE_CRUSH_RISE)
            _blip_hw = GRAD_RISE_XS + FSE_GPE_BLIP_FLAT / 2
            _enc_win = t_gfe_start - t_crush_end
            _enc_c   = ((t_crush_end + t_gfe_start) / 2 if _enc_win >= 2 * _blip_hw
                        else t_crush_end + _blip_hw)
            trap(ax_pe, _enc_c - _blip_hw, GRAD_RISE_XS, FSE_GPE_BLIP_FLAT,
                 GRAD_RISE_XS, pe_amp, c_pe)
            trap(ax_fe, t_gfe_start, GRAD_RISE_SM, FSE_GFE_READ_FLAT,
                 GRAD_RISE_SM, FSE_GFE_ECHO_AMP, c_fe)
            _rew_win = t_next_crush_start - t_gfe_end
            _rew_c   = ((t_gfe_end + t_next_crush_start) / 2 if _rew_win >= 2 * _blip_hw
                        else t_gfe_end + _blip_hw)
            trap(ax_pe, _rew_c - _blip_hw, GRAD_RISE_XS, FSE_GPE_BLIP_FLAT,
                 GRAD_RISE_XS, -pe_amp, c_pe)
            eamp = max(FSE_SIG_AMP_EFF * np.exp(-_dist * FSE_SIG_DECAY), 0.15)
            spin_echo(ax_sig, techo, eamp, SIGNAL_HW_SE)
            if i == eff_idx:
                te_pos = techo

        # ── Annotations ──────────────────────────────────────────────────────
        t_diagram_end = t_exc + 2 * n_echoes_to_show * tau_s
        if te_pos:
            ann_te(t_exc, te_pos, label=f"TEeff = {TE} ms", draw_axes=(ax_sig,))
        if ETL > FSE_MAX_ECHOES_TO_SHOW:
            fig.text(0.67, 0.95,
                     f"(Showing {FSE_MAX_ECHOES_TO_SHOW} of {ETL} echoes)",
                     color="white", fontsize=FONT_TITLE, fontweight="bold",
                     ha="left", va="top")

        # TI1 arrow: centre of first inversion → centre of second inversion
        for _vax_ti in (ax_ss, ax_sig):
            _vax_ti.annotate("", xy=(t_inv2, ANN_TI_Y), xytext=(t_inv1, ANN_TI_Y),
                             arrowprops=dict(arrowstyle="<->", color="#FFAA44",
                                             lw=ARROW_LINEWIDTH))
        ax_sig.text((t_inv1 + t_inv2) / 2, ANN_TI_Y - ANN_LABEL_OFFSET,
                    f"TI1 = {_ti1} ms", color="#FFAA44",
                    fontsize=FONT_ANN, ha="center", va="top")

        # TI2 arrow: centre of second inversion → centre of 90° excitation
        _ti2_ann_y = ANN_TI_Y + 0.38   # slightly above TI1 arrow to avoid overlap
        for _vax_ti in (ax_ss, ax_sig):
            _vax_ti.annotate("", xy=(t_exc, _ti2_ann_y), xytext=(t_inv2, _ti2_ann_y),
                             arrowprops=dict(arrowstyle="<->", color="#FFAA44",
                                             lw=ARROW_LINEWIDTH))
        ax_sig.text((t_inv2 + t_exc) / 2, _ti2_ann_y - ANN_LABEL_OFFSET,
                    f"TI2 = {_ti2} ms", color="#FFAA44",
                    fontsize=FONT_ANN, ha="center", va="top")

        ann_tr(t_inv1, t_diagram_end, ax=ax_ss)
        axes[-1].set_xlim(0, t_diagram_end + DIR_XLIM_PAD)

    # ================================================================
    # MPRAGE — non-selective inversion + small-flip GRE-train readout
    # ================================================================
    elif seq == "MPRAGE":
        # ── Schematic TI span: inversion centre → first GRE-train RF centre ──
        ti_s   = MPRAGE_TI_MIN_S + (TI / TR) * (T_TOTAL - MPRAGE_TI_RANGE)
        ti_s   = max(MPRAGE_TI_CLAMP_MIN, min(ti_s, MPRAGE_TI_CLAMP_MAX))
        t_inv1 = MPRAGE_T_INV1
        t_exc1 = t_inv1 + ti_s

        # ── Non-selective 180° hard inversion pulse + simple non-selective Gss ──
        rect_rf(ax_rf, t_inv1, 1.0, RF_HW_INV, label="180°")
        _ns_gss_t0 = t_inv1 - DIR_NS_GSS_RISE - DIR_NS_GSS_FLAT / 2
        trap(ax_ss, _ns_gss_t0, DIR_NS_GSS_RISE, DIR_NS_GSS_FLAT,
             DIR_NS_GSS_RISE, DIR_NS_GSS_AMP, c_ss)

        # ── Faint dotted vlines at t_inv1 (all rows) and t_exc1 ───────────────
        for _vax in axes:
            _vax.axvline(t_inv1, color="#555555", linewidth=VMARK_LINEWIDTH,
                         linestyle=":", alpha=0.5)
        _exc_peak_ymax = (1.0 - PSD_Y_MIN) / (PSD_Y_MAX - PSD_Y_MIN)
        ax_rf.axvline(t_exc1, ymin=0, ymax=_exc_peak_ymax,
                      color="#555555", linewidth=VMARK_LINEWIDTH, linestyle=":", alpha=0.5)
        for _vax in (ax_ss, ax_pe, ax_fe, ax_sig):
            _vax.axvline(t_exc1, color="#555555", linewidth=VMARK_LINEWIDTH,
                         linestyle=":", alpha=0.5)

        # ── One GRE-train repetition: small-flip sinc + Gss select/rephase +
        # Gpe encode/rewind + Gfe dephase/readout — identical shape to the
        # standalone GRE case, sized compact for a repeating train ───────────
        rf_amp = MPRAGE_RF_AMP_MIN + MPRAGE_RF_AMP_PER_DEG * max(0, FA - 5)

        def gre_rep(t0, pe_amp):
            gss_t0       = t0 - MPRAGE_REP_RF_HW - MPRAGE_REP_GSS_RISE
            gss_ss_end   = gss_t0 + MPRAGE_REP_GSS_RISE + MPRAGE_REP_GSS_FLAT + MPRAGE_REP_GSS_RISE
            gss_rep_end  = (gss_ss_end + MPRAGE_REP_GSS_REP_RISE + MPRAGE_REP_GSS_REP_FLAT
                            + MPRAGE_REP_GSS_REP_RISE)
            gpe_hw       = MPRAGE_REP_GPE_RISE + MPRAGE_REP_GPE_FLAT / 2
            gpe_enc_c    = gss_ss_end + gpe_hw
            prep_w       = 2 * MPRAGE_REP_GFE_DEP_RISE + MPRAGE_REP_GFE_DEP_FLAT
            gfe_ro_hw    = MPRAGE_REP_GFE_RO_RISE + MPRAGE_REP_GFE_RO_FLAT / 2
            te_c         = gss_rep_end + prep_w + gfe_ro_hw
            gfe_ro_start = te_c - gfe_ro_hw
            gpe_rew_c    = 2 * te_c - gpe_enc_c
            gpe_rew_t0   = gpe_rew_c - gpe_hw

            sinc_rf(ax_rf, t0, rf_amp, MPRAGE_REP_RF_HW, label=f"{FA}°")
            trap(ax_ss, gss_t0, MPRAGE_REP_GSS_RISE, MPRAGE_REP_GSS_FLAT,
                 MPRAGE_REP_GSS_RISE, GRE_GSS_REF_AMP, c_ss)
            trap(ax_ss, gss_ss_end, MPRAGE_REP_GSS_REP_RISE, MPRAGE_REP_GSS_REP_FLAT,
                 MPRAGE_REP_GSS_REP_RISE, MPRAGE_REP_GSS_REP_AMP, c_ss)
            trap(ax_pe, gpe_enc_c - gpe_hw, MPRAGE_REP_GPE_RISE, MPRAGE_REP_GPE_FLAT,
                 MPRAGE_REP_GPE_RISE, pe_amp, c_pe)
            trap(ax_pe, gpe_rew_t0, MPRAGE_REP_GPE_RISE, MPRAGE_REP_GPE_FLAT,
                 MPRAGE_REP_GPE_RISE, -pe_amp, c_pe)
            trap(ax_fe, gss_rep_end, MPRAGE_REP_GFE_DEP_RISE, MPRAGE_REP_GFE_DEP_FLAT,
                 MPRAGE_REP_GFE_DEP_RISE, MPRAGE_REP_GFE_DEP_AMP, c_fe)
            trap(ax_fe, gfe_ro_start, MPRAGE_REP_GFE_RO_RISE, MPRAGE_REP_GFE_RO_FLAT,
                 MPRAGE_REP_GFE_RO_RISE, MPRAGE_REP_GFE_RO_AMP, c_fe)
            rep_end = max(gfe_ro_start + 2 * MPRAGE_REP_GFE_RO_RISE + MPRAGE_REP_GFE_RO_FLAT,
                          gpe_rew_t0 + 2 * MPRAGE_REP_GPE_RISE + MPRAGE_REP_GPE_FLAT)
            return te_c, rep_end

        # ── Sequential (linear) k-space fill: Gpe steps monotonically across
        # the readout train — 3 repetitions shown near the start of the train,
        # then the final repetition at the opposite k-space edge ─────────────
        _n_steps = MPRAGE_N_SHOW + 1
        echo_times   = []
        rep_end_last = t_exc1
        for i in range(MPRAGE_N_SHOW):
            t0 = t_exc1 + i * MPRAGE_REP_SPACING
            pe_amp = -MPRAGE_PE_AMP_MAX + 2 * MPRAGE_PE_AMP_MAX * i / (_n_steps - 1)
            te_c, rep_end_last = gre_rep(t0, pe_amp)
            echo_times.append(te_c)

        # "..." continuation indicator
        _dots_x = rep_end_last + MPRAGE_DOTS_GAP / 2
        for _ax_d, _col_d in ((ax_rf, c_rf), (ax_ss, c_ss), (ax_pe, c_pe),
                              (ax_fe, c_fe), (ax_sig, "#FFFFFF")):
            _ax_d.text(_dots_x, 0.0, "· · ·", color=_col_d,
                       fontsize=7, ha="center", va="center", fontstyle="italic")

        # Final repetition — last partition-encode line acquired
        t0_final = rep_end_last + MPRAGE_DOTS_GAP + MPRAGE_REP_RF_HW + MPRAGE_REP_GSS_RISE
        te_final, rep_end_final = gre_rep(t0_final, MPRAGE_PE_AMP_MAX)
        echo_times.append(te_final)

        # ── Signal: gradient echoes under each GRE readout — T1-weighted
        # amplitude reflecting Mz recovery after the inversion ────────────────
        _tau_sig = max(ti_s, 0.5)
        for _te in echo_times:
            _sig_amp = MPRAGE_SIG_MIN + (MPRAGE_SIG_MAX - MPRAGE_SIG_MIN) * (
                1 - np.exp(-(_te - t_inv1) / _tau_sig))
            grad_echo(ax_sig, _te, min(_sig_amp, MPRAGE_SIG_MAX), MPRAGE_SIG_HW)

        # ── Annotations ──────────────────────────────────────────────────────
        # TI: inversion pulse centre → first GRE pulse centre
        ann_ti(t_inv1, t_exc1)

        # TR_readout: one GRE-train repetition (first RF centre → second RF centre)
        _tr_ro_end = t_exc1 + MPRAGE_REP_SPACING
        ax_pe.annotate("", xy=(_tr_ro_end, ANN_TR_Y), xytext=(t_exc1, ANN_TR_Y),
                       arrowprops=dict(arrowstyle="<->", color=c_pe, lw=ARROW_LINEWIDTH))
        ax_pe.text((t_exc1 + _tr_ro_end) / 2, ANN_TR_Y + ANN_TR_LABEL_OFF,
                   f"TR_readout = {MPRAGE_TR_READOUT} ms", color=c_pe,
                   fontsize=FONT_ANN, ha="center", va="bottom")

        # TR_total: inversion pulse centre → next inversion pulse centre
        t_inv2 = rep_end_final + MPRAGE_RECOVERY_GAP_S
        ann_tr(t_inv1, t_inv2, ax=ax_ss)
        for _vax in axes:
            _vax.axvline(t_inv2, color="#555555", linewidth=VMARK_LINEWIDTH,
                         linestyle=":", alpha=0.5)
        rect_rf(ax_rf, t_inv2, GRE_GHOST_AMP, RF_HW_INV, col=c_rf + "55")
        trap(ax_ss, t_inv2 - DIR_NS_GSS_RISE - DIR_NS_GSS_FLAT / 2, DIR_NS_GSS_RISE,
             DIR_NS_GSS_FLAT, DIR_NS_GSS_RISE, DIR_NS_GSS_AMP, c_ss + "44")

        if Npartitions > MPRAGE_N_SHOW + 1:
            fig.text(0.67, 0.95,
                     f"(Showing 4 of {Npartitions} partition-encode readouts)",
                     color="white", fontsize=FONT_TITLE, fontweight="bold",
                     ha="left", va="top")

        axes[-1].set_xlim(0, t_inv2 + MPRAGE_XLIM_PAD)

    # ================================================================
    # bSSFP  — 3 TR repetitions, all gradient moments balanced to zero
    # ================================================================
    elif seq == "bSSFP":
        tr_w = T_TOTAL / N_BSSFP_REPS

        for i in range(N_BSSFP_REPS):
            t0   = i * tr_w
            tc   = t0 + tr_w * BSSFP_RF_OFF          # RF centre
            te_c = t0 + tr_w * BSSFP_SIG_OFF          # echo centre: TE = TR/2 after RF
            sign = (-1) ** i
            amp  = BSSFP_TR1_AMP if i == 0 else 1.0
            fa_lbl = f"+{FA}\u00b0" if sign > 0 else f"\u2212{FA}\u00b0"

            # RF — sinc, alternating phase (+FA, −FA, +FA, …)
            sinc_rf(ax_rf, tc, sign * amp, RF_HW_BSSFP, label=fa_lbl)

            # ── Gss: pre-phaser(−) → slice-select(+) → re-phaser(−) ──────────
            #   prephaser area = SS_area/2, rephaser area = SS_area/2
            #   net moment: −½ + 1 − ½ = 0
            t_ss_c0    = tc - GRAD_RISE_XS - BSSFP_GSS_MAIN_FLAT / 2   # start of SS main lobe
            t_ss_end   = tc + GRAD_RISE_XS + BSSFP_GSS_MAIN_FLAT / 2   # end   of SS main lobe
            dur_pre_ss = 2 * GRAD_RISE_XS + BSSFP_GSS_HALFPRE_FLAT
            t_ss_pre0  = t_ss_c0 - dur_pre_ss                            # start of prephaser
            trap(ax_ss, t_ss_pre0, GRAD_RISE_XS, BSSFP_GSS_HALFPRE_FLAT, GRAD_RISE_XS, -BSSFP_SS_AMP, c_ss)
            trap(ax_ss, t_ss_c0,   GRAD_RISE_XS, BSSFP_GSS_MAIN_FLAT,    GRAD_RISE_XS,  BSSFP_SS_AMP, c_ss)
            trap(ax_ss, t_ss_end,  GRAD_RISE_XS, BSSFP_GSS_HALFPRE_FLAT, GRAD_RISE_XS, -BSSFP_SS_AMP, c_ss)

            # ── Gfe: neg(−) simultaneous with Gss pre-phaser,
            #         neg(−) simultaneous with Gss re-phaser,
            #         readout(+) centred at TE = TR/2
            #   area_neg1 + area_neg2 = readout_area → net Gfe moment = 0
            t_ro_c0 = te_c - GRAD_RISE_XS - BSSFP_GFE_RO_FLAT / 2    # start of readout lobe
            trap(ax_fe, t_ss_pre0, GRAD_RISE_XS, BSSFP_GSS_HALFPRE_FLAT, GRAD_RISE_XS, -BSSFP_FE_AMP,     c_fe)
            trap(ax_fe, t_ss_end,  GRAD_RISE_XS, BSSFP_GSS_HALFPRE_FLAT, GRAD_RISE_XS, -BSSFP_FE_AMP,     c_fe)
            trap(ax_fe, t_ro_c0,   GRAD_RISE_XS, BSSFP_GFE_RO_FLAT,      GRAD_RISE_XS,  GRAD_FE_READ_AMP, c_fe)

            # ── Gpe: encode(−) → rewind(+) bipolar pair (net moment = 0) ─────
            #   Timing (rise, flat, fall) matches Gss pre/rephaser and the
            #   simultaneous Gfe lobes exactly; only amplitude differs.
            trap(ax_pe, t_ss_pre0, GRAD_RISE_XS, BSSFP_GSS_HALFPRE_FLAT, GRAD_RISE_XS, -BSSFP_PE_AMP * sign, c_pe)
            trap(ax_pe, t_ss_end,  GRAD_RISE_XS, BSSFP_GSS_HALFPRE_FLAT, GRAD_RISE_XS,  BSSFP_PE_AMP * sign, c_pe)

            # Signal: gradient echo centred at TE = TR/2
            grad_echo(ax_sig, te_c, amp * BSSFP_SIG_SCALE, SIGNAL_HW_BSSFP)

            # TR boundary dashed marker
            ax_rf.axvline(t0, color=BSSFP_VMARK_COLOR,
                          linewidth=BASELINE_LINEWIDTH, linestyle=":")

        # ── Annotations ──────────────────────────────────────────────────────
        tc0  = tr_w * BSSFP_RF_OFF               # centre of 1st RF pulse
        te_c0 = tr_w * BSSFP_SIG_OFF             # centre of 1st Gfe readout (TE = TR/2)
        tc1  = tr_w + tc0                         # centre of 2nd RF pulse

        # TR arrow: centre of 1st RF → centre of 2nd RF
        # Drawn on ax_ss (top of Gss row) so it sits between the RF and Gss rows
        # and does not overlap the flip-angle text labels in ax_rf.
        ann_tr(tc0, tc1, ax=ax_ss)

        # TE = TR/2 arrow on Gfe row: 1st RF centre → 1st readout centre
        # Label is placed ABOVE the arrow to keep it within the axes clip region.
        ax_fe.annotate("", xy=(te_c0, ANN_TE_Y), xytext=(tc0, ANN_TE_Y),
                       arrowprops=dict(arrowstyle="<->", color="#AAAAAA",
                                       lw=ARROW_LINEWIDTH))
        ax_fe.text((tc0 + te_c0) / 2, ANN_TE_Y + 0.15,
                   "TE = TR/2", color="#AAAAAA", fontsize=FONT_ANN,
                   ha="center", va="bottom")

        # Dashed 'Repeating unit' rectangle spanning all rows around 2nd TR
        _trans = mtransforms.blended_transform_factory(
            ax_rf.transData, fig.transFigure
        )
        rep_rect = mpatches.Rectangle(
            (tr_w, PSD_MARGIN_BOT), tr_w,
            PSD_MARGIN_TOP - PSD_MARGIN_BOT,
            transform=_trans, clip_on=False,
            linewidth=BSSFP_RECT_LINEWIDTH, edgecolor=BSSFP_REPEAT_COLOR,
            facecolor="none", linestyle="--", zorder=10
        )
        fig.add_artist(rep_rect)
        ax_rf.text(1.5 * tr_w, ANN_TR_Y + 0.10, "Repeating unit",
                   color=BSSFP_REPEAT_COLOR, fontsize=FONT_ANN,
                   ha="center", va="center", fontstyle="italic")

        # ×N TRs label (placed in 3rd TR period to avoid crowding)
        ax_rf.text(2.5 * tr_w, ANN_TR_Y, f"\u00d7{N_BSSFP_REPS} TRs shown",
                   color="#777777", fontsize=BSSFP_FONT_REPS, ha="center")

        axes[-1].set_xlim(0, T_TOTAL)

    # ================================================================
    # EPI (single-shot) — physically accurate gradient-echo EPI
    # ================================================================
    elif seq == "EPI (single-shot)":
        # ── Timing landmarks ─────────────────────────────────────────────────
        # EPI uses its own Gss rise/flat (shorter than FSE/GRE global constants)
        gss_end     = EPI_GSS_T0 + EPI_GSS_RISE + EPI_GSS_FLAT + EPI_GSS_RISE    # 0.70
        gss_rep_end = gss_end + EPI_GSS_REP_RISE + EPI_GSS_REP_FLAT + EPI_GSS_REP_RISE  # 0.99
        prep_end    = gss_rep_end + EPI_FE_PREP_RISE + EPI_FE_PREP_FLAT + EPI_FE_PREP_RISE  # 1.224
        esp_s       = EPI_RO_FLAT + 2 * EPI_RO_RISE  # schematic ESP = 0.29

        # ── RF: 90° sinc excitation ──────────────────────────────────────────
        sinc_rf(ax_rf, EPI_RF_TC, 1.0, RF_HW_EXCITE, label="90°")

        # ── Gss: positive SS lobe directly connected to negative half-area rephaser ──
        # EPI_GSS_RISE/FLAT are EPI-specific (shorter than global GRAD_RISE/GRAD_SS_FLAT)
        # Rephaser: same amplitude as SS lobe, flat = 0.17 → area = SS_area / 2
        trap(ax_ss, EPI_GSS_T0, EPI_GSS_RISE, EPI_GSS_FLAT, EPI_GSS_RISE, GRAD_SS_AMP, c_ss)
        trap(ax_ss, gss_end, EPI_GSS_REP_RISE, EPI_GSS_REP_FLAT,
             EPI_GSS_REP_RISE, EPI_GSS_REP_AMP, c_ss)

        # ── Gpe: large negative prephaser (ky → −ky_max), labelled Aₚ,ₚ ─────
        trap(ax_pe, gss_rep_end, EPI_PE_PREP_RISE, EPI_PE_PREP_FLAT,
             EPI_PE_PREP_RISE, EPI_PE_PREP_AMP, c_pe)
        _pe_prep_cx = gss_rep_end + EPI_PE_PREP_RISE + EPI_PE_PREP_FLAT / 2
        ax_pe.text(_pe_prep_cx, EPI_PE_PREP_AMP / 2, "Ap,p",
                   color=c_pe, fontsize=FONT_ANN - 0.5, ha="center", va="center",
                   fontstyle="italic")

        # ── Gfe: large negative prephaser (kx → −kx_max) ────────────────────
        trap(ax_fe, gss_rep_end, EPI_FE_PREP_RISE, EPI_FE_PREP_FLAT,
             EPI_FE_PREP_RISE, EPI_FE_PREP_AMP, c_fe)

        # ── Gfe: continuous zigzag readout waveform (single polyline, no gaps) ──
        # Structure: 0 → +A (rise) → flat → −A (2×rise) → flat → +A (2×rise) → …
        # Each inter-lobe ramp passes through 0 exactly once — the PE blip sits here.
        _t_zz  = [prep_end]
        _y_zz  = [0.0]
        _t_cur = prep_end
        echo_centers = []
        blip_starts  = []

        for i in range(EPI_N_SHOW):
            _amp = EPI_RO_AMP if (i % 2 == 0) else -EPI_RO_AMP
            if i == 0:
                # First lobe: initial rise from 0
                _t_zz.append(_t_cur + EPI_RO_RISE)
                _y_zz.append(_amp)
                _t_cur += EPI_RO_RISE
            # else: already at _amp from previous transition — flat begins immediately
            echo_centers.append(_t_cur + EPI_RO_FLAT / 2)
            _t_zz.append(_t_cur + EPI_RO_FLAT)
            _y_zz.append(_amp)
            _t_cur += EPI_RO_FLAT
            if i < EPI_N_SHOW - 1:
                # Ramp directly to next polarity through zero (2 × EPI_RO_RISE)
                blip_starts.append(_t_cur)
                _t_zz.append(_t_cur + 2 * EPI_RO_RISE)
                _y_zz.append(-_amp)
                _t_cur += 2 * EPI_RO_RISE
            else:
                # Fall to zero after the last explicitly shown lobe
                _t_zz.append(_t_cur + EPI_RO_RISE)
                _y_zz.append(0.0)
                _t_cur += EPI_RO_RISE

        ax_fe.plot(_t_zz, _y_zz, color=c_fe, linewidth=WAVEFORM_LINEWIDTH)

        # ── Gpe: equal positive blips, one per inter-readout transition ───────
        for _bs in blip_starts:
            trap(ax_pe, _bs, EPI_BLIP_RISE, EPI_BLIP_FLAT,
                 EPI_BLIP_RISE, EPI_BLIP_AMP, c_pe)

        # ── "..." continuation indicator ──────────────────────────────────────
        for _ax_d, _col_d in ((ax_fe, c_fe), (ax_pe, c_pe), (ax_sig, "#FFFFFF")):
            _ax_d.text(EPI_DOTS_X, 0.0, "· · ·", color=_col_d, fontsize=7,
                       ha="center", va="center", fontstyle="italic")

        # ── Final (Nth) readout — separate trap starting from 0 after "..." ───
        final_center = EPI_FINAL_RO_T0 + EPI_RO_RISE + EPI_RO_FLAT / 2
        trap(ax_fe, EPI_FINAL_RO_T0, EPI_RO_RISE, EPI_RO_FLAT,
             EPI_RO_RISE, EPI_RO_AMP, c_fe)

        # ── Signal: monotonically decreasing gradient echoes (T2* decay) ─────
        _all_centers = echo_centers + [final_center]
        _all_indices = list(range(EPI_N_SHOW)) + [epi_n_readouts - 1]
        for _ec, _ei in zip(_all_centers, _all_indices):
            _eamp = max(0.10, EPI_SIG_PEAK * np.exp(-_ei * EPI_SIG_DECAY))
            grad_echo(ax_sig, _ec, _eamp, SIGNAL_HW_GRE * EPI_SIG_HW_SCALE)

        # ── T2* decay dotted envelope above signal echoes ─────────────────────
        _tau_env  = esp_s / EPI_SIG_DECAY                        # schematic time constant
        _t_env_0  = echo_centers[0]                              # time of echo-0 centre
        _t_env_end = echo_centers[-1] + EPI_RO_FLAT / 2         # end of last shown echo
        _t_env    = np.linspace(_t_env_0, _t_env_end, 200)
        _y_env    = EPI_SIG_PEAK * np.exp(-(_t_env - _t_env_0) / _tau_env)
        ax_sig.plot(_t_env, _y_env, color="#AAAAAA", linewidth=0.5,
                    linestyle=":", alpha=0.85)

        # ── TEeff annotation: arrow from RF centre to epi_eff_idx echo ────────
        _final_idx = epi_n_readouts - 1
        if epi_eff_idx < EPI_N_SHOW:
            te_pos = echo_centers[epi_eff_idx]
        elif epi_eff_idx == _final_idx:
            te_pos = final_center
        else:
            _frac  = ((epi_eff_idx - (EPI_N_SHOW - 1))
                      / max(1, _final_idx - (EPI_N_SHOW - 1)))
            te_pos = echo_centers[-1] + _frac * (final_center - echo_centers[-1])

        vmark(EPI_RF_TC, c_rf)
        vmark(te_pos, "#AAAAAA")
        ax_sig.annotate("", xy=(te_pos, ANN_TE_Y), xytext=(EPI_RF_TC, ANN_TE_Y),
                        arrowprops=dict(arrowstyle="<->", color="#AAAAAA",
                                        lw=ARROW_LINEWIDTH))
        ax_sig.text((EPI_RF_TC + te_pos) / 2, ANN_TE_Y - ANN_LABEL_OFFSET,
                    f"TEeff = {TE} ms", color="#AAAAAA", fontsize=FONT_ANN,
                    ha="center", va="top")

        # ── TR arrow drawn on Gss row (between RF and Gss) ───────────────────
        _epi_tr_end = EPI_FINAL_RO_T0 + EPI_RO_RISE + EPI_RO_FLAT + EPI_RO_RISE + 0.10
        ann_tr(EPI_RF_TC, _epi_tr_end, ax=ax_ss)

        # ── N/2 ghosting annotation ──────────────────────────────────────────
        fig.text(0.50, 0.055,
                 "N/2 ghost: phase inconsistencies between alternating ±Gfe readout directions "
                 "shift the ghost N/2 pixels in the phase-encode direction",
                 color="#FFAA44", fontsize=4.5, ha="center", va="bottom", style="italic")

        axes[-1].set_xlim(0, EPI_XLIM_RIGHT)

    axes[-1].set_xticks([])
    fig.text(0.5, 0.01, "Time  →", color="#555555",
             fontsize=FONT_FOOTER, ha="center")
    fig.text(0.5, 0.95, f"Pulse Sequence — {seq}", color="#CCCCCC",
             fontsize=FONT_TITLE, ha="center", va="top", fontweight="bold")
    return fig


# ---------------------------------------------------------------------------
# Brain phantom — BrainWeb with synthetic fallback
# ---------------------------------------------------------------------------
# Loads the BrainWeb subject 04 crisp tissue segmentation and caches the full
# 3-D labeled volume in memory. A separate cached function extracts and resizes
# the requested slice for display. If BrainWeb is unavailable, a simple
# concentric-ellipse phantom is used instead.

def _synthetic_phantom(size=256):
    phantom = np.zeros((size, size), dtype=int)
    cy, cx = size // 2, size // 2
    Y, X = np.ogrid[:size, :size]
    s = size / 2
    def ellipse(rx, ry, ox=0, oy=0):
        return ((X - cx - ox)**2 / rx**2 + (Y - cy - oy)**2 / ry**2) <= 1
    phantom[ellipse(0.44*s, 0.48*s)]              = 3
    phantom[ellipse(0.40*s, 0.44*s)]              = 2
    phantom[ellipse(0.32*s, 0.36*s)]              = 1
    phantom[ellipse(0.07*s, 0.13*s, -0.11*s, 0)] = 3
    phantom[ellipse(0.07*s, 0.13*s, +0.11*s, 0)] = 3
    return phantom

@st.cache_data
def get_phantom_volume():
    """Returns the full labeled volume as int8: WM=1, GM=2, CSF=3, else 0."""
    try:
        import brainweb
        fname = brainweb.get_file(
            "subject_04.bin.gz",
            brainweb.LINKS["subject_04.bin.gz"]
        )
        vol = brainweb.load_file(fname)         # shape (362, 434, 362), uint16
        labeled = np.zeros(vol.shape, dtype=np.int8)
        labeled[vol == brainweb.Act.whiteMatter] = 1
        labeled[vol == brainweb.Act.greyMatter]  = 2
        labeled[vol == brainweb.Act.csf]         = 3
        return labeled
    except Exception:
        return None

@st.cache_data
def get_phantom_slice(plane, idx, size=256):
    from skimage.transform import resize
    vol = get_phantom_volume()
    if vol is None:
        return _synthetic_phantom(size)
    if plane == "Sagittal":
        slc = vol[:, :, idx]
    elif plane == "Coronal":
        slc = vol[:, idx, :]
    else:  # Axial
        slc = vol[idx, :, :]
    return resize(slc.astype(float), (size, size), order=0,
                  preserve_range=True, anti_aliasing=False).astype(int)

# Applies a consistent dark theme to every matplotlib axes object so all
# plots match the dark Streamlit background.
def style_ax(ax):
    ax.set_facecolor("#1e1e1e")
    ax.tick_params(colors="white", labelsize=7)
    for spine in ax.spines.values():
        spine.set_edgecolor("#555")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    ax.title.set_color("white")

def _parameter_snapshot(
        protocol_name, seq, field_strength, TR, TE, TI, TI1, TI2, FA, b,
        NEX, BW, freq_matrix, phase_matrix, FOV_read, FOV_phase, slice_mm,
        Npartitions, fat_sat, pf_fraction, scan_sec):
    """Capture teaching-relevant state for local change explanations."""
    return {
        "protocol_name": protocol_name,
        "seq": seq,
        "field_strength": field_strength,
        "TR": TR,
        "TE": TE,
        "TI": TI,
        "TI1": TI1,
        "TI2": TI2,
        "FA": FA,
        "b": b,
        "NEX": NEX,
        "BW": BW,
        "freq_matrix": freq_matrix,
        "phase_matrix": phase_matrix,
        "FOV_read": FOV_read,
        "FOV_phase": FOV_phase,
        "slice_mm": slice_mm,
        "Npartitions": Npartitions,
        "fat_sat": fat_sat,
        "pf_fraction": pf_fraction,
        "scan_sec": scan_sec,
    }

def _changed(prev, curr, key):
    return prev is not None and prev.get(key) != curr.get(key)

def explain_parameter_changes(prev, curr):
    """Return short deterministic teaching notes for changed MRI parameters."""
    if prev is None:
        return []

    notes = []

    if _changed(prev, curr, "protocol_name"):
        notes.append(
            f"Protocol preset changed to {curr['protocol_name']}. The controls now start from a typical teaching example, but every parameter remains editable."
        )

    if _changed(prev, curr, "seq"):
        notes.append(
            f"Sequence changed from {prev['seq']} to {curr['seq']}. Signal weighting and the available timing controls now follow the selected pulse sequence."
        )

    if _changed(prev, curr, "field_strength"):
        notes.append(
            f"Field strength changed from {prev['field_strength']} to {curr['field_strength']}. Tissue T1/T2/T2* values and the simulated SNR scale update with B0."
        )

    if _changed(prev, curr, "TR"):
        if curr["TR"] > prev["TR"]:
            notes.append(
                f"TR increased from {prev['TR']} to {curr['TR']} ms. Longer TR allows more T1 recovery, usually reducing T1 weighting and increasing signal."
            )
        else:
            notes.append(
                f"TR decreased from {prev['TR']} to {curr['TR']} ms. Shorter TR increases T1 saturation, often strengthening T1 weighting but lowering signal."
            )

    if _changed(prev, curr, "TE"):
        te_label = "TEeff" if curr["seq"] in ("FSE", "FLAIR", "STIR", "DIR", "EPI (single-shot)") else "TE"
        decay = "T2*" if curr["seq"] in ("GRE", "EPI (single-shot)", "MPRAGE") else "T2"
        if curr["TE"] > prev["TE"]:
            notes.append(
                f"{te_label} increased from {prev['TE']} to {curr['TE']} ms. Longer echo time allows more {decay} decay, lowering signal and strengthening {decay}-weighted contrast."
            )
        else:
            notes.append(
                f"{te_label} decreased from {prev['TE']} to {curr['TE']} ms. Shorter echo time preserves signal and reduces {decay}-decay weighting."
            )

    if _changed(prev, curr, "TI") and curr["seq"] in ("FLAIR", "STIR", "MPRAGE"):
        target = "CSF" if curr["seq"] in ("FLAIR", "MPRAGE") else "fat"
        notes.append(
            f"TI changed from {prev['TI']} to {curr['TI']} ms. In inversion recovery imaging, TI controls which tissue is near its null point; here the main teaching target is {target} suppression."
        )

    if (_changed(prev, curr, "TI1") or _changed(prev, curr, "TI2")) and curr["seq"] == "DIR":
        notes.append(
            "DIR inversion timing changed. TI1 and TI2 jointly shape double-inversion contrast and determine which tissue signal is suppressed."
        )

    if _changed(prev, curr, "FA") and curr["seq"] in ("GRE", "bSSFP", "MPRAGE"):
        if curr["FA"] > prev["FA"]:
            notes.append(
                f"Flip angle increased from {prev['FA']}° to {curr['FA']}°. Larger flip angles increase transverse signal up to an optimum, but also increase saturation in short-TR sequences."
            )
        else:
            notes.append(
                f"Flip angle decreased from {prev['FA']}° to {curr['FA']}°. Smaller flip angles reduce saturation and can improve steady-state efficiency, but may lower immediate signal."
            )

    if _changed(prev, curr, "b") and curr["seq"] == "DWI":
        if curr["b"] > prev["b"]:
            notes.append(
                f"b-value increased from {prev['b']} to {curr['b']} s/mm². Diffusion attenuation is stronger, so freely diffusing tissues such as CSF lose more signal."
            )
        else:
            notes.append(
                f"b-value decreased from {prev['b']} to {curr['b']} s/mm². Diffusion weighting is reduced and the image moves closer to T2-weighted signal."
            )

    if _changed(prev, curr, "NEX"):
        ratio = np.sqrt(curr["NEX"] / prev["NEX"])
        if curr["NEX"] > prev["NEX"]:
            notes.append(
                f"NEX increased from {prev['NEX']} to {curr['NEX']}. SNR improves by about {ratio:.2f}×, while scan time increases in proportion to NEX."
            )
        else:
            notes.append(
                f"NEX decreased from {prev['NEX']} to {curr['NEX']}. Scan time falls, but SNR drops by about {1/ratio:.2f}×."
            )

    if _changed(prev, curr, "BW"):
        if curr["BW"] > prev["BW"]:
            notes.append(
                f"Bandwidth increased from {prev['BW']} to {curr['BW']} Hz/px. Higher bandwidth reduces chemical shift and distortion sensitivity, but increases noise."
            )
        else:
            notes.append(
                f"Bandwidth decreased from {prev['BW']} to {curr['BW']} Hz/px. Lower bandwidth improves SNR, but increases chemical shift and distortion sensitivity."
            )

    if _changed(prev, curr, "freq_matrix") or _changed(prev, curr, "phase_matrix"):
        notes.append(
            f"Matrix changed to {curr['freq_matrix']}×{curr['phase_matrix']}. Larger matrices improve in-plane resolution but reduce voxel size and can lower SNR; phase matrix also affects scan time."
        )

    if _changed(prev, curr, "FOV_read") or _changed(prev, curr, "FOV_phase"):
        notes.append(
            f"FOV changed to {curr['FOV_read']}×{curr['FOV_phase']} mm. Smaller FOV improves pixel size for a fixed matrix but can increase wraparound risk in the phase direction."
        )

    if _changed(prev, curr, "slice_mm"):
        if curr["slice_mm"] > prev["slice_mm"]:
            notes.append(
                f"Slice thickness increased from {prev['slice_mm']} to {curr['slice_mm']} mm. Thicker slices improve SNR but increase partial-volume blurring."
            )
        else:
            notes.append(
                f"Slice thickness decreased from {prev['slice_mm']} to {curr['slice_mm']} mm. Thinner slices reduce partial volume but lower SNR."
            )

    if _changed(prev, curr, "Npartitions") and curr["seq"] == "MPRAGE":
        notes.append(
            f"Partitions changed to {curr['Npartitions']}. More partitions make thinner 3D partitions for the fixed slab, improving through-plane sampling but reducing voxel volume and SNR."
        )

    if _changed(prev, curr, "fat_sat"):
        state = "enabled" if curr["fat_sat"] else "disabled"
        notes.append(
            f"Fat saturation {state}. Fat signal is selectively suppressed, which changes fat contrast and can make fluid-sensitive sequences easier to interpret."
        )

    if _changed(prev, curr, "pf_fraction"):
        if curr["pf_fraction"] < prev["pf_fraction"]:
            notes.append(
                f"Partial Fourier was reduced to {curr['pf_fraction']:.3f}. Fewer phase lines shorten scan time but reduce SNR and can increase ringing or reconstruction artifacts."
            )
        else:
            notes.append(
                f"Partial Fourier was increased to {curr['pf_fraction']:.3f}. More k-space is acquired, improving SNR/artifact behavior at the cost of scan time."
            )

    if _changed(prev, curr, "scan_sec") and prev["scan_sec"] > 0:
        pct = 100 * (curr["scan_sec"] - prev["scan_sec"]) / prev["scan_sec"]
        if abs(pct) >= 10:
            notes.append(
                f"Estimated scan time changed by {pct:+.0f}%. Timing, phase encodes, NEX, partitions, ETL, and partial Fourier all contribute to this estimate."
            )

    return notes[:5]

# ---------------------------------------------------------------------------
# Session state — window/level auto-update on parameter change
# ---------------------------------------------------------------------------
# wl_window and wl_level are stored in session state so they can be updated
# before the keyed sliders render. params_sig is a tuple of all parameters
# that affect signal intensity; when it changes, W/L is recomputed. When
# only the W/L sliders themselves move, params_sig is unchanged and the
# user's manual value is preserved.
if "params_sig"    not in st.session_state:
    st.session_state.params_sig    = None
if "wl_window"     not in st.session_state:
    st.session_state.wl_window     = 1.0
if "wl_level"      not in st.session_state:
    st.session_state.wl_level      = 0.5
if "wl_opt_window" not in st.session_state:
    st.session_state.wl_opt_window = 1.0
if "wl_opt_level"  not in st.session_state:
    st.session_state.wl_opt_level  = 0.5
# K-space reconstructed image independent W/L controls
if "ks_reco_window" not in st.session_state:
    st.session_state.ks_reco_window = 1.0
if "ks_reco_level"  not in st.session_state:
    st.session_state.ks_reco_level  = 0.5
if "ga4_prev_seq"   not in st.session_state:
    st.session_state.ga4_prev_seq   = None
if "ga4_prev_slice" not in st.session_state:
    st.session_state.ga4_prev_slice = (None, None, None)

# Events queued during sidebar rendering and fired once in the main area.
_ga4_events = []

# ---------------------------------------------------------------------------
# Sidebar — parameters
# ---------------------------------------------------------------------------
# Two sections: MRI Parameters (sequence, timing, acquisition) and Slice
# Position (x/y/z sliders that drive the crosshair and plane selection).
with st.sidebar:
    st.markdown("### MRI Parameters")

    st.markdown("**Protocol Preset**")
    protocol_name = st.selectbox(
        "Protocol Preset",
        list(PROTOCOL_PRESETS.keys()),
        index=0,
        label_visibility="collapsed",
        key="protocol_preset",
    )
    _preset = PROTOCOL_PRESETS[protocol_name]

    if st.session_state.get("_last_protocol_preset") != protocol_name:
        st.session_state.field_strength = _preset.get("field_strength", "3.0T")
        st.session_state.sequence = _preset.get("seq", "FSE")
        if "n2_ghost_pct" in _preset:
            st.session_state.n2_ghost_slider = _preset["n2_ghost_pct"]
        st.session_state._last_protocol_preset = protocol_name

    st.markdown("**Field Strength**")
    _field_options = ["0.064T", "0.5T", "1.0T", "1.5T", "3.0T", "4.7T", "7.0T", "9.4T", "11.7T"]
    field_strength = st.radio(
        "",
        _field_options,
        index=_field_options.index(_preset.get("field_strength", "3.0T")),
        horizontal=True,
        key="field_strength",
    )
    _tr_max = TR_MAX_BY_FIELD[field_strength]

    st.markdown("**Tissue Properties**")
    _fs_ref_tissues = FIELD_STRENGTH_TISSUES[field_strength]
    _tissue_ref_rows = "\n".join(
        f"| {_name} | {_fs_ref_tissues[_name]['T1']} | {_fs_ref_tissues[_name]['T2']} |"
        for _name in ("WM", "GM", "CSF", "Fat")
    )
    st.markdown(
        f"| Tissue | T1 (ms) | T2 (ms) |\n"
        f"|---|---|---|\n"
        f"{_tissue_ref_rows}"
    )
    st.caption(f"Values at {field_strength} — Rooney et al. (2007) and Bottomley et al. (1984)")

    st.markdown("**Sequence**")
    _seq_options = ["FSE", "GRE", "FLAIR", "STIR", "bSSFP", "DIR", "DWI", "MPRAGE", "EPI (single-shot)"]
    seq = st.radio(
        "",
        _seq_options,
        index=_seq_options.index(_preset.get("seq", "FSE")),
        horizontal=True,
        key="sequence",
    )

    if st.session_state.ga4_prev_seq is not None and seq != st.session_state.ga4_prev_seq:
        _ga4_events.append(("sequence_change", {"sequence": seq}))
    st.session_state.ga4_prev_seq = seq

    is_3d = (seq == "MPRAGE")
    dim_label = "3D" if is_3d else "2D"
    dim_color = "#4CAF50" if is_3d else "#2196F3"
    st.markdown(
        f"<span style='background:{dim_color};color:white;padding:2px 10px;"
        f"border-radius:4px;font-size:0.8rem;font-weight:bold;'>{dim_label}</span>",
        unsafe_allow_html=True
    )

    st.markdown("**Parameters**")

    # TR — sequence-specific range, capped at the field-strength maximum.
    # Each slider uses max(sequence_min + step, _tr_max) as its ceiling so
    # that the slider always has a valid range even at very low field strengths
    # where _tr_max may be shorter than the sequence's physiological minimum TR.
    if   seq == "FLAIR":
        TR = st.slider("TR (ms)",  3000, 10000, _preset.get("TR", 9000), 100)  # fixed range — FLAIR needs long TR at all field strengths
    elif seq == "STIR":
        _fm = max(1050, _tr_max)   # STIR min TR = 1000 ms
        TR = st.slider("TR (ms)",  1000, _fm, min(_preset.get("TR", 3000), _fm),  50)
    elif seq == "bSSFP":
        TR = st.slider("TR (ms)",     3,   20, _preset.get("TR", 5),   1)   # hardware-constrained; field-strength cap irrelevant
    elif seq == "DIR":
        _fm = max(5100, _tr_max)   # DIR min TR = 5000 ms
        TR = st.slider("TR (ms)",  5000, _fm, min(_preset.get("TR", 8000), _fm), 100)
    elif seq == "DWI":
        _fm = max(3100, _tr_max)   # DWI min TR = 3000 ms
        TR = st.slider("TR (ms)",  3000, _fm, min(_preset.get("TR", 5000), _fm), 100)
    elif seq == "MPRAGE":
        _fm = max(2100, _tr_max)   # MPRAGE min TR = 2000 ms
        TR = st.slider("TR (ms)",  2000, _fm, min(_preset.get("TR", 2300), _fm), 100)
    elif seq == "EPI (single-shot)":
        _fm = max(600, min(3000, _tr_max))   # EPI: cap at min(3000, field max)
        TR = st.slider("TR (ms)",   500, _fm, min(_preset.get("TR", 2000), _fm), 100)
    else:                                    # FSE, GRE — directly tied to T1 recovery curve
        TR = st.slider("TR (ms)",   500, _tr_max, min(_preset.get("TR", 4000), _tr_max),  50)

    # Initialise variables that may not be set by every sequence branch
    b                 = 0
    TI                = 0
    TI1               = 0
    TI2               = 0
    n2_ghost_intensity = 0.0   # EPI N/2 ghost — overridden in EPI branch
    _epi_eff_idx      = 1      # EPI k-space centre echo index — overridden in EPI branch
    _epi_esp_ms       = EPI_NOMINAL_ESP_MS  # nominal EPI echo spacing (ms)

    if seq == "FSE":
        TE  = st.slider("TE eff. (ms)", 10, 300, _preset.get("TE", 80), 5)
        ETL = st.slider("ETL",           1,  32, _preset.get("ETL", 16), 1)
        st.caption("Pulse sequence diagram shows a maximum of 6 echoes for readability.")
        FA  = 90
        # Derived FSE timing parameters
        _fse_eff_idx = max(0, min(min(ETL, FSE_MAX_ECHOES_TO_SHOW) - 1,
                                  round(TE / FSE_NOMINAL_ESP_MS) - 1))
        _fse_esp_ms  = TE / (_fse_eff_idx + 1)
        _fse_min_tr  = ETL * _fse_esp_ms
        st.caption(f"Echo spacing (ESP) ≈ {_fse_esp_ms:.1f} ms")
        if TR < _fse_min_tr:
            st.error(f"TR too short: minimum TR for current ETL and TEeff is {int(_fse_min_tr)} ms")
        if _fse_esp_ms < 10.0:
            st.error("Echo spacing too short to be physically realistic — minimum ESP is approximately 10 ms")

    elif seq == "GRE":
        TE  = st.slider("TE (ms)",       2, 100, _preset.get("TE", 5), 1)
        FA  = st.slider("Flip Angle (°)",1,  90, _preset.get("FA", 30), 1)
        ETL = 1
        _gre_gss_end  = GRE_GSS_T0 + GRE_GSS_RISE + GRE_GSS_FLAT + GRE_GSS_RISE
        _gre_rep_rise = abs(GRE_GSS_REP_AMP) / GRE_GSS_REF_AMP * GRE_GSS_RISE
        _gre_rep_end  = _gre_gss_end + _gre_rep_rise + GRE_GSS_REP_FLAT + _gre_rep_rise
        _gre_dep_rise = abs(GRAD_FE_DEP_AMP) / GRAD_FE_READ_AMP * GRAD_RISE
        _gre_te_min_ms = int(np.ceil((_gre_rep_end + 2*_gre_dep_rise + GRAD_FE_DEP_FLAT
                                      + GRAD_RISE + GRAD_FE_READ_FLAT/2 - GRE_TE_MIN)
                                     * GRE_TE_MAX_MS / GRE_TE_RANGE))
        if TE < _gre_te_min_ms:
            st.error(f"TE too short: minimum achievable TE is {_gre_te_min_ms:.0f} ms")

    elif seq == "FLAIR":
        TI  = st.slider("TI (ms)",     500, 4000, _preset.get("TI", 2500),  50)
        TE  = st.slider("TE eff. (ms)", 25,  250, _preset.get("TE", 90),   5)
        ETL = st.slider("ETL",           1,   32, _preset.get("ETL", 16),   1)
        FA  = 90
        _flair_eff_idx = max(0, min(min(ETL, FSE_MAX_ECHOES_TO_SHOW) - 1,
                                    round(TE / FLAIR_NOMINAL_ESP_MS) - 1))
        _flair_esp_ms  = TE / (_flair_eff_idx + 1)
        _flair_min_tr  = TI + ETL * _flair_esp_ms
        st.caption(f"Echo spacing (ESP) ≈ {_flair_esp_ms:.1f} ms")
        if TR < _flair_min_tr:
            st.error(f"TR too short: minimum TR for current TI, ETL and TEeff is {int(_flair_min_tr)} ms")
        if _flair_esp_ms < 10.0:
            st.error("Echo spacing too short to be physically realistic — minimum ESP is approximately 10 ms")

    elif seq == "STIR":
        TI  = st.slider("TI (ms)",      50,  400, _preset.get("TI", 210),  10)
        TE  = st.slider("TE eff. (ms)", 25,  250, _preset.get("TE", 60),   5)
        ETL = st.slider("ETL",           1,   32, _preset.get("ETL", 8),   1)
        FA  = 90
        _stir_eff_idx = max(0, min(min(ETL, FSE_MAX_ECHOES_TO_SHOW) - 1,
                                   round(TE / FLAIR_NOMINAL_ESP_MS) - 1))
        _stir_esp_ms  = TE / (_stir_eff_idx + 1)
        _stir_min_tr  = TI + ETL * _stir_esp_ms
        st.caption(f"Echo spacing (ESP) ≈ {_stir_esp_ms:.1f} ms")
        if TR < _stir_min_tr:
            st.error(f"TR too short: minimum TR for current TI, ETL and TEeff is {int(_stir_min_tr)} ms")
        if _stir_esp_ms < 10.0:
            st.error("Echo spacing too short to be physically realistic — minimum ESP is approximately 10 ms")

    elif seq == "bSSFP":
        FA  = st.slider("Flip Angle (°)", 10, 90, _preset.get("FA", 50), 1)
        TE  = TR / 2
        ETL = 1
        st.caption(f"TE = TR/2 = {TE:.1f} ms  (fixed by sequence design)")

    elif seq == "DIR":
        TI1 = st.slider("TI1 (ms)",  500, 5000, _preset.get("TI1", 3400), 50)
        TI2 = st.slider("TI2 (ms)",  100, 2500, _preset.get("TI2", 800), 50)
        TE  = st.slider("TE eff. (ms)", 10, 100, _preset.get("TE", 25),  5)
        ETL = st.slider("ETL",           1,  32, _preset.get("ETL", 8),  1)
        FA  = 90
        TI  = TI1
        _dir_eff_idx = max(0, min(min(ETL, FSE_MAX_ECHOES_TO_SHOW) - 1,
                                  round(TE / DIR_NOMINAL_ESP_MS) - 1))
        _dir_esp_ms  = TE / (_dir_eff_idx + 1)
        _dir_min_tr  = TI1 + TI2 + ETL * _dir_esp_ms
        st.caption(f"Echo spacing (ESP) ≈ {_dir_esp_ms:.1f} ms")
        if TR < _dir_min_tr:
            st.error(
                f"TR too short: minimum TR for current TI1, TI2, ETL and TEeff "
                f"is {int(_dir_min_tr)} ms")
        if _dir_esp_ms < 10.0:
            st.error(
                "Echo spacing too short to be physically realistic "
                "— minimum ESP is approximately 10 ms")

    elif seq == "DWI":
        b   = st.slider("b-value (s/mm²)", 0, 3000, _preset.get("b", 1000), 100)
        TE  = st.slider("TE (ms)",        50,  150, _preset.get("TE", 80),   5)
        ETL = 1
        FA  = 90
        st.caption(f"Diffusion attenuation: exp(−b×ADC)  |  b={b} s/mm²")

    elif seq == "MPRAGE":
        TI  = st.slider("TI (ms)",        200, 2500, _preset.get("TI", 900), 10)
        TE  = st.slider("TE (ms)",          2,    18, _preset.get("TE", 3),  1)
        FA  = st.slider("Flip Angle (°)",   5,   15, _preset.get("FA", 9),  1)
        ETL = 1

    else:  # EPI (single-shot)
        TE  = st.slider("TEeff (ms)", 1, 200, _preset.get("TE", 30), 1)
        FA  = 90
        ETL = 1
        n2_ghost_intensity = st.slider(
            "N/2 Ghost Intensity (%)", 0, 30, _preset.get("n2_ghost_pct", 15), 1,
            key="n2_ghost_slider"
        ) / 100.0
        st.caption(f"Optimal TEeff for BOLD ≈ T2* of GM = {TISSUES['GM']['T2s']} ms")

    freq_matrix  = st.slider("Frequency Matrix (px)", 64, 512, _preset.get("freq_matrix", 256), 32)
    phase_matrix = st.slider("Phase Matrix (px)",     64, 512, _preset.get("phase_matrix", 256), 32)
    if seq == "EPI (single-shot)":
        ETL = phase_matrix  # single-shot EPI: all phase encodes in one TR, so scan time = TR × NEX
        _epi_esp_ms   = EPI_NOMINAL_ESP_MS
        _epi_eff_idx  = max(0, min(phase_matrix - 1,
                                   round(TE / _epi_esp_ms) - 1))
        _epi_max_teeff = phase_matrix * _epi_esp_ms
        st.caption(
            f"Echo spacing (ESP) = {_epi_esp_ms:.1f} ms  |  "
            f"k-space centre: echo #{_epi_eff_idx + 1} of {phase_matrix}"
        )
        if TE > _epi_max_teeff:
            st.error(
                f"TEeff too long: maximum is {_epi_max_teeff:.0f} ms "
                f"for {phase_matrix} phase encodes at ESP {_epi_esp_ms:.1f} ms"
            )
            st.stop()

    FOV_read  = st.slider("FOV Read (mm)",  180, 400, _preset.get("FOV_read", 400), 10)
    FOV_phase = st.slider("FOV Phase (mm)", 180, 400, _preset.get("FOV_phase", 400), 10)
    st.caption(
        f"Frequency pixel: {FOV_read / freq_matrix:.2f} mm  |  "
        f"Phase pixel: {FOV_phase / phase_matrix:.2f} mm"
    )

    if is_3d:
        Npartitions = st.slider("Partitions", 32, 256, _preset.get("Npartitions", 176), 8)
        slice_mm    = 1.0
    else:
        slice_mm    = st.slider("Slice (mm)", 1, 10, _preset.get("slice_mm", 5), 1)
        Npartitions = 1

    if seq == "MPRAGE":
        readout_train_ms = MPRAGE_TR_READOUT * Npartitions
        st.caption(f"TR_readout: {MPRAGE_TR_READOUT} ms (fixed) — gradient echo spacing within partition train")
        st.caption(f"Readout train duration: {MPRAGE_TR_READOUT} ms × {Npartitions} partitions = {readout_train_ms} ms")
        if TR < TI + readout_train_ms:
            st.markdown(
                f":red[**⚠ Timing violation:** TR ({TR} ms) < TI ({TI} ms) + readout train "
                f"({readout_train_ms} ms) = {TI + readout_train_ms} ms. "
                f"Increase TR or decrease TI / Partitions.]"
            )
        st.caption(
            "MPRAGE timing: inversion pulse → TI delay → partition encode readout train "
            f"({readout_train_ms} ms) → remaining T1 recovery → next inversion pulse."
        )

    NEX     = st.slider("NEX",               1,   8, _preset.get("NEX", 1),  1)
    BW      = st.slider("Bandwidth (Hz/px)", 50, 500, _preset.get("BW", 200), 10)
    fat_sat = st.checkbox("Fat Saturation", value=_preset.get("fat_sat", False))

    _PF_OPTIONS = {"8/8 (Full)": 1.0, "7/8": 7/8, "6/8": 6/8, "5/8": 5/8}
    if seq in ("FSE", "GRE", "EPI (single-shot)"):
        _pf_labels   = list(_PF_OPTIONS.keys())
        _pf_label    = st.selectbox(
            "Partial Fourier (phase)",
            _pf_labels,
            index=_pf_labels.index(_preset.get("pf_label", "8/8 (Full)")),
        )
        pf_fraction  = _PF_OPTIONS[_pf_label]
        if pf_fraction < 1.0:
            st.caption(
                f"k-space: {pf_fraction:.4f} → "
                f"SNR ×{np.sqrt(pf_fraction):.3f}, "
                f"scan time ×{pf_fraction:.4f}"
            )
    else:
        pf_fraction = 1.0
    if seq == "MPRAGE" and fat_sat:
        st.caption(
            "Note: chemical shift selective (CHESS) fat suppression is "
            "uncommonly used with MPRAGE. Fat suppression alters the "
            "inversion recovery preparation, potentially disrupting the "
            "T1-null point optimised for CSF or WM. Fat signal in MPRAGE "
            "is typically managed via the inversion pulse itself."
        )

    # Compute optimal W/L before sliders are instantiated (session state must
    # be updated before keyed widgets render). Track a signature of all
    # signal-affecting parameters; recompute only when something changes.
    #
    # W/L is derived from pure physics signal equations — no voxel-volume,
    # SNR, noise, or field-strength SNR-scaling factors are applied.
    # Tissue T1/T2/T2s are taken from FIELD_STRENGTH_TISSUES[field_strength]
    # so the computed W/L always matches the tissue parameters used to render
    # the phantom image (which are also set from FIELD_STRENGTH_TISSUES).
    # PD and ADC are field-strength-independent and come from the base TISSUES dict.
    # field_strength is included in _params_sig so W/L recomputes whenever B0 changes.
    _params_sig = (seq, TR, TE, FA, TI, TI1, TI2, b, fat_sat,
                   FOV_read, FOV_phase, freq_matrix, phase_matrix, slice_mm,
                   field_strength)
    if _params_sig != st.session_state.params_sig:
        _sigs = []
        _fs_t = FIELD_STRENGTH_TISSUES[field_strength]   # T1/T2/T2s for current B0
        for _t in ["WM", "GM", "CSF"]:
            _p = {**TISSUES[_t], **_fs_t[_t]}   # PD/ADC from base; T1/T2/T2s from current B0
            if seq == "FSE":
                _s = fse_signal(TR, TE, _p["T1"], _p["T2"], _p["PD"], fat_sat=fat_sat, is_fat=False)
            elif seq == "GRE":
                _s = gre_signal(TR, TE, FA, _p["T1"], _p["T2s"], _p["PD"], fat_sat=fat_sat, is_fat=False)
            elif seq in ("FLAIR", "STIR"):
                _s = flair_signal(TR, TI, TE, _p["T1"], _p["T2"], _p["PD"], fat_sat=fat_sat, is_fat=False)
            elif seq == "bSSFP":
                _s = bssfp_signal(TR, FA, _p["T1"], _p["T2"], _p["PD"])
            elif seq == "DIR":
                _s = dir_signal(TR, TI1, TI2, TE, _p["T1"], _p["T2"], _p["PD"])
            elif seq == "DWI":
                _s = dwi_signal(TR, TE, b, _p["T1"], _p["T2"], _p["PD"], _p["ADC"])
            elif seq == "MPRAGE":
                _s = mprage_signal(TR, TI, TE, FA, _p["T1"], _p["T2s"], _p["PD"])
            else:  # EPI
                _s = epi_signal(TR, TE, _p["T1"], _p["T2s"], _p["PD"])
            _sigs.append(_s)
        _arr = np.array(_sigs)
        _opt_w = float(np.clip(2.0 * np.std(_arr),  0.01, 2.0))
        _opt_l = float(np.clip(np.mean(_arr),        0.0,  1.0))
        st.session_state.params_sig    = _params_sig
        st.session_state.wl_opt_window = _opt_w
        st.session_state.wl_opt_level  = _opt_l
        st.session_state.wl_window     = _opt_w
        st.session_state.wl_level      = _opt_l

    st.markdown("""
    <style>
    div.stButton > button {
        background-color: #2196F3;
        color: white;
        border: none;
        border-radius: 4px;
    }
    div.stButton > button:hover {
        background-color: #1976D2;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)
    if st.button("Reset W/L to Optimal"):
        st.session_state.wl_window = st.session_state.wl_opt_window
        st.session_state.wl_level  = st.session_state.wl_opt_level

    st.markdown("**Window / Level**")
    st.caption("Auto-resets to optimal values when sequence changes.")
    wl_window = st.slider("Window", 0.01, 2.0, 1.0, 0.01, key="wl_window")
    wl_level  = st.slider("Level",  0.0,  1.0, 0.5, 0.01, key="wl_level")

    st.markdown("---")
    st.markdown("### Slice Position")
    st.caption("red = sagittal (X) · green = coronal (Y) · blue = axial (Z)")
    st.caption("Click anywhere on a phantom image below to move the crosshair.")
    # A click on one of the phantom images (Row 1, below) stashes its target
    # slice indices in "_pending_slice_update" and triggers a rerun. A widget's
    # session_state value can only be changed on the run *before* the widget is
    # instantiated, so that pending update must be applied here, before the
    # sliders below are created.
    _pending_slice = st.session_state.pop("_pending_slice_update", None)
    if _pending_slice:
        for _pk, _pv in _pending_slice.items():
            st.session_state[_pk] = _pv
    st.session_state.setdefault("x_pos", 181)
    st.session_state.setdefault("y_pos", 217)
    st.session_state.setdefault("z_pos", 181)
    x_pos = st.slider("X  (Sagittal)", 0, 361, step=1, key="x_pos")
    y_pos = st.slider("Y  (Coronal)",  0, 433, step=1, key="y_pos")
    z_pos = st.slider("Z  (Axial)",    0, 361, step=1, key="z_pos")

    _cur_slice = (x_pos, y_pos, z_pos)
    if st.session_state.ga4_prev_slice != (None, None, None) and _cur_slice != st.session_state.ga4_prev_slice:
        _ga4_events.append(("slice_position_change", {"x": x_pos, "y": y_pos, "z": z_pos}))
    st.session_state.ga4_prev_slice = _cur_slice

# ---------------------------------------------------------------------------
# Fire any queued GA4 events
# ---------------------------------------------------------------------------
# st.components.v1.html() runs in a sandboxed iframe on the same origin, so
# window.parent.gtag() reaches the GA4 instance loaded in the parent page.
if _ga4_events:
    _js = "\n".join(
        f"window.parent.gtag('event', {json.dumps(name)}, {json.dumps(params)});"
        for name, params in _ga4_events
    )
    st.components.v1.html(f"<script>{_js}</script>", height=0)

# ---------------------------------------------------------------------------
# Apply field-strength-dependent tissue parameters
# ---------------------------------------------------------------------------
# Override the module-level TISSUES dict with values for the selected B0.
# Only T1, T2, and T2s change with field strength; PD, colour, and ADC are
# field-strength-independent and are inherited from the base TISSUES dict.
_fs_params = FIELD_STRENGTH_TISSUES[field_strength]
for _t in TISSUES:
    TISSUES[_t]["T1"]  = _fs_params[_t]["T1"]
    TISSUES[_t]["T2"]  = _fs_params[_t]["T2"]
    TISSUES[_t]["T2s"] = _fs_params[_t]["T2s"]

# Effective noise reference scaled by inverse of field-strength SNR factor.
# At 3 T (scale = 1.000) this equals NOISE_REF exactly.
# At lower fields (smaller scale) noise increases, reducing SNR.
_field_snr_scale  = FIELD_SNR_SCALE[field_strength]
_noise_ref_scaled = NOISE_REF / _field_snr_scale

# ---------------------------------------------------------------------------
# Compute signals and SNR
# ---------------------------------------------------------------------------
# Three-step Signal / Noise / SNR calculation — no circular dependencies
# ---------------------------------------------------------------------------

# STEP 1 — Physics signals (pure tissue contrast, 0–1 scale)
base_signals = {}
for tissue, props in TISSUES.items():
    is_fat = (tissue == "Fat")
    if seq == "FSE":
        S = fse_signal(TR, TE, props["T1"], props["T2"], props["PD"],
                       fat_sat=fat_sat, is_fat=is_fat)
    elif seq == "GRE":
        S = gre_signal(TR, TE, FA, props["T1"], props["T2s"], props["PD"],
                       fat_sat=fat_sat, is_fat=is_fat)
    elif seq in ("FLAIR", "STIR"):
        S = flair_signal(TR, TI, TE, props["T1"], props["T2"], props["PD"],
                         fat_sat=fat_sat, is_fat=is_fat)
    elif seq == "bSSFP":
        S = bssfp_signal(TR, FA, props["T1"], props["T2"], props["PD"])
        if fat_sat and is_fat:
            S *= 0.05
    elif seq == "DIR":
        S = dir_signal(TR, TI1, TI2, TE, props["T1"], props["T2"], props["PD"])
        if fat_sat and is_fat:
            S *= 0.05
    elif seq == "DWI":
        S = dwi_signal(TR, TE, b, props["T1"], props["T2"], props["PD"], props["ADC"])
        if fat_sat and is_fat:
            S *= 0.05
    elif seq == "MPRAGE":
        S = mprage_signal(TR, TI, TE, FA, props["T1"], props["T2s"], props["PD"])
        if fat_sat and is_fat:
            S *= 0.05
    else:  # EPI
        S = epi_signal(TR, TE, props["T1"], props["T2s"], props["PD"])
    base_signals[tissue] = S

# STEP 1 (cont.) — Multiply by voxel-volume scale factor.
# For 3D (MPRAGE): partition_thickness = slab / Npartitions, so more partitions
# → thinner slices → smaller voxel → lower Signal.  Npartitions is purely a
# voxel-volume parameter here, exactly analogous to slice_mm in 2D.
# For 2D: voxel thickness = slice_mm as before.
# Reference voxel uses the default partition thickness (1 mm) for 3D and the
# default slice (5 mm) for 2D, so signals sit at baseline at default settings.
if is_3d:
    _slice_for_vol = MPRAGE_SLAB_MM / Npartitions          # e.g. 176/176 = 1 mm at default
    _ref_slice     = MPRAGE_SLAB_MM / MPRAGE_NPART_DEFAULT  # = 1.0 mm
else:
    _slice_for_vol = slice_mm
    _ref_slice     = 5.0
_ref_vox   = (240.0 / 256.0) ** 2 * _ref_slice
voxel_vol  = (FOV_read / freq_matrix) * (FOV_phase / phase_matrix) * _slice_for_vol
_vol_scale = voxel_vol / _ref_vox
# Partial Fourier reduces SNR ∝ sqrt(pf_fraction) — fewer k-space lines acquired.
signals    = {t: round(base_signals[t] * _vol_scale * np.sqrt(pf_fraction), 2) for t in base_signals}

# STEP 2 — System noise floor.
# Depends ONLY on BW, NEX, and field strength — not on FOV, matrix, slice,
# Npartitions, TR, TE, TI, or flip angle.
# _noise_ref_scaled incorporates the B0-dependent SNR penalty (SNR ∝ B0).
noise_floor = _noise_ref_scaled * np.sqrt(BW / BW_REF) / np.sqrt(NEX / NEX_REF)

# STEP 3 — SNR = Signal / Noise, derived directly with no other dependencies.
# Keep noise_floor full precision so low-noise settings do not round to zero.
snrs = {t: signals[t] / noise_floor if noise_floor > 0 else 0.0 for t in signals}

names    = list(TISSUES.keys())
colors   = [TISSUES[t]["color"] for t in names]
freq_pixel_mm  = FOV_read  / freq_matrix
phase_pixel_mm = FOV_phase / phase_matrix
scan_sec = calc_scan_time(TR, phase_matrix, ETL, NEX, seq, Npartitions) * pf_fraction
scan_min = int(scan_sec // 60)
scan_s   = int(scan_sec % 60)
cnr_wm_gm  = abs(snrs["WM"] - snrs["GM"])
cnr_wm_csf = abs(snrs["WM"] - snrs["CSF"])
cnr_gm_csf = abs(snrs["GM"] - snrs["CSF"])

_current_teaching_snapshot = _parameter_snapshot(
    protocol_name, seq, field_strength, TR, TE, TI, TI1, TI2, FA, b,
    NEX, BW, freq_matrix, phase_matrix, FOV_read, FOV_phase, slice_mm,
    Npartitions, fat_sat, pf_fraction, scan_sec,
)
_what_changed_notes = explain_parameter_changes(
    st.session_state.get("teaching_prev_snapshot"),
    _current_teaching_snapshot,
)
st.session_state.teaching_prev_snapshot = _current_teaching_snapshot

# ---------------------------------------------------------------------------
# Brain phantom image
# ---------------------------------------------------------------------------
# Precompute the signal lookup table and noise level shared by all three plane
# renders. Each plane fetches its slice and builds its own image in the loop below.
# Phantom image uses base_signals (unscaled) so the W/L, which is computed in
# base-signal space (0–1), correctly maps to image pixel values.
# noise_std derived from base_signals gives SNR ∝ voxel_vol × sqrt(NEX)/sqrt(BW),
# so the phantom visibly gets less noisy as voxel size or NEX increases.
sig_lookup = np.array([0.0, base_signals["WM"], base_signals["GM"], base_signals["CSF"]])
# Phantom noise in base-signal space: noise_floor scaled back to the base-signal
# reference (undo _vol_scale) so it is consistent with the 0–1 pixel values.
noise_std  = noise_floor / _vol_scale if _vol_scale > 0 else 0.0

vmin = wl_level - wl_window / 2
vmax = wl_level + wl_window / 2

# ---------------------------------------------------------------------------
# Page title
# ---------------------------------------------------------------------------
# Centred heading displayed at the top of the main content area.
st.markdown(
    f"<h3 style='text-align: center;'>MRI Simulator — Brain ({field_strength})</h3>",
    unsafe_allow_html=True,
)

if _what_changed_notes:
    with st.expander("What changed?", expanded=True):
        for _note in _what_changed_notes:
            st.info(_note)

# ---------------------------------------------------------------------------
# Main layout: three planes (row 1) + bars/curves (row 2)
# ---------------------------------------------------------------------------
# Row 1: Axial, Coronal, and Sagittal images side by side, each with a
# crosshair overlay showing the intersection of the other two planes.
# Crosshair colours: red = sagittal (X), green = coronal (Y), blue = axial (Z).
# After np.flipud the display y-coordinate for original row r is (255 - r).
# FOV zoom: visible half-width in 256-px image space scales linearly with FOV.
# At FOV_MAX (400 mm) the full image is visible; smaller FOV zooms in on the
# crosshair position, larger FOV (up to max) zooms out.
FOV_MAX      = 400.0
half_w_read  = 128.0 * FOV_read  / FOV_MAX   # half-width  in image-px for read  direction
half_h_phase = 128.0 * FOV_phase / FOV_MAX   # half-height in image-px for phase direction
# Number of phantom rows the phase FOV covers (for aliasing calculation)
n_phase_rows = max(1, round(256 * FOV_phase / FOV_MAX))
# row_key/row_max and col_key/col_max identify which slice-position slider
# (and its max index) each image axis controls, so a click on the image can be
# inverted back into a slider update. This is the exact inverse of the
# crosshair_row/col display formulas: row uses the "255 - v/max*255" flip
# (see the np.flipud note above), column uses the plain "v/max*255" mapping.
plane_configs = [
    # (name, slice_idx, crosshair_row_display, crosshair_col_display, hcolor, vcolor,
    #  row_key, row_max, col_key, col_max)
    ("Axial",    z_pos, 255 - y_pos/433*255, x_pos/361*255, "#44FF44", "#FF4444",
     "y_pos", 433, "x_pos", 361),
    ("Coronal",  y_pos, 255 - z_pos/361*255, x_pos/361*255, "#4488FF", "#FF4444",
     "z_pos", 361, "x_pos", 361),
    ("Sagittal", x_pos, 255 - z_pos/361*255, y_pos/433*255, "#4488FF", "#44FF44",
     "z_pos", 361, "y_pos", 433),
]

col_ax, col_cor, col_sag = st.columns(3)
for col, (plane_name, idx, ch_y, ch_x, hcol, vcol, row_key, row_max, col_key, col_max) in zip(
        [col_ax, col_cor, col_sag], plane_configs):
    ph  = get_phantom_slice(plane_name, idx)
    img = sig_lookup[ph].copy()
    # In-plane resolution effect: downsample to matrix size then upsample back
    # to 256 px with nearest-neighbour to produce a pixelated/blocky look at
    # low matrix sizes and a sharp look at high matrix sizes.
    step_x = max(1, 256 // freq_matrix)   # frequency → columns (axis=1)
    step_y = max(1, 256 // phase_matrix)  # phase     → rows    (axis=0)
    if step_x > 1 or step_y > 1:
        img = img[::step_y, ::step_x]
        img = np.repeat(np.repeat(img, step_y, axis=0), step_x, axis=1)
        img = img[:256, :256]  # trim any overshoot from rounding
    # Rectangular FOV + phase-direction aliasing.
    # When FOV_phase < FOV_read the phase direction covers fewer anatomy rows.
    # Anatomy outside the phase FOV wraps back in (Nyquist/aliasing).
    if n_phase_rows < 256:
        phase_start = 128 - n_phase_rows // 2
        aliased = np.zeros_like(img)
        n_periods = int(np.ceil(256 / n_phase_rows)) + 1
        for k in range(-n_periods, n_periods + 1):
            src_rows = np.arange(n_phase_rows) + phase_start + k * n_phase_rows
            dst_rows = np.arange(n_phase_rows) + phase_start
            valid    = (src_rows >= 0) & (src_rows < 256)
            aliased[dst_rows[valid], :] += img[src_rows[valid], :]
        img = aliased
    # Gibbs ringing artifact from partial Fourier k-space truncation (phase dir).
    # Truncate the central pf_fraction of k-space rows; ringing intensity
    # increases as the fraction decreases toward 5/8.
    if pf_fraction < 1.0:
        n_rows  = img.shape[0]
        n_keep  = max(1, round(n_rows * pf_fraction))
        ksp     = np.fft.fftshift(np.fft.fft(img, axis=0), axes=0)
        center  = n_rows // 2
        mask    = np.zeros(n_rows)
        mask[center - n_keep // 2 : center + n_keep // 2] = 1.0
        ksp    *= mask[:, np.newaxis]
        img     = np.clip(np.real(np.fft.ifft(np.fft.ifftshift(ksp, axes=0), axis=0)), 0, 1)
    # Partial volume averaging: blur proportional to slice thickness.
    # sigma = 0 at 1 mm (sharp), rising to ~2.5 at 10 mm (heavy blurring).
    pva_sigma = (slice_mm - 1) * 0.25
    if pva_sigma > 0:
        img = gaussian_filter(img, sigma=pva_sigma)
    # EPI N/2 ghost: phase inconsistencies between alternating ±Gfe readout
    # directions produce a ghost shifted N/2 pixels in the phase-encode direction.
    # The ghost is slightly blurred relative to the main image.
    if seq == "EPI (single-shot)" and n2_ghost_intensity > 0:
        _ghost_shift = img.shape[0] // 2            # N/2 = half the display height
        _ghost = np.roll(img, _ghost_shift, axis=0)  # shift in phase-encode (row) direction
        _ghost = gaussian_filter(_ghost, sigma=1.5)  # slight additional blur
        img = np.clip(img + n2_ghost_intensity * _ghost, 0, 1)
    # Capture the clean axial image for k-space visualisation (no noise, full
    # processing applied, including ghost if EPI).  Must be saved before noise.
    if plane_name == "Axial":
        _kspace_source_img = img.copy()
    if noise_std > 0:
        img = np.clip(img + np.random.normal(0, noise_std, img.shape), 0, 1)
    with col:
        fig_p, ax_p = plt.subplots(figsize=(2.0, 2.0), facecolor="#1e1e1e")
        ax_p.imshow(np.flipud(img), cmap="gray", vmin=vmin, vmax=vmax,
                    interpolation="nearest")
        ax_p.axhline(ch_y, color=hcol, linewidth=0.8, alpha=0.8)
        ax_p.axvline(ch_x, color=vcol, linewidth=0.8, alpha=0.8)
        ax_p.set_xlim(128 - half_w_read,  128 + half_w_read)
        ax_p.set_ylim(128 + half_h_phase, 128 - half_h_phase)
        ax_p.axis("off")
        ax_p.set_title(plane_name, color="white", fontsize=8, pad=2)
        ax_p.text(0.5, 0.02, f"W:{wl_window:.2f} L:{wl_level:.2f}",
                  transform=ax_p.transAxes, ha="center", va="bottom",
                  color="white", fontsize=5,
                  bbox=dict(facecolor="black", alpha=0.5, edgecolor="none", pad=1))

        # Render to an in-memory PNG (rather than st.pyplot) so the exact same
        # raster can be handed to streamlit_image_coordinates for click capture.
        fig_p.canvas.draw()
        _png_buf = BytesIO()
        fig_p.savefig(_png_buf, format="png", dpi=fig_p.dpi,
                      facecolor=fig_p.get_facecolor())
        _png_buf.seek(0)
        _pil_img = Image.open(_png_buf)
        _nat_w, _nat_h = _pil_img.size

        _click = streamlit_image_coordinates(
            _pil_img, use_column_width=True, key=f"phantom_click_{plane_name}",
        )

        if _click is not None:
            # (x, y) are relative to the *displayed* image size (which
            # use_column_width may have scaled up/down from the natural PNG
            # size) — rescale back to natural pixel coordinates first.
            _disp_w = _click.get("width")  or _nat_w
            _disp_h = _click.get("height") or _nat_h
            _click_pt = (round(_click["x"] * _nat_w / _disp_w),
                         round(_click["y"] * _nat_h / _disp_h))
            _last_click_key = f"_last_phantom_click_{plane_name}"
            if st.session_state.get(_last_click_key) != _click_pt:
                # A genuinely new click (not just the component replaying its
                # last value on an unrelated rerun) — process it once.
                st.session_state[_last_click_key] = _click_pt
                # Undo the top-left-origin/DOM pixel space back into
                # matplotlib's bottom-left-origin display space, then invert
                # through this axes' transData to get image-index (data)
                # coordinates — the exact reverse of the crosshair placement.
                _disp_x = _click_pt[0]
                _disp_y = _nat_h - _click_pt[1]
                _data_x, _data_y = ax_p.transData.inverted().transform((_disp_x, _disp_y))
                _new_row = int(round((255 - _data_y) * row_max / 255))
                _new_col = int(round(_data_x * col_max / 255))
                _new_row = min(max(_new_row, 0), row_max)
                _new_col = min(max(_new_col, 0), col_max)
                _pending_update = dict(st.session_state.get("_pending_slice_update", {}))
                _pending_update[row_key] = _new_row
                _pending_update[col_key] = _new_col
                st.session_state["_pending_slice_update"] = _pending_update
                plt.close(fig_p)
                st.rerun()

        plt.close(fig_p)

# EPI N/2 ghost educational caption below the three-plane display
if seq == "EPI (single-shot)":
    st.caption(
        "**N/2 ghosting (EPI):** Phase inconsistencies between alternating positive and negative "
        "EPI readout directions produce a ghost image shifted N/2 pixels in the phase-encode "
        "direction. Adjust the N/2 Ghost Intensity slider to explore how gradient calibration "
        "errors affect image quality."
    )

# --- Pulse sequence diagram (between phantom images and signal bars) ---
_psd_fig = draw_pulse_sequence(seq, TR, TE, TI, FA, ETL, slice_mm=slice_mm, BW=BW, FOV_read=FOV_read,
                               epi_eff_idx=_epi_eff_idx, epi_n_readouts=phase_matrix, TI2=TI2,
                               Npartitions=Npartitions)
if _psd_fig is not None:
    st.pyplot(_psd_fig, use_container_width=True)
    plt.close(_psd_fig)

# --- Row 2: Signal/SNR bars and relaxation curves ---
col_bars, col_curves = st.columns([2, 3])

with col_bars:
    fig_sig, ax_sig = plt.subplots(figsize=(2.5, 1.5), facecolor="#1e1e1e")
    sig_vals = [signals[t] for t in names]
    bars = ax_sig.bar(names, sig_vals, color=colors, width=0.5)
    ax_sig.set_title(f"Signal (Noise floor = {noise_floor:.3g})", fontsize=9)

    # noise_floor already computed in Step 2 — use directly.
    _sig_max    = max(max(sig_vals), noise_floor) if sig_vals else 1.0
    _sig_offset = _sig_max * 0.025  # vertical offset for bar labels, scaled to max of bars and noise floor
    ax_sig.set_ylim(0, _sig_max * 1.25)
    for bar, val in zip(bars, sig_vals):
        ax_sig.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + _sig_offset,
                    f"{val:.2f}", ha="center", color="white", fontsize=5)

    # Draw noise floor as a horizontal dashed reference line; label via legend.
    ax_sig.axhline(noise_floor, color="#FF6B6B", linewidth=1.0,
                   linestyle="--", alpha=0.85)
    ax_sig.legend(fontsize=5, loc="upper right",
                  facecolor="#2e2e2e", edgecolor="none", labelcolor="white")

    style_ax(ax_sig)
    st.pyplot(fig_sig, use_container_width=True)
    plt.close(fig_sig)

    fig_snr, ax_snr = plt.subplots(figsize=(2.5, 1.5), facecolor="#1e1e1e")
    snr_vals = [snrs[t] for t in names]
    bars2 = ax_snr.bar(names, snr_vals, color=colors, width=0.5)
    ax_snr.set_title("SNR (= Signal / Noise floor)", fontsize=9)
    for bar, val in zip(bars2, snr_vals):
        ax_snr.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.2,
                    f"{val:.2f}", ha="center", color="white", fontsize=5)
    style_ax(ax_snr)
    st.pyplot(fig_snr, use_container_width=True)
    plt.close(fig_snr)

# --- Relaxation curves ---
brain_tissues = ["WM", "GM", "CSF"]
tr_range = np.linspace(100, _tr_max, 500)
te_range = np.linspace(2,   300,   500)

with col_curves:
    # ---- Upper plot ----
    fig_t1, ax_t1 = plt.subplots(figsize=(3.5, 1.5), facecolor="#1e1e1e")

    if seq in ("FLAIR", "STIR"):
        null_tissue = "CSF" if seq == "FLAIR" else "Fat"
        if seq == "STIR":
            ti_range_plot  = np.linspace(10, 500, 500)
            top_tissues    = brain_tissues + ["Fat"]
        else:
            ti_range_plot  = np.linspace(100, 4500, 500)
            top_tissues    = brain_tissues
        for t in top_tissues:
            p = TISSUES[t]
            s = p["PD"] * np.abs(1 - 2*np.exp(-ti_range_plot/p["T1"]) + np.exp(-TR/p["T1"])) * np.exp(-TE/p["T2"])
            ax_t1.plot(ti_range_plot, s, color=p["color"], label=t, linewidth=1.5)
        ax_t1.axvline(TI, color="white", linestyle="--", linewidth=1, alpha=0.6, label="Current TI")
        ti_null = csf_null_ti(TR, TISSUES[null_tissue]["T1"])
        ax_t1.axvline(ti_null, color=TISSUES[null_tissue]["color"], linestyle=":", linewidth=1, alpha=0.8, label=f"{null_tissue} null TI")
        ax_t1.set_title(f"Signal vs TI ({seq})", fontsize=9)
        ax_t1.set_xlabel("TI (ms)", fontsize=8)

    elif seq == "bSSFP":
        fa_range  = np.linspace(1, 90, 500)
        fa_opt_wm = bssfp_optimal_fa(TR, TISSUES["WM"]["T1"], TISSUES["WM"]["T2"])
        for t in brain_tissues:
            p = TISSUES[t]
            s = bssfp_signal(TR, fa_range, p["T1"], p["T2"], p["PD"])
            ax_t1.plot(fa_range, s, color=p["color"], label=t, linewidth=1.5)
        ax_t1.axvline(FA, color="white", linestyle="--", linewidth=1, alpha=0.6, label="Current FA")
        ax_t1.axvline(fa_opt_wm, color=TISSUES["WM"]["color"], linestyle=":", linewidth=1, alpha=0.8, label="Opt FA (WM)")
        ax_t1.set_title("Signal vs Flip Angle (bSSFP)", fontsize=9)
        ax_t1.set_xlabel("FA (°)", fontsize=8)

    elif seq == "DIR":
        ti2_range = np.linspace(10, TI1 - 10, 500)
        wm_null   = dir_null_ti2(TR, TI1, TISSUES["WM"]["T1"])
        for t in brain_tissues:
            p = TISSUES[t]
            s = p["PD"] * np.abs(1 - 2*np.exp(-ti2_range/p["T1"]) + 2*np.exp(-TI1/p["T1"]) - np.exp(-TR/p["T1"])) * np.exp(-TE/p["T2"])
            ax_t1.plot(ti2_range, s, color=p["color"], label=t, linewidth=1.5)
        ax_t1.axvline(TI2, color="white", linestyle="--", linewidth=1, alpha=0.6, label="Current TI2")
        if wm_null:
            ax_t1.axvline(wm_null, color=TISSUES["WM"]["color"], linestyle=":", linewidth=1, alpha=0.8, label="WM null TI2")
        ax_t1.set_title("Signal vs TI2 (DIR)", fontsize=9)
        ax_t1.set_xlabel("TI2 (ms)", fontsize=8)

    elif seq == "DWI":
        b_range = np.linspace(0, 3000, 500)
        for t in brain_tissues:
            p  = TISSUES[t]
            s0 = p["PD"] * (1 - np.exp(-TR/p["T1"])) * np.exp(-TE/p["T2"])
            s  = s0 * np.exp(-b_range * p["ADC"])
            ax_t1.plot(b_range, s, color=p["color"], label=t, linewidth=1.5)
        ax_t1.axvline(b, color="white", linestyle="--", linewidth=1, alpha=0.6, label="Current b")
        ax_t1.set_title("Signal vs b-value (DWI)", fontsize=9)
        ax_t1.set_xlabel("b (s/mm²)", fontsize=8)

    elif seq == "MPRAGE":
        ti_range_plot = np.linspace(100, 3000, 500)
        FA_r      = np.radians(FA)
        csf_null  = csf_null_ti(TR, TISSUES["CSF"]["T1"])
        for t in brain_tissues:
            p = TISSUES[t]
            s = p["PD"] * np.abs(1 - 2*np.exp(-ti_range_plot/p["T1"]) + np.exp(-TR/p["T1"])) * np.sin(FA_r) * np.exp(-TE/p["T2s"])
            ax_t1.plot(ti_range_plot, s, color=p["color"], label=t, linewidth=1.5)
        ax_t1.axvline(TI, color="white", linestyle="--", linewidth=1, alpha=0.6, label="Current TI")
        ax_t1.axvline(csf_null, color=TISSUES["CSF"]["color"], linestyle=":", linewidth=1, alpha=0.8, label="CSF null TI")
        ax_t1.set_title("Signal vs TI (MPRAGE)", fontsize=9)
        ax_t1.set_xlabel("TI (ms)", fontsize=8)

    else:
        # FSE, GRE, EPI: signal vs TR
        for t in brain_tissues:
            p = TISSUES[t]
            if seq == "GRE":
                FA_r = np.radians(FA)
                E1   = np.exp(-tr_range / p["T1"])
                s    = p["PD"] * np.sin(FA_r) * (1 - E1) / (1 - np.cos(FA_r)*E1) * np.exp(-TE/p["T2s"])
            elif seq == "EPI (single-shot)":
                s = p["PD"] * (1 - np.exp(-tr_range/p["T1"])) * np.exp(-TE/p["T2s"])
            else:  # FSE
                s = p["PD"] * (1 - np.exp(-tr_range/p["T1"])) * np.exp(-TE/p["T2"])
            ax_t1.plot(tr_range, s, color=p["color"], label=t, linewidth=1.5)
        ax_t1.axvline(TR, color="white", linestyle="--", linewidth=1, alpha=0.6, label="Current TR")
        ax_t1.set_xlim(0, _tr_max)
        ax_t1.set_title("T1 Recovery", fontsize=9)
        ax_t1.set_xlabel("TR (ms)", fontsize=8)

    ax_t1.set_ylabel("Signal", fontsize=8)
    ax_t1.set_ylim(bottom=0)
    ax_t1.legend(fontsize=5, labelcolor="white", facecolor="#2b2b2b",
                 edgecolor="#555", loc="upper left")
    style_ax(ax_t1)
    st.pyplot(fig_t1, use_container_width=True)
    plt.close(fig_t1)

    # ---- Lower plot ----
    fig_t2, ax_t2 = plt.subplots(figsize=(3.5, 1.5), facecolor="#1e1e1e")

    if seq == "bSSFP":
        tr_range_b = np.linspace(3, 50, 500)
        for t in brain_tissues:
            p = TISSUES[t]
            s = bssfp_signal(tr_range_b, FA, p["T1"], p["T2"], p["PD"])
            ax_t2.plot(tr_range_b, s, color=p["color"], label=t, linewidth=1.5)
        ax_t2.axvline(TR, color="white", linestyle="--", linewidth=1, alpha=0.6, label="Current TR")
        ax_t2.set_title("Signal vs TR (bSSFP)", fontsize=9)
        ax_t2.set_xlabel("TR (ms)", fontsize=8)

    elif seq == "EPI (single-shot)":
        te_range_epi = np.linspace(1, 90, 500)
        for t in brain_tissues:
            p = TISSUES[t]
            s = p["PD"] * (1 - np.exp(-TR/p["T1"])) * np.exp(-te_range_epi/p["T2s"])
            ax_t2.plot(te_range_epi, s, color=p["color"], label=t, linewidth=1.5)
            if p["T2s"] <= 90:
                ax_t2.axvline(p["T2s"], color=p["color"], linestyle=":", linewidth=1, alpha=0.6)
        ax_t2.axvline(TE, color="white", linestyle="--", linewidth=1, alpha=0.6, label="Current TE")
        ax_t2.set_title("T2* Decay (EPI)", fontsize=9)
        ax_t2.set_xlabel("TE (ms)", fontsize=8)

    elif seq == "DIR":
        for t in brain_tissues:
            p         = TISSUES[t]
            t1_factor = abs(1 - 2*np.exp(-TI2/p["T1"]) + 2*np.exp(-TI1/p["T1"]) - np.exp(-TR/p["T1"]))
            s         = p["PD"] * t1_factor * np.exp(-te_range/p["T2"])
            ax_t2.plot(te_range, s, color=p["color"], label=t, linewidth=1.5)
        ax_t2.axvline(TE, color="white", linestyle="--", linewidth=1, alpha=0.6, label="Current TE")
        ax_t2.set_title("T2 Decay (DIR-weighted)", fontsize=9)
        ax_t2.set_xlabel("TE (ms)", fontsize=8)

    elif seq == "DWI":
        for t in brain_tissues:
            p = TISSUES[t]
            s = p["PD"] * (1 - np.exp(-TR/p["T1"])) * np.exp(-te_range/p["T2"]) * np.exp(-b*p["ADC"])
            ax_t2.plot(te_range, s, color=p["color"], label=t, linewidth=1.5)
        ax_t2.axvline(TE, color="white", linestyle="--", linewidth=1, alpha=0.6, label="Current TE")
        ax_t2.set_title("T2 Decay (DWI-weighted)", fontsize=9)
        ax_t2.set_xlabel("TE (ms)", fontsize=8)

    elif seq == "MPRAGE":
        FA_r           = np.radians(FA)
        te_range_mprage = np.linspace(1, 20, 500)
        for t in brain_tissues:
            p         = TISSUES[t]
            t1_factor = abs(1 - 2*np.exp(-TI/p["T1"]) + np.exp(-TR/p["T1"])) * np.sin(FA_r)
            s         = p["PD"] * t1_factor * np.exp(-te_range_mprage/p["T2s"])
            ax_t2.plot(te_range_mprage, s, color=p["color"], label=t, linewidth=1.5)
        ax_t2.axvline(TE, color="white", linestyle="--", linewidth=1, alpha=0.6, label="Current TE")
        ax_t2.set_title("T2* Decay (MPRAGE)", fontsize=9)
        ax_t2.set_xlabel("TE (ms)", fontsize=8)

    else:
        # FSE, GRE, FLAIR, STIR
        t2_key   = "T2s" if seq == "GRE" else "T2"
        t2_title = "T2* Decay" if seq == "GRE" else "T2 Decay"
        for t in brain_tissues:
            p = TISSUES[t]
            if seq in ("FLAIR", "STIR"):
                t1_factor = abs(1 - 2*np.exp(-TI/p["T1"]) + np.exp(-TR/p["T1"]))
                s = p["PD"] * t1_factor * np.exp(-te_range/p[t2_key])
            else:
                s = p["PD"] * (1 - np.exp(-TR/p["T1"])) * np.exp(-te_range/p[t2_key])
            ax_t2.plot(te_range, s, color=p["color"], label=t, linewidth=1.5)
        ax_t2.axvline(TE, color="white", linestyle="--", linewidth=1, alpha=0.6, label="Current TE")
        if seq == "GRE":
            ax_t2.set_xlim(0, 150)
        ax_t2.set_title(t2_title, fontsize=9)
        ax_t2.set_xlabel("TE (ms)", fontsize=8)

    ax_t2.set_ylabel("Signal", fontsize=8)
    ax_t2.set_ylim(bottom=0)
    ax_t2.legend(fontsize=5, labelcolor="white", facecolor="#2b2b2b",
                 edgecolor="#555", loc="upper right")
    style_ax(ax_t2)
    st.pyplot(fig_t2, use_container_width=True)
    plt.close(fig_t2)

# ---------------------------------------------------------------------------
# Compact results row — metrics across the bottom
# ---------------------------------------------------------------------------
# Six summary metrics displayed in a single row: three CNR values, pixel size,
# estimated scan time, and one sequence-specific value (e.g. Ernst angle for
# GRE, null TI for FLAIR, optimal TE for EPI).
st.markdown("<div style='margin-top: 0.5rem;'></div>", unsafe_allow_html=True)
st.markdown("---")
m1, m2, m3, m4, m5, m6, m7 = st.columns(7)
m1.metric("CNR WM–GM",  f"{cnr_wm_gm:.1f}")
m2.metric("CNR WM–CSF", f"{cnr_wm_csf:.1f}")
m3.metric("CNR GM–CSF", f"{cnr_gm_csf:.1f}")
m4.metric("Pixel size", f"{freq_pixel_mm:.2f}×{phase_pixel_mm:.2f} mm")
m5.metric("Scan time",  f"{scan_min}m {scan_s:02d}s")
m6.metric("Voxel size", f"{freq_pixel_mm:.2f}×{phase_pixel_mm:.2f}×{slice_mm:.1f} mm")

if seq == "GRE":
    m7.metric("Ernst ∠ WM", f"{ernst_angle(TR, TISSUES['WM']['T1']):.1f}°")
elif seq == "FLAIR":
    m7.metric("CSF null TI", f"{csf_null_ti(TR, TISSUES['CSF']['T1']):.0f} ms")
elif seq == "STIR":
    m7.metric("Fat null TI", f"{csf_null_ti(TR, TISSUES['Fat']['T1']):.0f} ms")
elif seq == "bSSFP":
    m7.metric("Opt FA WM", f"{bssfp_optimal_fa(TR, TISSUES['WM']['T1'], TISSUES['WM']['T2']):.1f}°")
elif seq == "DIR":
    wm_null_m = dir_null_ti2(TR, TI1, TISSUES["WM"]["T1"])
    m7.metric("WM null TI2", f"{wm_null_m:.0f} ms" if wm_null_m else "n/a")
elif seq == "DWI":
    m7.metric("WM ADC", f"{TISSUES['WM']['ADC']*1000:.2f} ×10⁻³ mm²/s")
elif seq == "MPRAGE":
    m7.metric("CSF null TI", f"{csf_null_ti(TR, TISSUES['CSF']['T1']):.0f} ms")
elif seq == "EPI (single-shot)":
    m7.metric("Opt TEeff (GM)", f"{TISSUES['GM']['T2s']} ms  (T2*)")
else:
    m7.metric("Sequence", "FSE")

# ---------------------------------------------------------------------------
# Explain This Contrast — Claude API
# ---------------------------------------------------------------------------
# When the user clicks the button, the current sequence parameters and tissue
# signal values are packaged into a prompt and sent to Claude. The response is
# a plain-English explanation of why the image looks the way it does, written
# at a level suitable for a radiology resident or MRI student.
st.markdown("---")

if "explanation" not in st.session_state:
    st.session_state.explanation = ""

col_btn, col_status = st.columns([1, 3])

with col_btn:
    clicked = st.button("Explain This Contrast")

with col_status:
    status = st.empty()

if clicked:
    st.components.v1.html(
        f"<script>window.parent.gtag('event', 'explain_contrast_click', "
        f"{json.dumps({'sequence': seq})});</script>",
        height=0
    )
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        status.error("ANTHROPIC_API_KEY environment variable not set.")
    else:
        if seq == "GRE":
            extra_params = f"Flip Angle: {FA}°  (Ernst angle for WM ≈ {ernst_angle(TR, TISSUES['WM']['T1']):.1f}°)"
        elif seq == "FLAIR":
            extra_params = f"TI: {TI} ms  (CSF null TI ≈ {csf_null_ti(TR, TISSUES['CSF']['T1']):.0f} ms)"
        elif seq == "STIR":
            extra_params = f"TI: {TI} ms  (Fat null TI ≈ {csf_null_ti(TR, TISSUES['Fat']['T1']):.0f} ms)"
        elif seq == "bSSFP":
            fa_opt = bssfp_optimal_fa(TR, TISSUES['WM']['T1'], TISSUES['WM']['T2'])
            extra_params = (f"Flip Angle: {FA}°  (Optimal FA for WM ≈ {fa_opt:.1f}°)  "
                            f"TE = TR/2 = {TE:.1f} ms")
        elif seq == "DIR":
            wm_n = dir_null_ti2(TR, TI1, TISSUES['WM']['T1'])
            wm_n_str = f"{wm_n:.0f} ms" if wm_n else "n/a"
            extra_params = f"TI1: {TI1} ms  TI2: {TI2} ms  (WM null TI2 ≈ {wm_n_str})"
        elif seq == "DWI":
            extra_params = (f"b-value: {b} s/mm²  ADC: WM={TISSUES['WM']['ADC']*1000:.2f}, "
                            f"GM={TISSUES['GM']['ADC']*1000:.2f}, "
                            f"CSF={TISSUES['CSF']['ADC']*1000:.2f} ×10⁻³ mm²/s")
        elif seq == "MPRAGE":
            csf_n = csf_null_ti(TR, TISSUES['CSF']['T1'])
            extra_params = (f"TI: {TI} ms  TE: {TE} ms  FA: {FA}°  "
                            f"Partitions: {Npartitions}  3D acquisition  "
                            f"(CSF null TI ≈ {csf_n:.0f} ms)")
        elif seq == "EPI (single-shot)":
            extra_params = (f"TEeff: {TE} ms  FA: 90° (fixed)  "
                            f"ESP: {_epi_esp_ms:.1f} ms  "
                            f"k-space centre echo: #{_epi_eff_idx + 1} of {phase_matrix}  "
                            f"T2* values: WM={TISSUES['WM']['T2s']} ms, "
                            f"GM={TISSUES['GM']['T2s']} ms  "
                            f"Optimal BOLD TEeff ≈ {TISSUES['GM']['T2s']} ms")
        else:  # FSE
            extra_params = f"ETL: {ETL}"

        prompt = f"""You are an MRI physics educator. Explain in plain language why the brain phantom
image currently looks the way it does, based on these scan parameters and tissue properties.

Sequence: {seq}
TR: {TR} ms
TE: {TE} ms
{extra_params}

Tissue properties at {field_strength}:
- White Matter (WM):  T1={TISSUES['WM']['T1']} ms, T2={TISSUES['WM']['T2']} ms, T2*={TISSUES['WM']['T2s']} ms, Signal={signals['WM']:.3f}
- Gray Matter (GM):   T1={TISSUES['GM']['T1']} ms, T2={TISSUES['GM']['T2']} ms, T2*={TISSUES['GM']['T2s']} ms, Signal={signals['GM']:.3f}
- CSF:                T1={TISSUES['CSF']['T1']} ms, T2={TISSUES['CSF']['T2']} ms, T2*={TISSUES['CSF']['T2s']} ms, Signal={signals['CSF']:.3f}

Explain:
1. What type of contrast weighting this represents (T1, T2, T2*, PD, diffusion, etc.)
2. Why each tissue appears bright or dark
3. What clinical information this contrast is useful for

Write at a level suitable for a radiology resident or MRI student. Be concise and clear."""

        status.markdown("⏳ Claude is generating an explanation...")
        try:
            client = anthropic.Anthropic(api_key=api_key)
            message = client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=500,
                messages=[{"role": "user", "content": prompt}]
            )
            st.session_state.explanation = message.content[0].text
            status.markdown("✅ Explanation ready")
        except anthropic.APIError as exc:
            status.error(f"Claude API error: {exc}")
        except Exception as exc:
            status.error(f"Could not generate explanation: {exc}")

if st.session_state.explanation:
    escaped_explanation = html.escape(st.session_state.explanation).replace("\n", "<br>")
    st.markdown(f"""
    <div style='background-color:#1e1e1e; border-left:4px solid #4a90d9;
                padding:1rem 1.2rem; border-radius:6px; color:white;
                font-size:0.95rem; line-height:1.6;'>
    {escaped_explanation}
    </div>
    """, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# K-space visualiser
# ---------------------------------------------------------------------------
# Displays (left) the 2-D FFT magnitude of the current axial phantom slice
# with log scaling and (right) the image reconstructed from only the filled
# portion of k-space.
#
# Two independent fill selectors:
#   • ky fill (rows, axis 0): symmetric centre-out for all sequences except
#     FSE, which uses centric ordering (ky = 0 first, then ±1, ±2, …).
#   • kx fill (columns, axis 1): always symmetric centre-out.
# Only the intersection of the selected ky rows and kx columns is filled.
# ---------------------------------------------------------------------------

st.markdown("---")
st.markdown("### K-space Visualiser")

# ── Helpers ──────────────────────────────────────────────────────────────────

def _ks_centric_rows(N: int, n_lines: int) -> list:
    """First n_lines row indices in FSE centric order (ky=0 first, outward)."""
    centre = N // 2
    order  = [centre]
    for d in range(1, N):
        if len(order) >= n_lines:
            break
        if centre + d < N:
            order.append(centre + d)
        if len(order) >= n_lines:
            break
        if centre - d >= 0:
            order.append(centre - d)
    return order[:n_lines]

def _render_kspace_panels(ksp_partial: np.ndarray, col_left, col_right,
                           ksp_recon: np.ndarray = None,
                           vmin: float = 0.0, vmax: float = 1.0) -> None:
    """Draw log-magnitude k-space (left) and IFFT reconstruction (right).

    ksp_partial  — used for the magnitude display (clean, no noise).
    ksp_recon    — used for the IFFT reconstruction; defaults to ksp_partial.
                   Pass a noise-added copy to show realistic image SNR.
    vmin / vmax  — window/level bounds in base-signal space, matching the main
                   phantom display so contrast is identical.
    """
    if ksp_recon is None:
        ksp_recon = ksp_partial

    # K-space magnitude (clean, no noise)
    ksp_mag = np.log1p(np.abs(ksp_partial))
    fig_k, ax_k = plt.subplots(figsize=(2.8, 2.8), facecolor="#1e1e1e")
    ax_k.imshow(ksp_mag, cmap="inferno", interpolation="nearest", origin="upper")
    ax_k.set_title("K-space (log |F|)", color="white", fontsize=8, pad=3)
    ax_k.axis("off")
    fig_k.tight_layout(pad=0.3)
    with col_left:
        st.pyplot(fig_k, use_container_width=True)
    plt.close(fig_k)

    # Reconstructed image via IFFT (from noisy k-space).
    # Values are in base-signal space (same scale as _ks_img), so vmin/vmax
    # from the main W/L sliders apply directly — no normalisation needed.
    recon = np.abs(np.fft.ifft2(np.fft.ifftshift(ksp_recon)))
    fig_r, ax_r = plt.subplots(figsize=(2.8, 2.8), facecolor="#1e1e1e")
    ax_r.imshow(np.flipud(recon), cmap="gray", vmin=vmin, vmax=vmax,
                interpolation="nearest")
    ax_r.set_title("Reconstruction of Current Axial Image", color="white", fontsize=8, pad=3)
    ax_r.axis("off")
    fig_r.tight_layout(pad=0.3)
    with col_right:
        st.pyplot(fig_r, use_container_width=True)
    plt.close(fig_r)


# ── Precompute full k-space from the clean axial image ───────────────────────
# _kspace_source_img is captured inside the phantom rendering loop (before
# noise), so it reflects current contrast, FOV, matrix, and slice parameters.
_ks_N = 128   # display matrix (power of 2, keeps rendering fast)
from scipy.ndimage import zoom as _zoom
_ks_scale = _ks_N / _kspace_source_img.shape[0]
if abs(_ks_scale - 1.0) > 0.01:
    _ks_img = _zoom(_kspace_source_img, _ks_scale, order=1)
else:
    _ks_img = _kspace_source_img.copy()
_ks_img      = np.clip(_ks_img, 0, None)
_kspace_full = np.fft.fftshift(np.fft.fft2(_ks_img))

# ── Controls row: Fill Region | ky fill | kx fill | Auto W/L ─────────────────
_ks_reg_col, _ks_ky_col, _ks_kx_col, _ks_wl_col = st.columns([1.5, 2, 2, 2.5])
with _ks_reg_col:
    _ks_region = st.radio(
        "Fill region",
        ["Center", "Edges"],
        index=0,
        horizontal=False,
        key="kspace_region",
    )
    if _ks_region == "Edges":
        st.caption(
            "If the reconstructed image is not apparent, please manually "
            "adjust the window and/or level settings to better visualize "
            "the image under the current k-space filling selections."
        )
with _ks_ky_col:
    _ky_order_name = (
        "ky fill — centric (FSE)" if seq == "FSE"
        else "ky fill (rows)"
    )
    _ky_pct_label = st.radio(
        _ky_order_name,
        ["10%", "25%", "50%", "75%", "100%"],
        index=4,
        horizontal=False,
        key="kspace_ky_pct",
    )
with _ks_kx_col:
    _kx_pct_label = st.radio(
        "kx fill (columns)",
        ["10%", "25%", "50%", "75%", "100%"],
        index=4,
        horizontal=False,
        key="kspace_kx_pct",
    )

with _ks_wl_col:
    _ks_auto_wl = st.checkbox(
        "Auto W/L", key="ks_auto_wl",
        help="Automatically compute optimal window/level from the reconstructed image on every run.",
    )

# ── Build partial k-space for the selected percentages ───────────────────────
_ky_frac = int(_ky_pct_label.rstrip("%")) / 100
_kx_frac = int(_kx_pct_label.rstrip("%")) / 100
_n_ky    = max(1, round(_ks_N * _ky_frac))
_n_kx    = max(1, round(_ks_N * _kx_frac))

# ky rows to fill
if _ks_region == "Center":
    if seq == "FSE":
        _ky_rows = _ks_centric_rows(_ks_N, _n_ky)
    else:
        _ky_ctr        = _ks_N // 2
        _ky_half_below = _n_ky // 2
        _ky_half_above = _n_ky - _ky_half_below
        _ky_r_start    = max(0, _ky_ctr - _ky_half_below)
        _ky_r_end      = min(_ks_N, _ky_ctr + _ky_half_above)
        _ky_rows       = list(range(_ky_r_start, _ky_r_end))
else:  # Edges — outermost _n_ky rows, split evenly top/bottom
    if seq == "FSE":
        # Complement of the innermost (_ks_N - _n_ky) centric rows
        _ky_inner = set(_ks_centric_rows(_ks_N, _ks_N - _n_ky))
        _ky_rows  = sorted(set(range(_ks_N)) - _ky_inner)
    else:
        _ky_top = _n_ky // 2
        _ky_bot = _n_ky - _ky_top
        _ky_rows = list(range(0, _ky_top)) + list(range(_ks_N - _ky_bot, _ks_N))

# kx columns to fill
if _ks_region == "Center":
    _kx_ctr        = _ks_N // 2
    _kx_half_below = _n_kx // 2
    _kx_half_above = _n_kx - _kx_half_below
    _kx_c_start    = max(0, _kx_ctr - _kx_half_below)
    _kx_c_end      = min(_ks_N, _kx_ctr + _kx_half_above)
    _kx_cols       = list(range(_kx_c_start, _kx_c_end))
else:  # Edges — outermost _n_kx columns, split evenly left/right
    _kx_left  = _n_kx // 2
    _kx_right = _n_kx - _kx_left
    _kx_cols  = list(range(0, _kx_left)) + list(range(_ks_N - _kx_right, _ks_N))

# Fill k-space:
#   Center — AND: intersection of selected rows × columns (central rectangle).
#   Edges, both axes partial — OR: any point where ky OR kx is in the outer
#     set, producing a continuous outer border rather than isolated corners.
#   Edges, one axis at 100% — AND via np.ix_: the 100 % axis selects all
#     rows/cols, so its partner axis constraint is always respected.  The OR
#     approach would fill the entire array via the "all rows × all cols" term,
#     ignoring the partial constraint on the other axis entirely.
_kspace_partial = np.zeros_like(_kspace_full)
if _ks_region == "Center":
    _kspace_partial[np.ix_(_ky_rows, _kx_cols)] = _kspace_full[np.ix_(_ky_rows, _kx_cols)]
elif _n_ky == _ks_N or _n_kx == _ks_N:
    # At least one axis is 100 %: use AND so the partial axis still constrains.
    _kspace_partial[np.ix_(_ky_rows, _kx_cols)] = _kspace_full[np.ix_(_ky_rows, _kx_cols)]
else:
    # Both axes partial: OR for continuous border.
    _kspace_partial[_ky_rows, :] = _kspace_full[_ky_rows, :]
    _kspace_partial[:, _kx_cols] = _kspace_full[:, _kx_cols]

# ── Add k-space noise to match the main display noise floor ──────────────────
# Physics: for an N×N IFFT, image-space noise std = σ_k / N.
# noise_std (= noise_floor / _vol_scale) is already in base-signal (0–1) space,
# matching the pixel values of _ks_img.  So σ_k = noise_std × _ks_N gives
# image-space noise equal to noise_std at 100 % k-space fill.
# Noise is added only to filled positions; unfilled (zero) positions are left
# at zero, so partial-fill images show both blurring and realistic noise.
_ks_sigma = noise_std * _ks_N
_ks_noise_real = np.random.normal(0.0, _ks_sigma, _kspace_partial.shape)
_ks_noise_imag = np.random.normal(0.0, _ks_sigma, _kspace_partial.shape)
_ks_filled_mask = _kspace_partial != 0
_kspace_noisy = _kspace_partial.copy()
_kspace_noisy[_ks_filled_mask] += (
    _ks_noise_real[_ks_filled_mask] + 1j * _ks_noise_imag[_ks_filled_mask]
)

# ── Reco W/L: auto or manual ──────────────────────────────────────────────────
# Reconstruct now so Auto W/L can compute stats before the sliders render.
_ks_recon_for_wl = np.abs(np.fft.ifft2(np.fft.ifftshift(_kspace_noisy)))
if _ks_auto_wl:
    # Compute optimal W/L directly from the current reconstruction.
    st.session_state.ks_reco_window = float(np.clip(2.0 * float(np.std(_ks_recon_for_wl)),  0.01, 2.0))
    st.session_state.ks_reco_level  = float(np.clip(float(np.mean(_ks_recon_for_wl)), 0.0,  1.0))
    _ks_reco_window = st.session_state.ks_reco_window
    _ks_reco_level  = st.session_state.ks_reco_level
else:
    _, _ks_slider_col = st.columns([6.5, 2.5])
    with _ks_slider_col:
        if st.button("Reset Reco Image W/L to Optimal", key="ks_reco_wl_reset"):
            st.session_state.ks_reco_window = float(np.clip(2.0 * float(np.std(_ks_recon_for_wl)), 0.01, 2.0))
            st.session_state.ks_reco_level  = float(np.clip(float(np.mean(_ks_recon_for_wl)), 0.0, 1.0))
        _ks_reco_window = st.slider(
            "Reco Window", 0.01, 2.0, step=0.01, key="ks_reco_window",
        )
        _ks_reco_level = st.slider(
            "Reco Level", 0.0, 1.0, step=0.01, key="ks_reco_level",
        )

# ── Caption row: left column above k-space figure ────────────────────────────
_ks_cap_l, _ks_cap_r = st.columns(2)
with _ks_cap_l:
    st.caption(
        "**ky** fills rows (phase-encode direction).  \n"
        "**kx** fills columns (frequency-encode direction).  \n"
        "**Center**: innermost lines outward.  \n"
        "**Edges**: outermost lines inward, excluding centre.  \n"
        "Only the intersection of selected rows × columns is filled."
    )

# ── Images row: both columns start at the same vertical position ──────────────
_ks_reco_vmin = _ks_reco_level - _ks_reco_window / 2
_ks_reco_vmax = _ks_reco_level + _ks_reco_window / 2

_col_ksp, _col_recon = st.columns(2)
_render_kspace_panels(_kspace_partial, _col_ksp, _col_recon,
                      ksp_recon=_kspace_noisy,
                      vmin=_ks_reco_vmin, vmax=_ks_reco_vmax)
st.caption(
    f"{len(_ky_rows)} of {_ks_N} ky rows filled ({_ky_pct_label})  ·  "
    f"{len(_kx_cols)} of {_ks_N} kx columns filled ({_kx_pct_label})  ·  "
    f"Left: k-space magnitude (log scale)  ·  "
    f"Right: image reconstructed from filled region only"
)
