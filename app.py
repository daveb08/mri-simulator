# Standard library and third-party imports needed for numerics, plotting, the
# web UI, and the Claude API.
import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.transforms as mtransforms
import streamlit as st
import anthropic
from scipy.ndimage import gaussian_filter

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
}

# SNR scales approximately linearly with B0 (for a given coil / bandwidth).
# Factors below represent SNR relative to 3 T (= 1.000).
FIELD_SNR_SCALE = {
    "0.064T": 0.021,
    "0.5T":   0.167,
    "1.0T":   0.333,
    "1.5T":   0.500,
    "3.0T":   1.000,
}

# Maximum clinically useful TR per field strength.
# T1 values shorten at lower B0, so very long TRs add no contrast benefit.
# These values set both the TR slider ceiling and the T1 recovery curve x-axis limit.
TR_MAX_BY_FIELD = {
    "0.064T": 1000,
    "0.5T":   2000,
    "1.0T":   3000,
    "1.5T":   4000,
    "3.0T":   6000,
}

SNR_SCALE = 200.0

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

def calc_snr(signal, FOV, matrix, slice_mm, NEX, BW, Npartitions=1):
    pixel_mm  = FOV / matrix
    voxel_vol = pixel_mm * pixel_mm * slice_mm
    return signal * voxel_vol * np.sqrt(NEX * Npartitions) / np.sqrt(BW) * SNR_SCALE

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


def draw_pulse_sequence(seq, TR, TE, TI, FA, ETL, slice_mm=5, BW=200, FOV_read=240):
    """Return a matplotlib Figure with an oscilloscope-style pulse sequence
    diagram for FSE, GRE, FLAIR, STIR, bSSFP.  Returns None for others."""
    if seq not in ("FSE", "GRE", "FLAIR", "STIR", "bSSFP"):
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

    st.markdown("**Field Strength**")
    field_strength = st.radio(
        "",
        ["0.064T", "0.5T", "1.0T", "1.5T", "3.0T"],
        index=4,          # default: 3.0T
        horizontal=True,
        key="field_strength",
    )
    _tr_max = TR_MAX_BY_FIELD[field_strength]

    st.markdown("**Sequence**")
    seq = st.radio("", ["FSE", "GRE", "FLAIR", "STIR", "bSSFP", "DIR", "DWI", "MPRAGE", "EPI (single-shot)"],
                   horizontal=True)

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
        TR = st.slider("TR (ms)",  3000, 10000, 9000, 100)  # fixed range — FLAIR needs long TR at all field strengths
    elif seq == "STIR":
        _fm = max(1050, _tr_max)   # STIR min TR = 1000 ms
        TR = st.slider("TR (ms)",  1000, _fm, min(3000, _fm),  50)
    elif seq == "bSSFP":
        TR = st.slider("TR (ms)",     3,   20,     5,   1)   # hardware-constrained; field-strength cap irrelevant
    elif seq == "DIR":
        _fm = max(5100, _tr_max)   # DIR min TR = 5000 ms
        TR = st.slider("TR (ms)",  5000, _fm, min(8000, _fm), 100)
    elif seq == "DWI":
        _fm = max(3100, _tr_max)   # DWI min TR = 3000 ms
        TR = st.slider("TR (ms)",  3000, _fm, min(5000, _fm), 100)
    elif seq == "MPRAGE":
        _fm = max(2100, _tr_max)   # MPRAGE min TR = 2000 ms
        TR = st.slider("TR (ms)",  2000, _fm, min(2300, _fm), 100)
    elif seq == "EPI":
        _fm = max(600, min(3000, _tr_max))   # EPI: cap at min(3000, field max)
        TR = st.slider("TR (ms)",   500, _fm, min(2000, _fm), 100)
    else:                                    # FSE, GRE — directly tied to T1 recovery curve
        TR = st.slider("TR (ms)",   500, _tr_max, min(4000, _tr_max),  50)

    # Initialise variables that may not be set by every sequence branch
    b   = 0
    TI  = 0
    TI1 = 0
    TI2 = 0

    if seq == "FSE":
        TE  = st.slider("TE eff. (ms)", 10, 300,  80, 5)
        ETL = st.slider("ETL",           1,  32,  16, 1)
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
        TE  = st.slider("TE (ms)",       2, 100,   5, 1)
        FA  = st.slider("Flip Angle (°)",1,  90,  30, 1)
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
        TI  = st.slider("TI (ms)",     500, 4000, 2500,  50)
        TE  = st.slider("TE eff. (ms)", 25,  250,   90,   5)
        ETL = st.slider("ETL",           1,   32,   16,   1)
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
        TI  = st.slider("TI (ms)",      50,  400,  210,  10)
        TE  = st.slider("TE eff. (ms)", 25,  250,   60,   5)
        ETL = st.slider("ETL",           1,   32,    8,   1)
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
        FA  = st.slider("Flip Angle (°)", 10, 90, 50, 1)
        TE  = TR / 2
        ETL = 1
        st.caption(f"TE = TR/2 = {TE:.1f} ms  (fixed by sequence design)")

    elif seq == "DIR":
        TI1 = st.slider("TI1 (ms)", 2000, 5000, 3400, 50)
        TI2 = st.slider("TI2 (ms)",  100, 2500,  800, 50)
        TE  = st.slider("TE eff. (ms)", 10, 100,  25,  5)
        ETL = st.slider("ETL",           1,  32,   8,  1)
        FA  = 90
        TI  = TI1

    elif seq == "DWI":
        b   = st.slider("b-value (s/mm²)", 0, 3000, 1000, 100)
        TE  = st.slider("TE (ms)",        50,  150,   80,   5)
        ETL = 1
        FA  = 90
        st.caption(f"Diffusion attenuation: exp(−b×ADC)  |  b={b} s/mm²")

    elif seq == "MPRAGE":
        TI  = st.slider("TI (ms)",        200, 2500,  900, 10)
        TE  = st.slider("TE (ms)",          2,    18,    3,  1)
        FA  = st.slider("Flip Angle (°)",   5,   15,    9,  1)
        ETL = 1
        MPRAGE_TR_READOUT = 7  # ms — fixed gradient echo spacing within partition train

    else:  # EPI
        TE  = st.slider("TE (ms)", 15, 80, 30, 1)
        FA  = 90
        ETL = 1
        st.caption(f"Optimal TE for BOLD ≈ T2* of GM = {TISSUES['GM']['T2s']} ms")

    freq_matrix  = st.slider("Frequency Matrix (px)", 64, 512, 256, 32)
    phase_matrix = st.slider("Phase Matrix (px)",     64, 512, 256, 32)
    if seq == "EPI":
        ETL = phase_matrix  # single-shot EPI: all phase encodes in one TR, so scan time = TR × NEX

    FOV_read  = st.slider("FOV Read (mm)",  180, 400, 400, 10)
    FOV_phase = st.slider("FOV Phase (mm)", 180, 400, 400, 10)
    st.caption(
        f"Frequency pixel: {FOV_read / freq_matrix:.2f} mm  |  "
        f"Phase pixel: {FOV_phase / phase_matrix:.2f} mm"
    )

    if is_3d:
        Npartitions = st.slider("Partitions", 32, 256, 176, 8)
        slice_mm    = 1.0
    else:
        slice_mm    = st.slider("Slice (mm)", 1, 10, 5, 1)
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

    NEX     = st.slider("NEX",               1,   8,   1,  1)
    BW      = st.slider("Bandwidth (Hz/px)", 50, 500, 200, 10)
    fat_sat = st.checkbox("Fat Saturation")

    _PF_OPTIONS = {"8/8 (Full)": 1.0, "7/8": 7/8, "6/8": 6/8, "5/8": 5/8}
    if seq in ("FSE", "GRE", "EPI (single-shot)"):
        _pf_label    = st.selectbox("Partial Fourier (phase)", list(_PF_OPTIONS.keys()), index=0)
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
    x_pos = st.slider("X  (Sagittal)", 0, 361, 181, 1)
    y_pos = st.slider("Y  (Coronal)",  0, 433, 217, 1)
    z_pos = st.slider("Z  (Axial)",    0, 361, 181, 1)

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
noise_floor = round(_noise_ref_scaled * np.sqrt(BW / BW_REF) / np.sqrt(NEX / NEX_REF), 2)

# STEP 3 — SNR = Signal / Noise, derived directly with no other dependencies.
# signals and noise_floor are pre-rounded to 2 dp so displayed values are exact inputs.
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
plane_configs = [
    # (name, slice_idx, crosshair_row_display, crosshair_col_display, hcolor, vcolor)
    ("Axial",    z_pos, 255 - y_pos/433*255, x_pos/361*255, "#44FF44", "#FF4444"),
    ("Coronal",  y_pos, 255 - z_pos/361*255, x_pos/361*255, "#4488FF", "#FF4444"),
    ("Sagittal", x_pos, 255 - z_pos/361*255, y_pos/433*255, "#4488FF", "#44FF44"),
]

col_ax, col_cor, col_sag = st.columns(3)
for col, (plane_name, idx, ch_y, ch_x, hcol, vcol) in zip(
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
    # Capture the clean axial image for k-space visualisation (no noise, full
    # processing applied).  Must be saved before noise is added below.
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
        st.pyplot(fig_p, use_container_width=True)
        plt.close(fig_p)

# --- Pulse sequence diagram (between phantom images and signal bars) ---
_psd_fig = draw_pulse_sequence(seq, TR, TE, TI, FA, ETL, slice_mm=slice_mm, BW=BW, FOV_read=FOV_read)
if _psd_fig is not None:
    st.pyplot(_psd_fig, use_container_width=True)
    plt.close(_psd_fig)

# --- Row 2: Signal/SNR bars and relaxation curves ---
col_bars, col_curves = st.columns([2, 3])

with col_bars:
    fig_sig, ax_sig = plt.subplots(figsize=(2.5, 1.5), facecolor="#1e1e1e")
    sig_vals = [signals[t] for t in names]
    bars = ax_sig.bar(names, sig_vals, color=colors, width=0.5)
    ax_sig.set_title(f"Signal (Noise floor = {noise_floor:.2f})", fontsize=9)

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
            elif seq == "EPI":
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

    elif seq == "EPI":
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
elif seq == "EPI":
    m7.metric("Opt TE (GM)", f"{TISSUES['GM']['T2s']} ms")
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
        elif seq == "EPI":
            extra_params = (f"TE: {TE} ms  FA: 90° (fixed)  "
                            f"T2* values: WM={TISSUES['WM']['T2s']} ms, "
                            f"GM={TISSUES['GM']['T2s']} ms  "
                            f"Optimal BOLD TE ≈ {TISSUES['GM']['T2s']} ms")
        else:  # FSE
            extra_params = f"ETL: {ETL}"

        prompt = f"""You are an MRI physics educator. Explain in plain language why the brain phantom
image currently looks the way it does, based on these scan parameters and tissue properties.

Sequence: {seq}
TR: {TR} ms
TE: {TE} ms
{extra_params}

Tissue properties at 3T:
- White Matter (WM):  T1={TISSUES['WM']['T1']} ms, T2={TISSUES['WM']['T2']} ms, T2*={TISSUES['WM']['T2s']} ms, Signal={signals['WM']:.3f}
- Gray Matter (GM):   T1={TISSUES['GM']['T1']} ms, T2={TISSUES['GM']['T2']} ms, T2*={TISSUES['GM']['T2s']} ms, Signal={signals['GM']:.3f}
- CSF:                T1={TISSUES['CSF']['T1']} ms, T2={TISSUES['CSF']['T2']} ms, T2*={TISSUES['CSF']['T2s']} ms, Signal={signals['CSF']:.3f}

Explain:
1. What type of contrast weighting this represents (T1, T2, T2*, PD, diffusion, etc.)
2. Why each tissue appears bright or dark
3. What clinical information this contrast is useful for

Write at a level suitable for a radiology resident or MRI student. Be concise and clear."""

        status.markdown("⏳ Claude is generating an explanation...")
        client = anthropic.Anthropic(api_key=api_key)
        message = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=500,
            messages=[{"role": "user", "content": prompt}]
        )
        st.session_state.explanation = message.content[0].text
        status.markdown("✅ Explanation ready")

if st.session_state.explanation:
    st.markdown(f"""
    <div style='background-color:#1e1e1e; border-left:4px solid #4a90d9;
                padding:1rem 1.2rem; border-radius:6px; color:white;
                font-size:0.95rem; line-height:1.6;'>
    {st.session_state.explanation.replace(chr(10), "<br>")}
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
