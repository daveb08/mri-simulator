# Standard library and third-party imports needed for numerics, plotting, the
# web UI, and the Claude API.
import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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

def calc_scan_time(TR, matrix, ETL, NEX, seq, Npartitions=1):
    if seq == "GRE":
        return TR * matrix * NEX / 1000.0
    else:
        return TR * (matrix / ETL) * Npartitions * NEX / 1000.0

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
if "params_sig" not in st.session_state:
    st.session_state.params_sig = None
if "wl_window"  not in st.session_state:
    st.session_state.wl_window  = 1.0
if "wl_level"   not in st.session_state:
    st.session_state.wl_level   = 0.5
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
    st.markdown("**Sequence**")
    seq = st.radio("", ["FSE", "GRE", "FLAIR", "STIR", "bSSFP", "DIR", "DWI", "MPRAGE", "EPI"],
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

    # TR — sequence-specific range
    if   seq == "FLAIR":   TR = st.slider("TR (ms)",  3000, 15000,  9000, 100)
    elif seq == "STIR":    TR = st.slider("TR (ms)",  1000,  6000,  3000,  50)
    elif seq == "bSSFP":   TR = st.slider("TR (ms)",     3,    20,     5,   1)
    elif seq == "DIR":     TR = st.slider("TR (ms)",  5000, 15000,  8000, 100)
    elif seq == "DWI":     TR = st.slider("TR (ms)",  3000,  8000,  5000, 100)
    elif seq == "MPRAGE":  TR = st.slider("TR (ms)",  2000,  4000,  2300, 100)
    elif seq == "EPI":     TR = st.slider("TR (ms)",   500,  3000,  2000, 100)
    else:                  TR = st.slider("TR (ms)",   500, 10000,  4000,  50)

    # Initialise variables that may not be set by every sequence branch
    b   = 0
    TI  = 0
    TI1 = 0
    TI2 = 0

    if seq == "FSE":
        TE  = st.slider("TE eff. (ms)", 10, 300,  80, 5)
        ETL = st.slider("ETL",           1,  32,  16, 1)
        FA  = 90

    elif seq == "GRE":
        TE  = st.slider("TE (ms)",       2, 100,   5, 1)
        FA  = st.slider("Flip Angle (°)",1,  90,  30, 1)
        ETL = 1

    elif seq == "FLAIR":
        TI  = st.slider("TI (ms)",     500, 4000, 2500,  50)
        TE  = st.slider("TE eff. (ms)", 50,  200,   90,   5)
        ETL = st.slider("ETL",           1,   32,   16,   1)
        FA  = 90

    elif seq == "STIR":
        TI  = st.slider("TI (ms)",      50,  400,  210,  10)
        TE  = st.slider("TE eff. (ms)", 10,  100,   30,   5)
        ETL = st.slider("ETL",           1,   32,    8,   1)
        FA  = 90

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
        TE  = st.slider("TE (ms)",          2,    6,    3,  1)
        FA  = st.slider("Flip Angle (°)",   5,   15,    9,  1)
        ETL = 1

    else:  # EPI
        TE  = st.slider("TE (ms)", 15, 80, 30, 1)
        FA  = 90
        ETL = 1
        st.caption(f"Optimal TE for BOLD ≈ T2* of GM = {TISSUES['GM']['T2s']} ms")

    matrix = st.slider("Matrix (px)", 64, 512, 256, 32)
    if seq == "EPI":
        ETL = matrix  # single-shot: one TR acquires full matrix

    FOV = st.slider("FOV (mm)", 180, 400, 240, 10)
    st.caption(f"In-plane pixel size: {FOV / matrix:.2f} mm")

    if is_3d:
        Npartitions = st.slider("Partitions", 32, 256, 176, 8)
        slice_mm    = 1.0
    else:
        slice_mm    = st.slider("Slice (mm)", 1, 10, 5, 1)
        Npartitions = 1

    NEX     = st.slider("NEX",               1,   8,   1,  1)
    BW      = st.slider("Bandwidth (Hz/px)", 50, 500, 200, 10)
    fat_sat = st.checkbox("Fat Saturation")
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
    _params_sig = (seq, TR, TE, FA, TI, TI1, TI2, b, fat_sat, FOV, matrix, slice_mm)
    if _params_sig != st.session_state.params_sig:
        _sigs = []
        for _t in ["WM", "GM", "CSF"]:
            _p = TISSUES[_t]
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
        st.session_state.params_sig = _params_sig
        st.session_state.wl_level   = float(np.clip(np.mean(_arr),       0.0,  1.0))
        st.session_state.wl_window  = float(np.clip(2.0 * np.std(_arr),  0.01, 2.0))

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
voxel_vol  = (FOV / matrix) ** 2 * _slice_for_vol
_vol_scale = voxel_vol / _ref_vox
signals    = {t: round(base_signals[t] * _vol_scale, 2) for t in base_signals}

# STEP 2 — System noise floor.
# Depends ONLY on BW and NEX — Npartitions has no influence whatsoever.
# Never changes with FOV, matrix, slice, Npartitions, TR, TE, TI, or flip angle.
noise_floor = round(NOISE_REF * np.sqrt(BW / BW_REF) / np.sqrt(NEX / NEX_REF), 2)

# STEP 3 — SNR = Signal / Noise, derived directly with no other dependencies.
# signals and noise_floor are pre-rounded to 2 dp so displayed values are exact inputs.
snrs = {t: signals[t] / noise_floor if noise_floor > 0 else 0.0 for t in signals}

names    = list(TISSUES.keys())
colors   = [TISSUES[t]["color"] for t in names]
pixel_mm = FOV / matrix
scan_sec = calc_scan_time(TR, matrix, ETL, NEX, seq, Npartitions)
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
st.markdown("<h3 style='text-align: center;'>MRI Simulator — Brain (3T)</h3>",
            unsafe_allow_html=True)

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
FOV_MAX = 400.0
half_w  = 128.0 * FOV / FOV_MAX
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
    if matrix < 256:
        step = max(1, 256 // matrix)
        img  = img[::step, ::step]
        img  = np.repeat(np.repeat(img, step, axis=0), step, axis=1)
        img  = img[:256, :256]  # trim any overshoot from rounding
    # Partial volume averaging: blur proportional to slice thickness.
    # sigma = 0 at 1 mm (sharp), rising to ~2.5 at 10 mm (heavy blurring).
    pva_sigma = (slice_mm - 1) * 0.25
    if pva_sigma > 0:
        img = gaussian_filter(img, sigma=pva_sigma)
    if noise_std > 0:
        img = np.clip(img + np.random.normal(0, noise_std, img.shape), 0, 1)
    with col:
        fig_p, ax_p = plt.subplots(figsize=(2.0, 2.0), facecolor="#1e1e1e")
        ax_p.imshow(np.flipud(img), cmap="gray", vmin=vmin, vmax=vmax,
                    interpolation="nearest")
        ax_p.axhline(ch_y, color=hcol, linewidth=0.8, alpha=0.8)
        ax_p.axvline(ch_x, color=vcol, linewidth=0.8, alpha=0.8)
        ax_p.set_xlim(128 - half_w, 128 + half_w)
        ax_p.set_ylim(128 + half_w, 128 - half_w)
        ax_p.axis("off")
        ax_p.set_title(plane_name, color="white", fontsize=8, pad=2)
        ax_p.text(0.5, 0.02, f"W:{wl_window:.2f} L:{wl_level:.2f}",
                  transform=ax_p.transAxes, ha="center", va="bottom",
                  color="white", fontsize=5,
                  bbox=dict(facecolor="black", alpha=0.5, edgecolor="none", pad=1))
        st.pyplot(fig_p, use_container_width=True)
        plt.close(fig_p)

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
    ax_snr.set_title("SNR", fontsize=9)
    for bar, val in zip(bars2, snr_vals):
        ax_snr.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.2,
                    f"{val:.2f}", ha="center", color="white", fontsize=5)
    style_ax(ax_snr)
    st.pyplot(fig_snr, use_container_width=True)
    plt.close(fig_snr)

# --- Relaxation curves ---
brain_tissues = ["WM", "GM", "CSF"]
tr_range = np.linspace(100, 10000, 500)
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
        ax_t1.set_xlim(0, 6000)
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
m4.metric("Pixel size", f"{pixel_mm:.2f} mm")
m5.metric("Scan time",  f"{scan_min}m {scan_s:02d}s")
voxel_vol = pixel_mm * pixel_mm * slice_mm
m6.metric("Voxel size", f"{pixel_mm:.2f}×{pixel_mm:.2f}×{slice_mm:.1f} mm")

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
