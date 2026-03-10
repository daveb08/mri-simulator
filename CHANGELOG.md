# MRI Simulator – Changelog

All notable changes to `app.py` are documented here.

---

## Session 2 — Partial Volume Averaging & In-Plane Resolution

### Added
- **Partial volume averaging (PVA)** tied to slice thickness slider: applies `scipy.ndimage.gaussian_filter` with `sigma = (slice_mm - 1) * 0.5` to the phantom before noise addition; no blur at 1 mm, increasing blur at thicker slices.
- **Voxel size readout**: inline caption beneath the Slice slider displays current voxel dimensions as `pixel_mm × pixel_mm × slice_mm mm`.
- **In-plane resolution simulation** tied to matrix size slider: when matrix < 256 the phantom is downsampled then upsampled with nearest-neighbour interpolation to simulate pixelation from reduced k-space coverage.
- **Pixel size caption** below the Matrix slider: `f"In-plane pixel size: {FOV / matrix:.2f} mm"`.
- Minimum matrix size lowered to **64 × 64**.

### Fixed
- Phantom viewport anchored to image centre (`set_xlim(128 ± half_w)`) so changing the slice position slider no longer moves the brain image.
- Slice indices in the three orthogonal views now update correctly when the slice position sliders change.

---

## Session 3 — GitHub Deployment Preparation & Google Analytics 4

### Added
- **`requirements.txt`** listing all Python dependencies with pinned versions: `streamlit`, `numpy`, `matplotlib`, `anthropic`, `scipy`, `scikit-image`, `brainweb`.
- **`.gitignore`** excluding `.env`, `__pycache__`, `*.py[cod]`, BrainWeb cache files (`*.bin.gz`, `*.mnc.gz`, `brainweb_data/`), `.streamlit/`, and OS files.
- **Google Analytics 4** tracking via `st.markdown()` injection of the GA4 script tag (Measurement ID `G-BH1W66432Q`) plus `window.parent.gtag()` event firing from `st.components.v1.html(height=0)` to handle Streamlit's iframe sandbox.
- Main file **renamed** from `mri_simulator_streamlit.py` to `app.py` for Streamlit Cloud compatibility.

---

## Session 4 — Fat Suppression, TI Null, Legend, Signal/SNR Physics

### Fixed
- **Fat suppression bug (MPRAGE, DIR, DWI)**: fat suppression toggle now correctly applies `S *= 0.05` to Fat tissue in all three sequences; previously only FSE/GRE/FLAIR/STIR were handled.
- **TI null calculation**: corrected formula from `T1 * log((1 + exp(−TR/T1)) / 2)` to `T1 * log(2 / (1 + exp(−TR/T1)))` for FLAIR, STIR, and MPRAGE null-point calculations.
- **TI slider range (MPRAGE)**: expanded from default narrow range to **200–2500 ms**.
- **TI2 slider range (DIR)**: expanded to **100–2500 ms**.
- **Signal bar labels outside plot**: replaced fixed `+0.02` offset with proportional `_sig_offset = _sig_max * 0.05`; y-limit set to `_sig_max * 1.25` so labels always appear inside the axes.
- **W/L auto-update on sequence change**: W/L computation reverted to use unscaled `base_signals` (0–1 space) rather than voxel-scaled signals, preventing spurious max-value W/L when switching sequences.

### Added
- **Voxel volume scaling for Signal**: `Signal = physics_signal × (voxel_vol / ref_voxel_vol)` so Signal responds to FOV, matrix, and slice thickness changes, not just SNR.
  - 2D sequences use `ref_slice = 5.0 mm`; 3D (MPRAGE) uses `ref_slice = 1.0 mm` to keep baseline signals equivalent.
- **Noise floor metric** added to Signal figure as a red dashed horizontal line labelled with its numeric value.
- **Clean three-step Signal/Noise/SNR architecture** with no circular dependencies:
  1. Physics signals computed per tissue per sequence.
  2. `noise_floor = NOISE_REF × sqrt(BW/BW_REF) / sqrt(NEX/NEX_REF)` — BW and NEX only.
  3. `SNR = Signal / noise_floor` — direct ratio, no helper function.
- **Npartitions (MPRAGE)** correctly treated as a voxel-volume parameter (`partition_thickness = 176 mm / Npartitions`) rather than a signal-averaging parameter; removed from noise formula entirely.
- **SNR displayed to 2 decimal places** in the SNR bar chart.
- **Legend font size** reduced to 5 pt across all decay/recovery curve figures to reduce clutter.

---

## Session 5 — SNR Consistency, EPI Figure Fixes

### Fixed
- **SNR math inconsistency**: `signals` dict values and `noise_floor` are now each rounded to 2 decimal places *before* SNR is computed, so that `Signal_displayed / Noise_displayed = SNR_displayed` exactly when a user verifies the arithmetic manually.
- **EPI T2* decay curve x-axis range**: changed from 0–150 ms to **0–90 ms** to match clinically relevant EPI TE range and better resolve the WM/GM T2* differences.
- **Spurious vertical line at 500 ms on EPI T2* figure**: CSF has T2* = 500 ms; the per-tissue T2* marker `axvline` was rendering as a line stuck to the right edge of the 0–90 ms plot. Fix: tissue T2* marker lines are now suppressed for any tissue whose T2* exceeds the 90 ms plot window, so only WM (26 ms) and GM (33 ms) markers appear.

---

## Session 6 — Acquisition Geometry: Rectangular Matrix, Rectangular FOV, MPRAGE Timing

### Added
- **Separate Frequency and Phase Matrix sliders** replacing the single Matrix slider:
  - `freq_matrix` controls columns (frequency-encode direction); `phase_matrix` controls rows (phase-encode direction).
  - In-plane resolution pixelation uses `step_x = 256 // freq_matrix` for columns and `step_y = 256 // phase_matrix` for rows independently.
  - Pixel size caption shows both: `Frequency pixel: FOV_read / freq_matrix mm | Phase pixel: FOV_phase / phase_matrix mm`.
- **Separate FOV Read and FOV Phase sliders** replacing the single FOV slider:
  - `FOV_read` drives the frequency-direction viewport and frequency pixel size.
  - `FOV_phase` drives the phase-direction viewport, phase pixel size, and rectangular FOV display geometry.
  - When `FOV_phase < FOV_read`, the phantom images display as rectangular (compressed phase dimension) via independent `set_xlim` / `set_ylim` values.
  - **Phase-direction aliasing simulation**: when `FOV_phase < FOV_read`, anatomy outside the phase FOV window is folded back into the image using periodic accumulation (`n_periods` wrap cycles), correctly simulating Nyquist wraparound artifact.
- **MPRAGE timing transparency**:
  - `TR_readout = 7 ms` (fixed) displayed as a labelled parameter — the gradient echo spacing within the partition encode train.
  - `Readout train duration = TR_readout × Npartitions` shown as a dynamic calculated field.
  - **Timing constraint warning** in red if `TR < TI + readout_train_duration`, identifying physically impossible parameter combinations.
  - Explanatory caption describing the MPRAGE timing cycle: inversion → TI delay → readout train → recovery → next inversion.
- **Reset W/L to Optimal button**: one-shot `st.button` that restores Window/Level sliders to the auto-computed optimal values for the current sequence after manual adjustment. Optimal values stored in `st.session_state.wl_opt_window` / `wl_opt_level` and refreshed whenever sequence or signal parameters change.

### Fixed
- **MPRAGE scan time formula**: previously multiplied by `Npartitions` via the shared scan time formula, giving an incorrect result (matrix × matrix equivalent). Fixed to `TR × phase_matrix × NEX / 1000` — Npartitions constrains minimum TR through the readout train duration but does not appear in scan time.
- **EPI scan time using frequency matrix**: `ETL` was set to `freq_matrix` for single-shot EPI, making scan time depend on frequency matrix. Fixed to `ETL = phase_matrix`, giving `Scan time = TR × NEX / 1000` independent of frequency matrix.
- **Voxel volume using square FOV**: voxel volume now correctly uses `(FOV_read / freq_matrix) × (FOV_phase / phase_matrix) × slice_thickness`, so each dimension updates independently.
- **Voxel size metric displaying duplicate pixel size**: metric now shows `freq_pixel_mm × phase_pixel_mm × slice_mm` with each dimension computed from its own FOV and matrix values.
- **TR recovery curve x-axis too wide for FSE/GRE/EPI**: added `ax_t1.set_xlim(0, 6000)` for those sequences only; FLAIR/STIR/DIR retain the wider range needed for their long TR values.
- **GRE TE slider max too narrow**: expanded from 50 ms to 100 ms; T2* decay figure x-axis capped at 150 ms for GRE only.
- **Noise floor label cluttering Signal figure**: removed inline `ax.text` label; noise floor value now appears as a legend entry (`fontsize=5`) via the `axhline` label parameter.
