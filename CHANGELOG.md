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

---

## Session 7 — FSE Pulse Sequence Diagram Rewrite & bSSFP Refactor

### Added
- **Physics-accurate FSE pulse sequence diagram** — complete rewrite of the FSE drawing block:
  - All 180° refocusing pulses rendered as **sinc pulses** (matching the 90° excitation style).
  - **Composite Gss crusher waveform** around each 180° pulse: single continuous 8-point bridge shape — rise to leading crusher plateau (+0.85) → fall to slice-select plateau (+0.50, coinciding with sinc pulse) → rise to trailing crusher plateau (+0.85) → fall to zero. Both crushers positive (equal area).
  - **Gradient timing order**: phase-encode blip → Gfe readout → phase-encode rewind within each echo period.
  - **PE blip placement**: encode blip centred in the window between the trailing crusher end and the Gfe readout start; rewind blip centred in the window between the Gfe readout end and the next leading crusher start.
  - **PE amplitude ordering**: amplitude scales linearly with distance from `eff_idx`; smallest blip at the effective echo (k-space centre), largest at the outermost echoes. Sign alternates per echo.
  - **Rewind lobes**: equal amplitude and opposite polarity to their corresponding encode blip.
  - **T2-decay signal envelope**: `eamp = FSE_SIG_AMP_EFF × exp(−dist × FSE_SIG_DECAY)`, minimum 0.15, where `dist = |i − eff_idx|`.
  - **TEeff annotation**: double-headed arrow on the signal row spanning from the 90° pulse centre to the effective echo centre, labelled `TEeff = TE ms`.
  - **Gfe readout always positive** for all echoes.
  - **Faint dotted vertical line** at the 90° pulse centre propagated across all five subplot rows.
- **Fixed τ independent of ETL**: `tau_s = FSE_MAX_ESP − FSE_RF_TC = 1.15` schematic units; `esp = 2 × tau_s = 2.30`, constant regardless of ETL. 180° pulse centres at `(2n−1)τ`, echo centres at `2nτ`. Adding more echoes extends the diagram rightward without shifting existing waveforms.
- **Maximum 6 echoes displayed** in the FSE PSD regardless of ETL slider value; sidebar caption notes the limit.
- **"Showing N of ETL echoes" annotation** displayed as `fig.text` at figure coordinates (0.67, 0.95) in bold white, to the right of the title, when ETL > 6.
- **ESP sidebar display**: calculated echo spacing `ESP ≈ TE / eff_idx+1 ms` shown as a `st.caption` in the FSE sidebar block.
- **FSE constraint warnings**:
  - Red `st.error` in sidebar if TR < ETL × ESP (TR too short) or ESP < 10 ms (physically unrealistic).
  - When either constraint is violated, all PSD subplot rows are hidden and a red-boxed warning is rendered via `fig.text` at figure centre; drawing returns early.
- **TR annotation arrow** moved to the Gss row (`ann_tr(..., ax=ax_ss)`), spanning from the 90° centre to the end of the last echo's period.
- **bSSFP drawing block refactored**: replaced all inline numeric literals with the `BSSFP_` global constants defined in the previous session.

### Fixed
- **Vline colour bug**: invalid hex colour `"#F6E606EC"` on the 90° excitation vline replaced with `"#555555"`.
- **`ann_te` helper**: added `draw_axes` parameter so the TEeff arrow can be restricted to specific subplot rows (signal row only for FSE).
- **`ann_tr` helper**: added `ax` parameter so the TR arrow can be drawn on any row rather than always on `ax_rf`.

---

## Session 8 — GRE Pulse Sequence Diagram: Physics Accuracy & Timing Improvements

### Added
- **Gss amplitude normalisation** (`GRE_GSS_MAX_AMP`): Gss and rephaser amplitudes now scale as `GRE_GSS_MAX_AMP / slice_mm`, capped at `GRE_GSS_MAX_AMP = 1.90` at minimum slice thickness (≈ 95 % of `PSD_Y_MAX`), making slice-thickness changes visually apparent across the full slider range.
- **Gfe readout amplitude scaling** (`GRE_GFE_REF_BW`, `GRE_GFE_REF_FOV`): Gfe readout and prephaser amplitudes scale as `BW / FOV_read`, normalised to current amplitude at reference values (200 Hz/px, 240 mm). Prephaser scales identically so the half-area echo-centring relationship is always preserved; the scale factor cancels in the `gfe_dep_rise` formula so TE_min is unaffected. `draw_pulse_sequence` signature extended with `BW=200, FOV_read=240`.
- **GRE-specific rephaser constants** (`GRE_GSS_REP_AMP = -1.00`, `GRE_GSS_REP_FLAT = 0.06`): rephaser amplitude raised to match the Gss main lobe (same slew rate, rise time = `GRE_GSS_RISE`); flat-top shortened so that rephaser area equals exactly the area of the second half of the Gss lobe (`GRE_GSS_FLAT/2 + GRE_GSS_RISE/2 = 0.14`). Total rephaser duration reduced from 0.32 → 0.14 schematic units. Constants are GRE-specific and do not affect the FSE rephaser.

### Fixed
- **Gss / RF centre misalignment**: `GRE_GSS_FLAT` widened from 0.18 → 0.24 (= `2 × GRE_RF_HW`) so the centre of the Gss main lobe coincides exactly with `GRE_RF_TC`. The Gss flat-top now spans precisely from RF start to RF end.
- **Spurious TE annotation arrow on Gfe row**: `ann_te` for GRE now passes `draw_axes=(ax_sig,)` so the TE double-headed arrow is drawn only on the Signal row, not on the Gfe row.
- **Gpe encode overlaps Gss rephaser**: `gpe_enc_t0` moved from `gss_rep_end` to `gss_ss_end` so the phase-encode lobe plays simultaneously with the slice-select rephaser (different gradient axes), pulling the rewind lobe earlier and further reducing minimum TE.

---

## Session 13 — K-Space Visualiser

### Added
- **K-space visualiser** below the phantom images — two panels side by side:
  - *Left*: log-magnitude 2D FFT of the current axial phantom slice (`np.log1p(|kspace|)`), displayed with the `inferno` colormap.
  - *Right*: IFFT reconstruction from the partially filled k-space, normalised to [0, 1] and displayed with the `gray` colormap.
- **`_ks_centric_rows(N, n_lines)`** helper: returns the first `n_lines` row indices in FSE centric order (ky = 0 first, then ±1, ±2, …).
- **`_render_kspace_panels(ksp_partial, col_left, col_right)`** helper: renders both k-space and reconstruction panels into Streamlit columns.
- **K-space fill radio button** (`"10%"`, `"25%"`, `"50%"`, `"75%"`, `"100%"`; default `"100%"`; horizontal): instantly displays partially filled k-space for the selected percentage without animation.
  - *Linear sequences* (GRE, FSE for non-centric, bSSFP, FLAIR, STIR, MPRAGE, EPI, DWI): symmetric centre-out fill — lines centred on ky = 0 (`_r_start` … `_r_end`).
  - *FSE*: centric ordering via `_ks_centric_rows` — ky = 0 filled first, then outward alternating.
- Caption below panels shows: `"N of 128 ky lines filled (X%) · Left: k-space magnitude · Right: reconstructed image"`.
- **`_kspace_source_img`** captured from the axial phantom slice (before noise) so k-space reflects the true contrast-weighted image.

---

## Session 12 — bSSFP Gpe Lobe Duration Fix

### Fixed
- **bSSFP Gpe lobe flat-top duration**: encode and rewind Gpe lobes previously used `BSSFP_GPE_ENC_FLAT = 0.22`, making them ~3× longer than the simultaneously-playing Gss and Gfe lobes. Fixed by replacing `BSSFP_GPE_ENC_FLAT` with `BSSFP_GSS_HALFPRE_FLAT = 0.075` for both encode and rewind lobes, so rise time, flat-top, and fall time match the gradient lobes they play alongside. The unused constant `BSSFP_GPE_ENC_FLAT` was removed.

---

## Session 11 — W/L Calculation Fix for Field Strength

### Fixed
- **Optimal W/L using stale 3 T tissue values**: the W/L computation block runs inside `with st.sidebar:` before the `TISSUES` dict is updated to the selected field strength, so it always used 3 T values. At low field (e.g. 0.064 T, WM T1 = 275 ms vs. 832 ms at 3 T) the signal contrast differs substantially and the auto-computed W/L was incorrect. Fix: directly reference `FIELD_STRENGTH_TISSUES[field_strength][tissue]` for T1/T2/T2s within the W/L computation block, bypassing the mutable `TISSUES` dict entirely. `field_strength` added to the `_params_sig` cache key so W/L recomputes when field strength changes.

---

## Session 10 — Dynamic TR Slider Max & T1 Recovery Curve Range

### Added
- **`TR_MAX_BY_FIELD` dictionary** mapping each field strength to its maximum TR: `0.064T → 1000 ms`, `0.5T → 2000 ms`, `1.0T → 3000 ms`, `1.5T → 4000 ms`, `3.0T → 6000 ms`.
- **Dynamic TR slider maximum**: all per-sequence TR sliders now use `_tr_max = TR_MAX_BY_FIELD[field_strength]` as their ceiling. Each sequence floor is preserved (`max(sequence_min + step, _tr_max)` guards against `min > max` at low field).
- **Dynamic T1 recovery curve x-axis**: `tr_range = np.linspace(100, _tr_max, 500)` and `ax_t1.set_xlim(0, _tr_max)` so the recovery curve always spans the physiologically relevant TR range for the selected field strength.

---

## Session 9 — Multi-Field-Strength Support & pytest Test Suite

### Added
- **Field strength radio button** in the sidebar under "MRI Parameters": options `0.064T`, `0.5T`, `1.0T`, `1.5T`, `3.0T`; default `3.0T`; horizontal layout.
- **`FIELD_STRENGTH_TISSUES` dictionary** containing literature-based T1 and T2 values for WM, GM, CSF, and Fat at all five field strengths:
  - *0.064 T*: O'Reilly & Webb, Magn. Reson. Med. 87(1), 2022 (in-vivo ULF measurements).
  - *0.5 T – 3.0 T*: Bottomley et al., Med. Phys. 11(4), 1984 (empirical power-law scaling), cross-referenced with Stanisz et al., Magn. Reson. Med. 54(3), 2005 (3 T reference values).
  - T2* values estimated as `min(T2, T2s_3T × (3.0 / B0))`: susceptibility dephasing decreases at lower field so T2* approaches T2 as B0 → 0.
- **`FIELD_SNR_SCALE` dictionary** mapping each field strength to its SNR factor relative to 3 T (linear with B0): `0.064T → 0.021`, `0.5T → 0.167`, `1.0T → 0.333`, `1.5T → 0.500`, `3.0T → 1.000`.
- **Dynamic TISSUES update**: T1, T2, and T2* entries in the live `TISSUES` dict are overwritten before signal calculations using the selected field strength; PD, colour, and ADC are field-strength-independent and unchanged.
- **Field-strength SNR scaling**: effective noise reference `_noise_ref_scaled = NOISE_REF / _field_snr_scale` replaces bare `NOISE_REF` in the Step 2 noise-floor formula so that SNR decreases at lower field strengths.
- **Dynamic page title**: heading now reads `MRI Simulator — Brain (X.XT)` and updates with the selected field strength.
- **`test_simulator.py`** — pytest test suite for core physics calculations (43 tests, all passing):
  - *Scan time*: formula correctness and physically realistic range (30–900 s) for FSE, GRE, FLAIR, bSSFP, MPRAGE, and DWI; NEX and phase-matrix scaling verified.
  - *SNR relationships*: SNR ∝ √NEX and SNR ∝ 1/√BW confirmed with exact-value assertions.
  - *Voxel volume*: SNR ∝ voxel area (halving matrix → 4× SNR), monotonic decrease with matrix size, thicker slice → higher SNR.
  - *Signal equations*: all eight sequence signal functions verified to return values in [0, 1] at typical 3 T clinical parameters for all four tissue types; fat-saturation 5 % attenuation confirmed; non-negativity checked across all tissues × all sequences.
  - *MPRAGE timing constraint*: valid and violating cases, strict boundary, linear readout-train scaling with Npartitions.
  - Uses `ast` to extract physics functions directly from `app.py` source without importing the Streamlit runtime.
