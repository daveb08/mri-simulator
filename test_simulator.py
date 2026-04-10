"""
pytest tests for the core physics calculations in app.py.

Because app.py is a Streamlit application that executes UI code at module
level, a direct import would fail outside a Streamlit process.  Instead this
module uses the ast library to parse app.py and extract only the physics
function and constant definitions, which are then compiled and executed in an
isolated namespace.  The tests therefore always run against the actual source
code in app.py — any change to a physics function will be caught immediately.
"""

import ast
import os
import re
import sys
import types

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Extract physics definitions from app.py via ast
# ---------------------------------------------------------------------------

def _load_physics(app_path: str) -> dict:
    """Parse app.py and return a namespace containing only the physics
    functions and constants needed for testing."""

    with open(app_path, encoding="utf-8") as fh:
        source = fh.read()

    tree = ast.parse(source, filename=app_path)

    # Names we want to lift out of app.py.
    _TARGETS = {
        # constants
        "SNR_SCALE", "BW_REF", "NEX_REF", "NOISE_REF",
        "_ref_px", "REF_VOXEL_VOL",
        "MPRAGE_SLAB_MM", "MPRAGE_NPART_DEFAULT",
        "TISSUES",
        # field-strength lookup tables
        "FIELD_STRENGTH_TISSUES", "FIELD_SNR_SCALE", "TR_MAX_BY_FIELD",
        # signal functions
        "fse_signal", "gre_signal", "flair_signal", "bssfp_signal",
        "bssfp_optimal_fa", "dir_signal", "dir_null_ti2",
        "dwi_signal", "mprage_signal", "epi_signal",
        "csf_null_ti", "ernst_angle",
        # SNR / scan-time helpers
        "calc_snr", "calc_scan_time",
    }

    # Walk the *top-level* body only (not nested nodes) so we preserve order
    # and pick up both simple assignments and augmented/annotated assignments.
    selected_nodes = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in _TARGETS:
            selected_nodes.append(node)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in _TARGETS:
                    selected_nodes.append(node)
                    break
        elif isinstance(node, ast.AugAssign):
            if isinstance(node.target, ast.Name) and node.target.id in _TARGETS:
                selected_nodes.append(node)
        elif isinstance(node, ast.AnnAssign):
            if isinstance(node.target, ast.Name) and node.target.id in _TARGETS:
                selected_nodes.append(node)

    # Build a tiny module: import numpy, then the selected definitions.
    import_np = ast.parse("import numpy as np").body[0]
    mini_module = ast.Module(body=[import_np] + selected_nodes, type_ignores=[])
    ast.fix_missing_locations(mini_module)

    code = compile(mini_module, app_path, "exec")
    namespace: dict = {}
    exec(code, namespace)  # noqa: S102  (intentional use of exec for ast extraction)
    return namespace


_APP_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "app.py")
_physics  = _load_physics(_APP_PATH)

# Bind physics symbols for use in tests
fse_signal        = _physics["fse_signal"]
gre_signal        = _physics["gre_signal"]
flair_signal      = _physics["flair_signal"]
bssfp_signal      = _physics["bssfp_signal"]
dir_signal        = _physics["dir_signal"]
dwi_signal        = _physics["dwi_signal"]
mprage_signal     = _physics["mprage_signal"]
epi_signal        = _physics["epi_signal"]
calc_snr          = _physics["calc_snr"]
calc_scan_time    = _physics["calc_scan_time"]
TISSUES                = _physics["TISSUES"]
FIELD_STRENGTH_TISSUES = _physics["FIELD_STRENGTH_TISSUES"]
FIELD_SNR_SCALE        = _physics["FIELD_SNR_SCALE"]
TR_MAX_BY_FIELD        = _physics["TR_MAX_BY_FIELD"]

# MPRAGE_TR_READOUT is defined inside the Streamlit UI block in app.py, not as
# a module-level constant.  Mirror the value here; if it changes in app.py
# update this line too.
MPRAGE_TR_READOUT = 7  # ms — fixed gradient-echo spacing within partition train

# Tissue shortcuts
WM  = TISSUES["WM"]
GM  = TISSUES["GM"]
CSF = TISSUES["CSF"]
FAT = TISSUES["Fat"]


# ===========================================================================
# 1.  Scan time calculations — all sequences, physically realistic ranges
# ===========================================================================

class TestScanTime:
    """calc_scan_time should return seconds within a physically plausible range.

    Typical clinical acquisitions run between 30 s and 15 min (900 s).
    """

    LOWER = 30.0    # seconds
    UPPER = 900.0   # seconds

    # ── FSE ─────────────────────────────────────────────────────────────────

    def test_fse_scan_time_realistic(self):
        # TR=3 000 ms, 256 phase encodes, ETL=16, NEX=1 → 48 s
        t = calc_scan_time(TR=3000, phase_matrix=256, ETL=16, NEX=1, seq="FSE")
        assert self.LOWER < t < self.UPPER, (
            f"FSE scan time {t:.1f} s is outside realistic range "
            f"[{self.LOWER}, {self.UPPER}] s"
        )

    def test_fse_scan_time_formula(self):
        t = calc_scan_time(TR=3000, phase_matrix=256, ETL=16, NEX=1, seq="FSE")
        expected = 3000 * (256 / 16) * 1 * 1 / 1000.0   # TR*(Npe/ETL)*Npart*NEX/1000
        assert t == pytest.approx(expected), (
            f"FSE scan time formula mismatch: got {t:.3f} s, expected {expected:.3f} s"
        )

    # ── GRE ─────────────────────────────────────────────────────────────────

    def test_gre_scan_time_realistic(self):
        # TR=500 ms, 256 phase encodes, NEX=1 → 128 s ≈ 2 min
        t = calc_scan_time(TR=500, phase_matrix=256, ETL=1, NEX=1, seq="GRE")
        assert self.LOWER < t < self.UPPER, (
            f"GRE scan time {t:.1f} s is outside realistic range"
        )

    def test_gre_scan_time_formula(self):
        t = calc_scan_time(TR=500, phase_matrix=256, ETL=1, NEX=1, seq="GRE")
        expected = 500 * 256 * 1 / 1000.0   # TR*Npe*NEX/1000
        assert t == pytest.approx(expected), (
            f"GRE scan time formula mismatch: got {t:.3f} s, expected {expected:.3f} s"
        )

    # ── FLAIR ────────────────────────────────────────────────────────────────

    def test_flair_scan_time_realistic(self):
        # TR=9 000 ms, 256 phase encodes, ETL=16, NEX=1 → 144 s
        t = calc_scan_time(TR=9000, phase_matrix=256, ETL=16, NEX=1, seq="FLAIR")
        assert self.LOWER < t < self.UPPER, (
            f"FLAIR scan time {t:.1f} s is outside realistic range"
        )

    # ── bSSFP ────────────────────────────────────────────────────────────────

    def test_bssfp_scan_time_positive(self):
        t = calc_scan_time(TR=5, phase_matrix=256, ETL=1, NEX=1, seq="bSSFP")
        assert t > 0, f"bSSFP scan time must be positive, got {t}"

    # ── MPRAGE ───────────────────────────────────────────────────────────────

    def test_mprage_scan_time_realistic(self):
        # TR=2 300 ms, 256 phase encodes, NEX=1 → 588.8 s ≈ 9.8 min
        t = calc_scan_time(TR=2300, phase_matrix=256, ETL=1, NEX=1, seq="MPRAGE")
        assert self.LOWER < t < self.UPPER, (
            f"MPRAGE scan time {t:.1f} s is outside realistic range"
        )

    def test_mprage_scan_time_formula(self):
        t = calc_scan_time(TR=2300, phase_matrix=256, ETL=1, NEX=1, seq="MPRAGE")
        expected = 2300 * 256 * 1 / 1000.0   # TR*Npe*NEX/1000 (same as GRE)
        assert t == pytest.approx(expected), (
            f"MPRAGE scan time formula mismatch: got {t:.3f} s, expected {expected:.3f} s"
        )

    # ── DWI ──────────────────────────────────────────────────────────────────

    def test_dwi_scan_time_positive(self):
        # Typical DWI: single-shot EPI readout, ETL = phase_matrix
        t = calc_scan_time(TR=5000, phase_matrix=128, ETL=128, NEX=4, seq="DWI")
        assert t > 0, f"DWI scan time must be positive, got {t}"

    # ── Common scaling relationships ─────────────────────────────────────────

    def test_scan_time_doubles_with_doubled_nex(self):
        t1 = calc_scan_time(TR=3000, phase_matrix=256, ETL=16, NEX=1, seq="FSE")
        t2 = calc_scan_time(TR=3000, phase_matrix=256, ETL=16, NEX=2, seq="FSE")
        assert t2 == pytest.approx(2 * t1), (
            "Doubling NEX should exactly double scan time"
        )

    def test_scan_time_increases_with_phase_matrix(self):
        t1 = calc_scan_time(TR=3000, phase_matrix=128, ETL=16, NEX=1, seq="FSE")
        t2 = calc_scan_time(TR=3000, phase_matrix=256, ETL=16, NEX=1, seq="FSE")
        assert t2 > t1, "Larger phase matrix should produce longer scan time"

    def test_scan_time_positive_for_all_sequences(self):
        params = dict(TR=2000, phase_matrix=256, ETL=16, NEX=1)
        for seq in ("FSE", "GRE", "FLAIR", "MPRAGE", "DWI"):
            t = calc_scan_time(**params, seq=seq)
            assert t > 0, f"Scan time for {seq} must be positive, got {t}"


# ===========================================================================
# 2.  SNR relationships — NEX and bandwidth scaling
# ===========================================================================

class TestSNRRelationships:
    """SNR should scale as √NEX and as 1/√BW."""

    _signal   = 0.5
    _FOV      = 240.0
    _matrix   = 256
    _slice_mm = 5.0
    _BW_ref   = 200.0
    _NEX_ref  = 1

    def _snr(self, NEX=None, BW=None):
        return calc_snr(
            self._signal,
            self._FOV,
            self._matrix,
            self._slice_mm,
            NEX if NEX is not None else self._NEX_ref,
            BW  if BW  is not None else self._BW_ref,
        )

    def test_increasing_nex_increases_snr(self):
        snr1 = self._snr(NEX=1)
        snr4 = self._snr(NEX=4)
        assert snr4 > snr1, (
            f"SNR with NEX=4 ({snr4:.2f}) should exceed SNR with NEX=1 ({snr1:.2f})"
        )

    def test_nex_snr_scales_as_square_root(self):
        """Quadrupling NEX should exactly double SNR (SNR ∝ √NEX)."""
        snr1 = self._snr(NEX=1)
        snr4 = self._snr(NEX=4)
        assert snr4 == pytest.approx(snr1 * 2, rel=1e-6), (
            f"Expected SNR×2 = {snr1*2:.4f} with NEX=4, got {snr4:.4f}"
        )

    def test_increasing_bandwidth_decreases_snr(self):
        snr_low  = self._snr(BW=50)
        snr_high = self._snr(BW=500)
        assert snr_low > snr_high, (
            f"Lower BW ({snr_low:.2f}) should yield higher SNR than higher BW ({snr_high:.2f})"
        )

    def test_bandwidth_snr_scales_as_inverse_square_root(self):
        """Quadrupling bandwidth should exactly halve SNR (SNR ∝ 1/√BW)."""
        snr_200 = self._snr(BW=200)
        snr_800 = self._snr(BW=800)
        assert snr_800 == pytest.approx(snr_200 / 2, rel=1e-6), (
            f"Expected SNR/2 = {snr_200/2:.4f} with BW=800, got {snr_800:.4f}"
        )

    def test_snr_positive_for_nonzero_signal(self):
        assert self._snr() > 0, "SNR should be positive for nonzero signal"

    def test_snr_zero_for_zero_signal(self):
        snr = calc_snr(0.0, self._FOV, self._matrix, self._slice_mm,
                       self._NEX_ref, self._BW_ref)
        assert snr == pytest.approx(0.0), "SNR should be zero when signal is zero"


# ===========================================================================
# 3.  Voxel volume — decreasing matrix size must increase voxel volume (SNR)
# ===========================================================================

class TestVoxelVolume:
    """Within calc_snr, voxel_vol = (FOV/matrix)² × slice_mm.
    Decreasing matrix at fixed FOV and slice thickness increases voxel volume
    and therefore increases SNR proportionally."""

    _signal   = 0.5
    _FOV      = 240.0
    _slice_mm = 5.0
    _NEX      = 1
    _BW       = 200.0

    def _snr(self, matrix: int) -> float:
        return calc_snr(self._signal, self._FOV, matrix, self._slice_mm,
                        self._NEX, self._BW)

    def test_smaller_matrix_gives_larger_snr(self):
        snr_64  = self._snr(64)
        snr_256 = self._snr(256)
        assert snr_64 > snr_256, (
            f"64-px matrix SNR ({snr_64:.2f}) should exceed 256-px matrix SNR ({snr_256:.2f})"
        )

    def test_halving_matrix_quadruples_snr(self):
        """Halving matrix → 4× larger pixel area → 4× voxel volume → 4× SNR."""
        snr_128 = self._snr(128)
        snr_256 = self._snr(256)
        assert snr_128 == pytest.approx(4 * snr_256, rel=1e-6), (
            f"Expected {4*snr_256:.4f} (4× SNR) for 128-px matrix, got {snr_128:.4f}"
        )

    def test_snr_decreases_monotonically_with_matrix(self):
        matrices = [64, 128, 192, 256, 320]
        snrs = [self._snr(m) for m in matrices]
        assert snrs == sorted(snrs, reverse=True), (
            f"SNR should decrease monotonically as matrix increases; got {snrs}"
        )

    def test_larger_slice_thickness_increases_snr(self):
        """Thicker slice → larger voxel volume → higher SNR."""
        snr_thin  = calc_snr(self._signal, self._FOV, 256, 1.0, self._NEX, self._BW)
        snr_thick = calc_snr(self._signal, self._FOV, 256, 5.0, self._NEX, self._BW)
        assert snr_thick > snr_thin, (
            f"5-mm slice SNR ({snr_thick:.2f}) should exceed 1-mm slice SNR ({snr_thin:.2f})"
        )


# ===========================================================================
# 4.  Signal equations — outputs in [0, 1] at typical parameter values
# ===========================================================================

class TestSignalEquations:
    """Each signal function should return a value in [0, 1] at typical 3 T
    clinical parameter settings, and should always be non-negative."""

    # White matter at 3 T
    T1  = WM["T1"]    # 1084 ms
    T2  = WM["T2"]    # 69 ms
    T2s = WM["T2s"]   # 26 ms
    PD  = WM["PD"]    # 0.70
    ADC = WM["ADC"]   # 0.00070 mm²/s

    def _assert_unit_interval(self, value: float, label: str) -> None:
        assert 0.0 <= value <= 1.0, (
            f"{label} signal {value:.4f} is outside [0, 1] at typical parameters"
        )

    # ── Individual sequence tests ────────────────────────────────────────────

    def test_fse_signal_in_range(self):
        s = fse_signal(TR=3000, TE=80, T1=self.T1, T2=self.T2, PD=self.PD)
        self._assert_unit_interval(s, "FSE")

    def test_gre_signal_in_range(self):
        s = gre_signal(TR=500, TE=5, FA_deg=15,
                       T1=self.T1, T2s=self.T2s, PD=self.PD)
        self._assert_unit_interval(s, "GRE")

    def test_flair_signal_in_range(self):
        s = flair_signal(TR=9000, TI=2500, TE=90,
                         T1=self.T1, T2=self.T2, PD=self.PD)
        self._assert_unit_interval(s, "FLAIR")

    def test_bssfp_signal_in_range(self):
        s = bssfp_signal(TR=5, FA_deg=60, T1=self.T1, T2=self.T2, PD=self.PD)
        self._assert_unit_interval(s, "bSSFP")

    def test_dir_signal_in_range(self):
        s = dir_signal(TR=7000, TI1=3000, TI2=500, TE=30,
                       T1=self.T1, T2=self.T2, PD=self.PD)
        self._assert_unit_interval(s, "DIR")

    def test_dwi_signal_in_range(self):
        s = dwi_signal(TR=3000, TE=80, b=1000,
                       T1=self.T1, T2=self.T2, PD=self.PD, ADC=self.ADC)
        self._assert_unit_interval(s, "DWI b=1000")

    def test_dwi_b0_signal_in_range(self):
        s = dwi_signal(TR=3000, TE=80, b=0,
                       T1=self.T1, T2=self.T2, PD=self.PD, ADC=self.ADC)
        self._assert_unit_interval(s, "DWI b=0")

    def test_dwi_b0_greater_than_dwi_b1000(self):
        """b=0 image should have higher signal than b=1000 due to diffusion attenuation."""
        s_b0   = dwi_signal(TR=3000, TE=80, b=0,
                            T1=self.T1, T2=self.T2, PD=self.PD, ADC=self.ADC)
        s_b1k  = dwi_signal(TR=3000, TE=80, b=1000,
                            T1=self.T1, T2=self.T2, PD=self.PD, ADC=self.ADC)
        assert s_b0 > s_b1k, (
            f"b=0 signal ({s_b0:.4f}) should exceed b=1000 signal ({s_b1k:.4f})"
        )

    def test_mprage_signal_in_range(self):
        s = mprage_signal(TR=2300, TI=900, TE=3, FA_deg=9,
                          T1=self.T1, T2s=self.T2s, PD=self.PD)
        self._assert_unit_interval(s, "MPRAGE")

    def test_epi_signal_in_range(self):
        # Use GM — EPI is the basis for BOLD/functional imaging
        s = epi_signal(TR=3000, TE=30,
                       T1=GM["T1"], T2s=GM["T2s"], PD=GM["PD"])
        self._assert_unit_interval(s, "EPI")

    # ── Fat saturation ───────────────────────────────────────────────────────

    def test_fat_saturation_reduces_fat_signal(self):
        s_off = fse_signal(TR=3000, TE=80, T1=FAT["T1"], T2=FAT["T2"],
                           PD=FAT["PD"], fat_sat=False, is_fat=True)
        s_on  = fse_signal(TR=3000, TE=80, T1=FAT["T1"], T2=FAT["T2"],
                           PD=FAT["PD"], fat_sat=True,  is_fat=True)
        assert s_on < s_off, (
            "Fat-saturated FSE should produce lower fat signal than unsaturated"
        )
        assert s_on == pytest.approx(s_off * 0.05, rel=1e-6), (
            "Fat saturation should attenuate fat signal to 5% of the unsuppressed value"
        )

    # ── Non-negativity across all tissues and sequences ──────────────────────

    def test_all_tissue_signals_non_negative(self):
        """All sequence signal functions must return ≥ 0 for all tissue types."""
        for tissue_name, t in TISSUES.items():
            cases = {
                "FSE"   : fse_signal(3000, 80, t["T1"], t["T2"],  t["PD"]),
                "GRE"   : gre_signal(500, 5, 15, t["T1"], t["T2s"], t["PD"]),
                "FLAIR" : flair_signal(9000, 2500, 90, t["T1"], t["T2"], t["PD"]),
                "bSSFP" : bssfp_signal(5, 60, t["T1"], t["T2"], t["PD"]),
                "DWI"   : dwi_signal(3000, 80, 1000, t["T1"], t["T2"],
                                     t["PD"], t["ADC"]),
                "MPRAGE": mprage_signal(2300, 900, 3, 9,
                                        t["T1"], t["T2s"], t["PD"]),
                "EPI"   : epi_signal(3000, 30, t["T1"], t["T2s"], t["PD"]),
            }
            for seq_name, s in cases.items():
                assert s >= 0, (
                    f"{seq_name} signal for {tissue_name} is negative: {s:.4f}"
                )


# ===========================================================================
# 5.  MPRAGE timing constraint
# ===========================================================================

class TestMPRAGETimingConstraint:
    """The app enforces: TR > TI + MPRAGE_TR_READOUT × Npartitions.

    Physically, the inversion pulse fires, TI elapses, the entire partition
    readout train runs (MPRAGE_TR_READOUT ms per partition), and then the
    magnetisation must have at least some time to recover before the next
    inversion.  Violating this constraint is physically impossible.
    """

    def _readout_train_ms(self, Npartitions: int) -> int:
        return MPRAGE_TR_READOUT * Npartitions

    # ── Valid timing ─────────────────────────────────────────────────────────

    def test_valid_timing_default_params(self):
        TR, TI, Npartitions = 2300, 900, 176
        rt = self._readout_train_ms(Npartitions)
        assert TR > TI + rt, (
            f"Default MPRAGE params should satisfy TR > TI + readout_train: "
            f"{TR} > {TI} + {rt} = {TI + rt}"
        )

    def test_valid_timing_short_npartitions(self):
        TR, TI, Npartitions = 2000, 900, 64
        rt = self._readout_train_ms(Npartitions)
        assert TR > TI + rt, (
            f"MPRAGE with 64 partitions should have valid timing: "
            f"{TR} > {TI} + {rt} = {TI + rt}"
        )

    # ── Timing violations ────────────────────────────────────────────────────

    def test_timing_violation_when_ti_too_long(self):
        TR, TI, Npartitions = 2000, 1500, 100
        rt = self._readout_train_ms(Npartitions)
        assert TR < TI + rt, (
            f"Expected timing violation: TR={TR} should be less than "
            f"TI+readout = {TI}+{rt} = {TI + rt}"
        )

    def test_timing_violation_when_npartitions_too_large(self):
        TR, TI, Npartitions = 2300, 900, 256
        rt = self._readout_train_ms(Npartitions)
        assert TR < TI + rt, (
            f"Expected timing violation with 256 partitions: TR={TR} < "
            f"TI+readout = {TI}+{rt} = {TI + rt}"
        )

    # ── Boundary condition ───────────────────────────────────────────────────

    def test_boundary_tr_equals_ti_plus_readout_is_invalid(self):
        """The constraint is strict (>), so equality is a violation."""
        TI, Npartitions = 900, 100
        rt = self._readout_train_ms(Npartitions)
        TR_boundary = TI + rt
        is_valid = TR_boundary > TI + rt   # strict inequality required
        assert not is_valid, (
            "TR equal to TI + readout_train should NOT satisfy the strict "
            "timing constraint TR > TI + readout_train"
        )

    # ── Parameter sensitivity ────────────────────────────────────────────────

    def test_more_partitions_increase_minimum_tr(self):
        """Each additional partition adds MPRAGE_TR_READOUT ms to the minimum TR."""
        rt_64  = self._readout_train_ms(64)
        rt_256 = self._readout_train_ms(256)
        assert rt_256 > rt_64, (
            f"Readout train for 256 partitions ({rt_256} ms) should exceed "
            f"that for 64 partitions ({rt_64} ms)"
        )

    def test_readout_train_scales_linearly_with_npartitions(self):
        rt_64  = self._readout_train_ms(64)
        rt_128 = self._readout_train_ms(128)
        assert rt_128 == pytest.approx(2 * rt_64), (
            "Readout train duration should scale linearly with Npartitions"
        )

    def test_longer_ti_demands_higher_minimum_tr(self):
        Npartitions = 176
        rt = self._readout_train_ms(Npartitions)
        min_tr_short_ti = 500  + rt
        min_tr_long_ti  = 1200 + rt
        assert min_tr_long_ti > min_tr_short_ti, (
            "Longer TI increases the minimum feasible TR"
        )

    # ── Physics self-consistency ──────────────────────────────────────────────

    def test_mprage_signal_nonnegative_with_valid_timing(self):
        """A timing-valid MPRAGE acquisition must yield a non-negative WM signal."""
        TR, TI, Npartitions = 2300, 900, 176
        assert TR > TI + self._readout_train_ms(Npartitions), (
            "Precondition: timing parameters must satisfy constraint"
        )
        s = mprage_signal(TR=TR, TI=TI, TE=3, FA_deg=9,
                          T1=WM["T1"], T2s=WM["T2s"], PD=WM["PD"])
        assert s >= 0, (
            f"MPRAGE WM signal should be non-negative with valid timing, got {s:.4f}"
        )


# ---------------------------------------------------------------------------
# Helper — extract FLAIR TR slider bounds directly from app.py source
# ---------------------------------------------------------------------------

def _get_flair_tr_slider_bounds(app_path: str) -> tuple[int, int]:
    """Return (min_ms, max_ms) from the FLAIR TR st.slider() call in app.py.

    Looks for the first st.slider call that immediately follows the
    ``if seq == "FLAIR":`` branch and extracts its second and third positional
    arguments (min and max).  Raises ValueError if the pattern is not found.
    """
    with open(app_path, encoding="utf-8") as fh:
        source = fh.read()
    # Match: if   seq == "FLAIR": (optional whitespace / newline)
    #            TR = st.slider("TR (ms)", <min>, <max>, ...
    pattern = (
        r'if\s+seq\s*==\s*"FLAIR"\s*:\s*\n'
        r'\s*TR\s*=\s*st\.slider\(\s*"TR \(ms\)"\s*,\s*(\d+)\s*,\s*(\d+)'
    )
    match = re.search(pattern, source)
    if not match:
        raise ValueError(
            "Could not locate the FLAIR TR st.slider() call in app.py. "
            "Check that the FLAIR branch still matches the expected pattern."
        )
    return int(match.group(1)), int(match.group(2))


# ===========================================================================
# 6.  FLAIR TR slider range — fixed at all field strengths
# ===========================================================================

class TestFLAIRTRRange:
    """FLAIR TR slider must use a fixed 3000–10000 ms range at every field
    strength, bypassing the dynamic TR_MAX_BY_FIELD scaling used by other
    sequences.

    Tests parse the actual st.slider() call in app.py so they will fail
    immediately if the bounds are accidentally reverted to the dynamic range.
    """

    FLAIR_TR_MIN = 3000
    FLAIR_TR_MAX = 10000
    FIELD_STRENGTHS = ["0.064T", "0.5T", "1.0T", "1.5T", "3.0T"]

    def test_flair_slider_min_is_3000(self):
        lo, _ = _get_flair_tr_slider_bounds(_APP_PATH)
        assert lo == self.FLAIR_TR_MIN, (
            f"FLAIR TR slider minimum should be {self.FLAIR_TR_MIN} ms, got {lo} ms"
        )

    def test_flair_slider_max_is_10000(self):
        _, hi = _get_flair_tr_slider_bounds(_APP_PATH)
        assert hi == self.FLAIR_TR_MAX, (
            f"FLAIR TR slider maximum should be {self.FLAIR_TR_MAX} ms, got {hi} ms"
        )

    @pytest.mark.parametrize("field_strength", ["0.064T", "0.5T", "1.0T", "1.5T", "3.0T"])
    def test_flair_tr_max_exceeds_dynamic_tr_max_at_all_field_strengths(self, field_strength):
        """FLAIR's fixed 10 000 ms ceiling must exceed every field-strength
        dynamic TR maximum, confirming the fix is meaningful at each B0."""
        _, hi = _get_flair_tr_slider_bounds(_APP_PATH)
        dyn_max = TR_MAX_BY_FIELD[field_strength]
        assert hi > dyn_max, (
            f"FLAIR TR max ({hi} ms) should exceed TR_MAX_BY_FIELD['{field_strength}'] "
            f"({dyn_max} ms) — otherwise the fix has no effect at this field strength"
        )

    @pytest.mark.parametrize("field_strength", ["0.064T", "0.5T", "1.0T", "1.5T", "3.0T"])
    def test_flair_tr_min_independent_of_field_strength(self, field_strength):
        """The FLAIR TR lower bound must be the same fixed value at every B0,
        not derived from TR_MAX_BY_FIELD."""
        lo, _ = _get_flair_tr_slider_bounds(_APP_PATH)
        assert lo == self.FLAIR_TR_MIN, (
            f"FLAIR TR min at {field_strength} should be {self.FLAIR_TR_MIN} ms, "
            f"got {lo} ms (must not vary with field strength)"
        )


# ===========================================================================
# 7.  Dynamic TR maximum per field strength (non-FLAIR sequences)
# ===========================================================================

class TestDynamicTRMaxByFieldStrength:
    """TR_MAX_BY_FIELD must match the documented per-B0 ceiling values used
    by all sequences except FLAIR (which uses a fixed range)."""

    EXPECTED = {
        "0.064T": 1000,
        "0.5T":   2000,
        "1.0T":   3000,
        "1.5T":   4000,
        "3.0T":   6000,
    }

    @pytest.mark.parametrize("field_strength,expected_max", EXPECTED.items())
    def test_tr_max_matches_expected(self, field_strength, expected_max):
        assert TR_MAX_BY_FIELD[field_strength] == expected_max, (
            f"TR_MAX_BY_FIELD['{field_strength}'] expected {expected_max} ms, "
            f"got {TR_MAX_BY_FIELD[field_strength]} ms"
        )

    def test_tr_max_increases_monotonically_with_field_strength(self):
        """Higher B0 should permit a longer maximum TR (T1 values increase)."""
        ordered = ["0.064T", "0.5T", "1.0T", "1.5T", "3.0T"]
        values  = [TR_MAX_BY_FIELD[fs] for fs in ordered]
        assert values == sorted(values), (
            f"TR_MAX_BY_FIELD values should increase monotonically with B0: {values}"
        )

    def test_all_field_strengths_present(self):
        for fs in ["0.064T", "0.5T", "1.0T", "1.5T", "3.0T"]:
            assert fs in TR_MAX_BY_FIELD, (
                f"Field strength '{fs}' missing from TR_MAX_BY_FIELD"
            )


# ===========================================================================
# 8.  Tissue T1 values at different field strengths
# ===========================================================================

class TestFieldStrengthTissueT1:
    """FIELD_STRENGTH_TISSUES must contain the correct WM T1 values at each B0.

    Reference values:
      0.064 T — O'Reilly & Webb MRM 2022  (WM T1 = 275 ms)
      1.5  T  — Stanisz et al. MRM 2005   (WM T1 = 650 ms)
      3.0  T  — Stanisz et al. MRM 2005   (WM T1 = 832 ms)
    """

    def test_wm_t1_at_0064T(self):
        t1 = FIELD_STRENGTH_TISSUES["0.064T"]["WM"]["T1"]
        assert t1 == 275, (
            f"WM T1 at 0.064 T should be 275 ms (O'Reilly & Webb MRM 2022), got {t1} ms"
        )

    def test_wm_t1_at_15T(self):
        t1 = FIELD_STRENGTH_TISSUES["1.5T"]["WM"]["T1"]
        assert t1 == 650, (
            f"WM T1 at 1.5 T should be 650 ms (Stanisz et al. MRM 2005), got {t1} ms"
        )

    def test_wm_t1_at_3T(self):
        t1 = FIELD_STRENGTH_TISSUES["3.0T"]["WM"]["T1"]
        assert t1 == 832, (
            f"WM T1 at 3.0 T should be 832 ms (Stanisz et al. MRM 2005), got {t1} ms"
        )

    def test_wm_t1_increases_with_field_strength(self):
        """WM T1 lengthens as B0 increases — a fundamental MRI physics relationship."""
        ordered = ["0.064T", "0.5T", "1.0T", "1.5T", "3.0T"]
        t1s = [FIELD_STRENGTH_TISSUES[fs]["WM"]["T1"] for fs in ordered]
        assert t1s == sorted(t1s), (
            f"WM T1 should increase monotonically with B0: "
            + ", ".join(f"{fs}={v}" for fs, v in zip(ordered, t1s))
        )

    def test_all_tissues_present_at_all_field_strengths(self):
        for fs in ["0.064T", "0.5T", "1.0T", "1.5T", "3.0T"]:
            for tissue in ["WM", "GM", "CSF", "Fat"]:
                assert tissue in FIELD_STRENGTH_TISSUES[fs], (
                    f"Tissue '{tissue}' missing from FIELD_STRENGTH_TISSUES['{fs}']"
                )

    def test_t1_t2_t2s_keys_present_for_all_entries(self):
        for fs, tissues in FIELD_STRENGTH_TISSUES.items():
            for tissue, params in tissues.items():
                for key in ("T1", "T2", "T2s"):
                    assert key in params, (
                        f"Key '{key}' missing from FIELD_STRENGTH_TISSUES['{fs}']['{tissue}']"
                    )


# ===========================================================================
# 9.  SNR scaling factors relative to 3 T
# ===========================================================================

class TestFieldStrengthSNRScaling:
    """FIELD_SNR_SCALE must contain the correct per-B0 SNR factors.

    SNR scales approximately linearly with B0; factors are normalised so
    that 3.0 T = 1.000.

    Expected values (from app.py):
      0.064 T → 0.021
      0.5  T  → 0.167
      1.5  T  → 0.500
      3.0  T  → 1.000
    """

    def test_snr_scale_at_0064T(self):
        assert FIELD_SNR_SCALE["0.064T"] == pytest.approx(0.021, rel=1e-3), (
            f"SNR scale at 0.064 T should be ≈0.021, got {FIELD_SNR_SCALE['0.064T']}"
        )

    def test_snr_scale_at_05T(self):
        assert FIELD_SNR_SCALE["0.5T"] == pytest.approx(0.167, rel=1e-3), (
            f"SNR scale at 0.5 T should be ≈0.167, got {FIELD_SNR_SCALE['0.5T']}"
        )

    def test_snr_scale_at_15T(self):
        assert FIELD_SNR_SCALE["1.5T"] == pytest.approx(0.500, rel=1e-3), (
            f"SNR scale at 1.5 T should be ≈0.500, got {FIELD_SNR_SCALE['1.5T']}"
        )

    def test_snr_scale_at_3T_is_unity(self):
        assert FIELD_SNR_SCALE["3.0T"] == pytest.approx(1.000, rel=1e-6), (
            f"SNR scale at 3.0 T should be exactly 1.000, got {FIELD_SNR_SCALE['3.0T']}"
        )

    def test_snr_scale_increases_monotonically_with_field_strength(self):
        """Higher B0 should produce proportionally higher SNR."""
        ordered = ["0.064T", "0.5T", "1.0T", "1.5T", "3.0T"]
        scales  = [FIELD_SNR_SCALE[fs] for fs in ordered]
        assert scales == sorted(scales), (
            f"FIELD_SNR_SCALE values should increase monotonically with B0: {scales}"
        )

    def test_all_field_strengths_present_in_snr_scale(self):
        for fs in ["0.064T", "0.5T", "1.0T", "1.5T", "3.0T"]:
            assert fs in FIELD_SNR_SCALE, (
                f"Field strength '{fs}' missing from FIELD_SNR_SCALE"
            )

    def test_snr_scale_values_positive_and_at_most_unity(self):
        for fs, scale in FIELD_SNR_SCALE.items():
            assert 0.0 < scale <= 1.0, (
                f"FIELD_SNR_SCALE['{fs}'] = {scale} is outside (0, 1]"
            )
