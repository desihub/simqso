# test_simqso.py
# Adapted from examples/SimpleSpecExample.ipynb and wrapped into a unit test
# by LBL cborg coder

import os
import unittest
import tempfile
import numpy as np

import astropy.units as u
from astropy.cosmology import Planck13
from simqso.sqgrids import *
from simqso import sqbase
from simqso.sqrun import buildSpectraBulk, buildQsoSpectrum

# ----------------------------------------------------------------------
# Helper function that configures and runs simqso
# ----------------------------------------------------------------------
def run_simqso_simulation():
    """
    Execute the SimQSO example code and return the wavelength grid and
    the generated spectra.

    Parameters
    ----------
    save_spectra : bool, optional
        Forwarded to ``buildSpectraBulk``.  The default (False) avoids
        writing large FITS files during testing.

    Returns
    -------
    wave : numpy.ndarray
        Wavelength grid (Å).
    spectra : numpy.ndarray
        Array of shape (n_qso, len(wave)) containing the simulated spectra.
    """
    # ------------------------------------------------------------------
    # Fixed random seed for deterministic output
    # ------------------------------------------------------------------
    np.random.seed(12345)

    # ------------------------------------------------------------------
    # Build the wavelength grid (R = 1000, λ_min = 1000 Å, λ_max = 10 000 Å)
    # ------------------------------------------------------------------
    wave = sqbase.fixed_R_dispersion(1000, 10000, 1000)

    # ------------------------------------------------------------------
    # Simulation parameters
    # ------------------------------------------------------------------
    nqso = 5
    M = AbsMagVar(
        FixedSampler(np.linspace(-27, -25, nqso)[::-1]),
        restWave=1450,
    )
    z = RedshiftVar(FixedSampler(np.linspace(2, 4, nqso)))

    qsos = QsoSimPoints([M, z], cosmo=Planck13, units="luminosity")

    # ------------------------------------------------------------------
    # Continuum and dust components
    # ------------------------------------------------------------------
    contVar = BrokenPowerLawContinuumVar(
        [GaussianSampler(-1.5, 0.3), GaussianSampler(-0.5, 0.3)],
        [1215.0],
    )

    subDustVar = DustBlackbodyVar([ConstSampler(0.05), ConstSampler(1800.0)],
                                 name="sublimdust")
    subDustVar.set_associated_var(contVar)

    hotDustVar = DustBlackbodyVar([ConstSampler(0.1), ConstSampler(880.0)],
                                 name="hotdust")
    hotDustVar.set_associated_var(contVar)

    # Emission‑line component (uses the absolute‑magnitude distribution)
    emLineVar = generateBEffEmissionLines(qsos.absMag)

    # Fe‑template component
    fescales = [
        (0, 1540, 0.5),
        (1540, 1680, 2.0),
        (1680, 1868, 1.6),
        (1868, 2140, 1.0),
        (2140, 3500, 1.0),
    ]
    feVar = FeTemplateVar(VW01FeTemplateGrid(qsos.z, wave, scales=fescales))

    # Register all variable components with the QSO container
    qsos.addVars([contVar, subDustVar, hotDustVar, emLineVar, feVar])

    # ------------------------------------------------------------------
    # Build the spectra in bulk
    # ------------------------------------------------------------------
    _, spectra = buildSpectraBulk(wave, qsos, saveSpectra=True)

    return wave, spectra


# ----------------------------------------------------------------------
# Test case using the standard unittest library
# ----------------------------------------------------------------------
class TestSimQSO(unittest.TestCase):
    """Unittest suite for the SimQSO example."""

    @classmethod
    def setUpClass(cls):
        """Skip the whole suite if optional dependencies are missing."""
        try:
            import astropy   # noqa: F401
            import simqso    # noqa: F401
        except Exception as exc:   # pragma: no cover
            raise unittest.SkipTest(
                f"Optional dependencies not available: {exc}"
            )

    def setUp(self):
        """Create a temporary directory to sandbox any file output."""
        self.tmp_dir = tempfile.TemporaryDirectory()
        # Remember the original cwd so we can restore it later
        self.orig_cwd = os.getcwd()
        os.chdir(self.tmp_dir.name)

    def tearDown(self):
        """Return to the original working directory and clean up."""
        os.chdir(self.orig_cwd)
        self.tmp_dir.cleanup()

    def test_simqso_runs_and_returns_valid_spectra(self):
        """Smoke‑test that the SimQSO code runs and returns sensible data."""
        wave, spectra = run_simqso_simulation()

        # 1. Types
        self.assertIsInstance(wave, np.ndarray, "wave should be a NumPy array")
        self.assertIsInstance(spectra, np.ndarray, "spectra should be a NumPy array")

        # 2. Shape – expect (n_qso, len(wave))
        expected_n_qso = 5
        self.assertEqual(
            spectra.shape,
            (expected_n_qso, len(wave)),
            f"Expected spectra shape {(expected_n_qso, len(wave))}, got {spectra.shape}",
        )

        # 3. No NaNs / Infs – spectra should be finite numbers
        self.assertTrue(np.isfinite(spectra).all(),
                        "Spectra contain NaN or infinite values")

        # 4. Physical sanity: fluxes should be non‑negative (tiny negative
        #    rounding errors are allowed)
        self.assertGreater(
            spectra.min(),
            -1e-12,
            "Spectra contain unexpectedly negative values",
        )

    def test_repeatability_of_random_seed(self):
        """Check that the fixed NumPy seed makes the output deterministic."""
        wave1, spec1 = run_simqso_simulation()
        wave2, spec2 = run_simqso_simulation()

        # Wavelength grid is deterministic by construction
        np.testing.assert_array_equal(wave1, wave2)

        # Spectra must be identical across runs because we reset the seed each time
        np.testing.assert_allclose(spec1, spec2, rtol=1e-12, atol=0)


# ----------------------------------------------------------------------
# Allow running the file directly:  python test_simqso.py
# ----------------------------------------------------------------------
if __name__ == "__main__":
    unittest.main()
