"""SandwichSP - Signal Peptide Prediction using Deep Learning.

A deep learning model for predicting signal peptides in protein sequences.

Example:
    >>> from sandwichsp import SandwichSP
    >>> model = SandwichSP()
    >>> result = model.predict("MKFLILLFNILCLFPVLAADNH...")
    >>> print(result.sp_type, result.cleavage_site)
    SP 21

Label meanings:
    S - Sec/SPI signal peptide
    T - Tat/SPI signal peptide
    L - Sec/SPII signal peptide (lipoprotein)
    I - Cytoplasm (no signal peptide)
    M - Transmembrane region
    O - Other/extracellular
"""

from .predictor import SandwichSP, PredictionResult
from .config import Config

__version__ = "0.1.0"
__all__ = ["SandwichSP", "PredictionResult", "Config"]
