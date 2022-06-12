from dataclasses import dataclass
from typing import List
import numpy as np


@dataclass
class InceptionSampleDTO:
    smiles: List[str]
    input: List[str]
    output: List[str]
    scores: np.ndarray
    likelihood: np.ndarray