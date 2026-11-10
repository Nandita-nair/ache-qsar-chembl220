"""Central configuration: every threshold used in the study lives here."""

# ChEMBL target: human acetylcholinesterase (UniProt P22303).
# NOTE: CHEMBL216 is the muscarinic M1 receptor - do not confuse the two.
DEFAULT_TARGET = "CHEMBL220"

SEED = 42

# Activity thresholds (pIC50 = 9 - log10(IC50 in nM))
ACTIVE_PIC50 = 6.0      # IC50 <= 1 uM
INACTIVE_PIC50 = 5.0    # IC50 >= 10 uM  (between the two = "intermediate")

# Curation
MAX_REPLICATE_STD = 1.0   # drop compounds whose replicate pIC50 values disagree by > 1 log unit
MIN_HEAVY_ATOMS = 5
MAX_HEAVY_ATOMS = 120

# Evaluation protocol
TEST_SIZE = 0.2
CV_FOLDS = 5
N_BOOTSTRAP = 1000
N_Y_RANDOMIZATIONS = 20

# Fingerprints
MORGAN_RADIUS = 2
MORGAN_BITS = 2048
