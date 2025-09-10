import sys
from pathlib import Path

# Ensure the src directory is on sys.path for tests
ROOT = Path(__file__).resolve().parent
SRC = ROOT / 'src'
if SRC.exists() and str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
