# conftest.py
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Optional: Print for debugging
print(f"✅ Project root added to Python path: {project_root}")
