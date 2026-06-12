"""Re-export of the top-level sizes.py budget helpers for the research dir.

The canonical helper lives in /sizes.py so the production build can import it
without depending on ez80research/. This shim just exposes it under the name
the framework docs reference.
"""
import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from sizes import (  # noqa: F401,E402
    RAM_BUDGET_BYTES, FLASH_BUDGET_BYTES, MAX_APPVAR_BYTES,
    RUNTIME_BUFFER_BYTES, BudgetError, weight_data_bytes, total_ram,
    estimate_memory, check_budget, validate_or_raise,
)
