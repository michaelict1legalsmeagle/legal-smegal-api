"""
ceiling_engine.py (root) — DEPRECATED

This file is NOT the executed ceiling engine.

The canonical, executed ceiling engine is:
    services/ceiling_engine.py

app.py imports from services.ceiling_engine only:
    from services.ceiling_engine import calculate_ceiling as _calc_ceiling

This root file is retained as a reference artefact only.
It must not be imported, called, or used in production.
Any test confirming the executed path should assert that
services.ceiling_engine is the active module.

DO NOT EDIT. DO NOT IMPORT.
"""

raise ImportError(
    "ceiling_engine (root) is deprecated. "
    "Import from services.ceiling_engine instead."
)
