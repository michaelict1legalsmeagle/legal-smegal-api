# gunicorn.conf.py
# Gunicorn configuration for LegalSmegal API
# LLM calls (document summary) can take 60-120 seconds
# Worker must not timeout during these requests

import os

# Workers
# 2026-07-01: REDUCED FROM 2 TO 1 — OOM fix.
# Root cause: extract_pdf_text() uses mp.get_context("fork") which copies the
# full gunicorn worker RSS (~175-210MB) to a child process for each PDF
# extraction. With 2 workers both simultaneously extracting (triggered by a
# 6-8 file concurrent upload session), peak memory = 2×worker + 2×fork_child
# + overhead = ~430-480MB → over the 512MB Render Starter ceiling.
# workers=1 ensures only one extraction fork runs at a time (~200MB total,
# ~300MB headroom). Cost: serialised requests — uploads queue behind analysis
# calls (60-120s LLM). Acceptable on Starter plan. Revert to 2 only after
# refactoring _extract_pdf_text_worker into a standalone module (imports only
# fitz, not all of app.py) so mp.get_context("spawn") can be used instead —
# spawn child RSS ~30-40MB vs fork child ~175MB, making 2 workers safe again.
workers = 1
worker_class = "sync"

# Timeouts — must be longer than ANALYSIS_TIMEOUT_SECONDS (120)
timeout = 180          # Worker timeout — kill worker if request takes > 3 mins
graceful_timeout = 30  # Grace period on shutdown
keepalive = 5

# Binding
bind = f"0.0.0.0:{os.getenv('PORT', '10000')}"

# Logging
accesslog = "-"
errorlog = "-"
loglevel = "info"

# Prevent worker crashes from killing the whole process
max_requests = 500          # Restart workers after 500 requests — reduced frequency protects long-running background threads (area fetch)
max_requests_jitter = 50    # Stagger restarts
