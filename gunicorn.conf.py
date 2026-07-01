# gunicorn.conf.py
# Gunicorn configuration for LegalSmegal API
# LLM calls (document summary) can take 60-120 seconds
# Worker must not timeout during these requests

import os

# Workers
# 2026-07-01: restored to 2 after workers=1 caused ERR_HTTP2_PROTOCOL_ERROR.
# The OOM on 2026-07-01 (Render instance 27crh, 10:03 AM) was caused by
# CONCURRENCY=4 in legalsmegal-upload.html sending 4 simultaneous PDF
# extractions. Each extraction forks the gunicorn worker (~175MB RSS) to a
# child process via mp.get_context("fork"). 4 concurrent forks = ~430MB+
# overhead → over the 512MB Starter ceiling. The fix is reducing frontend
# upload concurrency to 1 (one fork child at a time), NOT reducing workers.
# workers=1 made the OOM worse in a different direction: the single sync
# worker blocked in p.join() during extraction, causing all other connections
# (status polls, analysis, other uploads) to queue on Render's load balancer
# and eventually receive ERR_HTTP2_PROTOCOL_ERROR. workers=2 keeps worker 2
# available to serve those requests while worker 1 is extracting.
# Memory with CONCURRENCY=1: 2×175MB (workers) + 1×25MB (fork) + 30MB = ~405MB → safe.
# Follow-up (medium-term): refactor _extract_pdf_text_worker into a standalone
# module (imports only fitz, not all of app.py) and switch to
# mp.get_context("spawn") — spawn child RSS ~30-40MB vs fork child ~175MB,
# which allows restoring CONCURRENCY=4 if needed in future.
workers = 2
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
