# gunicorn.conf.py
# Gunicorn configuration for LegalSmegal API
# LLM calls (document summary) can take 60-120 seconds
# Worker must not timeout during these requests

import os

# Workers
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
