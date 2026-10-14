"""rsched: tracing and scheduling for concurrent RPent episodes.

Components
----------
- ``rsched.proxy``   HTTP proxy in front of RPent's model services (Pi0.5, SAM3).
                     Logs every request and decides dispatch order (the scheduler).
- ``rsched.launch``  Drop-in wrapper for the ``rpent`` CLI that logs tool calls
                     and tags each model request with episode / tool-call ids.
- ``rsched.nvml``    GPU memory and utilization sampler.
- ``rsched.sweep``   Runs a fixed number of concurrent episodes per level.
"""

__version__ = "0.1.0"

# Request headers the launcher adds and the proxy reads.
HDR_EPISODE = "X-RS-Episode"
HDR_CALL = "X-RS-Call"
HDR_TOOL = "X-RS-Tool"
HDR_PRIORITY = "X-RS-Priority"
