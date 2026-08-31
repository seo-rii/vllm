# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Standalone remote draft server.

Usage:
    python -m vllm.entrypoints.remote_draft_server --endpoint tcp://127.0.0.1:0

The implementation lives in ``vllm.v1.spec_decode.remote.server``; this
module only exposes it from the conventional entrypoints namespace.
"""

from vllm.v1.spec_decode.remote.server import main

__all__ = ["main"]

if __name__ == "__main__":
    main()
