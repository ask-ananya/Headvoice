"""Offline fallback: any cloud failure or timeout falls back to the fixed phrase list."""

from __future__ import annotations

import concurrent.futures as cf
import logging
from collections.abc import Callable

log = logging.getLogger(__name__)

# A shared pool so a hung request never blocks the caller beyond the timeout.
_POOL = cf.ThreadPoolExecutor(max_workers=2, thread_name_prefix="cloud")


def candidates_with_fallback(
    fetch: Callable[[], list[str]],
    fallback: list[str],
    valid_ids: set[str],
    timeout_s: float,
) -> tuple[list[str], str]:
    """Return (message ids, source). Source is "cloud" or "offline:<reason>".

    Cloud answers are filtered to known phrase IDs; an empty or invalid answer is a failure.
    """
    fut = _POOL.submit(fetch)
    try:
        ids = fut.result(timeout=timeout_s)
    except cf.TimeoutError:
        fut.cancel()
        log.warning("suggest timeout after %.1fs, using offline list", timeout_s)
        return list(fallback), "offline:timeout"
    except Exception as e:  # network errors, HTTP errors, parse errors
        log.warning("suggest failed (%s), using offline list", e)
        return list(fallback), "offline:error"
    ids = [i for i in ids if isinstance(i, str) and i in valid_ids] if isinstance(ids, list) else []
    if not ids:
        return list(fallback), "offline:invalid"
    return ids, "cloud"
