"""Dependency-free container health probe for the local FastAPI process."""

import json
import os
from urllib.error import URLError
from urllib.request import urlopen


def main() -> int:
    """Return zero only when the service reports a loaded deployment model."""
    url = os.getenv("MARITIME_CBM_HEALTHCHECK_URL", "http://127.0.0.1:8000/health")
    try:
        with urlopen(url, timeout=2.0) as response:  # noqa: S310 - fixed local URL by default.
            payload = json.loads(response.read())
    except (OSError, URLError, ValueError, json.JSONDecodeError):
        return 1
    if response.status != 200:
        return 1
    if payload.get("status") != "ok" or payload.get("model_loaded") is not True:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
