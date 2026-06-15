#!/usr/bin/env python3
"""Optional, throttled check for a newer MLatom / Aitomic Add-Ons on PyPI.

Designed to stay out of the way of a calculation and to be quiet until it has
something to say:

  * The PyPI request runs in a background thread, so it never adds to the run
    time -- even a fast calculation is not slowed by the check.
  * At most one request per day (cached under ``~/.mlatom``). A host with no
    internet just fails that one background attempt silently and is not retried
    until the next day; with no data there is simply no notice.
  * The notice itself is shown at most once per day, not on every run/import.
  * Set ``MLATOM_NO_VERSION_CHECK=1`` to turn it off entirely.

Only installed packages are checked (MLatom always; the Aitomic add-ons only
when present, read via importlib.metadata -- never ``import aitomic_addons``,
which would pull in torch and trip its first-use notice).
"""
import json
import os
import re
import threading
import time
import urllib.request

from . import _version

USER_DIR = os.path.join(os.path.expanduser("~"), ".mlatom")
CACHE_FILE = os.path.join(USER_DIR, "version_check.json")
CHECK_INTERVAL = 24 * 3600          # seconds between (background) PyPI requests
NOTIFY_INTERVAL = 24 * 3600         # seconds between showing the notice
NETWORK_TIMEOUT = 1.5               # seconds per request
_PYPI_JSON = "https://pypi.org/pypi/%s/json"


def _disabled():
    return os.environ.get("MLATOM_NO_VERSION_CHECK", "").strip().lower() \
        not in ("", "0", "false", "no")


def _installed_versions():
    """{distribution-name: version} for the packages we check -- installed only."""
    versions = {"mlatom": _version.__version__}
    try:
        import importlib.metadata as _md
        versions["aitomic-addons"] = _md.version("aitomic-addons")
    except Exception:
        pass                        # add-ons not installed -> nothing to check
    return versions


def _pypi_latest(pkg):
    try:
        with urllib.request.urlopen(_PYPI_JSON % pkg, timeout=NETWORK_TIMEOUT) as r:
            return json.load(r).get("info", {}).get("version")
    except Exception:
        return None                 # no internet / PyPI down / parse error


def _newer(latest, current):
    """True only if `latest` is a strictly newer release than `current`."""
    if not latest or not current or latest == current:
        return False
    try:
        from packaging.version import Version
        return Version(latest) > Version(current)
    except Exception:
        def nums(v):
            return [int(x) for x in re.findall(r"\d+", v)]
        try:
            return nums(latest) > nums(current)
        except Exception:
            return False


def _read_cache():
    try:
        with open(CACHE_FILE) as f:
            return json.load(f)
    except Exception:
        return {}


def _write_cache(data):
    try:
        os.makedirs(USER_DIR, exist_ok=True)
        with open(CACHE_FILE, "w") as f:
            json.dump(data, f)
    except Exception:
        pass


def _refresh_in_background(packages):
    """Fetch latest versions off the calculation's path; update `latest` only on
    success, so a failed/offline attempt leaves the cached data untouched."""
    def _work():
        try:
            fetched = {p: v for p in packages
                       for v in [_pypi_latest(p)] if v}
            if fetched:
                cache = _read_cache()
                latest = dict(cache.get("latest", {}))
                latest.update(fetched)
                cache["latest"] = latest
                _write_cache(cache)
        except Exception:
            pass
    try:
        threading.Thread(target=_work, daemon=True).start()
    except Exception:
        pass


def update_notice():
    """A one-line-per-package notice for available updates, or '' when there is
    nothing to say (disabled, up to date, no data yet, or already shown today).
    Best-effort: never raises and never blocks on the network."""
    try:
        if _disabled():
            return ""
        installed = _installed_versions()
        cache = _read_cache()
        now = time.time()
        changed = False

        # Throttle the network request to once a day and run it in the
        # background. Mark the attempt NOW -- before it finishes and even if the
        # host is offline -- so a failed attempt is not retried until tomorrow.
        if now - cache.get("checked_at", 0) > CHECK_INTERVAL:
            cache["checked_at"] = now
            changed = True
            _refresh_in_background(list(installed))

        latest = cache.get("latest", {})
        outdated = [(pkg, cur, latest.get(pkg)) for pkg, cur in installed.items()
                    if _newer(latest.get(pkg), cur)]

        notice = ""
        if outdated and now - cache.get("last_notified", 0) > NOTIFY_INTERVAL:
            cache["last_notified"] = now
            changed = True
            notice = "\n".join(
                "A newer %s is available: %s (you have %s) -- upgrade with "
                "`pip install -U %s`" % (pkg, lat, cur, pkg)
                for pkg, cur, lat in outdated)

        if changed:
            _write_cache(cache)
        return notice
    except Exception:
        return ""
