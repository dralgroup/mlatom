"""Version resolution and git-stamp probing.

``__version__`` is the pinned release number (from ``_version_static.py``).
``commit`` and ``build_date`` come from a live git probe when MLatom runs from
a source checkout, and fall back to the values pinned at release time.
"""
import os
import subprocess

from . import _version_static as _static

__version__ = _static.version


def _probe_git():
    """Return (short_commit, iso_date) from git, or (None, None) off-repo."""
    try:
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if not os.path.isdir(os.path.join(root, ".git")):
            return None, None

        def _git(*args):
            out = subprocess.run(["git", *args], cwd=root, capture_output=True,
                                 text=True, timeout=2)
            return out.stdout.strip()

        commit = _git("rev-parse", "--short", "HEAD") or None
        date = _git("log", "-1", "--format=%ad", "--date=short", "HEAD") or None
        return commit, date
    except Exception:
        return None, None


def commit_and_date():
    """Best-effort (commit, build_date): live git in a source checkout,
    else the values pinned at release time."""
    commit, date = _probe_git()
    return commit or _static.commit, date or _static.build_date


def aitomic_addons_version():
    """Display string for the installed Aitomic add-ons — the version, plus the
    commit and build date when the package records them — or None if not present.

    Two install shapes report the same way: a build that bundles the add-ons
    stamps a marker module; a separately-installed add-ons package exposes its
    own ``__version__`` (and optional ``__commit__`` / ``__build_date__``).
    """
    version = commit = build_date = None
    try:
        from .addons import _aitomic_version as _b
        version = getattr(_b, "version", None)
        commit = getattr(_b, "commit", None)
        build_date = getattr(_b, "build_date", None)
    except Exception:
        try:
            import aitomic_addons
            version = getattr(aitomic_addons, "__version__", None)
            commit = getattr(aitomic_addons, "__commit__", None)
            build_date = getattr(aitomic_addons, "__build_date__", None)
        except Exception:
            return None
    if version is None:
        return None
    if commit and build_date:
        return "%s (commit %s, built %s)" % (version, commit, build_date)
    return version
