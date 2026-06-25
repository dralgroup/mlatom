# Pinned release info.
#
# `version` is the source of truth for the release number and is a human
# judgement call (is the next release a patch or a minor?). To make a release:
# bump `version` here, commit, then tag `release-X.Y.Z`.
#
# `commit` and `build_date` are advisory and are refreshed automatically by
# `stamp_version.py` at release time -- do not hand-edit them. When MLatom runs
# from a source checkout they are filled live from git instead; an installed
# build (no .git) falls back to the values pinned here. `version` always wins.
version = "3.23.1"
commit = 'd0db1b34'
build_date = '2026-06-25'
