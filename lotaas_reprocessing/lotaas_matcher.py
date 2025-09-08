#!/usr/bin/env python3
"""
lotaas_matcher.py

Utilities to load your sealed LOTAAS reference and match candidates as
"redetections" based ONLY on that list.

- Loads `lotaas_ref_with_coords.csv` (path can be set via env LOTAAS_REF).
- Prefers DM_LOTAAS; falls back to DM_atnf when DM_LOTAAS is NaN.
- Provides:
    load_reference(path=None) -> pd.DataFrame
    lotaas_within(pointing_coord, radius_arcmin=3.0) -> pd.DataFrame
    match_redetection(dm_cand, local_ref, dm_abs_tol=5.0, dm_rel_tol=0.10) -> dict
"""

from __future__ import annotations
import os
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
from astropy import units as u

# Default to CSV one directory above this file; override with env LOTAAS_REF or explicit load_reference(path=...)
_DEFAULT_REF_PATH = os.getenv(
    "LOTAAS_REF",
    os.path.join(os.path.dirname(__file__), "..", "lotaas_ref_with_coords.csv")
)

_ref_cache: pd.DataFrame | None = None

def load_reference(path: str | None = None) -> pd.DataFrame:
    """
    Load the sealed LOTAAS reference CSV.
    Required cols: psr, ra_deg, dec_deg
    Optional: DM_LOTAAS, DM_atnf
    Returns a DataFrame with an added 'dm_ref' column (preferred DM).
    """
    global _ref_cache
    ref_path = path or _DEFAULT_REF_PATH
    if _ref_cache is not None and (path is None or ref_path == _DEFAULT_REF_PATH):
        return _ref_cache

    df = pd.read_csv(ref_path)

    missing = [c for c in ("psr", "ra_deg", "dec_deg") if c not in df.columns]
    if missing:
        raise RuntimeError(f"{ref_path} missing required columns: {', '.join(missing)}")

    # Normalize types
    df["psr"] = df["psr"].astype(str).str.strip()
    df["ra_deg"] = pd.to_numeric(df["ra_deg"], errors="coerce")
    df["dec_deg"] = pd.to_numeric(df["dec_deg"], errors="coerce")

    # Preferred DM for matching
    dm_l = df["DM_LOTAAS"] if "DM_LOTAAS" in df.columns else np.nan
    dm_a = df["DM_atnf"]   if "DM_atnf"   in df.columns else np.nan
    df["dm_ref"] = pd.Series(dm_l, dtype="float64")
    df["dm_ref"] = df["dm_ref"].where(~pd.isna(df["dm_ref"]), pd.Series(dm_a, dtype="float64"))

    # Cache only for default path usage
    if path is None or ref_path == _DEFAULT_REF_PATH:
        _ref_cache = df

    return df

def _coords_from_df(df: pd.DataFrame) -> SkyCoord:
    """Build a vector SkyCoord from numeric degrees in the DataFrame."""
    return SkyCoord(df["ra_deg"].values * u.deg, df["dec_deg"].values * u.deg)

def lotaas_within(pointing_coord: SkyCoord,
                  radius_arcmin: float = 300.0,
                  ref: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    Return LOTAAS rows within radius (arcmin) of a pointing coordinate, sorted by separation.
    """
    ref = ref if ref is not None else load_reference()
    # Drop any rows without coordinates
    ref = ref.dropna(subset=["ra_deg", "dec_deg"]).copy()
    if ref.empty:
        return ref

    coords = _coords_from_df(ref)  # proper SkyCoord array
    sep = pointing_coord.separation(coords)
    keep = sep <= (radius_arcmin * u.arcmin)
    if not np.any(keep):
        return ref.iloc[0:0].copy()

    out = ref.loc[keep].copy()
    out["sep_arcsec"] = sep[keep].arcsec
    return out.sort_values("sep_arcsec")

def match_redetection(dm_cand: float,
                      local_ref: pd.DataFrame,
                      dm_abs_tol: float = 1,
                      dm_rel_tol: float = 0.05) -> dict:
    """
    Given a candidate DM and a local, position-filtered slice of the LOTAAS ref,
    decide whether this is a redetection.

    A match occurs if |DM_cand - dm_ref| <= max(dm_abs_tol, dm_rel_tol * dm_ref).

    Returns dict with keys:
      is_match, psr, dm_ref, sep_arcsec
    """
    if local_ref is None or local_ref.empty:
        return {"is_match": False, "psr": None, "dm_ref": None, "sep_arcsec": None}

    ref = local_ref.dropna(subset=["dm_ref"]).copy()
    if ref.empty:
        return {"is_match": False, "psr": None, "dm_ref": None, "sep_arcsec": None}

    tol = np.maximum(dm_abs_tol, dm_rel_tol * ref["dm_ref"].clip(lower=1e-6))
    ok = (np.abs(ref["dm_ref"] - dm_cand) <= tol)
    ref = ref.loc[ok]
    if ref.empty:
        return {"is_match": False, "psr": None, "dm_ref": None, "sep_arcsec": None}

    ref = ref.assign(dm_diff=np.abs(ref["dm_ref"] - dm_cand))
    best = ref.sort_values(["dm_diff", "sep_arcsec"]).iloc[0]
    return {
        "is_match": True,
        "psr": str(best["psr"]),
        "dm_ref": float(best["dm_ref"]) if pd.notna(best["dm_ref"]) else None,
        "sep_arcsec": float(best["sep_arcsec"]) if "sep_arcsec" in best else None,
    }