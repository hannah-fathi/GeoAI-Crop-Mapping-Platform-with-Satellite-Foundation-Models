from __future__ import annotations
# ============================== SYSTEM / I/O ==============================
import os, sys, zipfile, tempfile, threading, queue, warnings, subprocess, math, time,json
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any, Callable
from dataclasses import dataclass, asdict
# robust browser handling + CLI path resolution + URL parsing
import webbrowser
import re

# === sklearn MLP fallback + probability calibration ===
from sklearn.neural_network import MLPClassifier
from sklearn.calibration import CalibratedClassifierCV


# --- Force stable EE HTTP timeouts + quieter TF logs (place BEFORE `import ee`) ---

# Use Python implementation for protobuf on Windows
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

# TensorFlow log level: 0=all, 1=errors, 2=warnings, 3=info
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# Earth Engine HTTP timeouts (in seconds)
os.environ["EE_URL_OPEN_TIMEOUT"] = "300"
os.environ["EE_URL_READ_TIMEOUT"]  = "1800"

# Optional: increase automatic retries for transient EE errors
os.environ["EE_NUM_RETRIES"] = "10"

# Silence non-critical user warnings
warnings.filterwarnings("ignore", category=UserWarning)


# ============================== NUMERIC / GEO =============================
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely import wkb as _wkb
from shapely.geometry import Polygon, MultiPolygon, GeometryCollection

try:
    from shapely.validation import make_valid as _make_valid
    _HAS_MAKE_VALID = True
except Exception:
    _HAS_MAKE_VALID = False

_USE_PYOGRIO = False
try:
    import pyogrio  # noqa: F401
    _USE_PYOGRIO = True
except Exception:
    pass

# ============================== ML / TF ==================================
import joblib

def _lazy_tf(use_gpu: bool = False):
    """
    Lazy-import TensorFlow/Keras.
    If TF import fails (e.g., protobuf mismatch), raise ImportError with
    a clear remediation hint. Callers can fall back to sklearn.
    """
    try:
        import tensorflow as tf
    except Exception as e:
        # Provide a precise hint for the common 'runtime_version' protobuf issue
        hint = (
            "TensorFlow import failed (likely protobuf mismatch). "
            "Recommended fix on Windows:\n"
            "  pip uninstall -y protobuf\n"
            "  pip install protobuf==4.25.3\n"
            "  pip install 'tensorflow>=2.15,<2.18'\n"
            "Then restart the app."
        )
        raise ImportError(hint) from e

    if not use_gpu:
        try: tf.config.set_visible_devices([], 'GPU')
        except Exception: pass
    else:
        try:
            for g in tf.config.list_physical_devices('GPU'):
                tf.config.experimental.set_memory_growth(g, True)
        except Exception:
            pass

    from tensorflow import keras
    from tensorflow.keras import layers, regularizers, callbacks
    return tf, keras, layers, regularizers, callbacks


# ============================= EARTH ENGINE ===============================
try:
    import ee
except Exception:
    ee = None

# ============================== GUI / MAP =================================
# Import map widget safely and expose a guard flag so other functions can check it.
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from tkinter.scrolledtext import ScrolledText

try:
    from tkintermapview import TkinterMapView
    _MAP_OK = True
except Exception:
    TkinterMapView = None
    _MAP_OK = False



# ============================== CONSTANTS =================================
ESRI_IMAGERY = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
ESRI_GRAY    = "https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Light_Gray_Base/MapServer/tile/{z}/{y}/{x}"

COLOR_VEG   = "#2e7d32"  # green
COLOR_NONV  = "#6d4c41"  # brown
COLOR_UNK   = "#9e9e9e"  # gray

_COLOR_NAME_MAP = {
    "#1f77b4": "blue", "#ff7f0e": "orange", "#2ca02c": "green", "#d62728": "red",
    "#9467bd": "purple", "#8c564b": "brown", "#e377c2": "pink", "#7f7f7f": "gray",
    "#bcbd22": "olive", "#17becf": "teal",
    COLOR_VEG.lower():  "green",
    COLOR_NONV.lower(): "brown",
    COLOR_UNK.lower():  "gray",
}
def _color_name(hex_code: str) -> str:
    """Return a short, friendly color name for a hex code; fallback to the hex itself."""
    if not isinstance(hex_code, str):
        return str(hex_code)
    return _COLOR_NAME_MAP.get(hex_code.lower(), hex_code)

def _palette_with_nonveg_unknown(class_names: list[str]) -> dict[str, str]:

    # --- Helper: canonicalize only for reserved labels (keep others as-is) ---
    def _canon_reserved(label: str) -> str:
        orig = str(label).strip()
        t = orig.lower().replace("-", "_").replace(" ", "_")
        if t in {"veg", "vegetation"}:
            return "VEG"
        if t in {"non_veg", "nonveg", "non_vegetation", "nonvegetation", "non"}:
            return "NON_VEG"
        if t in {"unknown", "unk", "none", "nan"}:
            return "UNKNOWN"
        return orig  # non-reserved: return cleaned original (preserve user label)

    # --- De-duplicate while preserving order (after reserved canonicalization) ---
    seen: set[str] = set()
    ordered: list[str] = []
    for name in (class_names or []):
        c = _canon_reserved(name)
        if c not in seen:
            seen.add(c)
            ordered.append(c)

    # --- Start with reserved buckets (always present with fixed colors) ---
    pal: dict[str, str] = {
        "VEG": COLOR_VEG,
        "NON_VEG": COLOR_NONV,
        "UNKNOWN": COLOR_UNK,
    }

    # Fallback if CLASS_COLORS is empty (shouldn't happen, but be safe)
    palette_cycle = CLASS_COLORS if 'CLASS_COLORS' in globals() and CLASS_COLORS else ["#1f77b4", "#ff7f0e", "#2ca02c"]

    # --- Assign rotating colors to non-reserved classes deterministically ---
    idx = 0
    for name in ordered:
        if name in pal:            # skip reserved; they already have fixed colors
            continue
        pal[name] = palette_cycle[idx % len(palette_cycle)]
        idx += 1

    return pal



SUPPORTED_EXT = {".gpkg": 0, ".geojson": 1, ".json": 1, ".shp": 2, ".kml": 3}

DATASET_ID = "GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL"   # AlphaEarth (annual)

if os.name == "nt":
    DEFAULT_OUTDIR = "C:/Users/irantech24/Desktop/Hana/Master of Artificial Intelligence/Remote sensing/task2/code/venv/outputs"
else:
    DEFAULT_OUTDIR = str((Path.home() / "Desktop" / "output").resolve())



# =============================================================================
#                           ENVIRONMENT GUARDS
# =============================================================================
def _check_protobuf_version():
    """
    Warn (don't crash) for protobuf>=5 which can be flaky with EE on Win+Py3.10.
    """
    try:
        from google.protobuf import __version__ as _pbv
        major = int(_pbv.split(".")[0])
        if major >= 5:
            # FIX: warn instead of raising; app should still open.
            warnings.warn(
                f"Incompatible protobuf {_pbv}. "
                "Recommend: pip install 'protobuf>=3.20.3,<5.0.0' -U",
                RuntimeWarning
            )
    except Exception:
        pass

_check_protobuf_version()

# =============================================================================
#                           GEOMETRY UTILITIES
# =============================================================================
def _force_2d(geom):
    """Drop Z/M dimensions via WKB round-trip (robust)."""
    try:
        return _wkb.loads(_wkb.dumps(geom, output_dimension=2))
    except Exception:
        return geom

def _validify(geom):
    """Return a valid polygon/multipolygon or None."""
    if geom is None or geom.is_empty:
        return None
    if geom.is_valid:
        return geom
    try:
        return _make_valid(geom) if _HAS_MAKE_VALID else geom.buffer(0)
    except Exception:
        try:
            return geom.buffer(0)
        except Exception:
            return None

def _m_to_deg(m: float) -> float:
    """Approx meters->degrees (OK for mid-latitudes)."""
    return float(m) / 111_320.0

def shapely_to_ee_geometry(geom, simplify_m: float = 2.0):
    """
    Convert shapely Polygon/MultiPolygon to ee.Geometry with closed rings.
    """
    if ee is None:
        raise RuntimeError("earthengine-api is not installed. `pip install earthengine-api`")
    g = _validify(_force_2d(geom))
    if g is None or g.is_empty:
        return None
    if simplify_m and simplify_m > 0:
        try:
            g = g.simplify(_m_to_deg(simplify_m), preserve_topology=True)
        except Exception:
            pass

    def poly_to_coords(p: Polygon):
        rings = []
        if p.exterior is not None:
            ext = [(float(x), float(y)) for (x, y, *_) in p.exterior.coords]
            if ext[0] != ext[-1]:
                ext.append(ext[0])
            rings.append([[round(x, 6), round(y, 6)] for x, y in ext])
        for r in p.interiors:
            ins = [(float(x), float(y)) for (x, y, *_) in r.coords]
            if ins[0] != ins[-1]:
                ins.append(ins[0])
            rings.append([[round(x, 6), round(y, 6)] for x, y in ins])
        return rings

    if isinstance(g, Polygon):
        return ee.Geometry.Polygon(poly_to_coords(g), None, False)
    elif isinstance(g, MultiPolygon):
        return ee.Geometry.MultiPolygon([poly_to_coords(p) for p in g.geoms], None, False)
    return None

# =============================================================================
#                           EE AUTH / INITIALIZE (robust)
# =============================================================================
_EE_READY = False
EE_LOCK = threading.Lock()  # prevent concurrent getInfo/Initialize from freezing Tk
def _ee_cli_bin() -> str:
    """
    Return path to the Earth Engine CLI binary within the current venv, if present.
    Fallback to 'earthengine' (assuming it's on PATH).
    """
    try:
        exe = Path(sys.executable)
        cand = exe.with_name("earthengine.exe") if os.name == "nt" else exe.with_name("earthengine")
        return str(cand) if Path(cand).exists() else "earthengine"
    except Exception:
        return "earthengine"

def _run_cli(args: List[str]) -> Tuple[bool, str, str]:
    """
    Run a CLI command and return (ok, stdout, stderr).
    Treat as success if returncode==0 OR stdout/stderr contains the
    known success string used by Earth Engine's authenticator.
    """
    try:
        r = subprocess.run(args, text=True, capture_output=True)
        blob = f"{r.stdout}\n{r.stderr}".lower()
        ok = (r.returncode == 0) or ("successfully saved authorization token" in blob)
        return ok, r.stdout, r.stderr
    except Exception as e:
        return False, "", str(e)


def _ee_init_with_retry(project_id: str, credentials=None, tries=4, log=None) -> bool:
    """Initialize EE with retries/backoff and long deadline."""
    global _EE_READY
    last = None
    for i in range(tries):
        try:
            with EE_LOCK:
                if credentials is None:
                    ee.Initialize(project=project_id, opt_url="https://earthengine.googleapis.com")
                else:
                    ee.Initialize(credentials=credentials, project=project_id, opt_url="https://earthengine.googleapis.com")
                ee.data.setDeadline(1800000)
            _EE_READY = True
            return True
        except Exception as e:
            last = e
            delay = 2 * (2 ** i)
            log and log(f"[EE] init error: {e} — retry {i+1}/{tries} in {delay}s")
            time.sleep(delay)
    log and log(f"[EE] init failed: {last}")
    _EE_READY = False
    return False

def ee_sign_in_browser(project_id: str, reset_token: bool = False) -> bool:
    """
    Robust EE sign-in for Windows:
      1) Try cached creds (Initialize).
      2) Try `earthengine authenticate` on a few alternate ports to avoid
         firewall/AV conflicts with localhost binding.
      3) If all fail, guide the user to console-mode auth (no local server).
    """
    if ee is None:
        print("[EE] earthengine-api is not installed.")
        return False

    os.environ["GOOGLE_CLOUD_PROJECT"] = project_id

    # 1) Cached creds first
    if _ee_init_with_retry(project_id, log=print):
        return True

    # 2) Try CLI auth on multiple ports; avoid the fragile module fallback.
    cli = _ee_cli_bin()
    base = [cli, "authenticate"]
    if reset_token:
        base.append("--force")
    base.append("--quiet")

    preferred_ports = [8085, 8086, 8765, 9005]
    for port in preferred_ports:
        args = base + [f"--port={port}"]
        ok, out, err = _run_cli(args)
        print(f"[EE] authenticate on port {port}: {'OK' if ok else 'FAIL'}")
        if ok:
            # Try to initialize immediately after a (possibly) successful CLI run
            return _ee_init_with_retry(project_id, log=print)

    # 3) Console-mode fallback instructions (no localhost HTTP server)
    #    We do not attempt to automate the two-step console flow here;
    #    we show precise steps that always work behind firewalls.
    msg = (
        "[EE] Could not open a local OAuth callback port (blocked by firewall/AV).\n"
        "Do a one-time console authentication:\n"
        "  1) Open Command Prompt and run:\n"
        "       earthengine authenticate --auth_mode=console\n"
        "  2) Follow the printed steps: open the URL, then run the SECOND command\n"
        "     it prints with --authorization-code and --code-verifier exactly as shown.\n"
        "  3) When you see 'Successfully saved authorization token', return to the app and\n"
        "     click 'Test EE (cached)'."
    )
    print(msg)
    return False

def ee_force_browser_auth(project_id: str, *, force_reset: bool = True, log=None) -> bool:
    """
    Always launch a browser for Earth Engine OAuth.

    Strategy
    --------
    A) Try 'earthengine authenticate' with a set of alternate ports. This opens
       the system browser and performs an HTTP callback on localhost.
    B) If all ports fail (blocked by firewall/AV and WinError 10013), fall back
       to 'console' mode: parse the printed URL and open it via webbrowser.open().
       The user then pastes the auth code into the *second* CLI command printed
       by the tool. This path guarantees a browser window as well.

    Returns True only if we can initialize EE after auth, otherwise False.
    """
    if ee is None:
        log and log("[EE] earthengine-api is not installed.")
        return False

    os.environ["GOOGLE_CLOUD_PROJECT"] = project_id

    # Always force an interactive flow (do not short-circuit on cached creds).
    cli = _ee_cli_bin()
    base = [cli, "authenticate"]
    if force_reset:
        base.append("--force")
    base.append("--quiet")

    # A) Try auto browser with HTTP callback on a few ports
    ports = [8085, 8086, 8765, 9005]
    for port in ports:
        args = base + [f"--port={port}"]
        ok, out, err = _run_cli(args)
        log and log(f"[EE] authenticate --port={port} => {'OK' if ok else 'FAIL'}")

        if ok:
            # Immediately try to Initialize
            if _ee_init_with_retry(project_id, log=log):
                return True
            # If init still fails, continue to next fallback.

    # B) Console-mode fallback: open the printed URL in the default browser.
    # We still guarantee a browser window for the user.
    args = [cli, "authenticate", "--auth_mode=console", "--quiet"]
    ok, out, err = _run_cli(args)

    # Try to extract the URL that the CLI prints in console mode
    blob = (out or "") + "\n" + (err or "")
    m = re.search(r"(https?://[^\s]+)", blob)
    if m:
        url = m.group(1)
        try:
            webbrowser.open(url, new=1, autoraise=True)
            log and log(f"[EE] Opened browser to: {url}")
        except Exception as e:
            log and log(f"[EE] Could not open browser automatically: {e}\nURL: {url}")

    # Show very clear next steps (user must run the printed second command)
    # Note: we cannot auto-complete console flow without the user's auth code.
    log and log(
        "[EE] Console auth: after approving in the browser, run the SECOND command\n"
        "printed by the CLI (with --authorization-code and --code-verifier).\n"
        "When you see 'Successfully saved authorization token', click 'Test EE (cached)'."
    )

    # Try init anyway (in case token was already cached from a previous run)
    if _ee_init_with_retry(project_id, log=log):
        return True
    return False

def ee_initialize(project_id: str,
                  sa_email: Optional[str] = None,
                  sa_key_json: Optional[str] = None,
                  interactive_fallback: bool = False,
                  log=None) -> bool:
    if ee is None:
        log and log("[EE] earthengine-api not installed.")
        return False

    if _EE_READY:  # already initialized
        return True

    os.environ["GOOGLE_CLOUD_PROJECT"] = project_id

    creds = None
    if sa_email and sa_key_json and Path(sa_key_json).exists():
        try:
            creds = ee.ServiceAccountCredentials(sa_email, sa_key_json)
        except Exception as e:
            # don't crash; continue with user creds.
            log and log(f"[EE] SA credentials error: {e}")

    # 1) Preferred: with project quota
    if _ee_init_with_retry(project_id, credentials=creds, log=log):
        return True

    # 2) Fallback: without project (user quota)
    log and log("[EE] retrying Initialize() without project …")
    try:
        with EE_LOCK:
            if creds is None:
                ee.Initialize()
            else:
                ee.Initialize(credentials=creds)
        ee.data.setDeadline(600000)
        globals()["_EE_READY"] = True
        return True
    except Exception as e:
        log and log(f"[EE] init failed (no project): {e}")

    # 3) Optional final fallback: interactive browser auth
    if interactive_fallback:
        log and log("[EE] trying interactive browser auth …")
        if ee_sign_in_browser(project_id, reset_token=False):
            return True

    # only now admit failure.
    return False



def aef_image_for_year(year: int):
    if not _EE_READY:
        raise RuntimeError("EE not initialized. Click 'Sign in (Google)'.")
    year = int(year)

    col = ee.ImageCollection(DATASET_ID).filter(ee.Filter.eq("year", year))
    img = col.mosaic()
    try:
        with EE_LOCK:
            names = img.bandNames().getInfo() or []
    except Exception:
        names = []

    if not names:
        col2 = ee.ImageCollection(DATASET_ID).filterDate(f"{year}-01-01", f"{year+1}-01-01")
        img = col2.mosaic()
        with EE_LOCK:
            names = img.bandNames().getInfo() or []

    if not names:
        raise RuntimeError(f"AlphaEarth image not found for year={year}. Try 2020–2023.")
    return img



# =============================================================================
#                          EE REDUCE / EMBEDDINGS (streaming)
# =============================================================================
def _ee_retry(call, max_retries=4, base_delay=2.0, log=None):
    for i in range(max_retries):
        try: return call()
        except Exception as e:
            if i == max_retries - 1: raise
            delay = base_delay * (2 ** i)
            log and log(f"[EE] transient error: {e} – retry in {delay:.1f}s")
            time.sleep(delay)

def reduce_regions_mean(img: ee.Image, fc: ee.FeatureCollection,
                        scale: int = 10, tile_scale: int = 4) -> List[Dict]:
    reducer = ee.Reducer.mean()
    out_fc = img.reduceRegions(
        collection=fc,
        reducer=reducer,
        scale=int(scale),
        tileScale=int(tile_scale)
    )
    def _get():
        with EE_LOCK: return out_fc.getInfo()
    info = _ee_retry(_get)
    return info.get("features", [])


def reduce_regions_stats_chunked(
    img: "ee.Image",
    fc: "ee.FeatureCollection",
    scale: int = 10,
    tile_scale: int = 4,
    chunk_size: int = 250,
    log=None,
) -> list[dict]:
    """
    Robust, chunked stats reducer for Earth Engine.

    Returns a flat Python list of feature dicts whose 'properties' include
    per-band statistics like A00 (mean), A00_p25, A00_p50, A00_p75, and poly_id.
    This is a stats-preserving fallback when a single large reduceRegions call
    would timeout.

    Strategy:
      - Split FeatureCollection by poly_id batches.
      - Try reduce_regions_stats on each batch.
      - On failure, shrink chunk_size and slightly increase tile_scale, recurse.
    """

    # Collect all polygon ids with a single round-trip
    try:
        with EE_LOCK:
            ids = ee.List(fc.aggregate_array("poly_id")).getInfo()
    except Exception as e:
        log and log(f"[EE] aggregate_array('poly_id') failed: {e}")
        raise

    ids = list(map(int, ids or []))
    if not ids:
        log and log("[EE] stats-chunked: no poly_id present; nothing to do.")
        return []

    results: list[dict] = []
    total = len(ids)
    chunk_size = int(chunk_size)
    n_chunks = (total + chunk_size - 1) // chunk_size

    for cidx, start in enumerate(range(0, total, chunk_size), 1):
        sub_ids = ids[start:start + chunk_size]
        sub_fc = fc.filter(ee.Filter.inList("poly_id", sub_ids))

        log and log(
            f"[EE] stats chunk {cidx}/{n_chunks} — {len(sub_ids)} polys "
            f"(tileScale={tile_scale}, scale={scale})"
        )
        try:
            part = reduce_regions_stats(img, sub_fc, scale=scale, tile_scale=int(tile_scale))
            if part:
                results.extend(part)
        except Exception as e:
            # If the batch fails (memory/timeout), try again with smaller chunks
            # and a slightly larger tileScale (reduces server work per pixel).
            log and log(f"[EE] stats chunk {cidx} failed: {e} — shrinking & retrying…")
            if chunk_size > 50:
                results.extend(
                    reduce_regions_stats_chunked(
                        img=img,
                        fc=sub_fc,
                        scale=scale,
                        tile_scale=min(16, int(tile_scale) + 2),
                        chunk_size=max(50, chunk_size // 2),
                        log=log,
                    )
                )
            else:
                # At this depth we give up to avoid infinite recursion
                raise

    return results


def reduce_regions_stats(img: ee.Image, fc: ee.FeatureCollection,
                         scale: int = 10, tile_scale: int = 4) -> List[Dict]:
    """
    Compute per-polygon statistics for all bands:
      - mean
      - percentiles at 25, 50 (median), 75
    Output properties per band follow GEE naming: <band>_mean, <band>_p25, <band>_p50, <band>_p75
    """
    reducer = ee.Reducer.mean().combine(
        ee.Reducer.percentile([25, 50, 75]), sharedInputs=True
    )
    out_fc = img.reduceRegions(
        collection=fc,
        reducer=reducer,
        scale=int(scale),
        tileScale=int(tile_scale)
    )
    with EE_LOCK:
        info = out_fc.getInfo()  # may raise transient errors; caller can wrap with retry if needed
    return (info or {}).get("features", [])


def reduce_regions_chunked(img: ee.Image, fc: ee.FeatureCollection,
                           scale: int = 10, tile_scale: int = 4,
                           chunk_size: int = 250, log=None) -> List[Dict]:
    with EE_LOCK: ids = fc.aggregate_array("poly_id").getInfo()
    results: List[Dict] = []; total = len(ids)
    for i in range(0, total, chunk_size):
        sub_ids = ids[i:i+chunk_size]
        sub_fc = fc.filter(ee.Filter.inList("poly_id", sub_ids))
        log and log(f"[EE] chunk {i//chunk_size+1}/{(total+chunk_size-1)//chunk_size} ({len(sub_ids)} polys)…")
        try:
            part = reduce_regions_mean(img, sub_fc, scale=scale, tile_scale=tile_scale)
            results.extend(part)
        except Exception as e:
            log and log(f"[EE] chunk failed: {e} — shrinking chunk/tileScale")
            if chunk_size > 50:
                results.extend(reduce_regions_chunked(img, sub_fc, scale=scale,
                                                      tile_scale=min(16, tile_scale + 2),
                                                      chunk_size=chunk_size // 2, log=log))
            else:
                raise
    return results

#===================================================================================================
def _atomic_npy_save(path: "Path", array: "np.ndarray") -> None:
    """Write a .npy atomically to reduce the chance of partial files on crashes."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    np.save(tmp, array)
    tmp.replace(path)


def _safe_write_excel(df: "pd.DataFrame", path: "Path", sheet: str, log=None) -> None:
    """Write an Excel file, with a temp file + rename for safety."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        with pd.ExcelWriter(tmp, engine="xlsxwriter") as writer:
            df.to_excel(writer, sheet_name=sheet, index=False)
        tmp.replace(path)
        log and log(f"[XLSX] saved: {path.name}")
    except Exception as e:
        log and log(f"[XLSX] write failed: {e}")


# ----------------------------- helpers -----------------------------
def _atomic_npy_save(path: "Path", arr: "np.ndarray") -> None:
    """
    Write a .npy file atomically so partial writes don't corrupt outputs.
    """
    import tempfile, os
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(prefix=".tmp_", suffix=".npy", dir=str(path.parent), delete=False) as tmp:
        np.save(tmp.name, arr.astype("float32"))
        tmp.flush()
        os.fsync(tmp.fileno())
        tmp_name = tmp.name
    os.replace(tmp_name, str(path))


def _safe_write_excel(df: "pd.DataFrame", xlsx_path: "Path", sheet_name: str, log=None) -> None:
    """
    Try to write a DataFrame to Excel using XlsxWriter; fall back to CSV if the
    engine is missing, and log a clear hint.
    """
    try:
        import xlsxwriter  # noqa: F401  (probe availability)
        with pd.ExcelWriter(xlsx_path, engine="xlsxwriter") as writer:
            df.to_excel(writer, index=False, sheet_name=sheet_name)
        log and log(f"[XLSX] saved: {Path(xlsx_path).name}")
    except Exception as e:
        log and log(f"[XLSX] write failed: {e} — install with: pip install XlsxWriter")
        # Optional: also leave a CSV next to it, so users still get a table.
        try:
            csv_fallback = Path(xlsx_path).with_suffix(".csv")
            df.to_csv(csv_fallback, index=False)
            log and log(f"[CSV] fallback saved: {csv_fallback.name}")
        except Exception as e2:
            log and log(f"[CSV] fallback failed: {e2}")


def reduce_regions_stats_chunked(
    img: "ee.Image",
    fc: "ee.FeatureCollection",
    scale: int = 10,
    tile_scale: int = 4,
    chunk_size: int = 250,
    log=None,
) -> "list[dict]":
    """
    Chunked fallback for the *stats* reducer.
    Unlike reduce_regions_chunked (which only does mean), this preserves p25/p50/p75.

    Strategy:
      - Split the feature collection by id batches.
      - Try reduce_regions_stats; on failure, shrink chunk_size and slightly bump tile_scale.
    """
    with EE_LOCK:
        ids = fc.aggregate_array("poly_id").getInfo()
    ids = list(map(int, ids or []))
    if not ids:
        return []

    results: list[dict] = []
    total = len(ids)
    n_chunks = (total + int(chunk_size) - 1) // int(chunk_size)

    for i in range(0, total, int(chunk_size)):
        part_ids = ids[i:i + int(chunk_size)]
        sub_fc = fc.filter(ee.Filter.inList("poly_id", part_ids))
        log and log(f"[EE] stats-chunk {i//int(chunk_size)+1}/{n_chunks} — {len(part_ids)} polys")
        try:
            res = reduce_regions_stats(img, sub_fc, scale=scale, tile_scale=tile_scale)
            results.extend(res or [])
        except Exception as e:
            log and log(f"[EE] stats-chunk failed: {e} — shrinking and retrying…")
            if chunk_size > 50:
                # Recursive retry with smaller chunk and a slightly larger tileScale
                results.extend(
                    reduce_regions_stats_chunked(
                        img, sub_fc, scale=scale,
                        tile_scale=min(16, tile_scale + 2),
                        chunk_size=max(50, chunk_size // 2),
                        log=log
                    )
                )
            else:
                raise
    return results
# -----------------------------------------------------------------------------


def stream_save_alpha_embeddings(
    img: "ee.Image",
    gdf: "gpd.GeoDataFrame",
    out_dir: "Path",
    tile_scale: int,
    simplify_m: float,
    chunk_size: int,
    log=None,
) -> int:
    """
    Save the classic AlphaEarth 64-D mean embeddings (one vector per polygon).

    Robustness upgrades:
      • Accept both naming styles for mean: <band> OR <band>_mean (EE can differ).
      • Atomic .npy writes, periodic manifest flush, optional Excel export.
      • Clear logs for chunks and progress.
    """

    # ---- Validate bands once ------------------------------------------------
    try:
        band_names = ee.List(img.bandNames()).getInfo()
    except Exception as e:
        raise RuntimeError(f"Could not read band names from AlphaEarth image: {e}")

    expected = [f"A{i:02d}" for i in range(64)]
    if not set(expected).issubset(set(band_names)):
        log and log(f"[AEF] Warning: image missing some A-bands. Have {len(band_names)}; expected 64.")
        keys = [k for k in expected if k in band_names]
        if not keys:
            raise RuntimeError("AlphaEarth bands A00..A63 not present on image.")
    else:
        keys = expected

    # ---- Prepare output -----------------------------------------------------
    emb_dir = Path(out_dir) / "embeddings"
    emb_dir.mkdir(parents=True, exist_ok=True)

    if "poly_id" not in gdf.columns:
        raise ValueError("GeoDataFrame must contain a 'poly_id' column.")

    manifest_rows: list[dict] = []
    excel_rows: list[dict] = []   # {poly_id, A00..A63}
    saved = 0

    ids = gdf["poly_id"].astype(int).tolist()
    total = len(ids)
    n_chunks = (total + int(chunk_size) - 1) // int(chunk_size)

    # ---- Chunked processing -------------------------------------------------
    for cidx, start in enumerate(range(0, total, int(chunk_size)), 1):
        sub_ids = ids[start: start + int(chunk_size)]
        sub = gdf[gdf["poly_id"].isin(sub_ids)]

        # Build FeatureCollection (simplify on-the-fly for safety/perf)
        feats = []
        for pid, geom in zip(sub["poly_id"], sub.geometry):
            ee_geom = shapely_to_ee_geometry(geom, simplify_m=simplify_m)
            if ee_geom is None:
                log and log(f"[AEF] invalid geometry pid={int(pid)}; skipped")
                continue
            feats.append(ee.Feature(ee_geom, {"poly_id": int(pid)}))
        if not feats:
            log and log(f"[AEF] chunk {cidx}/{n_chunks}: no valid geometries")
            continue

        fc = ee.FeatureCollection(feats)
        log and log(f"[AEF] reduceRegions mean — chunk {cidx}/{n_chunks} | polys={len(feats)} | tileScale={tile_scale}")

        # Main + fallback
        try:
            features = reduce_regions_mean(img, fc, scale=10, tile_scale=int(tile_scale))
        except Exception as e:
            log and log(f"[AEF] chunk {cidx} failed: {e} — fallback split…")
            features = reduce_regions_chunked(
                img, fc, scale=10, tile_scale=int(tile_scale),
                chunk_size=max(50, len(feats)//2), log=log
            )

        # ---- Assemble and save per polygon ----------------------------------
        for feat in features or []:
            props = feat.get("properties", {}) or {}
            pid = int(props.get("poly_id", -1))
            if pid < 0:
                continue

            # Accept both mean naming styles: <band> OR <band>_mean
            vals = []
            for k in keys:
                v = props.get(k, props.get(f"{k}_mean", np.nan))
                vals.append(v)

            vec = np.array(vals, dtype="float32")
            if np.isnan(vec).all():
                log and log(f"[AEF] all-NaN vector pid={pid} — skipped")
                continue
            if np.isnan(vec).any():
                vec = np.nan_to_num(vec, nan=0.0)

            nrm = float(np.linalg.norm(vec))
            if not np.isfinite(nrm) or nrm < 1e-8:
                log and log(f"[AEF] near-zero/invalid norm pid={pid} — skipped")
                continue
            vec = (vec / nrm).astype("float32")

            try:
                _atomic_npy_save(emb_dir / f"alpha_{pid}.npy", vec)
            except Exception as e:
                log and log(f"[AEF] save failed pid={pid}: {e}")
                continue

            manifest_rows.append({"poly_id": pid, "file": f"alpha_{pid}.npy", "provider": "alpha", "dim": 64})
            excel_rows.append({"poly_id": pid, **{f"A{i:02d}": float(vec[i]) for i in range(len(vec))}})
            saved += 1

        # Periodic manifest flush (every other chunk)
        if manifest_rows and (cidx % 2 == 0):
            try:
                pd.DataFrame(manifest_rows).to_csv(emb_dir / "manifest.csv", index=False)
            except Exception as e:
                log and log(f"[AEF] periodic manifest write failed: {e}")

        log and log(f"[AEF] saved so far (64-D): {saved}")

    # ---- Finalize: manifest + CSV + Excel ----------------------------------
    if manifest_rows:
        try:
            pd.DataFrame(manifest_rows).to_csv(emb_dir / "manifest.csv", index=False)
        except Exception as e:
            log and log(f"[AEF] manifest write (final) failed: {e}")

        # CSV (legacy; keep for backward-compat)
        try:
            if excel_rows:
                merged_csv = emb_dir / "alphaearth_all_embeddings.csv"
                pd.DataFrame(excel_rows).sort_values("poly_id").to_csv(merged_csv, index=False)
                log and log(f"[AEF] merged CSV saved: {merged_csv.name}")
        except Exception as e:
            log and log(f"[AEF] CSV merge failed: {e}")

        # NEW: Excel export (requires XlsxWriter; otherwise CSV fallback kicks in)
        try:
            if excel_rows:
                df_x = pd.DataFrame(excel_rows).sort_values("poly_id")
                _safe_write_excel(df_x, emb_dir / "alphaearth_64_embeddings.xlsx", "alpha64", log=log)
        except Exception as e:
            log and log(f"[AEF] Excel(64) failed: {e}")

    if saved == 0:
        log and log("[AEF] No embeddings were saved. Check EE year/region/bands/connectivity.")
    return saved


def stream_save_alpha_embeddings_four(
    img: "ee.Image",
    gdf: "gpd.GeoDataFrame",
    out_dir: "Path",
    tile_scale: int,
    simplify_m: float,
    chunk_size: int,
    log=None,
) -> int:
    """
    For each polygon, compute FOUR 64-D blocks from A00..A63:
      mean, median (p50), q1 (p25), q3 (p75)  →  concatenate → 256-D.
    Upgrades:
      • Robust to mean being named either <band> or <band>_mean.
      • Uses a *stats-specific* chunked fallback (keeps p25/p50/p75).
      • Atomic saves + optional Excel with f000..f255 columns.
    """

    # ---- Validate bands once -----------------------------------------------
    try:
        band_names = ee.List(img.bandNames()).getInfo()
    except Exception as e:
        raise RuntimeError(f"AlphaEarth band probing failed: {e}")

    expected = [f"A{i:02d}" for i in range(64)]
    keys = [k for k in expected if k in band_names]
    if not keys:
        raise RuntimeError("AlphaEarth A00..A63 bands not present in the image.")

    emb_dir = out_dir / "embeddings"
    emb_dir.mkdir(parents=True, exist_ok=True)

    if "poly_id" not in gdf.columns:
        raise ValueError("GeoDataFrame must contain a 'poly_id' column.")

    ids = gdf["poly_id"].astype(int).tolist()
    total = len(ids)
    n_chunks = (total + int(chunk_size) - 1) // int(chunk_size)

    saved = 0
    excel_batch: list[dict] = []  # rows with poly_id + f000..f255

    def _norm(vec_like) -> "np.ndarray | None":
        """Vectorize -> fill NaN -> safe L2-normalize; None if degenerate."""
        v = np.array(vec_like, dtype="float32")
        if np.isnan(v).all():
            return None
        if np.isnan(v).any():
            v = np.nan_to_num(v, nan=0.0)
        nrm = float(np.linalg.norm(v))
        if not np.isfinite(nrm) or nrm < 1e-8:
            return None
        return (v / nrm).astype("float32")

    for cidx, start in enumerate(range(0, total, int(chunk_size)), 1):
        sub_ids = ids[start: start + int(chunk_size)]
        sub = gdf[gdf["poly_id"].isin(sub_ids)]

        # Build FC
        feats = []
        for pid, geom in zip(sub["poly_id"], sub.geometry):
            ee_geom = shapely_to_ee_geometry(geom, simplify_m=simplify_m)
            if ee_geom is None:
                log and log(f"[AEF4] invalid geometry pid={int(pid)}; skipped")
                continue
            feats.append(ee.Feature(ee_geom, {"poly_id": int(pid)}))
        if not feats:
            log and log(f"[AEF4] chunk {cidx}/{n_chunks}: no valid geometries")
            continue

        fc = ee.FeatureCollection(feats)
        log and log(f"[AEF4] reduceRegions stats — chunk {cidx}/{n_chunks} | polys={len(feats)} | tileScale={tile_scale}")

        # Main stats call + stats-specific fallback
        try:
            features = reduce_regions_stats(img, fc, scale=10, tile_scale=int(tile_scale))
        except Exception as e:
            log and log(f"[AEF4] chunk {cidx} failed: {e} — fallback to stats-chunked…")
            features = reduce_regions_stats_chunked(
                img, fc, scale=10, tile_scale=int(tile_scale),
                chunk_size=max(50, len(feats)//2), log=log
            )

        for feat in features or []:
            props = (feat.get("properties") or {})
            pid = int(props.get("poly_id", -1))
            if pid < 0:
                continue

            # Collect 4 stats in fixed band order; mean may be <band> OR <band>_mean
            v_mean, v_mid, v_q1, v_q3 = [], [], [], []
            for k in keys:
                v_mean.append(props.get(k, props.get(f"{k}_mean", np.nan)))
                v_mid.append(props.get(f"{k}_p50",  np.nan))
                v_q1.append(props.get(f"{k}_p25",  np.nan))
                v_q3.append(props.get(f"{k}_p75",  np.nan))

            v_mean = _norm(v_mean)
            v_mid  = _norm(v_mid)
            v_q1   = _norm(v_q1)
            v_q3   = _norm(v_q3)
            if any(v is None for v in (v_mean, v_mid, v_q1, v_q3)):
                log and log(f"[AEF4] skip pid={pid} due to invalid block(s)")
                continue

            # Concatenate to 256-D and normalize as a whole
            v256 = np.concatenate([v_mean, v_mid, v_q1, v_q3], axis=0).astype("float32")
            n_cat = float(np.linalg.norm(v256))
            if not np.isfinite(n_cat) or n_cat < 1e-8:
                log and log(f"[AEF4] invalid 256-D norm pid={pid} — skipped")
                continue
            v256 = (v256 / n_cat).astype("float32")

            # Save all 5 vectors (4×64 + 256)
            try:
                _atomic_npy_save(emb_dir / f"alpha4_mean_{pid}.npy", v_mean)
                _atomic_npy_save(emb_dir / f"alpha4_mid_{pid}.npy",  v_mid)
                _atomic_npy_save(emb_dir / f"alpha4_q1_{pid}.npy",   v_q1)
                _atomic_npy_save(emb_dir / f"alpha4_q3_{pid}.npy",   v_q3)
                _atomic_npy_save(emb_dir / f"alpha4_{pid}.npy",      v256)
                saved += 1
            except Exception as e:
                log and log(f"[AEF4] save failed pid={pid}: {e}")
                continue

            # Row for Excel
            row = {"poly_id": pid}
            for j in range(256):
                row[f"f{j:03d}"] = float(v256[j])
            excel_batch.append(row)

        log and log(f"[AEF4] saved so far (256-D): {saved}")

        # Optional: periodic flush to limit RAM
        if (cidx % 8 == 0) and excel_batch:
            try:
                interim_csv = emb_dir / "_alpha4_interim.csv"
                df_batch = pd.DataFrame(excel_batch)
                if interim_csv.exists():
                    df_old = pd.read_csv(interim_csv)
                    df_all = pd.concat([df_old, df_batch], ignore_index=True).drop_duplicates(subset=["poly_id"])
                else:
                    df_all = df_batch
                df_all.sort_values("poly_id").to_csv(interim_csv, index=False)
                excel_batch.clear()
                log and log("[AEF4] interim CSV flushed.")
            except Exception as e:
                log and log(f"[AEF4] interim flush failed: {e}")

    # ---- Finalize: Excel (consolidate) -------------------------------------
    try:
        interim_csv = emb_dir / "_alpha4_interim.csv"
        if interim_csv.exists():
            df_all = pd.read_csv(interim_csv)
            interim_csv.unlink(missing_ok=True)
        else:
            df_all = pd.DataFrame(excel_batch) if excel_batch else None

        if (df_all is None) or df_all.empty:
            log and log("[AEF4] Excel consolidation skipped (no rows).")
        else:
            df_all.sort_values("poly_id", inplace=True)
            _safe_write_excel(df_all, emb_dir / "alphaearth_alpha4_embeddings.xlsx", "alpha4", log=log)
    except Exception as e:
        log and log(f"[AEF4] Excel(256) failed: {e}")

    if saved == 0:
        log and log("[AEF4] No embeddings saved. Check EE auth/year/region.")
    else:
        log and log(f"[AEF4] Done. Saved {saved} polygons (4×64 + 256) → {emb_dir}")
    return saved



# =============================================================================
#                             GALILEO (OPTIONAL)
# =============================================================================

from pathlib import Path
from typing import Optional


class GalileoEncoder:
    """
    Local Galileo polygon encoder (TorchScript).
    Fetches a Sentinel-2 chip via EE and returns a unit-norm embedding.
    """

    def __init__(
        self,
        weights_dir: Path,
        device: str = "cpu",
        year: int = 2024,
        chip_px: int = 128,
        log=None,
    ):
        self.log = log
        self.weights_dir = Path(weights_dir)
        self.year = int(year)
        self.chip_px = int(chip_px)
        self.device = "cpu"
        self.model = None
        self.expected_dim = 256
        self.ok_flag = False
        self.err: Optional[str] = None

        # --- torch import
        try:
            import torch
            self.torch = torch
        except Exception as e:
            self.err = f"PyTorch not available: {e}"
            self._log(f"[Galileo] {self.err}")
            return

        # --- device
        self.device = "cuda" if (device == "cuda" and self.torch.cuda.is_available()) else "cpu"

        # --- resolve TorchScript path
        ts_path = self._resolve_torchscript_path(self.weights_dir)
        if ts_path is None:
            self.err = ("No TorchScript file found. Provide 'encoder.pt', "
                        "or a folder that contains it.")
            self._log(f"[Galileo] {self.err}")
            return

        # --- load TorchScript
        try:
            self.model = self.torch.jit.load(str(ts_path), map_location=self.device)
            self.model.eval()
            self._log(f"[Galileo] TorchScript loaded: {ts_path.name}")
        except Exception as e:
            msg = str(e)
            if "PytorchStreamReader failed" in msg or "constants.pkl" in msg:
                self.err = ("Provided file is a PyTorch checkpoint (state_dict), not TorchScript. "
                            "Convert to TorchScript and retry.")
            else:
                self.err = f"TorchScript load failed: {e}"
            self.model = None
            self._log(f"[Galileo] {self.err}")
            return

        # --- probe output dim
        try:
            with self.torch.no_grad():
                dummy = self.torch.zeros(1, 4, self.chip_px, self.chip_px, device=self.device)
                out = self.model(dummy)
                if isinstance(out, (list, tuple)):
                    out = out[0]
                if hasattr(out, "ndim") and out.ndim == 4:
                    out = out.mean(dim=[2, 3])  # GAP
                self.expected_dim = int(out.reshape(1, -1).shape[1])
        except Exception as e:
            self.expected_dim = 256
            self._log(f"[Galileo] Could not probe output dim, fallback=256. Reason: {e}")

        self.ok_flag, self.err = True, None

    # ----------------------------- public API -----------------------------

    def ok(self) -> tuple[bool, str]:
        """Return (is_ok, error_message_if_any)."""
        return self.ok_flag, (self.err or "")

    def encode_polygon(self, geom: Polygon | MultiPolygon) -> np.ndarray:
        """
        Fetch a chip and run it through the encoder.
        Returns a L2-normalized 1D vector (float32) of size `expected_dim`.
        """
        if not self.ok_flag or self.model is None:
            self._log("[Galileo] encoder not ready; returning zeros")
            return np.zeros(self.expected_dim, dtype="float32")

        try:
            chip = self._fetch_s2_chip(geom)
            if chip is None or chip.size == 0:
                return np.zeros(self.expected_dim, dtype="float32")

            x = self.torch.from_numpy(chip).unsqueeze(0).to(self.device)  # (1, 4, H, W)

            with self.torch.no_grad():
                out = self.model(x)
                if isinstance(out, (tuple, list)):
                    out = out[0]
                if hasattr(out, "ndim") and out.ndim == 4:
                    out = out.mean(dim=[2, 3])  # GAP
                out = out.squeeze(0)
                vec = out.detach().float().cpu().numpy()

            n = float(np.linalg.norm(vec))
            if not np.isfinite(n) or n < 1e-8:
                return np.zeros(self.expected_dim, dtype="float32")
            return (vec / n).astype("float32")
        except Exception as e:
            self._log(f"[Galileo] encode failed: {e}")
            return np.zeros(self.expected_dim, dtype="float32")

    # ----------------------------- internals -----------------------------

    @staticmethod
    def _resolve_torchscript_path(p: Path) -> Optional[Path]:
        """Accepts a direct file or a directory containing a known TorchScript filename."""
        p = Path(p)
        if p.is_file():
            return p
        for name in ("encoder.pt", "encoder_scripted.pt", "encoder.ts.pt",
                     "model_scripted.pt", "model_ts.pt"):
            q = p / name
            if q.exists():
                return q
        return None

    def _log(self, msg: str):
        try:
            if callable(self.log):
                self.log(msg)
            else:
                print(msg)
        except Exception:
            pass

    def _center_crop_or_pad(self, arr: np.ndarray, h: int, w: int) -> np.ndarray:
        """Center-crop or zero-pad a (C,H,W) array to exactly (C,h,w)."""
        C, H, W = arr.shape
        y0 = max(0, (H - h) // 2)
        x0 = max(0, (W - w) // 2)
        arr = arr[:, y0:y0 + min(h, H), x0:x0 + min(w, W)]
        out = np.zeros((C, h, w), dtype=arr.dtype)
        oh = min(h, arr.shape[1]); ow = min(w, arr.shape[2])
        oy = (h - oh) // 2; ox = (w - ow) // 2
        out[:, oy:oy + oh, ox:ox + ow] = arr[:, :oh, :ow]
        return out

    def _fetch_s2_chip(self, geom: Polygon | MultiPolygon) -> Optional[np.ndarray]:
        """
        Robustly fetch a Sentinel-2 chip around the geometry.
        Strategy:
          1) Square region around centroid using max(chip footprint, bbox extent).
          2) Clip composite to the rectangle and sample at 10 m (B2,B3,B4,B8).
          3) Expand the rectangle (×1.0, ×1.5, ×2.0) if the chip is too empty.
          4) Align shapes, normalize to [0,1], and center-crop/pad to (chip_px, chip_px).
        """
        if ee is None:
            self._log("[Galileo] EE not available.")
            return None

        ee_geom = shapely_to_ee_geometry(geom, simplify_m=2.0)
        if ee_geom is None:
            self._log("[Galileo] Invalid geometry after simplification.")
            return None

        base_half_deg = _m_to_deg(self.chip_px * 10.0 / 2.0)
        minx, miny, maxx, maxy = geom.bounds
        bbox_half_deg = max(maxx - minx, maxy - miny) / 2.0
        half_deg_base = max(base_half_deg, bbox_half_deg * 1.10)

        cen = geom.centroid
        cx, cy = float(cen.x), float(cen.y)

        base_img = s2_median_cloudfree(self.year, ee_geom).select(["B2", "B3", "B4", "B8"])

        for factor in (1.0, 1.5, 2.0):
            half_deg = float(half_deg_base * factor)
            rect = ee.Geometry.Rectangle([cx - half_deg, cy - half_deg, cx + half_deg, cy + half_deg], None, False)
            img = base_img.clip(rect)

            try:
                with EE_LOCK:
                    blob = img.sampleRectangle(region=rect, scale=10).getInfo()
            except Exception as e:
                self._log(f"[Galileo] sampleRectangle failed (factor={factor:.1f}): {e}")
                continue

            try:
                mats = []
                for b in ("B2", "B3", "B4", "B8"):
                    arr = np.array(blob.get(b, None), dtype="float32")
                    if arr.size == 0:
                        raise ValueError(f"Empty array for band {b}")
                    arr = np.nan_to_num(arr, nan=0.0)
                    arr = np.clip(arr / 10000.0, 0.0, 1.0)
                    mats.append(arr)

                H = int(min(m.shape[0] for m in mats))
                W = int(min(m.shape[1] for m in mats))
                if H < 8 or W < 8:
                    self._log(f"[Galileo] tiny chip {H}x{W} at factor={factor:.1f}; expanding…")
                    continue

                mats = [m[:H, :W] for m in mats]
                chip = np.stack(mats, axis=0)  # (4,H,W)

            except Exception as e:
                self._log(f"[Galileo] Failed to assemble chip (factor={factor:.1f}): {e}")
                continue

            valid_frac = float(np.count_nonzero(chip)) / float(chip.size)
            if valid_frac < 0.05:
                self._log(f"[Galileo] Too few valid pixels ({valid_frac:.1%}) at factor={factor:.1f}; expanding…")
                continue

            if chip.shape[1:] != (self.chip_px, self.chip_px):
                chip = self._center_crop_or_pad(chip, self.chip_px, self.chip_px)

            return chip.astype("float32", copy=False)

        self._log("[Galileo] No valid S2 chip after progressive expansions.")
        return None

    # ------------------------ Forward encoding -------------------------
def stream_save_galileo_embeddings(
    gdf: gpd.GeoDataFrame,
    out_dir: Path,
    weights_dir: Path,
    device: str = "cpu",
    year: int | None = None,
    chip_px: int = 128,
    log=None,
) -> int:
    """
    Encode every polygon using the local Galileo encoder and
    save one .npy vector per polygon into <out_dir>/embeddings>.

    Now skips zero/invalid vectors to avoid poisoning downstream training.
    """
    emb_dir = Path(out_dir) / "embeddings"
    emb_dir.mkdir(parents=True, exist_ok=True)

    enc = GalileoEncoder(
        weights_dir=Path(weights_dir),
        device=device,
        year=int(year) if year is not None else 2024,
        chip_px=int(chip_px),
        log=log,
    )
    ok, err = enc.ok()
    if not ok:
        log and log(f"[Galileo] load failed: {err} — using zero-vectors.")
    rows, saved = [], 0

    if "poly_id" not in gdf.columns:
        raise ValueError("GeoDataFrame must contain a 'poly_id' column.")

    for pid, geom in zip(gdf["poly_id"], gdf.geometry):
        pid = int(pid)
        try:
            v = enc.encode_polygon(geom).astype("float32")

            # Skip truly empty/invalid vectors (do not poison training/prediction)
            if not np.isfinite(v).all() or float(np.linalg.norm(v)) < 1e-6:
                log and log(f"[Galileo] zero/invalid vector for pid={pid} — skipped")
                continue

            np.save(emb_dir / f"gali_{pid}.npy", v)
            rows.append({
                "poly_id": pid,
                "file": f"gali_{pid}.npy",
                "provider": "galileo",
                "dim": int(v.shape[0]),
            })
            saved += 1
            if saved % 200 == 0:
                log and log(f"[Galileo] saved: {saved}")

        except Exception as e:
            log and log(f"[Galileo] pid={pid}: {e}")

    # Update / create manifest
    man = emb_dir / "manifest.csv"
    try:
        if man.exists():
            df_old = pd.read_csv(man)
            pd.concat([df_old, pd.DataFrame(rows)], ignore_index=True).to_csv(man, index=False)
        else:
            pd.DataFrame(rows).to_csv(man, index=False)
    except Exception as e:
        log and log(f"[Galileo] manifest write failed: {e}")

    return saved

# =============================================================================
#                               IO / ZIP INGEST
# =============================================================================
def load_polygons_from_zip(zip_path: Path, min_area_m2: float = 0.0, simplify_m: float = 2.0) -> gpd.GeoDataFrame:
    tmpdir = Path(tempfile.mkdtemp(prefix="zip_in_"))
    with zipfile.ZipFile(zip_path, "r") as zf: zf.extractall(tmpdir)

    cands = []
    for root, _, files in os.walk(tmpdir):
        for f in files:
            if Path(f).suffix.lower() in SUPPORTED_EXT:
                cands.append(Path(root) / f)
    if not cands: raise FileNotFoundError("No supported vector layer found inside the ZIP.")

    dfs = []
    for vp in sorted(cands, key=lambda p: SUPPORTED_EXT[p.suffix.lower()]):
        try:
            gdf = gpd.read_file(vp, engine="pyogrio" if _USE_PYOGRIO else None)
            if gdf.crs is None:
                print(f"[INGEST] '{vp.name}' has no CRS. Assuming EPSG:4326."); gdf = gdf.set_crs(4326, allow_override=True)
            gdf = gdf.to_crs(4326)

            outs = []
            for g in gdf.geometry:
                if isinstance(g, (Polygon, MultiPolygon)): outs.append(g)
                elif isinstance(g, GeometryCollection): outs.extend([x for x in g.geoms if isinstance(x, (Polygon, MultiPolygon))])
            gdf2 = gpd.GeoDataFrame(geometry=outs, crs=4326)

            if min_area_m2 > 0 and len(gdf2):
                a = gdf2.to_crs(3857).area.astype("float64")
                gdf2 = gdf2[a >= float(min_area_m2)]

            if simplify_m and simplify_m > 0 and len(gdf2):
                gdf2 = gdf2.to_crs(3857)
                gdf2["geometry"] = gdf2.geometry.simplify(float(simplify_m), preserve_topology=True)
                gdf2 = gdf2[gdf2.geometry.notnull() & ~gdf2.geometry.is_empty].to_crs(4326)

            if len(gdf2): dfs.append(gdf2)
            print(f"[INGEST] {vp.name} -> {len(gdf2)} polygon(s)")
        except Exception as e:
            print(f"[INGEST] skip {vp.name}: {e}")

    if not dfs: raise RuntimeError("No polygons ingested from the ZIP.")
    g = pd.concat(dfs, ignore_index=True).set_crs(4326, allow_override=True)
    g["poly_id"] = np.arange(len(g))
    return g

# =============================================================================
#                 Polygon-level NDVI labels (for MLP training)
# =============================================================================
def s2_median_cloudfree(year: int, region) -> ee.Image:
    """
    Median S2 SR for [year-01-01, (year+1)-01-01) with QA60 cloud/cirrus mask.
    End date is *exclusive* to cover the full calendar year without spillover.
    """
    bands = ["B2", "B3", "B4", "B8", "QA60"]
    start = f"{year}-01-01"
    end   = f"{year+1}-01-01"  # end-exclusive

    col = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
           .filterBounds(region)
           .filterDate(start, end)
           .select(bands))

    def mask_clouds(img):
        qa = img.select("QA60")
        cloud  = qa.bitwiseAnd(1 << 10).neq(0)
        cirrus = qa.bitwiseAnd(1 << 11).neq(0)
        return img.updateMask(cloud.Or(cirrus).Not())  # keep clear pixels

    return (col.map(mask_clouds).median()
               .select(["B2", "B3", "B4", "B8"])
               .rename(["B2", "B3", "B4", "B8"]))



def compute_polygon_labels_ndvi(gdf: gpd.GeoDataFrame, year: int, thr: float,
                                tile_scale: int, out_dir: Path, log=None) -> pd.DataFrame:
    """
    Advanced, timeout-resilient NDVI labeling:
      - Chunks polygons to keep each Earth Engine request small
      - Retries getInfo() with exponential backoff
      - Simplifies geometries client-side before sending to EE
      - Writes partial results to CSV after every chunk (so you always get output)
      - Keeps the same signature and final CSV path: out_dir/labels/ndvi_{year}.csv

    Expects helper functions present in your codebase:
      - shapely_to_ee_geometry(geom, simplify_m=...)
      - s2_median_cloudfree(year:int, region:ee.Geometry) -> ee.Image
    """
    # Local imports (kept inside to avoid global pollution)
    import os, math, time, contextlib
    from pathlib import Path
    import pandas as pd
    import ee

    # ------------------------- I/O setup -------------------------
    out_dir = Path(out_dir)
    labels_dir = out_dir / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)
    csv_path = labels_dir / f"ndvi_{year}.csv"

    # Start from a clean file so appends have a single header
    try:
        if csv_path.exists():
            csv_path.unlink()
    except Exception as e:
        # Non-fatal: if removal fails we'll still append below
        pass

    # Small helper to log safely if a logger is provided
    def _log(msg: str):
        if callable(log):
            try:
                log(msg)
            except Exception:
                pass

    # EE getInfo retry with exponential backoff (caps sleep to 30s)
    def _ee_retry(fn, max_retries: int = 6, base_delay: float = 2.0, growth: float = 1.8):
        last_err = None
        for attempt in range(1, max_retries + 1):
            try:
                return fn()
            except Exception as e:
                last_err = e
                sleep_s = min(base_delay * (growth ** (attempt - 1)), 30.0)
                _log(f"[LABEL] getInfo retry {attempt}/{max_retries} after: {e.__class__.__name__}: {e}")
                time.sleep(sleep_s)
        raise last_err

    # Simple list chunker
    def _chunks(lst, size: int):
        for i in range(0, len(lst), size):
            yield i, lst[i:i + size]

    # ------------------------- Build EE features -------------------------
    feats = []
    # Geometry simplification tolerance on the client (meters)
    simplify_m = 10.0

    for pid, geom in zip(gdf["poly_id"], gdf.geometry):
        # Skip null/empty shapes to avoid EE server-side failures
        if geom is None or geom.is_empty:
            _log(f"[LABEL] skipped empty geometry pid={int(pid)}")
            continue

        ee_geom = shapely_to_ee_geometry(geom, simplify_m=simplify_m)
        if ee_geom is None:
            _log(f"[LABEL] skipped invalid geometry pid={int(pid)}")
            continue

        feats.append(ee.Feature(ee_geom, {"poly_id": int(pid)}))

    if not feats:
        # Explicit error helps the UI show a clear message
        raise RuntimeError("No valid geometries found for NDVI labeling.")

    # Limit the S2 mosaic to the union region of your polygons
    region_fc = ee.FeatureCollection(feats)

    # Sentinel-2 median composite (cloud-free) then NDVI
    ndvi = (
        s2_median_cloudfree(year, region_fc.geometry())
        .normalizedDifference(["B8", "B4"])
        .rename("ndvi")
    )

    # ------------------------- EE runtime tuning -------------------------
    # Give EE more time; ignore if not supported in the current backend
    try:
        ee.data.setDeadline(1_200_000)  # 20 minutes in ms
    except Exception:
        pass

    # Use a global EE lock if your project defines one; otherwise a no-op context
    EE_LOCK = globals().get("EE_LOCK")
    lock_ctx = EE_LOCK if EE_LOCK is not None else contextlib.nullcontext()

    # ------------------------- Chunked reduceRegions -------------------------
    CHUNK = 200          # Tune down to 100 / 50 if you still see timeouts
    SCALE_M = 20         # 20 m is usually enough for mean NDVI and far cheaper than 10 m
    TILE_SCALE = int(tile_scale)

    recs = []
    wrote_header = False
    total = len(feats)
    total_chunks = (total + CHUNK - 1) // CHUNK

    for base_idx, sub in _chunks(feats, CHUNK):
        # Prepare a small FeatureCollection for this chunk
        fc = ee.FeatureCollection(sub)

        # Keep the reducer minimal (single-band mean)
        out_fc = ndvi.reduceRegions(
            collection=fc,
            reducer=ee.Reducer.mean(),
            scale=SCALE_M,
            tileScale=TILE_SCALE
        )

        # Lock + retry around the (expensive) client-side materialization
        with lock_ctx:
            features = _ee_retry(lambda: (out_fc.getInfo() or {}).get("features", []),
                                 max_retries=6, base_delay=3.0, growth=1.8)

        # Parse server response defensively
        chunk_rows = []
        for r in features or []:
            p = (r.get("properties") or {})
            pid = int(p.get("poly_id", -1))
            mean_val = p.get("mean", None)  # reducer mean of single-band 'ndvi'
            if mean_val is None:
                continue
            try:
                nd = float(mean_val)
            except Exception:
                continue
            if math.isnan(nd):
                continue
            chunk_rows.append({
                "poly_id": pid,
                "ndvi_mean": nd,
                "y": 1 if nd >= float(thr) else 0
            })

        # Append to in-memory list (for return value)
        recs.extend(chunk_rows)

        # Append to disk immediately (so you have output even if the next chunk fails)
        if chunk_rows:
            df_chunk = pd.DataFrame(chunk_rows).sort_values("poly_id")
            df_chunk.to_csv(csv_path, mode="a", header=not wrote_header, index=False)
            wrote_header = True

        _log(f"[LABEL] chunk {base_idx // CHUNK + 1}/{total_chunks}: "
             f"in={len(sub)}  out={len(chunk_rows)}  cum_out={len(recs)}")

    # ------------------------- Finalize -------------------------
    df = pd.DataFrame(recs).sort_values("poly_id").reset_index(drop=True)

    # If nothing was written yet (e.g., 0 rows matched), still create an empty file
    if not csv_path.exists():
        df.to_csv(csv_path, index=False)

    _log(f"[LABEL] polygon NDVI labels saved: {len(df)} rows -> {csv_path}")
    return df



# =============================================================================
#                 Labeled polygons + interpretable/tabular features
# =============================================================================
from collections import defaultdict
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt

# A stable color palette for multi-class mapping
CLASS_COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
]
def load_labeled_polygons_from_zip(
    zip_path: Path,
    label_col: Optional[str] = None,
    code_map: Optional[Dict[Any, str]] = None,
    min_area_m2: float = 0.0,
    simplify_m: float = 2.0,
    drop_unlabeled: bool = True,
    log: Optional[Callable[[str], None]] = None,
) -> gpd.GeoDataFrame:
    """
    Robustly read a labeled polygon dataset from a ZIP archive (SHP/GPKG/GeoJSON…),
    preserve *all* attributes, and produce:
      - poly_id           : sequential ID after merging all layers
      - crop_label_raw    : the raw value from the label column (best-effort)
      - crop_label        : normalized/mapped string label
      - geometry          : valid 2D Polygon/MultiPolygon in EPSG:4326

    Behavior:
    ---------
    - If `label_col` is provided but not found in a layer, the function tries to
      GUESS a label column (crop/class/product/label/type/...); if none is found,
      that layer's records are kept with NaN labels (and can be dropped via
      `drop_unlabeled=True`).
    - Small polygons are removed by `min_area_m2` (metric) *after* reprojecting
      to EPSG:3857; geometries are simplified with `simplify_m` (meters) and
      reprojected back to EPSG:4326.
    - All non-geometry attributes are preserved; an extra `src_layer` column is
      added so you can trace where each feature came from.

    Parameters
    ----------
    zip_path : Path
        Path to a ZIP containing a vector layer (SHP family, GPKG, GeoJSON…).
    label_col : Optional[str]
        Preferred label column name. If not present, a best-effort guess is used.
    code_map : Optional[dict]
        Optional mapping to convert raw codes -> canonical string labels.
    min_area_m2 : float
        Minimum polygon area in square meters (filter).
    simplify_m : float
        Douglas–Peucker tolerance in meters for simplification (0 disables).
    drop_unlabeled : bool
        If True, rows with missing label are dropped (recommended for training).
    log : Optional[callable]
        Optional logger function; called with short progress messages.

    Returns
    -------
    GeoDataFrame
        Columns: ['poly_id','crop_label_raw','crop_label','geometry', ...other attributes...]
    """

    def _log(msg: str):  # tiny safe logger
        if log:
            try: log(msg)
            except Exception: pass

    def _guess_label_col_from_names(cols: list[str]) -> Optional[str]:
        """Heuristic scoring to pick a likely label column."""
        if not cols: return None
        tokens = ["crop", "class", "label", "product", "type", "cultivar", "species", "plant"]
        ranked = []
        for c in cols:
            lc = c.lower()
            score = 0
            for t in tokens:
                if lc == t or lc == f"{t}_code":
                    score += 4
                elif t in lc:
                    score += 2
            if lc.endswith("_code"):
                score += 1
            ranked.append((score, c))
        ranked.sort(reverse=True)
        return ranked[0][1] if ranked and ranked[0][0] > 0 else None

    # --- Extract the ZIP into a temp folder
    tmpdir = Path(tempfile.mkdtemp(prefix="zip_labeled_"))
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(tmpdir)

    # --- Find all candidate vector layers
    cands: list[Path] = []
    for root, _, files in os.walk(tmpdir):
        for f in files:
            suf = Path(f).suffix.lower()
            if suf in (".shp", ".gpkg", ".geojson", ".json", ".kml"):
                cands.append(Path(root) / f)
    if not cands:
        raise FileNotFoundError("No vector layer found inside the ZIP.")

    # Prefer modern containers first
    order = {".gpkg": 0, ".geojson": 1, ".json": 1, ".shp": 2, ".kml": 3}
    cands.sort(key=lambda p: order.get(p.suffix.lower(), 9))

    # --- Read, clean and normalize each layer
    gdfs: list[gpd.GeoDataFrame] = []
    for vp in cands:
        try:
            _log(f"[LABELED INGEST] Reading {vp.name}")
            gdf = gpd.read_file(vp, engine="pyogrio" if _USE_PYOGRIO else None)

            # Ensure CRS; work in WGS84 for IO, reproject to 3857 for metric ops
            if gdf.crs is None:
                _log(f"[LABELED INGEST] '{vp.name}' has no CRS; assuming EPSG:4326")
                gdf = gdf.set_crs(4326, allow_override=True)
            gdf = gdf.to_crs(4326)

            # Keep only (Multi)Polygon; drop empties/nulls early
            rows, geoms = [], []
            for _, row in gdf.iterrows():
                geom = row.geometry
                if isinstance(geom, (Polygon, MultiPolygon)) and (geom is not None) and (not geom.is_empty):
                    # Force 2D + make valid (uses utilities defined elsewhere)
                    g_clean = _validify(_force_2d(geom))
                    if (g_clean is None) or g_clean.is_empty:
                        continue
                    rows.append(row)
                    geoms.append(g_clean)
            if not rows:
                _log(f"[LABELED INGEST] {vp.name}: no polygonal geometries; skipped")
                continue

            g2 = gpd.GeoDataFrame(rows, geometry=geoms, crs=4326)
            g2["src_layer"] = vp.name  # provenance

            # Metric area filter + simplify (do in EPSG:3857 then back to 4326)
            if len(g2) and (min_area_m2 > 0 or simplify_m > 0):
                g_metric = g2.to_crs(3857)
                if min_area_m2 > 0:
                    a = g_metric.area.astype("float64")
                    g_metric = g_metric[a >= float(min_area_m2)]
                if simplify_m and simplify_m > 0 and len(g_metric):
                    g_metric["geometry"] = g_metric.geometry.simplify(float(simplify_m), preserve_topology=True)
                g2 = g_metric[g_metric.geometry.notnull() & ~g_metric.geometry.is_empty].to_crs(4326)
                if not len(g2):
                    _log(f"[LABELED INGEST] {vp.name}: all polygons filtered out by area/simplify")
                    continue

            # --- Resolve label column robustly
            avail_cols = [c for c in g2.columns if c != "geometry"]
            chosen = None
            if label_col and (label_col in avail_cols):
                chosen = label_col
            else:
                chosen = _guess_label_col_from_names(avail_cols)
                if not chosen:
                    _log(f"[LABELED INGEST] {vp.name}: no label column found; rows will be unlabeled")

            g2 = g2.copy()
            if chosen:
                # raw label value (keep original dtype), plus a string-normalized version
                g2["crop_label_raw"] = g2[chosen]
                if code_map:
                    # map codes to human-readable labels; fallback to string
                    g2["crop_label"] = g2["crop_label_raw"].map(lambda v: code_map.get(v, str(v)))
                else:
                    g2["crop_label"] = g2["crop_label_raw"].astype(str).str.strip()
            else:
                g2["crop_label_raw"] = np.nan
                g2["crop_label"] = np.nan

            gdfs.append(g2)

        except Exception as e:
            _log(f"[LABELED INGEST] skip {vp.name}: {e}")

    if not gdfs:
        raise RuntimeError("No labeled polygons found in the ZIP.")

    # --- Merge all layers, assign global poly_id, and (optionally) drop unlabeled
    out = pd.concat(gdfs, ignore_index=True).set_crs(4326, allow_override=True)
    out["poly_id"] = np.arange(len(out))

    if drop_unlabeled:
        missing = int(out["crop_label"].isna().sum())
        if missing > 0:
            _log(f"[LABELED INGEST] rows without label dropped: {missing}")
        out = out.dropna(subset=["crop_label"]).reset_index(drop=True)
        out["poly_id"] = np.arange(len(out))  # reindex after drop to keep it dense

    # --- Return with all attributes preserved (geometry last for readability)
    base_cols = ["poly_id", "crop_label_raw", "crop_label", "geometry"]
    other_cols = [c for c in out.columns if c not in base_cols]
    return out[base_cols + other_cols]

def _shape_features_for_gdf(gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    """
    Compute interpretable shape features per polygon (no EE calls).
    Features:
      - area_m2, perimeter_m, compactness (4πA/P^2), rectangularity (A/rect_area),
        elongation (major/minor), n_holes
    """
    g = gdf.to_crs(3857).copy()
    records = []
    for pid, geom in zip(gdf["poly_id"].astype(int), g.geometry):
        try:
            if isinstance(geom, MultiPolygon):
                geom2 = max(geom.geoms, key=lambda p: p.area)  # take largest for shape metrics
            else:
                geom2 = geom
            area = float(geom2.area)
            per  = float(geom2.length)
            compact = float(4.0 * math.pi * area / (per ** 2)) if per > 0 else 0.0
            # min. rotated rectangle to approximate length/width
            mrr = geom2.minimum_rotated_rectangle
            mrr_coords = list(mrr.exterior.coords)
            edges = [math.dist(mrr_coords[i], mrr_coords[(i+1) % 4]) for i in range(4)]
            L, W = (max(edges), min(edges)) if edges else (0.0, 0.0)
            elong = float(L / W) if W > 0 else 0.0
            rect_area = float(mrr.area) if hasattr(mrr, "area") else 0.0
            rectangularity = float(area / rect_area) if rect_area > 0 else 0.0
            holes = int(len(geom2.interiors)) if isinstance(geom2, Polygon) else 0
            records.append({"poly_id": int(pid),
                            "area_m2": area, "perimeter_m": per,
                            "compactness": compact, "elongation": elong,
                            "rectangularity": rectangularity, "n_holes": holes})
        except Exception:
            records.append({"poly_id": int(pid),
                            "area_m2": np.nan, "perimeter_m": np.nan,
                            "compactness": np.nan, "elongation": np.nan,
                            "rectangularity": np.nan, "n_holes": np.nan})
    return pd.DataFrame(records)

def _to_float_or_nan(v):
    if v is None:
        return np.nan
    try:
        x = float(v)
        return x if np.isfinite(x) else np.nan
    except Exception:
        return np.nan

def compute_s2_tabular_features(gdf: gpd.GeoDataFrame, year: int,
                                tile_scale: int = 4, chunk_size: int = 300,
                                log=None) -> pd.DataFrame:
    """
    Interpretable spectral features from Sentinel-2 annual median:
      - Mean of B2,B3,B4,B8
      - NDVI mean + stdDev

    Done in chunks to stay well under EE's 5k cap.
    """
    if ee is None or not _EE_READY:
        raise RuntimeError("EE not initialized; cannot compute S2 features.")

    base_img = s2_median_cloudfree(year, ee.Geometry.Rectangle([-180,-89.9,180,89.9]))  # global; we'll clip per geom
    ndvi = base_img.normalizedDifference(["B8", "B4"]).rename("ndvi")

    feats_rows = []

    ids = gdf["poly_id"].astype(int).tolist()
    total = len(ids)
    for i in range(0, total, int(chunk_size)):
        sub = gdf.iloc[i:i+chunk_size]
        ee_feats = []
        for pid, geom in zip(sub["poly_id"], sub.geometry):
            ee_geom = shapely_to_ee_geometry(geom, simplify_m=2.0)
            if ee_geom is None:
                log and log(f"[S2] invalid geometry pid={int(pid)}; skipped")
                continue
            ee_feats.append(ee.Feature(ee_geom, {"poly_id": int(pid)}))
        if not ee_feats:
            continue

        fc = ee.FeatureCollection(ee_feats)

        # Means of B2..B8 (we only need 4 bands)
        reducer_b = ee.Reducer.mean()
        img_b = base_img.select(["B2","B3","B4","B8"])
        out_b = img_b.reduceRegions(collection=fc, reducer=reducer_b,
                                    scale=10, tileScale=int(tile_scale))

        # NDVI mean + stdDev
        reducer_ndvi = ee.Reducer.mean().combine(ee.Reducer.stdDev(), sharedInputs=True)
        out_ndvi = ndvi.reduceRegions(collection=fc, reducer=reducer_ndvi,
                                      scale=10, tileScale=int(tile_scale))

        with EE_LOCK:
            rows_b = out_b.getInfo().get("features", [])
            rows_n = out_ndvi.getInfo().get("features", [])

        # Merge by poly_id
        tmp_b = {int(r["properties"]["poly_id"]): r["properties"] for r in rows_b}
        tmp_n = {int(r["properties"]["poly_id"]): r["properties"] for r in rows_n}
        for pid in tmp_b.keys():
            rb = tmp_b.get(pid, {})
            rn = tmp_n.get(pid, {})
            rec = {
                "poly_id": pid,
                "B2_mean": _to_float_or_nan(rb.get("B2")),
                "B3_mean": _to_float_or_nan(rb.get("B3")),
                "B4_mean": _to_float_or_nan(rb.get("B4")),
                "B8_mean": _to_float_or_nan(rb.get("B8")),
                "ndvi_mean": _to_float_or_nan(rn.get("mean")),
                "ndvi_std": _to_float_or_nan(rn.get("stdDev")),
            }

            feats_rows.append(rec)

        log and log(f"[S2] chunk {i//chunk_size+1}/{(total+chunk_size-1)//chunk_size} -> {len(ee_feats)} polys")

    return pd.DataFrame(feats_rows)

def _load_embeddings_block(gdf: gpd.GeoDataFrame, out_dir: Path,
                           provider_mode: str, log=None) -> Tuple[Optional[np.ndarray], List[int], List[str]]:
    """
    Load embeddings for given provider_mode in {"alpha","gali","concat"}.
    Returns (X_emb, pids, feature_names) or (None, [], []) if unavailable.
    """
    pids = []
    vecs = []
    if provider_mode not in {"alpha","gali","concat"}:
        return None, [], []

    for pid in gdf["poly_id"].astype(int):
        if provider_mode == "alpha":
            va = _load_embedding_for_pid(out_dir, int(pid), "alpha")
            if va is None: continue
            vecs.append(va); pids.append(int(pid))
        elif provider_mode == "gali":
            vg = _load_embedding_for_pid(out_dir, int(pid), "gali")
            if vg is None: continue
            vecs.append(vg); pids.append(int(pid))
        else:
            va = _load_embedding_for_pid(out_dir, int(pid), "alpha")
            vg = _load_embedding_for_pid(out_dir, int(pid), "gali")
            if (va is None) or (vg is None): continue
            vecs.append(np.concatenate([va, vg], axis=0)); pids.append(int(pid))

    if not vecs:
        return None, [], []
    X = np.stack(vecs, axis=0)
    if provider_mode == "alpha":
        names = [f"A{i:02d}" for i in range(X.shape[1])]
    elif provider_mode == "gali":
        names = [f"G{i:02d}" for i in range(X.shape[1])]
    else:
        d = X.shape[1] // 2
        names = [f"A{i:02d}" for i in range(d)] + [f"G{i:02d}" for i in range(X.shape[1]-d)]
    return X, pids, names

from typing import Optional, Tuple, List, Callable
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import LabelEncoder

def build_crop_feature_matrix(
    gdf_labeled: "gpd.GeoDataFrame",
    out_dir: Path,
    year: int,
    tile_scale: int,
    feature_mode: str,                      # one of: alpha | alpha4 | alpha4x64 | gali | concat
    ndvi_filter: Optional[float] = None,    # if set, keep only polygons with NDVI>=thr
    log: Optional[Callable[[str], None]] = None,
) -> Tuple[np.ndarray, np.ndarray, List[int], List[str], List[str], LabelEncoder]:


    def _l(msg: str):
        if log:
            log(msg)

    # ---------------- 0) Canonicalize mode & sanity ----------------
    aliases = {
        "alphaearth": "alpha",
        "alphaearth (mean 64-d)": "alpha",
        "alphaearth (4× stats)": "alpha4",
        "alphaearth (4x stats)": "alpha4",
        "alphaearth 4x": "alpha4",
        "alpha (64+4×)": "alpha4x64",
        "alpha (64+4x)": "alpha4x64",
        "alpha64+alpha4": "alpha4x64",
        "galileo": "gali",
        "concat (alpha+galileo)": "concat",
        "a+g": "concat",
    }
    mode_raw = (feature_mode or "concat").strip().lower()
    mode = aliases.get(mode_raw, mode_raw)
    if mode not in {"alpha", "alpha4", "alpha4x64", "gali", "concat"}:
        raise RuntimeError(f"Unsupported feature_mode: {feature_mode!r}")

    g = gdf_labeled.copy()

    # Ensure we have a stable primary key
    if "poly_id" not in g.columns:
        g["poly_id"] = np.arange(len(g), dtype=int)

    # Normalize the label column name to 'crop_label'
    label_col = None
    for c in ("crop_label", "label", "class", "label_num", "label_name"):
        if c in g.columns:
            label_col = c
            break
    if label_col is None:
        raise RuntimeError("No label column found (expected one of: crop_label, label, class, label_num, label_name).")
    if label_col != "crop_label":
        g = g.rename(columns={label_col: "crop_label"})

    # ---------------- 1) Optional NDVI gate (filter only) ----------------
    # We may need S2 tabular just to compute NDVI; we do NOT add these to X.
    if isinstance(ndvi_filter, (int, float)) and float(ndvi_filter) > 0.0:
        _l(f"[Crop-MLP] Applying NDVI gate (thr={ndvi_filter})…")
        s2 = compute_s2_tabular_features(g, year, tile_scale=tile_scale, log=log)
        g = g.merge(s2[["poly_id", "ndvi_mean"]], on="poly_id", how="left")
        g = g[(~g["ndvi_mean"].isna()) & (g["ndvi_mean"].astype(float) >= float(ndvi_filter))].copy()
        g = g.drop(columns=["ndvi_mean"])
        if g.empty:
            raise RuntimeError("All polygons were removed by NDVI gate; nothing to train.")

    # ---------------- 2) Load embeddings per polygon ----------------
    pids_all = g["poly_id"].astype(int).tolist()

    def _load_alpha64_vec(pid: int) -> Optional[np.ndarray]:
        # Uses your existing helper for 64-D AlphaEarth
        return _load_embedding_for_pid(out_dir, pid, kind="alpha")

    def _load_gali_vec(pid: int) -> Optional[np.ndarray]:
        # Uses your existing helper for Galileo
        return _load_embedding_for_pid(out_dir, pid, kind="gali")

    def _load_alpha4_vec(pid: int) -> Optional[np.ndarray]:
        # Uses your existing helper for 256-D AlphaEarth (4× stats)
        return _load_alpha4_for_pid(out_dir, pid)

    rows: List[np.ndarray] = []
    kept_pids: List[int] = []

    if mode == "alpha":
        for pid in pids_all:
            v = _load_alpha64_vec(pid)
            if v is not None:
                rows.append(v.astype("float32", copy=False))
                kept_pids.append(pid)
        feat_names = [f"a64_{i:02d}" for i in range(64)]

    elif mode == "alpha4":
        for pid in pids_all:
            v = _load_alpha4_vec(pid)
            if v is not None:
                rows.append(v.astype("float32", copy=False))
                kept_pids.append(pid)
        feat_names = [f"a4_{i:03d}" for i in range(256)]

    elif mode == "alpha4x64":
        # Require *both* vectors; skip polygons missing either one.
        for pid in pids_all:
            v64 = _load_alpha64_vec(pid)
            v256 = _load_alpha4_vec(pid)
            if v64 is not None and v256 is not None:
                rows.append(np.concatenate(
                    [v64.astype("float32", copy=False), v256.astype("float32", copy=False)], axis=0
                ))
                kept_pids.append(pid)
        feat_names = [f"a64_{i:02d}" for i in range(64)] + [f"a4_{i:03d}" for i in range(256)]

    elif mode == "gali":
        for pid in pids_all:
            v = _load_gali_vec(pid)
            if v is not None:
                rows.append(v.astype("float32", copy=False))
                kept_pids.append(pid)
        # We don't know Galileo dim a priori; infer after stacking.
        feat_names = []  # will fill after stack

    else:  # mode == "concat"  →  [alpha64 | gali]
        # Intersect on poly_id: only polygons that have both vectors are kept.
        alpha_vecs, alpha_pids = [], []
        for pid in pids_all:
            v = _load_alpha64_vec(pid)
            if v is not None:
                alpha_vecs.append(v.astype("float32", copy=False))
                alpha_pids.append(pid)

        gali_vecs, gali_pids = [], []
        for pid in pids_all:
            v = _load_gali_vec(pid)
            if v is not None:
                gali_vecs.append(v.astype("float32", copy=False))
                gali_pids.append(pid)

        common = [pid for pid in alpha_pids if pid in set(gali_pids)]
        if not common:
            rows, kept_pids, feat_names = [], [], []
        else:
            # Build aligned matrices then hstack
            ia = [alpha_pids.index(pid) for pid in common]
            ig = [gali_pids.index(pid) for pid in common]
            Xa = np.stack([alpha_vecs[i] for i in ia], axis=0)
            Xg = np.stack([gali_vecs[i] for i in ig], axis=0)
            rows = [np.concatenate([Xa[i], Xg[i]], axis=0) for i in range(len(common))]
            kept_pids = common
            feat_names = [f"a64_{i:02d}" for i in range(Xa.shape[1])] + [f"gali_{i:03d}" for i in range(Xg.shape[1])]

    if not rows:
        raise RuntimeError(f"No embeddings available for provider '{mode}'. Compute embeddings first.")

    X = np.stack(rows, axis=0).astype("float32", copy=False)
    if mode == "gali" and not feat_names:
        # Infer Galileo dimension after stack
        feat_names = [f"gali_{i:03d}" for i in range(X.shape[1])]

    # ---------------- 3) Align labels with kept_pids ----------------
    g = g[g["poly_id"].isin(kept_pids)].copy()
    # preserve the same order as X (kept_pids)
    g = g.set_index("poly_id").loc[kept_pids].reset_index()

    y_str = g["crop_label"].astype(str).tolist()
    le = LabelEncoder()
    y_int = le.fit_transform(y_str)
    class_names = list(le.classes_)
    pids = kept_pids

    _l(f"[Crop-MLP] samples={len(X)} | features={X.shape[1]} | classes={class_names}")
    return X, y_int, pids, feat_names, class_names, le

# =============================================================================
#                         MLP (Polygon Embeddings) Train / Predict
# =============================================================================

# ----------------------------------------------------------------
# Low-level loader for a single pid for AlphaEarth (64D) or Galileo
# ----------------------------------------------------------------
def _load_embedding_for_pid(
    base: Path,
    pid: int,
    provider: str,
    *,
    mmap: bool = True,
    renorm_non_alpha: bool = True,
) -> Optional[np.ndarray]:
    """
    Load one embedding vector by poly_id for a given provider.

    - AlphaEarth 64D files:  embeddings/alpha_{pid}.npy  (L2-normalized already)
    - Galileo     ~256D     : embeddings/gali_{pid}.npy   (we normalize defensively)

    Returns a 1D float32 vector or None if missing/invalid.
    """
    f = (base / "embeddings" /
         (f"alpha_{pid}.npy" if provider == "alpha" else f"gali_{pid}.npy"))
    if not f.exists():
        return None

    try:
        v = np.load(f, mmap_mode="r" if mmap else None)
        v = np.asarray(v, dtype="float32").reshape(-1)  # force 1-D
        if provider != "alpha" and renorm_non_alpha:
            # Galileo may not be normalized; do a safe L2 renorm
            n = float(np.linalg.norm(v))
            if not np.isfinite(n) or n < 1e-8:
                return None
            v = (v / (n + 1e-8)).astype("float32", copy=False)
        # Replace any accidental NaN/Inf (shouldn't happen, but be defensive)
        if not np.isfinite(v).all():
            v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0).astype("float32", copy=False)
        return v
    except Exception:
        return None


# ----------------------------------------------------------------
# AlphaEarth-4x loader (256D): from concatenated file or 4 parts
# ----------------------------------------------------------------
def _load_alpha4_for_pid(
    base: Path,
    pid: int,
    *,
    # File/layout knobs
    parts: Iterable[str] = ("mean", "mid", "q1", "q3"),
    part_prefix: str = "alpha4_{part}_{pid}.npy",
    concat_name: str = "alpha4_{pid}.npy",
    # Shape/validation knobs
    part_dim: int = 64,
    expected_dim: int = 256,
    strict_parts: bool = True,            # True: any missing/invalid slice ⇒ None
    allow_rebuild_from_parts: bool = True,
    # IO/Perf knobs
    use_mmap: bool = True,                # memory-map .npy reads
    # Misc
    cache: Optional[Dict[int, np.ndarray]] = None,   # optional cache {pid: vec}
    log: Optional[Callable[[str], None]] = None,
) -> Optional[np.ndarray]:
    """
    Load a 256-D AlphaEarth 'alpha4' embedding for a polygon id.

    Priority:
      1) embeddings/alpha4_{pid}.npy                         (single 256-D file)
      2) build from 4 slices: mean/mid/q1/q3 (each 64-D)     (if allowed)

    Returns a 1D float32 vector (L2-normalized) or None.
    """
    def _log(msg: str) -> None:
        try:
            if callable(log):
                log(msg)
        except Exception:
            pass

    # Cache short-circuit
    if cache is not None and pid in cache:
        v = cache[pid]
        if isinstance(v, np.ndarray) and v.ndim == 1 and v.dtype == np.float32 and v.size == expected_dim:
            return v.copy()

    emb_dir = Path(base) / "embeddings"
    eps = 1e-8

    # 1) try the pre-concatenated file
    p_cat = emb_dir / concat_name.format(pid=pid)
    try:
        if p_cat.exists():
            v = np.load(p_cat, mmap_mode="r" if use_mmap else None).astype("float32", copy=False).reshape(-1)
            if v.size == expected_dim and np.isfinite(v).all():
                n = float(np.linalg.norm(v))
                if n < eps:
                    return None
                v = (v / (n + eps)).astype("float32", copy=False)
                if cache is not None:
                    cache[pid] = v.copy()
                return v
            else:
                _log(f"[alpha4] {p_cat.name}: invalid shape/values → trying slices…")
    except Exception as e:
        _log(f"[alpha4] {p_cat.name}: load error ({e}); will try slices…")

    # 2) rebuild from four 64-D parts
    if not allow_rebuild_from_parts:
        return None

    def _fix_shape_1d(x: np.ndarray, target: int) -> Optional[np.ndarray]:
        x = np.asarray(x, dtype="float32").reshape(-1)
        if x.size == target:
            return x
        if strict_parts:
            return None
        # best-effort truncate/pad if strict=False
        out = np.zeros((target,), dtype="float32")
        n = min(target, x.size)
        out[:n] = x[:n]
        return out

    slices: List[np.ndarray] = []
    for part in parts:
        p = emb_dir / part_prefix.format(part=part, pid=pid)
        try:
            if not p.exists():
                _log(f"[alpha4] missing slice: {p.name}")
                return None
            x = np.load(p, mmap_mode="r" if use_mmap else None).astype("float32", copy=False)
            if not np.isfinite(x).all():
                x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).astype("float32", copy=False)

            x = _fix_shape_1d(x, part_dim)
            if x is None:
                _log(f"[alpha4] slice shape mismatch ({p.name}); strict_parts=True → abort")
                return None

            n = float(np.linalg.norm(x))
            if n < eps:
                if strict_parts:
                    _log(f"[alpha4] degenerate slice ({p.name}); strict_parts=True → abort")
                    return None
                x = np.zeros_like(x, dtype="float32")
            else:
                x = (x / (n + eps)).astype("float32", copy=False)

            slices.append(x)
        except Exception as e:
            _log(f"[alpha4] {p.name}: load/normalize error ({e})")
            return None

    if not slices:
        return None

    try:
        v256 = np.concatenate(slices, axis=0).astype("float32", copy=False)
        if v256.size != expected_dim:
            if strict_parts:
                _log(f"[alpha4] concatenated size {v256.size} != {expected_dim} → abort")
                return None
            # best-effort fix
            tmp = np.zeros((expected_dim,), dtype="float32")
            n = min(expected_dim, v256.size)
            tmp[:n] = v256[:n]
            v256 = tmp

        n_cat = float(np.linalg.norm(v256))
        if n_cat < eps:
            return None
        v256 = (v256 / (n_cat + eps)).astype("float32", copy=False)

        if cache is not None:
            cache[pid] = v256.copy()
        return v256
    except Exception as e:
        _log(f"[alpha4] concatenate/normalize error ({e})")
        return None


# ----------------------------------------------------------------
# Build X matrix for Polygon-MLP from saved embeddings on disk
# ----------------------------------------------------------------
def build_feature_matrix_from_embeddings(
    gdf: gpd.GeoDataFrame,
    out_dir: Path,
    provider_mode: str,
    log: Optional[Callable[[str], None]] = None
) -> Tuple[np.ndarray, List[int]]:

    base = Path(out_dir)
    emb_dir = base / "embeddings"

    # --- normalize provider_mode with a friendly alias map
    mode_raw = (provider_mode or "").lower().strip()
    aliases = {
        "alphaearth": "alpha",
        "alphaearth (4× stats)": "alpha4",
        "alphaearth (4x stats)": "alpha4",
        "alpha4": "alpha4",
        "galileo": "gali",
        "gal": "gali",
        "concat (alpha+galileo)": "concat",
        "a+g": "concat",
    }
    mode = aliases.get(mode_raw, mode_raw)
    if mode not in {"alpha", "gali", "concat", "alpha4"}:
        raise ValueError(
            f"Unsupported provider_mode='{provider_mode}'. "
            "Use one of: 'alpha', 'gali', 'concat', 'alpha4'."
        )

    def _log(msg: str) -> None:
        if callable(log):
            try: log(msg)
            except Exception: pass

    # loader per mode
    def _load_vec(pid: int) -> Optional[np.ndarray]:
        if mode == "alpha":
            return _load_embedding_for_pid(base, pid, "alpha")        # 64-D
        elif mode == "gali":
            return _load_embedding_for_pid(base, pid, "gali")         # ~256-D
        elif mode == "concat":
            va = _load_embedding_for_pid(base, pid, "alpha")
            vg = _load_embedding_for_pid(base, pid, "gali")
            if va is None or vg is None:
                return None
            return np.concatenate([va, vg], axis=0).astype("float32", copy=False)
        else:  # alpha4
            return _load_alpha4_for_pid(base, pid)

    # Pass 1: iterate and collect good rows (dim-consistent)
    X_rows: List[np.ndarray] = []
    pids: List[int] = []

    poly_ids = [int(pid) for pid in gdf["poly_id"].astype(int)]
    total = len(poly_ids)

    missing: List[int] = []
    badshape: List[int] = []
    naninf: List[int] = []
    probed_dim: Optional[int] = None

    for pid in poly_ids:
        try:
            v = _load_vec(pid)
            if v is None:
                missing.append(pid); continue
            if v.ndim != 1:
                badshape.append(pid); continue
            if probed_dim is None:
                probed_dim = int(v.size)
            if int(v.size) != probed_dim:
                badshape.append(pid); continue
            if not np.isfinite(v).all():
                v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0).astype("float32", copy=False)
                if float(np.linalg.norm(v)) < 1e-8:
                    naninf.append(pid); continue

            X_rows.append(v.astype("float32", copy=False))
            pids.append(pid)
        except Exception:
            missing.append(pid)

    matched = len(X_rows)
    _log(f"[Poly-MLP] embeddings matched: {matched}/{total} for provider={provider_mode}")
    if matched == 0:
        raise RuntimeError(
            "No embeddings found for provider='{pm}'. Checked folder: {d}\n"
            "Hints:\n"
            "  • Ensure the Output folder points to where you saved embeddings.\n"
            "  • Do not re-run 'Run' after saving embeddings (poly_id may change).\n"
            "  • For 'alpha4', run 'Compute AlphaEarth (4× stats)' first.\n"
            "  • For 'concat', BOTH AlphaEarth and Galileo vectors must exist."
            .format(pm=provider_mode, d=emb_dir)
        )

    skipped = len(missing) + len(badshape) + len(naninf)
    if skipped:
        parts = []
        if missing:  parts.append(f"missing={len(missing)}")
        if badshape: parts.append(f"bad-shape={len(badshape)}")
        if naninf:   parts.append(f"nan/inf={len(naninf)}")
        _log(f"[Poly-MLP] skipped: {skipped} ({', '.join(parts)})")

    # Final stack into (N×D)
    try:
        X = np.stack(X_rows, axis=0).astype("float32", copy=False)
    except Exception as e:
        raise RuntimeError(f"Failed to stack embeddings: {e}")

    return X, pids


# ----------------------------------------------------------------
# TrainSummary: dataclass fix (solves “takes no arguments”)
# ----------------------------------------------------------------
@dataclass
class TrainSummary:
    """Compact record of what happened during training."""
    backend: str                 # "tf" or "sklearn"
    provider_mode: str
    best_val_auc: Optional[float]  # may be None for sklearn
    n_samples: int
    n_pos: int
    n_neg: int
    timestamp: float


def _now() -> float:
    """UTC epoch seconds."""
    return time.time()


def _write_meta(sidecar: Path, meta: dict | TrainSummary) -> None:
    """
    Write a sidecar JSON file atomically where possible.
    Accepts either a dict or a TrainSummary dataclass.
    """
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    payload = asdict(meta) if isinstance(meta, TrainSummary) else meta
    tmp = sidecar.with_suffix(sidecar.suffix + ".tmp")
    try:
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(sidecar)  # atomic on most filesystems
    except Exception:
        # Fallback (non-atomic) if replace fails on some Windows setups
        sidecar.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _safe_log(log: Optional[Callable[[str], None]], msg: str) -> None:
    if log:
        try:
            log(msg)
        except Exception:
            # Never let logging crash the pipeline.
            pass

def _class_weights(y: np.ndarray) -> Dict[int, float]:
    """Balanced weights that sum to ~1 across classes regardless of skew."""
    pos = float((y == 1).sum())
    neg = float((y == 0).sum())
    if pos == 0 or neg == 0:
        # Degenerate case: single-class labels -> avoid divide by zero
        return {0: 1.0, 1: 1.0}
    total = pos + neg
    return {
        0: 0.5 * total / neg,
        1: 0.5 * total / pos,
    }

def _seed_everything(seed: int = 42) -> None:
    """Make behavior as deterministic as feasible across backends."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf  # noqa
        tf.random.set_seed(seed)
    except Exception:
        pass


# ------------------------------- Training --------------------------------- #

def train_polygon_mlp(
    gdf: gpd.GeoDataFrame,
    out_dir: Path,
    year: int,
    ndvi_thr: float,
    provider_mode: str,
    use_gpu_tf: bool,
    tile_scale: int,
    model_path: Path,
    scaler_path: Path,
    log=None,
):

    _seed_everything(42)

    # === 1) Derive labels via NDVI (delegated to your existing helper) ===
    labels_df = compute_polygon_labels_ndvi(
        gdf, year, ndvi_thr, tile_scale, out_dir, log=log
    )

    # === 2) Build features from saved embeddings (delegated helper) ===
    X, pids = build_feature_matrix_from_embeddings(gdf, out_dir, provider_mode)
    if X.size == 0:
        raise RuntimeError("No embeddings found for training. Compute embeddings first.")

    # Map polygon ids -> labels
    lab_map = dict(
        zip(labels_df["poly_id"].astype(int).tolist(),
            labels_df["y"].astype(int).tolist())
    )
    idx: List[int] = [i for i, pid in enumerate(pids) if int(pid) in lab_map]
    if not idx:
        raise RuntimeError("No overlap between embeddings and NDVI labels.")

    X = X[idx]
    y = np.array([lab_map[int(pids[i])] for i in idx], dtype="int8")

    # Basic dataset stats for logging / sanity checks
    n_samples = int(X.shape[0])
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    _safe_log(log, f"[Poly-MLP] samples={n_samples}, pos={n_pos}, neg={n_neg}")

    # Persist dirs
    model_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path = Path(str(model_path) + ".meta.json")

    # Fit scaler once, upstream of backend branches
    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)

    # Compute class weights
    class_weight = _class_weights(y)
    _safe_log(log, f"[Poly-MLP] class weights: {class_weight}")

    # ================== 3) Try TF/Keras; if it fails, fall back ================== #
    try:
        # lazy import (may raise ImportError on TF/protobuf problems)
        _, keras, layers, regularizers, callbacks = _lazy_tf(use_gpu_tf)

        # --- Model: 2 hidden layers with mild L2 & dropout ---
        reg = regularizers.l2(1e-4)
        inp = keras.Input(shape=(Xs.shape[1],))
        h = layers.Dense(256, activation="relu", kernel_regularizer=reg)(inp)
        h = layers.BatchNormalization()(h)
        h = layers.Dropout(0.25)(h)
        h = layers.Dense(128, activation="relu", kernel_regularizer=reg)(h)
        out = layers.Dense(1, activation="sigmoid")(h)
        model = keras.Model(inp, out)

        # Use class weights; track robust metrics
        model.compile(
            optimizer=keras.optimizers.Adam(1e-3),
            loss="binary_crossentropy",
            metrics=[
                keras.metrics.AUC(name="AUC"),
                keras.metrics.Precision(name="Precision"),
                keras.metrics.Recall(name="Recall"),
            ],
        )

        # Early stopping + LR decay on plateau
        es = callbacks.EarlyStopping(
            monitor="val_loss", patience=10, restore_best_weights=True
        )
        rlrop = callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6
        )

        # Slight shuffle to avoid order effects on validation split
        Xs_shuf, y_shuf = sk_shuffle(Xs, y, random_state=42)

        hist = model.fit(
            Xs_shuf,
            y_shuf,
            validation_split=0.2,
            epochs=200,
            batch_size=256,
            verbose=0,
            class_weight=class_weight,
            callbacks=[es, rlrop],
        )

        # Save artifacts
        model.save(model_path)
        joblib.dump(scaler, scaler_path)

        # Best validation AUC (if present)
        best_auc = float(np.max(hist.history.get("val_AUC", hist.history.get("val_auc", [0.0]))))
        _safe_log(
            log,
            f"[Poly-MLP/{provider_mode}] Keras trained. Best val AUC≈{best_auc:.3f}. Saved.",
        )

        summary = TrainSummary(
            backend="tf",
            provider_mode=provider_mode,
            best_val_auc=best_auc,
            n_samples=n_samples,
            n_pos=n_pos,
            n_neg=n_neg,
            timestamp=_now(),
        )
        _write_meta(meta_path, json.loads(json.dumps(summary.__dict__)))
        return

    except ImportError as tf_err:
        # Informative message; proceed to sklearn fallback
        _safe_log(
            log,
            f"[Poly-MLP] TensorFlow unavailable: {tf_err}. Falling back to sklearn…",
        )
    except Exception as tf_other:
        # Unexpected TF error; do not abort training, still fall back
        _safe_log(
            log,
            f"[Poly-MLP] TensorFlow training failed ({type(tf_other).__name__}: {tf_other}). "
            "Falling back to sklearn…",
        )

    # ======================= 4) Fallback: sklearn + calibration ======================= #
    # Architecture: similar hidden sizes; early stopping on internal val split.
    base = MLPClassifier(
        hidden_layer_sizes=(256, 128),
        activation="relu",
        alpha=1e-4,               # L2 regularization
        batch_size=256,
        learning_rate_init=1e-3,
        max_iter=400,
        early_stopping=True,
        n_iter_no_change=20,
        validation_fraction=0.2,
        random_state=42,
    )

    # Calibrate probabilities using CV for better reliability (Platt scaling)
    clf = CalibratedClassifierCV(base_estimator=base, cv=3, method="sigmoid")
    clf.fit(Xs, y, sample_weight=np.vectorize(class_weight.get)(y))

    # Persist artifacts (note: model_path becomes a joblib .pkl effectively)
    joblib.dump(scaler, scaler_path)
    joblib.dump(clf, model_path)

    # Estimate out-of-fold AUC (mean across calibrator folds if available)
    try:
        # Not strictly guaranteed by sklearn API, but commonly available:
        # we can compute an in-sample AUC as a rough quality signal.
        p_in = clf.predict_proba(Xs)[:, 1]
        auc_in = float(roc_auc_score(y, p_in)) if len(np.unique(y)) > 1 else None
    except Exception:
        auc_in = None

    _safe_log(
        log,
        f"[Poly-MLP/{provider_mode}] sklearn MLP trained + calibrated. "
        + (f"In-sample AUC≈{auc_in:.3f}. " if auc_in is not None else "")
        + "Saved."
    )

    summary = TrainSummary(
        backend="sklearn",
        provider_mode=provider_mode,
        best_val_auc=auc_in,
        n_samples=n_samples,
        n_pos=n_pos,
        n_neg=n_neg,
        timestamp=_now(),
    )
    _write_meta(meta_path, json.loads(json.dumps(summary.__dict__)))


# -------------------------------- Inference -------------------------------- #

def predict_polygons_with_polygon_mlp(
    gdf: gpd.GeoDataFrame,
    out_dir: Path,
    provider_mode: str,
    model_path: Path,
    scaler_path: Path,
    log=None,
) -> gpd.GeoDataFrame:
    """
    Predict per-polygon probability and label using the trained polygon-MLP.

    Works with both backends:
      * backend="tf": Keras model saved as .keras/.h5
      * backend="sklearn": Calibrated MLPClassifier saved via joblib

    Output columns:
      - final_label: "vegetation" | "non_vegetation" | "unknown"
      - confidence : calibrated confidence in [0, 1] (|2p-1|)
      - p_veg_mean : predicted probability of vegetation (np.nan if missing)
    """
    # Detect backend from sidecar JSON or file suffix heuristic
    meta_path = Path(str(model_path) + ".meta.json")
    backend: Optional[str] = None

    if meta_path.exists():
        try:
            backend = json.loads(meta_path.read_text()).get("backend")
        except Exception:
            backend = None

    if backend is None:
        # Fallback heuristic: TF models usually end with .keras/.h5
        backend = "tf" if model_path.suffix.lower() in {".keras", ".h5"} else "sklearn"

    # Load scaler
    if not Path(scaler_path).exists():
        raise FileNotFoundError(
            f"Scaler not found at {scaler_path}. Did you run training successfully?"
        )
    scaler: StandardScaler = joblib.load(scaler_path)

    # Build features (same provider_mode used during training)
    X, pids = build_feature_matrix_from_embeddings(gdf, out_dir, provider_mode)
    if X.size == 0:
        raise RuntimeError("No embeddings available for prediction.")
    Xs = scaler.transform(X)

    # Predict probabilities using the detected backend
    if backend == "tf":
        try:
            _, keras, _, _, _ = _lazy_tf(False)  # may raise ImportError
            model = keras.models.load_model(model_path, compile=False)
            probs = model.predict(Xs, verbose=0).reshape(-1)
        except ImportError as tf_err:
            raise RuntimeError(
                f"TensorFlow backend required to load Keras model: {tf_err}"
            )
    else:
        clf: CalibratedClassifierCV = joblib.load(model_path)
        probs = clf.predict_proba(Xs)[:, 1]

    # Map polygon id -> probability for fast lookup
    pred_map: Dict[int, float] = {int(pid): float(p) for pid, p in zip(pids, probs)}

    # Construct output GeoDataFrame (preserve row order)
    labels: List[str] = []
    confs: List[float] = []
    p_means: List[float] = []

    for pid in gdf["poly_id"].astype(int):
        p = pred_map.get(int(pid))
        if p is None or not (p == p):  # None or NaN
            labels.append("unknown")
            confs.append(np.nan)
            p_means.append(np.nan)
            continue

        lbl = "vegetation" if p >= 0.5 else "non_vegetation"
        conf = abs(2.0 * p - 1.0)  # calibrated confidence in [0,1]
        labels.append(lbl)
        confs.append(conf)
        p_means.append(float(p))

    out = gdf.copy()
    out["final_label"] = labels
    out["confidence"] = confs
    out["p_veg_mean"] = p_means

    _safe_log(
        log,
        f"[Poly-MLP/{provider_mode}] Predicted {sum(l != 'unknown' for l in labels)} / "
        f"{len(labels)} polygons (backend={backend})."
    )
    return out



def train_evaluate_crop_mlp(
    X: np.ndarray,
    y_int: np.ndarray,
    class_names: List[str],
    feature_names: List[str],
    model_path: Path,
    scaler_path: Path,
    meta_path: Path,
    use_gpu_tf: bool = False,
    test_size: float = 0.30,
    random_state: int = 42,
    log=None,
):
    """
    Train a multi-class MLP with 70/30 stratified split.
    - Primary backend: TF/Keras (softmax).
    - Fallback: sklearn MLPClassifier (multi-class).
    - Saves: model, scaler, meta.json, metrics.json, confusion_matrix.png
    """
    # Split (stratified)
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(splitter.split(X, y_int))
    Xtr, Xte = X[train_idx], X[test_idx]
    ytr, yte = y_int[train_idx], y_int[test_idx]

    # Scale
    scaler = StandardScaler().fit(Xtr)
    Xtr_s = scaler.transform(Xtr)
    Xte_s = scaler.transform(Xte)

    model_path.parent.mkdir(parents=True, exist_ok=True)

    # Class weights (balanced)
    classes, counts = np.unique(ytr, return_counts=True)
    total = float(len(ytr))
    class_weight = {int(c): float(total / (len(classes) * cnt)) for c, cnt in zip(classes, counts)}

    backend_used = "sklearn"
    yte_pred_proba = None

    # Try TF first
    try:
        tf, keras, layers, regularizers, callbacks = _lazy_tf(use_gpu_tf)
        reg = regularizers.l2(1e-4)
        inp = keras.Input(shape=(Xtr_s.shape[1],))
        h = layers.Dense(512, activation="relu", kernel_regularizer=reg)(inp)
        h = layers.BatchNormalization()(h); h = layers.Dropout(0.35)(h)
        h = layers.Dense(256, activation="relu", kernel_regularizer=reg)(h)
        h = layers.BatchNormalization()(h); h = layers.Dropout(0.25)(h)
        out = layers.Dense(len(class_names), activation="softmax")(h)
        model = keras.Model(inp, out)
        model.compile(optimizer=keras.optimizers.Adam(1e-3),
                      loss="sparse_categorical_crossentropy",
                      metrics=["accuracy"])
        es = callbacks.EarlyStopping(monitor="val_loss", patience=12, restore_best_weights=True)
        rl = callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=6)
        model.fit(Xtr_s, ytr, validation_split=0.2, epochs=250, batch_size=512,
                  verbose=0, callbacks=[es, rl], class_weight=class_weight)
        model.save(model_path)
        backend_used = "tf"
        yte_pred_proba = model.predict(Xte_s, verbose=0)
    except Exception as e:
        log and log(f"[Crop-MLP] TF unavailable -> sklearn fallback: {e}")
        base = MLPClassifier(hidden_layer_sizes=(512, 256),
                             activation="relu",
                             alpha=1e-4,
                             batch_size=512,
                             learning_rate_init=1e-3,
                             max_iter=400,
                             early_stopping=True,
                             n_iter_no_change=25,
                             validation_fraction=0.2,
                             random_state=42)
        base.fit(Xtr_s, ytr)
        joblib.dump(base, model_path)
        yte_pred_proba = base.predict_proba(Xte_s)

    # Save scaler
    joblib.dump(scaler, scaler_path)

    # Metrics
    yte_pred = np.argmax(yte_pred_proba, axis=1)
    acc = accuracy_score(yte, yte_pred)
    f1m = f1_score(yte, yte_pred, average="macro")
    rep = classification_report(yte, yte_pred, target_names=class_names, output_dict=True)
    cm = confusion_matrix(yte, yte_pred, labels=list(range(len(class_names))))

    # Save metrics JSON
    metrics = {
        "backend": backend_used,
        "accuracy": acc,
        "f1_macro": f1m,
        "per_class": rep,
        "n_classes": len(class_names),
        "classes": class_names,
    }
    metrics_path = Path(str(model_path) + ".metrics.json")
    metrics_path.write_text(json.dumps(metrics, indent=2))

    # Save confusion matrix figure
    try:
        fig = plt.figure(figsize=(6,6), dpi=140)
        plt.imshow(cm, interpolation="nearest")
        plt.title("Confusion matrix")
        plt.xticks(range(len(class_names)), class_names, rotation=45, ha="right")
        plt.yticks(range(len(class_names)), class_names)
        for (i,j), v in np.ndenumerate(cm):
            plt.text(j, i, str(v), ha='center', va='center')
        plt.tight_layout()
        png = Path(str(model_path) + ".cm.png")
        fig.savefig(png)
        plt.close(fig)
    except Exception as e:
        log and log(f"[Crop-MLP] Could not save CM: {e}")

    # Save meta
    meta = {
        "backend": backend_used,
        "feature_names": feature_names,
        "classes": class_names,
        "timestamp": time.time(),
    }
    meta_path.write_text(json.dumps(meta, indent=2))

    return {
        "acc": acc, "f1_macro": f1m, "report": rep, "cm": cm,
        "yte_true": yte.tolist(), "yte_pred": yte_pred.tolist()
    }

def predict_crops_with_mlp(
    gdf: gpd.GeoDataFrame,
    out_dir: Path,
    model_path: Path,
    scaler_path: Path,
    feature_mode: str,
    year: int,
    tile_scale: int,
    ndvi_filter: Optional[float],
    log=None,
    veg_mask_df: Optional[pd.DataFrame] = None,  # <-- NEW
) -> gpd.GeoDataFrame:
    """
    Predict crop class labels with a gate:

      If 'veg_mask_df' is provided:
        - Use it as the vegetation mask (from page-2 Polygon-MLP).
        - Classify ONLY polygons with veg_flag=True.
        - NON_VEG -> 'NON_VEG'; unknown (missing flag) -> 'UNKNOWN'.
        - No NDVI is computed.

      Else (legacy):
        - Compute ndvi_mean for ALL polygons (EE required).
        - Gate by ndvi_filter as before.

    Returns GeoDataFrame with: pred_crop, pred_conf, (and ndvi_mean if legacy path).
    """
    # --- read class meta (names & backend) ---
    meta_path = Path(str(model_path) + ".meta.json")
    if not meta_path.exists():
        raise FileNotFoundError("Meta file missing for crop model.")
    meta = json.loads(meta_path.read_text())
    class_names = meta["classes"]
    backend = meta.get("backend", "sklearn")

    # === Branch 1: Use page-2 vegetation mask (preferred) =====================
    if veg_mask_df is not None and len(veg_mask_df):
        g_all = gdf.merge(veg_mask_df[["poly_id", "veg_flag"]], on="poly_id", how="left")
        veg_mask = g_all["veg_flag"].fillna(False).astype(bool)
        # Build features for ALL (we'll subselect vegetated indices)
        X_all, _, pids_all, feat_names, _, _ = build_crop_feature_matrix(
            gdf, Path(out_dir), year, tile_scale, feature_mode, ndvi_filter=None, log=log
        )
        pid_to_idx = {pid: i for i, pid in enumerate(pids_all)}

        # Load scaler + predictor
        scaler: StandardScaler = joblib.load(scaler_path)
        Xs_all = scaler.transform(X_all)
        if backend == "tf":
            _, keras, _, _, _ = _lazy_tf(False)
            model = keras.models.load_model(model_path, compile=False)
            predict_proba = lambda batch: model.predict(batch, verbose=0)
        else:
            clf = joblib.load(model_path)
            predict_proba = lambda batch: clf.predict_proba(batch)

        # Predict only for veg polygons
        pred_lbl, pred_conf = [], []
        veg_pids = [int(pid) for pid, isveg in zip(g_all["poly_id"].astype(int), veg_mask) if isveg and int(pid) in pid_to_idx]
        veg_idx = [pid_to_idx[p] for p in veg_pids]
        proba_veg = predict_proba(Xs_all[veg_idx]) if len(veg_idx) else None
        if proba_veg is not None and proba_veg.ndim == 1:
            proba_veg = np.stack([1.0 - proba_veg, proba_veg], axis=1)
        it_veg = iter(proba_veg) if proba_veg is not None else iter(())

        for pid, isveg in zip(g_all["poly_id"].astype(int), veg_mask):
            if not bool(isveg):
                pred_lbl.append("NON_VEG")
                pred_conf.append(1.0)
                continue
            idx = pid_to_idx.get(int(pid), None)
            if idx is None or proba_veg is None:
                pred_lbl.append("UNKNOWN")
                pred_conf.append(np.nan)
                continue
            p = next(it_veg)
            j = int(np.argmax(p))
            pred_lbl.append(class_names[j])
            pred_conf.append(float(np.max(p)))

        out = g_all.copy()
        out["pred_crop"] = pred_lbl
        out["pred_conf"] = pred_conf
        # No ndvi_mean here (we didn't compute NDVI)
        return out

    # === Branch 2: Legacy NDVI gate (kept for backward compatibility) ========
    s2_df = compute_s2_tabular_features(gdf, year, tile_scale=tile_scale, log=log)
    g_all = gdf.merge(s2_df[["poly_id", "ndvi_mean"]], on="poly_id", how="left")
    thr = float(ndvi_filter) if ndvi_filter is not None else -1.0
    veg_mask = g_all["ndvi_mean"].astype(float) >= thr if ndvi_filter is not None else pd.Series(True, index=g_all.index)

    X_all, _, pids_all, feat_names, _, _ = build_crop_feature_matrix(
        gdf, Path(out_dir), year, tile_scale, feature_mode, ndvi_filter=None, log=log
    )
    pid_to_idx = {pid: i for i, pid in enumerate(pids_all)}

    scaler: StandardScaler = joblib.load(scaler_path)
    Xs_all = scaler.transform(X_all)
    if backend == "tf":
        _, keras, _, _, _ = _lazy_tf(False)
        model = keras.models.load_model(model_path, compile=False)
        predict_proba = lambda batch: model.predict(batch, verbose=0)
    else:
        clf = joblib.load(model_path)
        predict_proba = lambda batch: clf.predict_proba(batch)

    pred_lbl, pred_conf = [], []
    veg_pids = [int(pid) for pid, isveg in zip(g_all["poly_id"].astype(int), veg_mask) if bool(isveg) and int(pid) in pid_to_idx]
    veg_idx = [pid_to_idx[p] for p in veg_pids]
    proba_veg = predict_proba(Xs_all[veg_idx]) if len(veg_idx) else None
    if proba_veg is not None and proba_veg.ndim == 1:
        proba_veg = np.stack([1.0 - proba_veg, proba_veg], axis=1)
    it_veg = iter(proba_veg) if proba_veg is not None else iter(())

    for pid, isveg in zip(g_all["poly_id"].astype(int), veg_mask):
        if not bool(isveg):
            pred_lbl.append("NON_VEG")
            pred_conf.append(1.0)
            continue
        idx = pid_to_idx.get(int(pid), None)
        if idx is None or proba_veg is None:
            pred_lbl.append(None)
            pred_conf.append(np.nan)
            continue
        p = next(it_veg)
        j = int(np.argmax(p))
        pred_lbl.append(class_names[j])
        pred_conf.append(float(np.max(p)))

    out = g_all.copy()
    out["pred_crop"] = pred_lbl
    out["pred_conf"] = pred_conf
    return out


# =============================================================================
#                 Column probing (list all fields inside labeled ZIPs)
# =============================================================================
def _guess_label_col_from_names(cols: list[str]) -> Optional[str]:
    """
    Heuristics to guess a label column name among available fields.
    Prefers names containing these tokens (case-insensitive).
    """
    if not cols: return None
    tokens = ["crop", "class", "label", "product", "type", "cultivar", "species", "plant"]
    ranked = []
    for c in cols:
        lc = c.lower()
        score = 0
        for t in tokens:
            if lc == t or lc == f"{t}_code":
                score += 4
            elif t in lc:
                score += 2
        if lc.endswith("_code"):
            score += 1
        ranked.append((score, c))
    ranked.sort(reverse=True)
    return ranked[0][1] if ranked and ranked[0][0] > 0 else None

def probe_columns_in_zip(zip_path: Path) -> list[str]:
    """
    Open a ZIP containing vector layers (SHP/GPKG/GeoJSON) and return the
    list of attribute column names (excluding 'geometry').
    Simplicity over micro-optimization: read with GeoPandas, then extract columns.
    """
    tmpdir = Path(tempfile.mkdtemp(prefix="zip_probe_"))
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(tmpdir)

    cands = []
    for root, _, files in os.walk(tmpdir):
        for f in files:
            suf = Path(f).suffix.lower()
            if suf in (".shp", ".gpkg", ".geojson", ".json", ".kml"):
                cands.append(Path(root) / f)
    if not cands:
        return []
    # Prefer GPKG/GeoJSON; fallback to SHP
    cands = sorted(cands, key=lambda p: {".gpkg":0, ".geojson":1, ".json":1, ".shp":2, ".kml":3}[p.suffix.lower()])
    cols: set[str] = set()
    for vp in cands:
        try:
            gdf = gpd.read_file(vp, engine="pyogrio" if _USE_PYOGRIO else None)
            for c in gdf.columns:
                if c != "geometry":
                    cols.add(str(c))
        except Exception:
            pass
    return sorted(cols)

# =============================================================================
#                                   GUI
# =============================================================================
class App(tk.Tk):
    """
    Main Tkinter application.
    - Centralized async job runner with busy-state handling
    - Clean logging pipeline (queue-based)
    - Robust UI wiring
    """

    def __init__(self):
        """
        Main Tkinter application bootstrap.
        This version adds a robust Page-2 ↔ Page-3 sync, including the new
        'alpha4x64' feature mode (64-D mean + 256-D 4× stats → 320-D).
        """

        super().__init__()
        self.title("Geo Embedding — AlphaEarth (EE) + Galileo (local)")
        self.geometry("1540x1000")

        # ------------------------- Runtime state -------------------------
        # Thread-safe queue for background logging
        self._log_q = queue.Queue()
        # Handles to map polygons (to clear later)
        self._drawn_polys = []
        # Loaded polygons (Page 1 output). None until 'Run' is pressed.
        self._gdf = None  # type: Optional[gpd.GeoDataFrame]
        # Reentrancy guard for long-running jobs
        self._running_jobs = set()
        # Small status line (e.g., vegetation/non-vegetation counters)
        self._counts_var = tk.StringVar(value="—")

        # ------------------------- Output location -----------------------
        # All outputs live under out_dir
        self.out_dir = tk.StringVar(value=DEFAULT_OUTDIR)
        out_root = Path(self.out_dir.get())
        # Create a minimal directory structure (idempotent)
        for sub in ("", "embeddings", "models", "maps", "logs"):
            (out_root / sub).mkdir(parents=True, exist_ok=True)

        # ------------------------- Page-1 filters ------------------------
        # Geometry pre-filters for ingestion
        self.min_area_m2 = tk.IntVar(value=0)  # drop polygons below this area (m²)
        self.simplify_m = tk.IntVar(value=2)  # Douglas–Peucker tolerance (meters)
        self.max_rings = tk.IntVar(value=15000)  # safety cap for map drawing
        self.basemap = tk.StringVar(value="Esri Imagery")

        # ------------------------- EE / AlphaEarth -----------------------
        # Server-side params used by Earth Engine jobs
        self.ee_project = tk.StringVar(value="roi-iran-project")
        self.ee_year = tk.IntVar(value=2024)  # AlphaEarth V1 ~ 2017–2024
        self.ee_tilescale = tk.IntVar(value=4)  # EE reduceRegions tileScale
        self.ee_chunk = tk.IntVar(value=250)  # max features per server batch

        # ------------------------- Providers -----------------------------
        # Feature providers available in Page 2
        self.use_alpha = tk.BooleanVar(value=True)  # AlphaEarth is required for alpha/alpha4
        self.use_gali = tk.BooleanVar(value=False)  # Galileo off by default (until weights exist)

        # ------------------------- Service Account (optional) -----------
        self.sa_email = tk.StringVar(value="")  # EE service account email (optional)
        self.sa_json = tk.StringVar(value="")  # path to JSON keyfile (optional)

        # ------------------------- Galileo (optional) --------------------
        # Local TorchScript weights (nano/tiny/small...) — not bundled
        default_gali = Path.cwd() / "galileo" / "data" / "models" / "nano"
        self.gali_weights = tk.StringVar(value=str(default_gali) if default_gali.exists() else "")
        # "cpu" or "cuda" (validated on 'Test Galileo')
        self.gali_device = tk.StringVar(value="cpu")

        # ------------------------- Polygon-MLP source (Page 2) ----------
        # Feature source label shown in Page 2 (Combobox).
        # Supported labels:
        #   "AlphaEarth (mean 64-D)", "Galileo",
        #   "Concat (Alpha+Galileo)", "AlphaEarth (4× stats)"
        # Keep a NON-empty default to avoid downstream ValueErrors.
        self.use_alpha4 = tk.BooleanVar(value=False)  # 4× stats toggle (256-D)
        self.poly_feat_src = tk.StringVar(value="AlphaEarth (mean 64-D)")

        # ------------------------- Polygon-MLP (Page 2) ------------------
        # TensorFlow GPU toggle is best-effort; availability is checked later
        self.tf_use_gpu = tk.BooleanVar(value=False)
        # Optional NDVI gate for building supervision (<=0 disables gate)
        self.ndvi_thresh = tk.DoubleVar(value=0.30)

        # ------------------------- Model dirs ---------------------------
        models_dir = out_root / "models"
        models_dir.mkdir(parents=True, exist_ok=True)

        # ---- Helper: normalize Page-2 label → canonical mode ------------
        # Returns one of: {"alpha", "alpha4", "gali", "concat"}
        def _map_page2_label_to_mode(lbl: str) -> str:
            t = (lbl or "").strip().lower()
            if "concat" in t or "alpha+gal" in t:
                return "concat"
            if "galileo" in t:
                return "gali"
            if "4×" in t or "4x" in t or "stats" in t or "quartile" in t:
                return "alpha4"
            if "alpha" in t:
                return "alpha"
            return "alpha"

        # ---- Helper: quick file existence probe under Outputs/embeddings
        def _have_embeddings(pattern: str) -> bool:
            try:
                emb = Path(self.out_dir.get()) / "embeddings"
                return any(emb.glob(pattern))
            except Exception:
                return False

        # ---- Helper: set Page-2 Polygon-MLP model/scaler paths ----------
        # (kept here so the UI always shows a consistent path per mode)
        def _set_poly_model_paths(mode: str) -> None:
            if mode == "alpha4":
                self.poly_model_path = getattr(self, "poly_model_path",
                                               tk.StringVar(value=str(models_dir / "mlp_poly_alpha4.keras")))
                self.poly_scaler_path = getattr(self, "poly_scaler_path",
                                                tk.StringVar(value=str(models_dir / "scaler_poly_alpha4.pkl")))
                try:
                    self.poly_model_path.set(str(models_dir / "mlp_poly_alpha4.keras"))
                    self.poly_scaler_path.set(str(models_dir / "scaler_poly_alpha4.pkl"))
                except Exception:
                    pass
                # Ensure provider toggles reflect the mode
                try:
                    self.use_alpha.set(True)
                    self.use_alpha4.set(True)
                except Exception:
                    pass
            elif mode == "gali":
                self.poly_model_path = getattr(self, "poly_model_path",
                                               tk.StringVar(value=str(models_dir / "mlp_poly_gali.keras")))
                self.poly_scaler_path = getattr(self, "poly_scaler_path",
                                                tk.StringVar(value=str(models_dir / "scaler_poly_gali.pkl")))
                try:
                    self.poly_model_path.set(str(models_dir / "mlp_poly_gali.keras"))
                    self.poly_scaler_path.set(str(models_dir / "scaler_poly_gali.pkl"))
                except Exception:
                    pass
                try:
                    self.use_alpha4.set(False)
                except Exception:
                    pass
            elif mode == "concat":
                self.poly_model_path = getattr(self, "poly_model_path",
                                               tk.StringVar(value=str(models_dir / "mlp_poly_concat.keras")))
                self.poly_scaler_path = getattr(self, "poly_scaler_path",
                                                tk.StringVar(value=str(models_dir / "scaler_poly_concat.pkl")))
                try:
                    self.poly_model_path.set(str(models_dir / "mlp_poly_concat.keras"))
                    self.poly_scaler_path.set(str(models_dir / "scaler_poly_concat.pkl"))
                except Exception:
                    pass
                try:
                    # concat requires BOTH providers
                    self.use_alpha.set(True)
                    self.use_gali.set(True)
                    self.use_alpha4.set(False)
                except Exception:
                    pass
            else:  # "alpha" (mean 64-D)
                self.poly_model_path = getattr(self, "poly_model_path",
                                               tk.StringVar(value=str(models_dir / "mlp_poly_alpha.keras")))
                self.poly_scaler_path = getattr(self, "poly_scaler_path",
                                                tk.StringVar(value=str(models_dir / "scaler_poly_alpha.pkl")))
                try:
                    self.poly_model_path.set(str(models_dir / "mlp_poly_alpha.keras"))
                    self.poly_scaler_path.set(str(models_dir / "scaler_poly_alpha.pkl"))
                except Exception:
                    pass
                try:
                    self.use_alpha.set(True)
                    self.use_alpha4.set(False)
                except Exception:
                    pass

        # ---- Initialize Page-2 model paths from current label/toggle ----
        _set_poly_model_paths(_map_page2_label_to_mode(self.poly_feat_src.get()))

        # ------------------------- Page-3 (Crop classification) ----------
        # Multi-zip dataset ingestion; label column is chosen after preview
        self.crop_zip_list = []  # type: list[str]
        self.crop_label_col = tk.StringVar(value="")  # selected label field name
        self._crop_available_cols = []  # type: list[str]

        # Feature mode used by Crop-MLP on Page-3.
        # Canonical values we use internally:
        #   "alpha"    → 64-D mean (AlphaEarth)
        #   "alpha4"   → 256-D 4× stats (AlphaEarth)
        #   "alpha4x64"→ 320-D concat(64 + 256)  ← NEW
        #   "gali"     → Galileo
        #   "concat"   → Alpha + Galileo
        self.crop_feat_mode = tk.StringVar(value="concat")

        # Default Crop-MLP model/scaler paths (auto-updated by sync below)
        self.crop_model_path = tk.StringVar(value=str(models_dir / "mlp_crop_concat.keras"))
        self.crop_scaler_path = tk.StringVar(value=str(models_dir / "scaler_crop_concat.pkl"))

        # ---- Helper: set Page-3 Crop-MLP model/scaler paths --------------
        def _set_crop_model_paths(mode: str) -> None:
            """Reflect feature mode → model/scaler filenames for Crop-MLP."""
            mapping = {
                "alpha": ("mlp_crop_alpha.keras", "scaler_crop_alpha.pkl"),
                "alpha4": ("mlp_crop_alpha4.keras", "scaler_crop_alpha4.pkl"),
                "alpha4x64": ("mlp_crop_alpha4x64.keras", "scaler_crop_alpha4x64.pkl"),  # 320-D
                "gali": ("mlp_crop_gali.keras", "scaler_crop_gali.pkl"),
                "concat": ("mlp_crop_concat.keras", "scaler_crop_concat.pkl"),
            }
            fname_m, fname_s = mapping.get(mode, mapping["concat"])
            try:
                self.crop_model_path.set(str(models_dir / fname_m))
                self.crop_scaler_path.set(str(models_dir / fname_s))
            except Exception:
                pass

        # ---- Helper: keep Page-3 mode in sync with Page-2 selection ------
        def _sync_crop_feat_from_page2(*_):
            """
            Read Page-2 label (poly_feat_src) → canonical mode,
            then auto-upgrade to 'alpha4x64' when both 64-D and 256-D embeddings exist.
            """
            mode = _map_page2_label_to_mode(self.poly_feat_src.get())

            # If user is on AlphaEarth 4×, and BOTH 64-D and 256-D embeddings exist,
            # prefer the richer 320-D combo for Crop-MLP.
            if mode == "alpha4":
                have64 = _have_embeddings("alpha_*.npy")
                # Accept either pre-concatenated alpha4_*.npy or 4 parts
                have256 = (_have_embeddings("alpha4_*.npy") or
                           (_have_embeddings("alpha4_mean_*.npy") and
                            _have_embeddings("alpha4_mid_*.npy") and
                            _have_embeddings("alpha4_q1_*.npy") and
                            _have_embeddings("alpha4_q3_*.npy")))
                if have64 and have256:
                    mode = "alpha4x64"

            self.crop_feat_mode.set(mode)
            _set_crop_model_paths(mode)

        # React to user switching "Features" on Page-2 (Combobox there)
        try:
            self.poly_feat_src.trace_add("write", _sync_crop_feat_from_page2)
        except Exception:
            pass
        # Run once now to initialize Page-3 from current Page-2 selection
        try:
            _sync_crop_feat_from_page2()
        except Exception:
            pass

        # ------------------------- Cached vegetation mask ------------------
        # If Page-2 built a vegetation/non-vegetation mask, keep it here to reuse
        # in Page-3 (avoids recomputing NDVI just for a quick preview)
        self._veg_mask_df = None  # type: Optional[pd.DataFrame]

        # ------------------------- Build UI -------------------------------
        self._build_ui()

    # ============================= Utilities ==============================

    def _log(self, msg: str):
        """Append a line to the log view and keep it scrolled."""
        self.log.insert("end", msg + "\n")
        self.log.see("end")
        self.update_idletasks()

    def _log_async(self, msg: str):
        """Thread-safe logging via queue."""
        self._log_q.put(msg)

    def _drain_logs(self):
        """Pull all pending logs from the queue and print them."""
        try:
            while True:
                self._log(self._log_q.get_nowait())
        except queue.Empty:
            pass
        # Keep draining while jobs are running
        if self._running_jobs:
            self.after(80, self._drain_logs)

    def _set_busy(self, btn: ttk.Button, key: str, busy: bool, text_busy="Working…"):
        """
        Toggle a 'busy' look on a button and remember previous state.
        This is robust to themes without 'Accent.TButton'.
        """
        # FIX: Configure the style once and guard against unsupported themes.
        style = ttk.Style(self)
        try:
            style.configure("Busy.TButton", foreground="white", background="#f39c12")
            style.map("Busy.TButton", background=[("disabled", "#f39c12")])
        except Exception:
            pass

        if busy:
            self._running_jobs.add(key)
            btn._orig_text = btn.cget("text")
            btn._orig_style = btn.cget("style")
            try:
                btn.config(text=text_busy, style="Busy.TButton", state="disabled")
            except Exception:
                btn.config(text=text_busy, state="disabled")
            # Start draining logs when the very first job begins
            if len(self._running_jobs) == 1:
                self._drain_logs()
        else:
            btn.config(text=getattr(btn, "_orig_text", btn.cget("text")),
                       style=getattr(btn, "_orig_style", ""))
            btn.config(state="normal")
            self._running_jobs.discard(key)

    def _run_bg(self, *, key: str, btn: ttk.Button, text_busy: str, target: callable):
        def _runner():
            try:
                target()
            except Exception as e:
                self._log_async(f"[ERROR] {key}: {e}")
                msg = f"{key}: {e}"  # freeze
                self.after(0, lambda m=msg: messagebox.showerror("Error", m))
            finally:
                self.after(0, lambda: self._set_busy(btn, key, False))

        self._set_busy(btn, key, True, text_busy)
        threading.Thread(target=_runner, daemon=True).start()

    def crop_refresh_columns(self):
        zips = list(self.lb_crop.get(0, "end"))
        if not zips:
            self._log_async("[COLUMNS] No ZIPs added.")
            self.cmb_crop_label.configure(values=[])
            self.crop_label_col.set("")
            return

        union_cols: set[str] = set()
        for zp in zips:
            try:
                cols = probe_columns_in_zip(Path(zp))
                union_cols.update(cols)
            except Exception:
                pass

        all_cols = sorted(union_cols)
        self._crop_available_cols = all_cols
        self.cmb_crop_label.configure(values=all_cols)

        # keep or guess default
        cur = self.crop_label_col.get().strip()
        default = cur if cur in all_cols else (
                    _guess_label_col_from_names(all_cols) or (all_cols[0] if all_cols else ""))
        self.crop_label_col.set(default)

        # (optional) log some samples from the first zip that contains the selected column
        for zp in zips:
            try:
                cols = probe_columns_in_zip(Path(zp))
                if default in cols:
                    tmpdir = Path(tempfile.mkdtemp(prefix="zip_peek_"))
                    with zipfile.ZipFile(zp, "r") as zf:
                        zf.extractall(tmpdir)
                    shp = next(
                        (p for p in tmpdir.rglob("*") if p.suffix.lower() in (".shp", ".gpkg", ".geojson", ".json")),
                        None)
                    if shp:
                        gdf = gpd.read_file(shp, engine="pyogrio" if _USE_PYOGRIO else None)
                        vals = list(pd.Series(gdf[default]).dropna().unique())[:10]
                        self._log_async(f"[COLUMNS] sample({default}): {vals}")
                    break
            except Exception:
                pass

        self._log_async(f"[COLUMNS] union: {all_cols} | selected: {self.crop_label_col.get()}")

    # =============================== UI =================================

    def _apply_basemap(self):
        """Switch tile server based on the combobox selection."""
        if not _MAP_OK or self.map is None:
            return
        self.map.set_tile_server(ESRI_GRAY if self.basemap.get() == "Esri Gray" else ESRI_IMAGERY)

    def _build_ui(self):
        """Create and place all widgets."""
        pad = {"padx": 10, "pady": 6}
        top = ttk.Frame(self)
        top.pack(fill="x", **pad)

        # File controls + outputs path
        self.btn_add = ttk.Button(top, text="Add ZIP(s)…", command=self.add_zips)
        self.btn_add.pack(side="left")
        ttk.Button(top, text="Clear List", command=self.clear_list).pack(side="left", padx=6)
        ttk.Button(top, text="Output Folder…", command=self.pick_outdir).pack(side="left", padx=6)
        ttk.Label(top, text="Outputs:").pack(side="left", padx=(12, 2))
        ttk.Label(top, textvariable=self.out_dir, foreground="#555").pack(side="left")

        # EE sign-in and global run button
        self.btn_sign = ttk.Button(top, text="Sign in (Google)", command=self.sign_in_google, style="Accent.TButton")
        self.btn_sign.pack(side="right", padx=(6, 0))
        self.btn_run = ttk.Button(top, text="Run", command=self.run, style="Accent.TButton")
        self.btn_run.pack(side="right")
        self.progress = ttk.Progressbar(top, mode="determinate", maximum=100)
        self.progress.pack(side="right", padx=10, fill="x", expand=True)

        # Notebook with two pages
        self.nb = ttk.Notebook(self)
        self.nb.pack(fill="both", expand=True, padx=10, pady=6)

        # ---------------- Page 1: Load & Preview ----------------
        page = ttk.Frame(self.nb)
        self.nb.add(page, text="1) Load & Preview")
        page.columnconfigure(0, weight=1)
        page.columnconfigure(1, weight=3)
        page.rowconfigure(2, weight=1)
        page.rowconfigure(3, weight=0)

        # Options (area/simplify/basemap)
        opt = ttk.LabelFrame(page, text="Options")
        opt.grid(row=0, column=0, columnspan=2, sticky="ew", padx=4, pady=4)

        r1 = ttk.Frame(opt)
        r1.pack(fill="x", padx=6, pady=3)

        ttk.Label(r1, text="Min area (m²):").pack(side="left")
        ttk.Spinbox(r1, from_=0, to=1_000_000, increment=50, textvariable=self.min_area_m2, width=10).pack(side="left")

        ttk.Label(r1, text="Simplify tol (m):").pack(side="left", padx=(12, 2))
        ttk.Spinbox(r1, from_=0, to=100, increment=1, textvariable=self.simplify_m, width=8).pack(side="left")

        ttk.Label(r1, text="Max rings drawn:").pack(side="left", padx=(12, 2))
        ttk.Spinbox(r1, from_=2000, to=50_000, increment=1000, textvariable=self.max_rings, width=10).pack(side="left")

        ttk.Label(r1, text="Basemap:").pack(side="left")
        cmb = ttk.Combobox(r1, textvariable=self.basemap, width=16,
                           values=["Esri Imagery", "Esri Gray"], state="readonly")
        cmb.pack(side="left")
        cmb.bind("<<ComboboxSelected>>", lambda _e: self._apply_basemap())

        # Left column: zip list
        left = ttk.Frame(page)
        left.grid(row=1, column=0, rowspan=2, sticky="nsew", padx=(0, 10))
        ttk.Label(left, text="Selected ZIP files:").pack(anchor="w")
        self.listbox = self._make_listbox(left)
        self.listbox.pack(fill="both", expand=True)

        # Right column: log + map
        ttk.Label(page, text="Log:").grid(row=1, column=1, sticky="w")
        self.log = ScrolledText(page, height=10, wrap="word")
        self.log.grid(row=1, column=1, sticky="nsew")

        map_box = ttk.LabelFrame(page, text="Map (Esri imagery)")
        map_box.grid(row=2, column=1, sticky="nsew")
        map_box.rowconfigure(0, weight=1)
        map_box.columnconfigure(0, weight=1)
        self.map = TkinterMapView(map_box, corner_radius=0)
        self.map.grid(row=0, column=0, sticky="nsew")
        self._apply_basemap()
        self.map.set_position(32.0, 53.0)
        self.map.set_zoom(5)

        bottom = ttk.Frame(page)
        bottom.grid(row=3, column=0, columnspan=2, sticky="ew", padx=8, pady=6)
        ttk.Button(bottom, text="Zoom to polygons", command=self.zoom_to_polygons).pack(side="left")
        ttk.Button(bottom, text="Clear map", command=self.clear_map_polygons).pack(side="left", padx=6)
        ttk.Label(bottom, textvariable=self._counts_var, foreground="#555").pack(side="left", padx=12)

        # ---------------- Page 2: Embeddings + MLP (FINAL / ADVANCED) ----------------
        p3 = ttk.Frame(self.nb)
        self.nb.add(p3, text="2) Embeddings + MLP")

        # =================== Embedding providers (AlphaEarth / Galileo) ===================
        cfg = ttk.LabelFrame(p3, text="Embedding providers")
        cfg.pack(fill="x", padx=8, pady=6)

        # -------- AlphaEarth (EE) row ------------------------------------------------------
        a = ttk.Frame(cfg)
        a.pack(fill="x", padx=6, pady=3)

        # Main toggle for AlphaEarth provider (server-side features on Earth Engine)
        ttk.Checkbutton(a, text="Use AlphaEarth (EE)", variable=self.use_alpha).pack(side="left")

        # Optional 4× statistics toggle (mean/median(q50)/q1/q3) → 256-D alpha4 vectors
        # (keep idempotent in case __init__ already created the var)
        if not hasattr(self, "use_alpha4"):
            self.use_alpha4 = tk.BooleanVar(value=False)

        ttk.Checkbutton(
            a, text="Also compute 4× stats (alpha4)", variable=self.use_alpha4
        ).pack(side="left", padx=(12, 0))

        # EE project / year / tiling controls
        ttk.Label(a, text="Project").pack(side="left", padx=(12, 4))
        ttk.Entry(a, textvariable=self.ee_project, width=26).pack(side="left")

        ttk.Label(a, text="Year").pack(side="left", padx=(8, 2))
        # NOTE: AlphaEarth V1 covers roughly 2017–2024; keep the upper bound at 2024.
        ttk.Spinbox(a, from_=2017, to=2024, textvariable=self.ee_year, width=6).pack(side="left")

        ttk.Label(a, text="tileScale").pack(side="left", padx=(8, 2))
        ttk.Spinbox(a, from_=1, to=16, textvariable=self.ee_tilescale, width=6).pack(side="left")

        ttk.Label(a, text="Chunk size").pack(side="left", padx=(8, 2))
        ttk.Spinbox(a, from_=50, to=1000, increment=50, textvariable=self.ee_chunk, width=8).pack(side="left")

        # EE connectivity test buttons (cached vs. forced OAuth-in-browser)
        self.btn_test_browser = ttk.Button(a, text="Test EE (force browser)", command=self.test_ee_force_browser)
        self.btn_test_browser.pack(side="right")
        self.btn_test_cache = ttk.Button(a, text="Test EE (cached)", command=self.test_ee_cached)
        self.btn_test_cache.pack(side="right", padx=4)

        # -------- Optional Service Account row --------------------------------------------
        a2 = ttk.Frame(cfg)
        a2.pack(fill="x", padx=6, pady=3)

        ttk.Label(a2, text="(Optional) SA Email").pack(side="left")
        ttk.Entry(a2, textvariable=self.sa_email, width=32).pack(side="left", padx=(4, 10))

        ttk.Label(a2, text="SA Key JSON").pack(side="left")
        ttk.Entry(a2, textvariable=self.sa_json, width=40).pack(side="left", padx=(4, 6))

        ttk.Button(  # file picker for SA key JSON
            a2, text="Browse…",
            command=lambda: self._pick_file(self.sa_json, [("JSON", "*.json")])
        ).pack(side="left")

        # -------- Galileo (local TorchScript encoder) row ---------------------------------
        g = ttk.Frame(cfg)
        g.pack(fill="x", padx=6, pady=3)

        ttk.Checkbutton(g, text="Use Galileo (local)", variable=self.use_gali).pack(side="left")

        ttk.Label(g, text="Weights").pack(side="left", padx=(8, 2))
        entry_gali_w = ttk.Entry(g, textvariable=self.gali_weights, width=52)
        entry_gali_w.pack(side="left", padx=(4, 6))

        # Place the browse button next to the Weights entry (better UX)
        btn_gali_browse = ttk.Button(g, text="Browse…", command=lambda: self._pick_weights(self.gali_weights))
        btn_gali_browse.pack(side="left", padx=(0, 8))

        ttk.Label(g, text="Device").pack(side="left", padx=(12, 2))
        cmb_gali_dev = ttk.Combobox(g, values=["cpu", "cuda"], textvariable=self.gali_device, width=6, state="readonly")
        cmb_gali_dev.pack(side="left")

        self.btn_test_gali = ttk.Button(g, text="Test Galileo", command=self.test_galileo)
        self.btn_test_gali.pack(side="right")

        # Small helper to enable/disable Galileo controls when the toggle changes
        def _toggle_gali_controls(*_):
            state = "normal" if self.use_gali.get() else "disabled"
            try:
                entry_gali_w.config(state=state)
                btn_gali_browse.config(state=state)
                cmb_gali_dev.config(state="readonly" if state == "normal" else "disabled")
                self.btn_test_gali.config(state=state)
            except Exception:
                pass

        # Trace Galileo toggle so the controls reflect the current state
        try:
            self.use_gali.trace_add("write", lambda *_: _toggle_gali_controls())
        except Exception:
            pass
        _toggle_gali_controls()  # initialize once

        # =================== Workflow (compute embeddings + TF options) ===================
        flow = ttk.LabelFrame(p3, text="Workflow")
        flow.pack(fill="x", padx=8, pady=6)

        # Compute buttons (AlphaEarth + Galileo together, or individually)
        self.btn_embed_both = ttk.Button(
            flow, text="Compute embeddings (AlphaEarth + Galileo)",
            command=self.compute_embeddings_both
        )
        self.btn_embed_both.pack(side="left", padx=6)

        self.btn_embed_alpha = ttk.Button(
            flow, text="Compute AlphaEarth only",
            command=self.compute_embeddings_alpha_only
        )
        self.btn_embed_alpha.pack(side="left", padx=6)

        self.btn_embed_gali = ttk.Button(
            flow, text="Compute Galileo only",
            command=self.compute_embeddings_galileo_only
        )
        self.btn_embed_gali.pack(side="left", padx=6)

        # NDVI threshold used for auto-labeling vegetation vs. non-vegetation during polygon-MLP training
        ttk.Label(flow, text="NDVI≥ (label threshold)").pack(side="left", padx=(18, 4))
        ttk.Spinbox(flow, from_=-1.0, to=1.0, increment=0.05, width=6, textvariable=self.ndvi_thresh).pack(side="left")

        # Allow TensorFlow to see GPU (best-effort); sklearn fallback ignores this
        ttk.Checkbutton(flow, text="Use GPU for TF (if available)", variable=self.tf_use_gpu).pack(side="left", padx=12)

        # =================== Polygon MLP (train/predict from saved embeddings) =============
        poly = ttk.LabelFrame(p3, text="Polygon MLP (from saved embeddings)")
        poly.pack(fill="x", padx=8, pady=6)

        r0 = ttk.Frame(poly)
        r0.pack(fill="x", padx=6, pady=4)

        # ---- Feature source combobox: supports alpha / gali / concat / alpha4 -------------
        ttk.Label(r0, text="Features").pack(side="left")

        feat_combo = ttk.Combobox(
            r0,
            textvariable=self.poly_feat_src,  # tk.StringVar created in __init__
            width=28,
            values=[
                "AlphaEarth (mean 64-D)",
                "Galileo",
                "Concat (Alpha+Galileo)",
                "AlphaEarth (4× stats)",
            ],
            state="readonly"
        )
        feat_combo.pack(side="left", padx=(4, 18))

        # Keep model/scaler paths consistent with chosen provider mode.
        # Also toggle 'use_alpha4' when user selects the 4×-stats option.
        def _on_feat_change(_e=None):
            choice = (self.poly_feat_src.get() or "").strip().lower()
            models_dir = Path(self.out_dir.get()) / "models"
            models_dir.mkdir(parents=True, exist_ok=True)

            if "4×" in choice or "4x" in choice or "stats" in choice:
                # AlphaEarth (4× stats) → 256-D
                self.poly_model_path.set(str(models_dir / "mlp_poly_alpha4.keras"))
                self.poly_scaler_path.set(str(models_dir / "scaler_poly_alpha4.pkl"))
                try:
                    self.use_alpha.set(True)
                    self.use_alpha4.set(True)
                except Exception:
                    pass
            elif choice.startswith("galileo"):
                self.poly_model_path.set(str(models_dir / "mlp_poly_gali.keras"))
                self.poly_scaler_path.set(str(models_dir / "scaler_poly_gali.pkl"))
                try:
                    self.use_alpha4.set(False)
                except Exception:
                    pass
            elif choice.startswith("concat"):
                self.poly_model_path.set(str(models_dir / "mlp_poly_concat.keras"))
                self.poly_scaler_path.set(str(models_dir / "scaler_poly_concat.pkl"))
                try:
                    # concat requires BOTH providers to have embeddings on disk
                    self.use_alpha.set(True)
                    self.use_gali.set(True)
                    self.use_alpha4.set(False)
                    _toggle_gali_controls()
                except Exception:
                    pass
            else:
                # Plain AlphaEarth (mean 64-D)
                self.poly_model_path.set(str(models_dir / "mlp_poly_alpha.keras"))
                self.poly_scaler_path.set(str(models_dir / "scaler_poly_alpha.pkl"))
                try:
                    self.use_alpha.set(True)
                    self.use_alpha4.set(False)
                except Exception:
                    pass

        # Bind selection-change once and initialize paths now
        feat_combo.bind("<<ComboboxSelected>>", _on_feat_change)
        _on_feat_change()

        # ---- Model / scaler paths (auto-updated after training) --------------------------
        ttk.Label(r0, text="Model").pack(side="left")
        ttk.Entry(r0, textvariable=self.poly_model_path, width=48).pack(side="left", padx=(4, 10))

        ttk.Label(r0, text="Scaler").pack(side="left")
        ttk.Entry(r0, textvariable=self.poly_scaler_path, width=36).pack(side="left", padx=(4, 0))

        # ---- Train / Predict actions (polygon-level veg vs. non-veg) ---------------------
        actp = ttk.Frame(p3)
        actp.pack(fill="x", padx=8, pady=6)

        self.btn_poly_train = ttk.Button(
            actp, text="Train Polygon-MLP",
            command=self.train_polygon_mlp_btn, style="Accent.TButton"
        )
        self.btn_poly_train.pack(side="left", padx=6)

        self.btn_poly_predict = ttk.Button(
            actp, text="Predict polygons (Polygon-MLP) + map",
            command=self.predict_polygon_mlp_btn
        )
        self.btn_poly_predict.pack(side="left", padx=12)

        # ---------------- Page 3: Crop Classification ----------------
        p4 = ttk.Frame(self.nb)
        self.nb.add(p4, text="3) Crop Classification")

        # Layout: header fixed, list grows, actions fixed (no widgets get hidden)
        p4.columnconfigure(0, weight=1)
        p4.rowconfigure(1, weight=1)  # only the list area expands

        # ========== Top controls (dataset / options / paths) ==========
        ds = ttk.LabelFrame(p4, text="Labeled dataset (ZIP with SHP components)")
        ds.grid(row=0, column=0, sticky="ew", padx=8, pady=(6, 3))

        # -- Add / Clear ZIPs
        btn_add_zip = ttk.Button(ds, text="Add labeled ZIP(s)…", command=self.crop_add_zips)
        btn_add_zip.pack(side="left", padx=4)
        ttk.Button(ds, text="Clear", command=self.crop_clear_list).pack(side="left", padx=4)

        # -- Label-column selector (values are filled by crop_refresh_columns)
        ttk.Label(ds, text="Label column:").pack(side="left", padx=(12, 4))
        self.cmb_crop_label = ttk.Combobox(
            ds,
            textvariable=self.crop_label_col,  # set in __init__
            state="readonly",
            width=22,
            values=[]  # filled by crop_refresh_columns()
        )
        self.cmb_crop_label.pack(side="left", padx=(0, 4))
        ttk.Button(ds, text="Refresh cols", command=self.crop_refresh_columns).pack(side="left", padx=(2, 10))

        ttk.Button(ds, text="Preview table", command=self.crop_show_table).pack(side="left", padx=(2, 10))
        self.cmb_crop_label.bind(
            "<<ComboboxSelected>>",
            lambda _e: self.crop_col_info.set(f"Selected: {self.crop_label_col.get()}")
        )

        # (Optional) live info text – shows the selected label column
        self.crop_col_info = tk.StringVar(value="")
        ttk.Label(ds, textvariable=self.crop_col_info, foreground="#666").pack(side="left", padx=(0, 6))

        # -------------------- Feature mode (Page-3 view) --------------------
        # We expose a user-facing selector that mirrors Page-2 and ALSO offers the new 320-D combo.
        # Internally we keep using self.crop_feat_mode with canonical tokens:
        #   'alpha' | 'alpha4' | 'alpha4x64' | 'gali' | 'concat'

        # Seed canonical mode from Page-2 (best effort)
        try:
            self.crop_feat_mode.set(self._provider_mode())  # returns: alpha|gali|concat|alpha4
        except Exception:
            pass

        # UI label var (decoupled from the canonical mode). Default: auto mirror from Page-2.
        _ui_feat_label = tk.StringVar(value="Auto (from Page 2)")

        # Mapping between user-facing labels and internal modes
        label_to_mode = {
            "Auto (from Page 2)": "auto",  # resolves from self.crop_feat_mode / page-2
            "AlphaEarth (mean 64-D)": "alpha",
            "AlphaEarth (4× stats)": "alpha4",
            "AlphaEarth (64+4× = 320-D)": "alpha4x64",  # NEW combined 320-D
            "Galileo": "gali",
            "Concat (Alpha+Galileo)": "concat",
        }
        mode_to_label = {v: k for k, v in label_to_mode.items() if v != "auto"}

        def _set_crop_model_paths_for(mode: str) -> None:
            """
            Update Page-3 model/scaler paths to match the chosen feature mode.
            Keeps files under Outputs/models. This avoids accidental overwrite.
            """
            mapping = {
                "alpha": ("mlp_crop_alpha.keras", "scaler_crop_alpha.pkl"),
                "alpha4": ("mlp_crop_alpha4.keras", "scaler_crop_alpha4.pkl"),
                "alpha4x64": ("mlp_crop_alpha4x64.keras", "scaler_crop_alpha4x64.pkl"),  # 320-D
                "gali": ("mlp_crop_gali.keras", "scaler_crop_gali.pkl"),
                "concat": ("mlp_crop_concat.keras", "scaler_crop_concat.pkl"),
            }
            m_name, s_name = mapping.get(mode, mapping["concat"])
            try:
                models_dir = Path(self.out_dir.get()) / "models"
                models_dir.mkdir(parents=True, exist_ok=True)
                self.crop_model_path.set(str(models_dir / m_name))
                self.crop_scaler_path.set(str(models_dir / s_name))
            except Exception:
                pass

        def _on_crop_feat_change(_e=None):
            """
            User changed the Page-3 combo:
              - resolve to canonical token
              - update self.crop_feat_mode
              - point model/scaler to proper filenames
            """
            lbl = (_ui_feat_label.get() or "").strip()
            key = label_to_mode.get(lbl, "auto")
            # "auto" → keep the already-resolved canonical mode (mirrored from Page-2)
            final_mode = (self.crop_feat_mode.get() or "concat") if key == "auto" else key
            self.crop_feat_mode.set(final_mode)
            _set_crop_model_paths_for(final_mode)

        ttk.Label(ds, text="Features:").pack(side="left", padx=(8, 4))
        cmb_crop_features = ttk.Combobox(
            ds,
            textvariable=_ui_feat_label,
            width=28,
            state="readonly",
            values=list(label_to_mode.keys())
        )
        cmb_crop_features.pack(side="left", padx=(4, 12))
        cmb_crop_features.bind("<<ComboboxSelected>>", _on_crop_feat_change)

        def _sync_ui_from_canonical(*_):
            """
            Whenever self.crop_feat_mode changes (e.g., Page-2 selection),
            reflect it in Page-3 user-facing label and model/scaler paths.
            """
            mode = (self.crop_feat_mode.get() or "concat").strip().lower()
            _ui_feat_label.set(mode_to_label.get(mode, "Auto (from Page 2)"))
            _set_crop_model_paths_for(mode)

        # Keep Page-3 UI in sync with canonical mode
        try:
            self.crop_feat_mode.trace_add("write", _sync_ui_from_canonical)
        except Exception:
            pass
        _sync_ui_from_canonical()  # initial sync

        # -- Model / Scaler output paths (reflect current mode)
        ttk.Label(ds, text="Model").pack(side="left", padx=(12, 4))
        ttk.Entry(ds, textvariable=self.crop_model_path, width=42).pack(side="left", padx=(0, 8))
        ttk.Label(ds, text="Scaler").pack(side="left", padx=(4, 4))
        ttk.Entry(ds, textvariable=self.crop_scaler_path, width=34).pack(side="left")

        # ========== Middle: scrollable ZIP list (this row expands) ==========
        box = ttk.Frame(p4)
        box.grid(row=1, column=0, sticky="nsew", padx=8, pady=3)
        box.columnconfigure(0, weight=1)
        box.rowconfigure(0, weight=1)

        # Grid-based listbox wrapper (avoid mixing pack/grid in the same parent)
        frame = ttk.Frame(box)
        frame.grid(row=0, column=0, sticky="nsew")
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(0, weight=1)

        self.lb_crop = tk.Listbox(frame, width=48, height=14, selectmode="extended")
        sb_crop = ttk.Scrollbar(frame, orient="vertical", command=self.lb_crop.yview)
        self.lb_crop.configure(yscrollcommand=sb_crop.set)
        self.lb_crop.grid(row=0, column=0, sticky="nsew")
        sb_crop.grid(row=0, column=1, sticky="ns")

        # ========== Bottom actions (always visible) ==========
        act = ttk.LabelFrame(p4, text="Training / Evaluation")
        act.grid(row=2, column=0, sticky="ew", padx=8, pady=(3, 6))

        # Train with a 70/30 stratified split; save model, scaler, meta, metrics, CM PNG
        self.btn_crop_build = ttk.Button(
            act,
            text="Train (70/30) + Evaluate + Map + Save",
            command=self.crop_train_eval,
            style="Accent.TButton"
        )
        self.btn_crop_build.pack(side="left", padx=6)

        # Direct NDVI-gated prediction & map from Page-3 (uses crop_predict_map)
        self.btn_crop_predict = ttk.Button(
            act, text="Predict (NDVI-gated) + Map", command=self.crop_predict_map
        )
        self.btn_crop_predict.pack(side="left", padx=6)

        # ========== Results area (inline, single place; no duplicate boxes) ==========
        res = ttk.LabelFrame(p4, text="Results (preview & metrics)")
        res.grid(row=3, column=0, sticky="nsew", padx=8, pady=(3, 8))
        res.columnconfigure(0, weight=1)
        res.columnconfigure(1, weight=1)
        p4.rowconfigure(3, weight=1)

        self.res_preview = ttk.Frame(res)  # left: preview table
        self.res_preview.grid(row=0, column=0, sticky="nsew", padx=(0, 8), pady=6)
        self.res_preview.columnconfigure(0, weight=1)
        self.res_preview.rowconfigure(1, weight=1)

        self.res_metrics = ttk.Frame(res)  # right: metrics table
        self.res_metrics.grid(row=0, column=1, sticky="nsew", padx=(8, 0), pady=6)
        self.res_metrics.columnconfigure(0, weight=1)
        self.res_metrics.rowconfigure(1, weight=1)

        # Backward-compat: some helpers still expect `results_area`; point it to res_preview
        self.results_area = self.res_preview

        # ---------- Final: try to populate label-column combobox now ----------
        try:
            self.crop_refresh_columns()
            if self.crop_label_col.get():
                self.crop_col_info.set(f"Selected: {self.crop_label_col.get()}")
        except Exception:
            pass

    # ========================== Small helpers ===========================

    def _make_listbox(self, parent):
        """
        Create a Listbox wrapped in a frame with its own vertical scrollbar.
        The frame's .pack() is rebound to the Listbox for a simpler call site.
        """
        frame = ttk.Frame(parent)
        lb = tk.Listbox(frame, width=48, height=22, selectmode="extended")
        sb = ttk.Scrollbar(frame, orient="vertical", command=lb.yview)
        lb.configure(yscrollcommand=sb.set)

        # Grid layout inside the wrapper frame
        lb.grid(row=0, column=0, sticky="nsew")
        sb.grid(row=0, column=1, sticky="ns")
        frame.grid_columnconfigure(0, weight=1)
        frame.grid_rowconfigure(0, weight=1)

        # Let callers do: self.listbox.pack(...)
        lb.pack = frame.pack
        return lb

    def _clear_frame(self, frame):
        # Remove all children from a container frame safely
        for w in frame.winfo_children():
            try:
                w.destroy()
            except Exception:
                pass

    def _show_preview_inpage(self, df: pd.DataFrame, title="Preview"):
        """Render a small table (Treeview) inside Page 3 (left panel)."""
        self._clear_frame(self.res_preview)
        hdr = ttk.Label(self.res_preview, text=f"{title} — showing {len(df)} row(s)", foreground="#555")
        hdr.grid(row=0, column=0, sticky="w", padx=4, pady=(0, 4))
        cols = list(df.columns)
        tv = ttk.Treeview(self.res_preview, columns=cols, show="headings", height=14)
        for c in cols:
            tv.heading(c, text=c)
            tv.column(c, width=120, stretch=True)
        vs = ttk.Scrollbar(self.res_preview, orient="vertical", command=tv.yview)
        tv.configure(yscrollcommand=vs.set)
        tv.grid(row=1, column=0, sticky="nsew")
        vs.grid(row=1, column=1, sticky="ns")
        for _, r in df.head(2000).iterrows():
            tv.insert("", "end", values=[r.get(c, "") for c in cols])

    def _clear_results(self):
        """Remove all widgets from the inline results area."""
        if hasattr(self, "results_area"):
            for child in self.results_area.winfo_children():
                try:
                    child.destroy()
                except Exception:
                    pass

    def _show_table_inpage(self, df: pd.DataFrame):
        """Render a table inside Page-3 (left results pane: res_preview)."""
        # Use the visible left pane instead of the old 'results_area'
        try:
            parent = self.res_preview  # visible pane
        except Exception:
            # Fallback: keep old behavior if res_preview is missing
            parent = getattr(self, "results_area", None) or self

        # Clear pane
        for w in parent.winfo_children():
            try:
                w.destroy()
            except Exception:
                pass

        # Header
        hdr = ttk.Label(parent, text=f"Preview — {len(df)} row(s)", foreground="#555")
        hdr.grid(row=0, column=0, sticky="w", padx=4, pady=(0, 4))

        # Table
        cols = list(df.columns)
        tv = ttk.Treeview(parent, columns=cols, show="headings", height=14)
        vs = ttk.Scrollbar(parent, orient="vertical", command=tv.yview)
        tv.configure(yscrollcommand=vs.set)

        tv.grid(row=1, column=0, sticky="nsew")
        vs.grid(row=1, column=1, sticky="ns")

        # Allow the table to expand
        try:
            parent.columnconfigure(0, weight=1)
            parent.rowconfigure(1, weight=1)
        except Exception:
            pass

        for c in cols:
            tv.heading(c, text=c)
            tv.column(c, width=140 if c != "src_layer" else 220, anchor="center")

        for _, r in df.iterrows():
            tv.insert("", "end", values=[r.get(c, "") for c in cols])

    def _show_metrics_inpage(self, result: dict, class_names: list[str]):
        """Render accuracy/F1 and per-class table inside Page 3 (right panel)."""
        self._clear_frame(self.res_metrics)
        acc = float(result.get("acc", 0.0)) * 100.0
        f1m = float(result.get("f1_macro", 0.0)) * 100.0
        ttk.Label(self.res_metrics, text=f"Accuracy: {acc:.2f}%   |   F1-macro: {f1m:.2f}%",
                  font=("Segoe UI", 10, "bold")).grid(row=0, column=0, sticky="w", padx=4, pady=(0, 6))

        tv = ttk.Treeview(self.res_metrics,
                          columns=["class", "precision", "recall", "f1", "support"],
                          show="headings", height=14)
        for c, w in [("class", 160), ("precision", 90), ("recall", 90), ("f1", 90), ("support", 80)]:
            tv.heading(c, text=c.title())
            tv.column(c, width=w, anchor="center")
        vs = ttk.Scrollbar(self.res_metrics, orient="vertical", command=tv.yview)
        tv.configure(yscrollcommand=vs.set)
        tv.grid(row=1, column=0, sticky="nsew")
        vs.grid(row=1, column=1, sticky="ns")

        rep = result.get("report", {})
        for cls in class_names:
            row = rep.get(cls, {})
            tv.insert("", "end", values=[
                cls,
                f"{float(row.get('precision', 0.0)) * 100:.1f}%",
                f"{float(row.get('recall', 0.0)) * 100:.1f}%",
                f"{float(row.get('f1-score', 0.0)) * 100:.1f}%",
                int(row.get('support', 0))
            ])

    def _pick_file(self, var, types):
        """Open a file picker and write the result into a tk.StringVar."""
        p = filedialog.askopenfilename(filetypes=types)
        if p:
            var.set(p)

    def _pick_dir(self, var):
        """Open a directory picker and write the result into a tk.StringVar."""
        d = filedialog.askdirectory()
        if d:
            var.set(d)

    def _pick_weights(self, var: tk.StringVar):
        """
        Allow user to select either a TorchScript file (encoder.pt) OR a folder that contains it.
        We first try file selection (more explicit), then fallback to directory selection.
        """
        p = filedialog.askopenfilename(
            title="Select TorchScript file (encoder.pt) or a folder",
            filetypes=[("TorchScript", "*.pt *.pth"), ("All files", "*.*")]
        )
        if p:
            var.set(p)
            return

        d = filedialog.askdirectory(title="Select folder that contains encoder.pt")
        if d:
            var.set(d)

    def _require_veg_mask(self) -> Optional[pd.DataFrame]:
        """
        Ensure we have a vegetation mask from page 2.
        If not in memory, try to load Outputs/veg_mask_last.csv.
        Return None if still unavailable.
        """
        # already in memory
        if isinstance(self._veg_mask_df, pd.DataFrame) and len(self._veg_mask_df):
            return self._veg_mask_df

        # try to load from disk
        try:
            p = Path(self.out_dir.get()) / "veg_mask_last.csv"
            if p.exists():
                df = pd.read_csv(p)
                # light schema check
                if {"poly_id", "final_label", "veg_flag"}.issubset(set(df.columns)):
                    self._veg_mask_df = df.copy()
                    self._log_async("[PAGE-3] Loaded vegetation mask from veg_mask_last.csv")
                    return self._veg_mask_df
        except Exception as e:
            self._log_async(f"[PAGE-3] Could not load veg mask CSV: {e}")

        return None  # caller will prompt the user to run page 2

    def _predict_crops_alpha4x64(self, g_in: "gpd.GeoDataFrame", model_path: Path,
                                 scaler_path: Path) -> "gpd.GeoDataFrame":
        """
        Predict crop classes for vegetated polygons using alpha4x64 (320-D) features.
        Returns a copy of g_in with columns: pred_crop (str), pred_conf (float).
        """
        import joblib
        import numpy as _np
        from tensorflow.keras.models import load_model

        # Build features for the given subset
        X, _, pids, feat_names, class_names, le = self._build_features_alpha4x64(g_in)

        # Load scaler + model
        scaler = joblib.load(scaler_path)
        model = load_model(model_path)

        Xs = scaler.transform(X)
        proba = model.predict(Xs, verbose=0)  # shape [N, C]
        idx_max = _np.argmax(proba, axis=1)
        conf = proba[_np.arange(len(proba)), idx_max]
        y_pred_labels = le.inverse_transform(idx_max)

        # Attach predictions back to GeoDataFrame order by poly_id
        dfp = pd.DataFrame({"poly_id": pids, "pred_crop": y_pred_labels.astype(str), "pred_conf": conf.astype(float)})
        gout = g_in.merge(dfp, on="poly_id", how="left")
        return gout

    # ========================= Map drawing helpers ======================

    def _draw_gdf(self, gdf: gpd.GeoDataFrame, color_field: Optional[str] = None):
        """
        Draw polygons onto the map with optional class-based coloring.
        To keep TkinterMapView responsive:
          - decimate overly dense rings
          - respect a global cap on total rings drawn
        """

        if not _MAP_OK or self.map is None:
            self._log_async("[Map] Map widget not available.")
            return

        self.clear_map_polygons()
        cap = int(self.max_rings.get() or 15000)
        drawn = 0

        def draw_one(poly_geom, color_hex: str):
            """Draw a single polygon ring after decimation."""
            nonlocal drawn
            if drawn >= cap:
                return

            # Exterior ring only (TkinterMapView does not support holes directly)
            coords = [(float(x), float(y)) for (x, y, *_) in poly_geom.exterior.coords]

            # Decimate super-dense rings to keep rendering light
            max_pts = 4000
            if len(coords) > max_pts:
                step = max(1, len(coords) // max_pts)
                coords = coords[::step]
                if coords[0] != coords[-1]:
                    coords.append(coords[0])

            # TkinterMapView expects (lat, lon)
            ring_latlon = [(y, x) for (x, y) in coords]

            # Use a solid fill to make classes easy to see
            fill = color_hex if color_field else "#ff9800"

            poly = self.map.set_polygon(
                ring_latlon, outline_color=color_hex, fill_color=fill, border_width=2
            )
            self._drawn_polys.append(poly)
            drawn += 1

        # Color mapping when a label/field is provided
        for _, row in gdf.iterrows():
            geom = row.geometry
            if geom is None or geom.is_empty:
                continue

            color = "#e65100"  # default orange
            if color_field and color_field in gdf.columns:
                v = str(row[color_field])
                color = (
                    COLOR_VEG if v == "vegetation"
                    else COLOR_NONV if v == "non_vegetation"
                    else COLOR_UNK
                )

            try:
                if isinstance(geom, Polygon):
                    draw_one(geom, color)
                elif isinstance(geom, MultiPolygon):
                    for p in geom.geoms:
                        draw_one(p, color)
                        if drawn >= cap:
                            break
            except Exception as ex:
                self._log_async(f"[Map] draw error: {ex}")

            if drawn >= cap:
                break

        # Fit the map to the overall bounds for better UX
        if len(gdf):
            minx, miny, maxx, maxy = gdf.total_bounds
            self.map.fit_bounding_box((maxy, minx), (miny, maxx))

        # Binary legend with color names
        self._log_async("Legend (class → color):")
        self._log_async(f"vegetation: {_color_name(COLOR_VEG)}")
        self._log_async(f"NON_VEG:   {_color_name(COLOR_NONV)}")
        self._log_async(f"unknown:   {_color_name(COLOR_UNK)}")

        self._log_async(f"Drawn rings: {drawn}")

    def clear_map_polygons(self):
        """Remove all polygons currently drawn on the map."""
        for p in getattr(self, "_drawn_polys", []):
            try:
                p.delete()
            except Exception:
                pass
        self._drawn_polys = []

    def zoom_to_polygons(self):
        """Zoom to the extent of the currently loaded GeoDataFrame."""
        if self._gdf is None or not len(self._gdf):
            return
        minx, miny, maxx, maxy = self._gdf.total_bounds
        self.map.fit_bounding_box((maxy, minx), (miny, maxx))



    # -------------------- Page 3 handlers --------------------

    def crop_add_zips(self):
        paths = filedialog.askopenfilenames(title="Select labeled ZIP files", filetypes=[("ZIP files", "*.zip")])
        existing = set(self.lb_crop.get(0, "end"))
        for p in paths:
            if p not in existing:
                self.lb_crop.insert("end", p)

    def crop_clear_list(self):
        self.lb_crop.delete(0, "end")

    def _load_all_labeled(self) -> gpd.GeoDataFrame:
        """
        Load and merge all labeled ZIPs using the user-provided label column.
        """
        zips = list(self.lb_crop.get(0, "end"))
        if not zips:
            raise RuntimeError("Add at least one labeled ZIP.")
        dfs = []
        for zp in zips:
            try:
                g = load_labeled_polygons_from_zip(
                    Path(zp),
                    label_col=self.crop_label_col.get().strip(),
                    min_area_m2=float(self.min_area_m2.get() or 0),
                    simplify_m=float(self.simplify_m.get() or 2.0),
                )
                dfs.append(g)
                self._log_async(f"[LABELED] {Path(zp).name}: {len(g)} polygons")
            except Exception as e:
                self._log_async(f"[LABELED] error {Path(zp).name}: {e}")
        if not dfs:
            raise RuntimeError("No labeled polygons loaded.")
        gdf = pd.concat(dfs, ignore_index=True).set_crs(4326, allow_override=True)
        gdf["poly_id"] = np.arange(len(gdf))

        missing = int(gdf["crop_label"].isna().sum())
        if missing > 0:
            self._log_async(f"[LABELED] rows without label skipped: {missing}")
        gdf = gdf.dropna(subset=["crop_label"]).reset_index(drop=True)

        return gdf

    def crop_show_table(self):
        """
        Preview labeled polygons inline in Page 3. If EE is available, also show
        ndvi_mean and a veg_flag (ndvi>=thr) so you can see the NDVI gate effect.
        """
        try:
            g = self._load_all_labeled()
        except Exception as e:
            self._log_async(f"[Preview] load error: {e}")
            return self._mbox_err("Preview", f"Cannot load labeled dataset:\n{e}")

        # Try to append NDVI; continue gracefully if EE fails
        ndvi_col = None
        try:
            ok = ee_initialize(
                self.ee_project.get().strip(),
                self.sa_email.get().strip() or None,
                self.sa_json.get().strip() or None,
                interactive_fallback=False,
                log=self._log_async,
            )
            if ok:
                s2tab = compute_s2_tabular_features(
                    g, int(self.ee_year.get()),
                    tile_scale=int(self.ee_tilescale.get()),
                    log=self._log_async,
                )
                g = g.merge(s2tab[["poly_id", "ndvi_mean"]], on="poly_id", how="left")
                # crop_show_table
                thr = float(self.ndvi_thresh.get())
                g["veg_flag"] = g["ndvi_mean"].astype(float) >= thr

                ndvi_col = "ndvi_mean"
        except Exception as e:
            self._log_async(f"[Preview] NDVI not shown: {e}")

        cols = ["poly_id", "crop_label", "src_layer"]
        if ndvi_col:
            cols += [ndvi_col, "veg_flag"]

        show_df = g[cols].copy()
        if len(show_df) > 2000:
            show_df = show_df.head(2000)

        self._show_table_inpage(show_df)
        self._log_async(f"[Preview] showing {len(show_df)} rows inline.")


    def _show_metrics_window(self, result: dict, class_names: list[str]):
        """
        Pop up a metrics window with percentages and a per-class table.
        `result` is the dict returned by train_evaluate_crop_mlp.
        """
        acc = float(result.get("acc", 0.0))
        f1m = float(result.get("f1_macro", 0.0))
        rep = result.get("report", {})

        win = tk.Toplevel(self)
        win.title("MLP Metrics")
        win.geometry("720x520")

        top = ttk.Frame(win);
        top.pack(fill="x", padx=12, pady=10)
        ttk.Label(top, text=f"Accuracy: {acc * 100:.2f}%", font=("Segoe UI", 11, "bold")).pack(side="left")
        ttk.Label(top, text=f"   |   F1-macro: {f1m * 100:.2f}%", font=("Segoe UI", 11, "bold")).pack(side="left",
                                                                                                      padx=12)

        # Per-class table
        tv = ttk.Treeview(win, columns=["class", "precision", "recall", "f1", "support"], show="headings", height=16)
        for c, w in [("class", 180), ("precision", 100), ("recall", 100), ("f1", 100), ("support", 90)]:
            tv.heading(c, text=c.title())
            tv.column(c, width=w, anchor="center")
        vsb = ttk.Scrollbar(win, orient="vertical", command=tv.yview)
        tv.configure(yscrollcommand=vsb.set)
        tv.pack(side="left", fill="both", expand=True, padx=(12, 0), pady=(0, 12))
        vsb.pack(side="right", fill="y", padx=(0, 12), pady=(0, 12))

        for cls in class_names:
            row = rep.get(cls, {})
            prec = float(row.get("precision", 0.0)) * 100.0
            rec = float(row.get("recall", 0.0)) * 100.0
            f1 = float(row.get("f1-score", 0.0)) * 100.0
            sup = int(row.get("support", 0))
            tv.insert("", "end", values=[cls, f"{prec:.1f}%", f"{rec:.1f}%", f"{f1:.1f}%", sup])

        hint = ttk.Label(win, text="Artifacts saved next to your model: .meta.json, .metrics.json, .cm.png",
                         foreground="#666")
        hint.pack(fill="x", padx=12, pady=(0, 10))

    def _class_palette(self, class_names: List[str]) -> Dict[str, str]:
        """
        Deterministic color assignment for class names.
        Always reserve brown for 'NON_VEG' (non-vegetation).
        """
        pal = {}
        for i, name in enumerate(class_names):
            pal[name] = CLASS_COLORS[i % len(CLASS_COLORS)]
        # --- keep non-vegetation brown no matter what ---
        pal.setdefault("NON_VEG", COLOR_NONV)  # <-- always include a brown entry
        return pal

    def _draw_gdf_multiclass(
            self,
            gdf: "gpd.GeoDataFrame",
            label_field: str,
            class_colors: "Dict[str, str]"
    ) -> None:
        """
        Draw class-colored polygons from a GeoDataFrame onto a TkinterMapView.

        This implementation is engineered for robustness on large layers:
        - Validates/repairs geometries (buffer(0) fallback).
        - Auto-reprojects to EPSG:4326 if CRS is known and different.
        - Decimates very dense rings while preserving ring closure.
        - Enforces a safety cap to keep the UI responsive.
        - Produces a human-readable legend and per-class stats before drawing.

        Parameters
        ----------
        gdf : GeoDataFrame
            Expected to be in EPSG:4326 (lon/lat). If a different CRS is detected,
            the layer will be reprojected on-the-fly for display.
        label_field : str
            Column containing the class label for each polygon.
        class_colors : Dict[str, str]
            Mapping from class name → HEX color (e.g. "#1f77b4").
            "NON_VEG" will be added (brown) if absent.

        Notes
        -----
        • TkinterMapView polygons do not support interior rings (holes). We draw
          the exterior ring only.
        • Coordinates passed to the widget are (lat, lon).
        • Extremely dense rings are decimated to maintain interactivity.
        """

        # ---- Imports kept local to avoid module-level coupling ----
        from typing import Iterable, Iterator, Tuple, Dict, Any
        import math
        try:
            from shapely.geometry import Polygon, MultiPolygon, GeometryCollection
        except Exception:  # pragma: no cover
            # Shapely is required for robust geometry handling.
            self._log_async("[Map] Shapely not available; aborting draw.")
            return

        # ---- 0) Guard: map availability / plugin state ----
        if not globals().get("_MAP_OK", True) or getattr(self, "map", None) is None:
            self._log_async("[Map] Map widget not available.")
            return

        # ---- 1) Normalize color mapping and friendly legend labels ----
        # Fallback to globals if your app defines these constants centrally.
        COLOR_NONV_FALLBACK = str(globals().get("COLOR_NONV", "#6d4c41")).lower()
        COLOR_VEG_FALLBACK = str(globals().get("COLOR_VEG", "#2ca02c")).lower()
        COLOR_UNK_FALLBACK = str(globals().get("COLOR_UNK", "#7f7f7f")).lower()

        # Normalize provided color map (lower-case hex).
        colors: Dict[str, str] = dict(class_colors or {})
        colors.setdefault("NON_VEG", COLOR_NONV_FALLBACK)

        def _normalize_hex(hx: str) -> str:
            """Return a lower-cased #rrggbb string; fall back to input if malformed."""
            if not isinstance(hx, str):
                return COLOR_UNK_FALLBACK
            hx = hx.strip().lower()
            if not hx.startswith("#"):
                hx = f"#{hx}"
            # Basic sanity: expect 7 chars (# + 6 hex)
            return hx if len(hx) == 7 else COLOR_UNK_FALLBACK

        colors = {k: _normalize_hex(v) for k, v in colors.items()}

        friendly_name = {
            "#1f77b4": "blue", "#ff7f0e": "orange", "#2ca02c": "green", "#d62728": "red",
            "#9467bd": "purple", "#8c564b": "brown", "#e377c2": "pink", "#7f7f7f": "gray",
            "#bcbd22": "olive", "#17becf": "teal",
            COLOR_NONV_FALLBACK: "brown",  # non-vegetation
            COLOR_VEG_FALLBACK: "green",  # generic vegetation
            COLOR_UNK_FALLBACK: "gray",  # unknown
        }

        def _color_name(hx: str) -> str:
            """Map a hex code to a human-friendly color name if known."""
            return friendly_name.get(_normalize_hex(hx), _normalize_hex(hx))

        # ---- 2) Input validation, CRS normalization, and stats logging ----
        if gdf is None or len(gdf) == 0:
            self.clear_map_polygons()
            self._log_async("[Map] Empty GeoDataFrame; nothing to draw.")
            return

        # Reproject to WGS84 if CRS is known and not EPSG:4326.
        try:
            if getattr(gdf, "crs", None) is not None and str(gdf.crs).lower() not in (
            "epsg:4326", "wgs84", "ogc:crs84"):
                self._log_async(f"[Map] Reprojecting layer from {gdf.crs} to EPSG:4326 for display.")
                gdf = gdf.to_crs(epsg=4326)
        except Exception as ex:
            self._log_async(f"[Map] CRS check/reproject failed: {ex}. Proceeding as-is.")

        # Per-class counts
        if label_field in gdf.columns:
            try:
                counts = gdf[label_field].astype("string").value_counts(dropna=False).to_dict()
            except Exception:
                counts = gdf[label_field].astype(str).value_counts(dropna=False).to_dict()
        else:
            counts = {"unknown": len(gdf)}

        self._log_async("[Predict] per-class counts: " + " | ".join(f"{k}:{v}" for k, v in counts.items()))
        self._log_async("Legend (class → color):")
        for cls, hexc in colors.items():
            self._log_async(f"{cls}: {_color_name(hexc)}")

        # ---- 3) Prepare drawing: clear old layer & compute caps ----
        self.clear_map_polygons()

        # Safety cap on number of polygon rings drawn (keeps UI smooth).
        try:
            cap = int(getattr(self, "max_rings").get())  # Tk variable in the original app
        except Exception:
            cap = 15_000
        cap = max(1, cap)

        # Cap for ring point count; keep generous but bounded.
        try:
            max_pts = int(getattr(self, "max_ring_points", 4000))
        except Exception:
            max_pts = 4000
        max_pts = max(100, min(max_pts, 50_000))

        drawn = 0  # number of exterior rings actually drawn

        # ---- 4) Geometry utilities ----
        def _iter_polygons(geom: Any) -> Iterator["Polygon"]:
            """
            Yield valid Polygon parts from an arbitrary geometry.
            - Repairs invalids via buffer(0) when possible.
            - Unpacks MultiPolygon/GeometryCollection gracefully.
            """
            if geom is None:
                return
            g = geom

            # Attempt to fix invalid geometries (common in vector data)
            try:
                if hasattr(g, "is_valid") and not g.is_valid:
                    g = g.buffer(0)
            except Exception:
                pass

            # Normalize collections into polygons
            if isinstance(g, Polygon):
                yield g
            elif isinstance(g, MultiPolygon):
                for part in g.geoms:
                    if isinstance(part, Polygon):
                        yield part
            elif isinstance(g, GeometryCollection):
                for part in g.geoms:
                    if isinstance(part, Polygon):
                        yield part
                    elif isinstance(part, MultiPolygon):
                        for sub in part.geoms:
                            if isinstance(sub, Polygon):
                                yield sub

        def _decimate_ring(xy: Iterable[Tuple[float, float]]) -> list[Tuple[float, float]]:
            """
            Down-sample a long ring by stride to a ceiling of `max_pts`.
            Preserves ring closure (first == last).
            """
            pts = [(float(x), float(y)) for (x, y, *_) in xy]
            if not pts:
                return pts

            # Ensure closure before decimation so we don't accidentally drop the last = first point.
            if pts[0] != pts[-1]:
                pts.append(pts[0])

            n = len(pts)
            if n <= max_pts:
                return pts

            step = max(1, math.ceil(n / max_pts))
            dec = pts[::step]

            # Re-ensure closure after slicing
            if dec and dec[0] != dec[-1]:
                dec.append(dec[0])
            return dec

        def _draw_one(poly: "Polygon", color_hex: str) -> None:
            """
            Draw a single polygon's exterior ring.
            - Holes are intentionally ignored (TkinterMapView limitation).
            - Coordinates are flipped to (lat, lon).
            """
            nonlocal drawn
            if drawn >= cap:
                return

            try:
                # Exterior coordinates (lon, lat)
                exterior_xy = list(poly.exterior.coords)
                ring_xy = _decimate_ring(exterior_xy)

                # Flip to (lat, lon) for the widget
                ring_latlon = [(y, x) for (x, y) in ring_xy]

                poly_handle = self.map.set_polygon(
                    ring_latlon,
                    outline_color=_normalize_hex(color_hex),
                    fill_color=_normalize_hex(color_hex),
                    border_width=2
                )
                self._drawn_polys.append(poly_handle)
                drawn += 1
            except Exception as ex:
                self._log_async(f"[Map] draw error: {ex}")

        # ---- 5) Iterate rows, resolve color, and draw ----
        for _, row in gdf.iterrows():
            # Resolve label → color, defaulting to unknown if unmapped/missing
            lbl = "unknown"
            if label_field in gdf.columns:
                try:
                    lbl = str(row[label_field])
                except Exception:
                    lbl = "unknown"

            color_hex = colors.get(lbl, COLOR_UNK_FALLBACK)

            geom = getattr(row, "geometry", None)
            try:
                for poly in _iter_polygons(geom):
                    if drawn >= cap:
                        break
                    _draw_one(poly, color_hex)
            except Exception as ex:
                self._log_async(f"[Map] geometry handling error: {ex}")

            if drawn >= cap:
                break

        # ---- 6) Fit map view to layer extent & report ----
        try:
            if len(gdf) > 0:
                minx, miny, maxx, maxy = gdf.total_bounds  # (lon_min, lat_min, lon_max, lat_max)
                # TkinterMapView expects ((north, west), (south, east)) = ((max_lat, min_lon), (min_lat, max_lon))
                self.map.fit_bounding_box((maxy, minx), (miny, maxx))
        except Exception as ex:
            self._log_async(f"[Map] fit view failed: {ex}")

        self._log_async(f"Drawn rings: {drawn}")

    def crop_train_eval(self):
        """
        70/30 training for multi-class crop MLP + metrics + save artifacts.
        Hardened against tiny classes and unstable splits.
        """
        key = "crop_train"
        self._set_busy(self.btn_crop_build, key, True, "Training…")
        self.progress["value"] = 10

        def worker():
            try:
                # --- EE init only if needed by selected feature mode ----------------
                feat_mode = self.crop_feat_mode.get().strip()
                need_ee = False
                if need_ee:
                    ok = ee_initialize(
                        self.ee_project.get().strip(),
                        self.sa_email.get().strip() or None,
                        self.sa_json.get().strip() or None,
                        interactive_fallback=False,
                        log=self._log_async
                    )
                    if not ok:
                        return self.after(0, lambda: messagebox.showerror("EE", "Not signed in."))

                # --- Load labeled polygons -----------------------------------------
                g = self._load_all_labeled()
                if g is None or g.empty:
                    return self.after(0, lambda: messagebox.showwarning(
                        "No labeled data",
                        "No rows with a valid label were found.\nCheck the label column selection."
                    ))

                # --- Build features (NO NDVI filter here) ---------------------------
                # do NOT pass an internal NDVI filter; prediction uses an external NDVI gate.
                X, y_int, pids, feat_names, class_names, le = build_crop_feature_matrix(
                    g, Path(self.out_dir.get()), int(self.ee_year.get()),
                    int(self.ee_tilescale.get()), feat_mode,
                    ndvi_filter=None,  # <--- critical change
                    log=self._log_async
                )
                self._log_async(f"[Crop-MLP] samples={len(X)} | features={len(feat_names)} | classes={class_names}")

                # --- Guard 1: need >= 2 classes overall -----------------------------
                if len(np.unique(y_int)) < 2:
                    return self.after(0, lambda: messagebox.showwarning(
                        "Not enough classes",
                        "Only one class is present after preprocessing.\nAdd more labeled data and try again."
                    ))

                # --- Guard 2: each class must have >= 2 samples for stratified split
                # drop ultra-rare classes (<2) to avoid Stratified errors
                counts = np.bincount(y_int, minlength=len(class_names))
                rare_ids = np.where(counts < 2)[0]  # classes with 0 or 1 sample

                if rare_ids.size > 0:
                    rare_names = [class_names[i] for i in rare_ids]
                    self._log_async(f"[Sanity] Dropping tiny classes (<2 samples): {rare_names}")

                    # build keep mask
                    keep_mask = ~np.isin(y_int, rare_ids)
                    # apply mask to X / y / pids
                    X = X[keep_mask]
                    y_int = y_int[keep_mask]
                    pids = [pid for pid, k in zip(pids, keep_mask) if k]
                    # shrink class_names and remap y to [0..C'-1]
                    kept_ids = sorted(np.unique(y_int))
                    new_id_map = {old: new for new, old in enumerate(kept_ids)}
                    y_int = np.array([new_id_map[v] for v in y_int], dtype=int)
                    class_names = [class_names[i] for i in kept_ids]

                    # re-check after dropping
                    if len(np.unique(y_int)) < 2:
                        return self.after(0, lambda: messagebox.showwarning(
                            "Not enough classes after drop",
                            "After removing tiny classes (<2 samples), fewer than two classes remain.\n"
                            "Add more labeled data or merge classes, then try again."
                        ))

                # Paths
                mp = Path(self.crop_model_path.get())
                sp = Path(self.crop_scaler_path.get())
                meta_path = Path(str(mp) + ".meta.json")

                # --- Train + evaluate ------------------------------------------------
                # training function should now receive a clean y with all classes >=2
                result = train_evaluate_crop_mlp(
                    X, y_int, class_names, feat_names, mp, sp, meta_path,
                    use_gpu_tf=self.tf_use_gpu.get(), test_size=0.30, log=self._log_async
                )
                self._show_metrics_inpage(result, class_names)
                self._log_async(f"[Crop-MLP] acc={result['acc']:.2%} | f1_macro={result['f1_macro']:.2%}")

                # --- Build a small test-geo to draw ---------------------------------
                # do a deterministic stratified split; safe because all classes >=2
                from sklearn.model_selection import StratifiedShuffleSplit
                splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.30, random_state=42)
                train_idx, test_idx = next(splitter.split(X, y_int))
                test_pids = [pids[i] for i in test_idx]
                g_test = g[g["poly_id"].isin(test_pids)].copy()

                # --- Predict on test subset for visualization -----------------------
                g_pred = predict_crops_with_mlp(
                    g_test, Path(self.out_dir.get()),
                    Path(self.crop_model_path.get()), Path(self.crop_scaler_path.get()),
                    feat_mode, int(self.ee_year.get()), int(self.ee_tilescale.get()),
                    ndvi_filter=None,  # <--- keep prediction free of internal NDVI filter
                    log=self._log_async
                )

                # --- Save predictions ------------------------------------------------
                out_dir = Path(self.out_dir.get())
                ts = pd.Timestamp.utcnow().strftime("%Y%m%dT%H%M%SZ")
                try:
                    g_pred.to_file(out_dir / f"crop_test_pred_{ts}.geojson", driver="GeoJSON")
                    g_pred.to_file(out_dir / f"crop_test_pred_{ts}.gpkg", driver="GPKG")
                    self._log_async(f"[SAVE] crop_test_pred_{ts}.* written to {out_dir}")
                except Exception as e:
                    self._log_async(f"[SAVE] {e}")

                # --- Draw on map -----------------------------------------------------
                pal = self._class_palette(class_names)
                self._draw_gdf_multiclass(g_pred.rename(columns={"pred_crop": "final_label"}), "final_label", pal)

                self.after(0, lambda: messagebox.showinfo(
                    "Crop-MLP",
                    f"Training done.\nAccuracy: {result['acc']:.2%}\nF1-macro: {result['f1_macro']:.2%}\n"
                    f"Artifacts:\n  {mp}\n  {sp}\n  {mp}.meta.json\n  {mp}.metrics.json\n  {mp}.cm.png"
                ))

            except Exception as e:
                self._log_async(f"[Crop-MLP] error: {e}")
                self.after(0, lambda m=str(e): messagebox.showerror("Crop-MLP", m))
            finally:
                self.after(0, lambda: self._set_busy(self.btn_crop_build, key, False))
                self.after(0, lambda: self.progress.configure(value=100))

        threading.Thread(target=worker, daemon=True).start()

    def crop_predict_map(self):
        """
        NDVI-gated crop inference (Page 3)
        ----------------------------------
        Pipeline (robust, UI-synced):
          1) Resolve feature mode (alpha | alpha4 | alpha4x64 | gali | concat).
             - "auto" falls back to Page-2 selection via self._provider_mode().
          2) Compute per-polygon mean NDVI (Sentinel-2 tabular) for selected year.
          3) Split polygons: VEGETATION (ndvi >= thr), NON_VEG (ndvi < thr), UNKNOWN (NaN).
          4) Run crop-MLP ONLY on VEGETATION; do NOT apply any internal NDVI filter.
          5) Merge results, assign colors (NON_VEG=brown, UNKNOWN=gray, crops=palette),
             save GeoJSON/GPKG, and paint the map with a readable legend.
        """
        from pathlib import Path
        import json
        import numpy as np
        import pandas as pd

        key = "crop_predict"
        # Use a safe button handle if the Page-3 predict button exists; otherwise no busy styling.
        btn = getattr(self, "btn_crop_predict", None)
        if btn:
            self._set_busy(btn, key, True, "Predicting…")
        try:
            self.progress["value"] = 10
        except Exception:
            pass

        def _resolve_feature_mode() -> str:
            """
            Resolve canonical feature mode used by the predictor.
            Accepts: "alpha", "alpha4", "alpha4x64", "gali", "concat".
            "auto" (or empty) → use Page-2 provider mode via self._provider_mode().
            """
            try:
                mode = (self.crop_feat_mode.get() or "").strip().lower()
            except Exception:
                mode = ""
            if mode in ("", "auto", "automatic"):
                try:
                    # Fall back to Page-2's normalized provider mode
                    mode = self._provider_mode()
                except Exception:
                    mode = "alpha"
            # Final safety
            if mode not in {"alpha", "alpha4", "alpha4x64", "gali", "concat"}:
                mode = "alpha"
            return mode

        def _ensure_model_paths_for(mode: str) -> None:
            """
            Make sure crop model/scaler paths are set consistently for the chosen mode.
            If user hasn't trained yet, this points to the expected filenames under /models.
            """
            mapping = {
                "alpha": ("mlp_crop_alpha.keras", "scaler_crop_alpha.pkl"),
                "alpha4": ("mlp_crop_alpha4.keras", "scaler_crop_alpha4.pkl"),
                "alpha4x64": ("mlp_crop_alpha4x64.keras", "scaler_crop_alpha4x64.pkl"),  # 320-D (64 + 256)
                "gali": ("mlp_crop_gali.keras", "scaler_crop_gali.pkl"),
                "concat": ("mlp_crop_concat.keras", "scaler_crop_concat.pkl"),
            }
            try:
                models_dir = Path(self.out_dir.get()) / "models"
                models_dir.mkdir(parents=True, exist_ok=True)
                m, s = mapping.get(mode, mapping["concat"])
                # If paths are empty or point to missing files, update to defaults (non-destructive if they exist)
                if not self.crop_model_path.get():
                    self.crop_model_path.set(str(models_dir / m))
                if not self.crop_scaler_path.get():
                    self.crop_scaler_path.set(str(models_dir / s))
            except Exception:
                pass

        def worker():
            try:
                # ----------------------------- 0) Data load -----------------------------
                g = self._load_all_labeled()
                if g is None or g.empty:
                    return self.after(0, lambda: messagebox.showwarning("No data", "Add at least one labeled ZIP."))
                if "poly_id" not in g.columns:
                    g = g.copy()
                    g["poly_id"] = np.arange(len(g), dtype=int)

                # ---------------- 1) Resolve feature mode + model/scaler paths ----------
                mode = _resolve_feature_mode()
                _ensure_model_paths_for(mode)

                mp = Path(self.crop_model_path.get())
                sp = Path(self.crop_scaler_path.get())
                if (not mp.exists()) or (not sp.exists()):
                    # Fail early with a clear message if the model is missing
                    return self.after(
                        0,
                        lambda: messagebox.showwarning(
                            "Model not found",
                            f"Model/Scaler not found for mode='{mode}'.\n"
                            f"Model:  {mp}\nScaler: {sp}\n\nPlease train on Page 3 first."
                        )
                    )

                # ----------------------- 2) Earth Engine init ---------------------------
                ok = ee_initialize(
                    self.ee_project.get().strip(),
                    self.sa_email.get().strip() or None,
                    self.sa_json.get().strip() or None,
                    interactive_fallback=False,
                    log=self._log_async
                )
                if not ok:
                    return self.after(0, lambda: messagebox.showerror("EE", "Not signed in."))

                try:
                    year = int(self.ee_year.get())
                except Exception:
                    year = 2024
                try:
                    tilescale = int(self.ee_tilescale.get())
                except Exception:
                    tilescale = 4
                try:
                    ndvi_thr = float(self.ndvi_thresh.get())
                except Exception:
                    ndvi_thr = 0.30

                # -------------------- 3) Compute NDVI & split masks --------------------
                s2tab = compute_s2_tabular_features(g, year, tile_scale=tilescale, log=self._log_async)
                g = g.merge(s2tab[["poly_id", "ndvi_mean"]], on="poly_id", how="left")

                ndvi = g["ndvi_mean"].astype(float)
                mask_unknown = ndvi.isna()
                mask_veg = (~mask_unknown) & (ndvi >= ndvi_thr)
                mask_nonveg = (~mask_unknown) & (ndvi < ndvi_thr)

                n_tot = int(len(g))
                n_veg = int(mask_veg.sum())
                n_nonveg = int(mask_nonveg.sum())
                n_unk = int(mask_unknown.sum())
                self._log_async(
                    f"[NDVI gate] total={n_tot} | veg={n_veg} | non_veg={n_nonveg} | unknown={n_unk} (thr={ndvi_thr})"
                )

                # -------------------- 4) Predict only on vegetation --------------------
                g_veg = g[mask_veg].copy()
                if len(g_veg) > 0:
                    g_pred_veg = predict_crops_with_mlp(
                        g_veg,
                        Path(self.out_dir.get()),
                        mp,
                        sp,
                        feature_mode=mode,  # <-- honored modes: alpha | alpha4 | alpha4x64 | gali | concat
                        year=year,
                        tile_scale=tilescale,
                        ndvi_filter=None,  # <-- keep predictor free of internal NDVI gating
                        log=self._log_async
                    )
                    # Normalize output columns
                    g_pred_veg["final_label"] = g_pred_veg.get("pred_crop", None).astype(str)
                    if "pred_conf" in g_pred_veg.columns:
                        g_pred_veg["pred_conf"] = g_pred_veg["pred_conf"].astype(float)
                    else:
                        g_pred_veg["pred_conf"] = np.nan
                else:
                    # Empty vegetated subset; keep schema consistent
                    g_pred_veg = g_veg.copy()
                    g_pred_veg["pred_crop"] = None
                    g_pred_veg["pred_conf"] = np.nan
                    g_pred_veg["final_label"] = None

                # ---------------------- 5) Build NON_VEG/UNKNOWN -----------------------
                g_nonveg = g[mask_nonveg].copy()
                g_nonveg["pred_crop"] = None
                g_nonveg["pred_conf"] = 1.0
                g_nonveg["final_label"] = "NON_VEG"

                g_unknown = g[mask_unknown].copy()
                g_unknown["pred_crop"] = None
                g_unknown["pred_conf"] = np.nan
                g_unknown["final_label"] = "UNKNOWN"

                # ---------------------- 6) Merge, save artifacts -----------------------
                g_out = pd.concat([g_pred_veg, g_nonveg, g_unknown], ignore_index=True)
                g_out = g_out.sort_values("poly_id").reset_index(drop=True)

                out_dir = Path(self.out_dir.get())
                out_dir.mkdir(parents=True, exist_ok=True)
                ts = pd.Timestamp.utcnow().strftime("%Y%m%dT%H%M%SZ")
                try:
                    g_out.to_file(out_dir / f"crop_pred_{mode}_{ts}.geojson", driver="GeoJSON")
                    g_out.to_file(out_dir / f"crop_pred_{mode}_{ts}.gpkg", driver="GPKG")
                    self._log_async(f"[SAVE] crop_pred_{mode}_{ts}.* written to {out_dir}")
                except Exception as e:
                    self._log_async(f"[SAVE] {e}")

                # ---------------------- 7) Palette & legend ----------------------------
                # Try to read class names from the model's meta; fall back to predictions.
                try:
                    meta = json.loads(Path(str(mp) + ".meta.json").read_text(encoding="utf-8"))
                    class_names = list(map(str, meta.get("classes", [])))
                except Exception:
                    # Use observed classes (exclude NON_VEG/UNKNOWN/None)
                    obs = (
                        g_pred_veg.get("final_label", pd.Series([], dtype=str))
                        .dropna().astype(str).unique().tolist()
                    )
                    class_names = sorted(obs)

                # Build palette, ensuring NON_VEG & UNKNOWN exist (brown/gray).
                pal = _palette_with_nonveg_unknown(class_names)

                # Per-class counts and friendly legend
                cnts = g_out["final_label"].value_counts(dropna=False).to_dict()
                self._log_async("[Predict] per-class counts: " + " | ".join(f"{k}:{v}" for k, v in cnts.items()))
                legend_lines = ["Legend (class → color):"] + [f"{k}: {_color_name(v)}" for k, v in pal.items()]
                self._log_async("\n".join(legend_lines))

                # ---------------------- 8) Draw on map ---------------------------------
                self._draw_gdf_multiclass(g_out, "final_label", pal)

            except Exception as e:
                self._log_async(f"[Crop-MLP] error: {e}")
                self.after(0, lambda m=str(e): messagebox.showerror("Crop-MLP", m))
            finally:
                # Always release busy state & complete progress safely
                if btn:
                    self.after(0, lambda: self._set_busy(btn, key, False))
                try:
                    self.after(0, lambda: self.progress.configure(value=100))
                except Exception:
                    pass

        threading.Thread(target=worker, daemon=True).start()

    # ============================ EE Sign-in ============================

    def sign_in_google(self):
        """
        Always trigger an interactive browser sign-in (even if cached creds exist).
        Uses multiple ports; falls back to console-mode and still opens the browser.
        """
        key = "signin"
        self._set_busy(self.btn_sign, key, True, "Opening browser…")

        def worker():
            ok = ee_force_browser_auth(self.ee_project.get().strip(), force_reset=True, log=self._log_async)
            self.after(0, lambda: self._set_busy(self.btn_sign, key, False))
            if ok:
                self.after(0, lambda: messagebox.showinfo("AlphaEarth", "Signed in. EE initialized."))
            else:
                self.after(0, lambda: messagebox.showerror(
                    "AlphaEarth",
                    "Sign-in did not complete.\n"
                    "If a firewall/antivirus blocks localhost, complete the console flow:\n"
                    "  1) Approve in the opened browser.\n"
                    "  2) Run the SECOND command that the CLI printed (with the code).\n"
                    "Then click 'Test EE (cached)'."
                ))

        threading.Thread(target=worker, daemon=True).start()

    def test_ee_cached(self):
        """
        Try to initialize EE using cached credentials (service account optional).
        Gives a clear error message if the dataset cannot be accessed.
        """
        key = "testcached"
        self._set_busy(self.btn_test_cache, key, True)

        def worker():
            self._log_async("[EE] Testing with cached creds…")
            ok = ee_initialize(
                self.ee_project.get().strip(),
                self.sa_email.get().strip() or None,
                self.sa_json.get().strip() or None,
                interactive_fallback=False,
                log=self._log_async
            )
            if not ok:
                self.after(0, lambda: self._set_busy(self.btn_test_cache, key, False))
                return self.after(0, lambda: messagebox.showerror(
                    "AlphaEarth", "EE init (cached) failed. Click 'Sign in (Google)'."
                ))
            try:
                img = aef_image_for_year(self.ee_year.get())
                with EE_LOCK:
                    _ = img.bandNames().size().getInfo()
                self.after(0, lambda: messagebox.showinfo("AlphaEarth", "EE connection OK (cached)."))
            except Exception as e:
                e_msg = str(e)
                self._log_async(f"[EE] dataset test error: {e_msg}")
                self.after(0, lambda msg=e_msg: messagebox.showerror("AlphaEarth", f"Dataset error: {msg}"))
            finally:
                self.after(0, lambda: self._set_busy(self.btn_test_cache, key, False))

        threading.Thread(target=worker, daemon=True).start()

    def test_ee_force_browser(self):
        """
        Force a browser-based sign-in, then ping the dataset to verify access.
        """
        key = "testbrowser"
        self._set_busy(self.btn_test_browser, key, True, "Signing in…")

        def worker():
            self._log_async("[EE] Browser sign-in (forced)…")
            ok = ee_force_browser_auth(self.ee_project.get().strip(), force_reset=True, log=self._log_async)
            if not ok:
                self.after(0, lambda: self._set_busy(self.btn_test_browser, key, False))
                return self.after(0, lambda: messagebox.showerror("AlphaEarth", "Sign-in failed."))

            try:
                img = aef_image_for_year(self.ee_year.get())
                with EE_LOCK:
                    _ = img.bandNames().size().getInfo()
                self.after(0, lambda: messagebox.showinfo("AlphaEarth", "EE connection OK (browser)."))
            except Exception as e:
                e_msg = str(e)
                self._log_async(f"[EE] dataset test error: {e_msg}")
                self.after(0, lambda msg=e_msg: messagebox.showerror("AlphaEarth", f"Dataset error: {msg}"))
            finally:
                self.after(0, lambda: self._set_busy(self.btn_test_browser, key, False))

        threading.Thread(target=worker, daemon=True).start()

    def test_galileo(self):
        """
        Light-check that a local Galileo encoder can be constructed.
        Also probes output dimension via a dummy forward (handled in __init__).
        """
        key = "testgali"
        self._set_busy(self.btn_test_gali, key, True)

        def worker():
            try:
                enc = GalileoEncoder(
                    Path(self.gali_weights.get()),
                    device=self.gali_device.get(),
                    year=int(self.ee_year.get()),
                    chip_px=128,
                    log=self._log_async
                )
                ok, err = enc.ok()
            finally:
                self.after(0, lambda: self._set_busy(self.btn_test_gali, key, False))

            if ok:
                self.after(0, lambda: messagebox.showinfo("Galileo",
                                                          f"OK. Device={enc.device} | out_dim≈{enc.expected_dim}"))
            else:
                self.after(0, lambda msg=err: messagebox.showerror("Galileo", f"Failed to load: {msg}"))

        threading.Thread(target=worker, daemon=True).start()

    # ============================ Ingest / Run ===========================

    def add_zips(self):
        """Append selected ZIP files to the listbox, skipping duplicates."""
        paths = filedialog.askopenfilenames(
            title="Select ZIP files", filetypes=[("ZIP files", "*.zip")]
        )
        existing = set(self.listbox.get(0, "end"))
        for p in paths:
            if p not in existing:
                self.listbox.insert("end", p)

    def clear_list(self):
        """Clear the ZIP listbox."""
        self.listbox.delete(0, "end")

    def pick_outdir(self):
        """Select and ensure the output directory exists."""
        d = filedialog.askdirectory(title="Select output folder")
        if d:
            self.out_dir.set(d)
            Path(d).mkdir(parents=True, exist_ok=True)

    def _worker_job(self, zips: List[str]):
        """
        Background job: ingest all ZIPs, merge polygons, assign poly_id,
        draw on the map, and move progress to 50%.
        """
        try:
            dfs = []
            n = len(zips)
            for i, zp in enumerate(zips, start=1):
                self._log_async(f"=== {Path(zp).name} ===")
                try:
                    gdf = load_polygons_from_zip(
                        Path(zp),
                        min_area_m2=float(self.min_area_m2.get() or 0),
                        simplify_m=float(self.simplify_m.get() or 2),
                    )
                    if len(gdf):
                        dfs.append(gdf)
                    self._log_async(f"  -> {len(gdf)} polygon(s)")
                except Exception as e:
                    self._log_async(f"  ingest error: {e}")

                # smooth-ish progress towards 50%
                self.after(0, lambda v=int(i * 50 / max(1, n)): self.progress.configure(value=v))

            if not dfs:
                self._log_async("No polygons ingested.")
                return

            gdf_all = pd.concat(dfs, ignore_index=True).set_crs(4326, allow_override=True)
            gdf_all["poly_id"] = np.arange(len(gdf_all))
            self._gdf = gdf_all

            # Draw on the map in the main thread
            self.after(0, lambda: self._draw_gdf(gdf_all))
        finally:
            self.after(0, lambda: self.progress.configure(value=50))

    def run(self):
        """
        Entry point for page-1: ingest selected ZIP(s) in the background.
        """
        zips = list(self.listbox.get(0, "end"))
        if not zips:
            return self.after(0, lambda: messagebox.showwarning("No input", "Add at least one ZIP."))

        # Reset progress + log; ensure output path exists
        self.progress["value"] = 0
        self.log.delete("1.0", "end")
        Path(self.out_dir.get()).mkdir(parents=True, exist_ok=True)

        key = "run_ingest"
        self._set_busy(self.btn_run, key, True)

        def finisher():
            self._worker_job(zips)
            self.after(0, lambda: self._set_busy(self.btn_run, key, False))

        threading.Thread(target=finisher, daemon=True).start()

    # ==================== Embeddings (split buttons) ====================

    def compute_embeddings_both(self):
        """Compute AlphaEarth + Galileo embeddings (if toggled in UI)."""
        if self._gdf is None or len(self._gdf) == 0:
            return self.after(0, lambda: messagebox.showwarning("No polygons", "Load polygons first."))
        key = "embed_both"
        self._set_busy(self.btn_embed_both, key, True)
        self.progress["value"] = 50
        threading.Thread(
            target=lambda: self._compute_embeddings_worker(
                alpha=self.use_alpha.get(), gali=self.use_gali.get(), key=key
            ),
            daemon=True,
        ).start()

    def compute_embeddings_alpha_only(self):
        """Compute AlphaEarth embeddings only."""
        if self._gdf is None or len(self._gdf) == 0:
            return self.after(0, lambda: messagebox.showwarning("No polygons", "Load polygons first."))
        key = "embed_alpha"
        self._set_busy(self.btn_embed_alpha, key, True)
        self.progress["value"] = 50
        threading.Thread(
            target=lambda: self._compute_embeddings_worker(alpha=True, gali=False, key=key),
            daemon=True,
        ).start()

    def compute_embeddings_galileo_only(self):
        """Compute Galileo embeddings only."""
        if self._gdf is None or len(self._gdf) == 0:
            return self.after(0, lambda: messagebox.showwarning("No polygons", "Load polygons first."))
        key = "embed_gali"
        self._set_busy(self.btn_embed_gali, key, True)
        self.progress["value"] = 50
        threading.Thread(
            target=lambda: self._compute_embeddings_worker(alpha=False, gali=True, key=key),
            daemon=True,
        ).start()

    def _compute_embeddings_worker(self, alpha: bool, gali: bool, key: str):
        """
        Background worker that computes embeddings for the selected providers.

        What's improved here:
        - Hardened parameter parsing & clamping (tileScale / chunk_size / simplify).
        - Single AlphaEarth image build reused by both 64-D and 4×(64-D) stats.
        - Clearer logging and progress updates (+ elapsed times).
        - Early guards (no polygons / no provider / missing Galileo weights / EE auth).
        - Post-run integrity checks + optional spreadsheet export (CSV/Excel) for
          BOTH 64-D and 256-D alpha embeddings if they exist (best-effort).
        - Robust final summary with delta file counts.
        """

        # ---------- local helpers (no UI calls inside these) ----------
        from pathlib import Path
        import time
        import numpy as np
        import pandas as pd

        def _count_files(pat: str) -> int:
            """Count files under Outputs/embeddings matching a glob pattern."""
            emb_dir = Path(self.out_dir.get()) / "embeddings"
            try:
                return len(list(emb_dir.glob(pat)))
            except Exception:
                return 0

        def _alpha4_enabled() -> bool:
            """Return True if the UI has a 'use_alpha4' toggle and it's enabled."""
            try:
                return bool(getattr(self, "use_alpha4").get())
            except Exception:
                return False

        def _export_alpha_spreadsheets(out_dir: Path, log):
            """
            Best-effort exporters for consolidated spreadsheets after a run.
            - 64-D mean vectors  → alphaearth_all_embeddings.{csv,xlsx}
            - 256-D concatenated → alphaearth_all_embeddings_alpha4_256.{csv,xlsx}
            Will not raise; useful if inner functions already exported CSVs but we still
            want to ensure .xlsx parity (user requested Excel for 256-D too).
            """
            emb_dir = out_dir / "embeddings"
            emb_dir.mkdir(parents=True, exist_ok=True)

            # 1) 64-D mean (alpha_*.npy)
            try:
                alpha_files = sorted(emb_dir.glob("alpha_*.npy"))
                if alpha_files:
                    rows = []
                    for f in alpha_files:
                        try:
                            pid = int(f.stem.split("_")[1])
                        except Exception:
                            continue
                        vec = np.load(f).astype("float32")
                        if vec.shape[0] == 64:
                            rows.append({"poly_id": pid, **{f"A{i:02d}": float(vec[i]) for i in range(64)}})
                    if rows:
                        df64 = pd.DataFrame(rows).sort_values("poly_id")
                        csv64 = emb_dir / "alphaearth_all_embeddings.csv"
                        xlsx64 = emb_dir / "alphaearth_all_embeddings.xlsx"
                        try:
                            df64.to_csv(csv64, index=False)
                        except Exception:
                            pass
                        try:
                            # Requires xlsxwriter; will fall back silently if missing
                            with pd.ExcelWriter(xlsx64, engine="xlsxwriter") as w:
                                df64.to_excel(w, sheet_name="embeddings64", index=False)
                        except Exception:
                            # If xlsxwriter is not available, we simply skip .xlsx
                            log and log("[AEF] Excel export (64-D) skipped (xlsxwriter missing?).")
            except Exception as e:
                log and log(f"[AEF] Post 64-D export failed: {e}")

            # 2) 256-D concatenated (alpha4_*.npy)
            try:
                alpha4_files = sorted(emb_dir.glob("alpha4_*.npy"))
                # Filter out the per-part files (alpha4_mean_*, alpha4_mid_*, ...)
                alpha4_full = [f for f in alpha4_files
                               if f.stem.startswith("alpha4_")
                               and not any(
                        f.stem.startswith(p) for p in ("alpha4_mean_", "alpha4_mid_", "alpha4_q1_", "alpha4_q3_"))]
                if alpha4_full:
                    rows = []
                    for f in alpha4_full:
                        try:
                            pid = int(f.stem.split("_")[1])
                        except Exception:
                            continue
                        vec = np.load(f).astype("float32")
                        if vec.shape[0] == 256:
                            rows.append({"poly_id": pid, **{f"A4_{i:03d}": float(vec[i]) for i in range(256)}})
                    if rows:
                        df256 = pd.DataFrame(rows).sort_values("poly_id")
                        csv256 = emb_dir / "alphaearth_all_embeddings_alpha4_256.csv"
                        xlsx256 = emb_dir / "alphaearth_all_embeddings_alpha4_256.xlsx"
                        try:
                            df256.to_csv(csv256, index=False)
                        except Exception:
                            pass
                        try:
                            with pd.ExcelWriter(xlsx256, engine="xlsxwriter") as w:
                                df256.to_excel(w, sheet_name="embeddings256", index=False)
                        except Exception:
                            log and log("[AEF4] Excel export (256-D) skipped (xlsxwriter missing?).")
            except Exception as e:
                log and log(f"[AEF4] Post 256-D export failed: {e}")

        # ------------------------- early guards -------------------------
        try:
            # 0) Polygons required
            if self._gdf is None or len(self._gdf) == 0:
                self.after(0, lambda: messagebox.showwarning("No polygons", "Load polygons first."))
                return

            # 1) At least one provider must be requested
            if not (alpha or gali):
                self._log_async("[EMB] Nothing to do: both providers are disabled.")
                return

            # 2) Output folder
            out_dir = Path(self.out_dir.get())
            out_dir.mkdir(parents=True, exist_ok=True)

            # 3) Parse tunables safely (with sane clamps)
            try:
                tile_scale = int(self.ee_tilescale.get())
            except Exception:
                tile_scale = 4
            tile_scale = max(1, min(16, tile_scale))

            try:
                chunk_size = int(self.ee_chunk.get())
            except Exception:
                chunk_size = 250
            chunk_size = max(25, min(2000, chunk_size))

            try:
                simplify_m = float(self.simplify_m.get() or 2.0)
            except Exception:
                simplify_m = 2.0
            simplify_m = max(0.0, simplify_m)

            # Counters & timers for summary
            saved_alpha_mean = 0
            saved_alpha4 = 0
            saved_gali = 0

            # Pre-run counts (useful to detect duplicate runs)
            pre_a = _count_files("alpha_*.npy")
            pre_a4 = _count_files("alpha4_*.npy")
            pre_g = _count_files("gali_*.npy")

            # Smooth base progress (page-1 ingest likely ended near 50)
            self.after(0, lambda: self.progress.configure(value=55))

            # --------------- EE initialization (required for Alpha & Galileo) ---------------
            need_ee = bool(alpha or gali)  # Galileo fetches S2 chips via EE
            if need_ee:
                ok = ee_initialize(
                    self.ee_project.get().strip(),
                    self.sa_email.get().strip() or None,
                    self.sa_json.get().strip() or None,
                    interactive_fallback=False,
                    log=self._log_async,
                )
                if not ok:
                    # Without EE we cannot proceed for either provider
                    self._log_async("[EE] Initialization failed. Click 'Sign in (Google)' and retry.")
                    self.after(0, lambda: messagebox.showerror(
                        "Earth Engine",
                        "EE is not initialized. Use 'Sign in (Google)' or 'Test EE (cached)'."
                    ))
                    return

            # ----------------------- AlphaEarth embeddings -----------------------
            if alpha:
                t0 = time.perf_counter()
                try:
                    # Build AlphaEarth annual image ONCE (used by mean and alpha4)
                    self._log_async("[AEF] Building annual image…")
                    img_alpha = aef_image_for_year(int(self.ee_year.get()))

                    # 1) Mean 64-D
                    self._log_async(
                        f"[AEF] Computing 64-D mean embeddings (chunked, tileScale={tile_scale}, chunk={chunk_size})…")
                    saved_alpha_mean = stream_save_alpha_embeddings(
                        img=img_alpha,
                        gdf=self._gdf.copy(),
                        out_dir=out_dir,
                        tile_scale=tile_scale,
                        simplify_m=simplify_m,
                        chunk_size=chunk_size,
                        log=self._log_async,
                    )
                    dt = time.perf_counter() - t0
                    self._log_async(f"[AEF] Saved mean embeddings: {saved_alpha_mean}  (in {dt:.1f}s)")

                    # 2) Optional 4×-stats 256-D (mean, median, q1, q3)
                    if _alpha4_enabled():
                        t1 = time.perf_counter()
                        self._log_async("[AEF4] Computing 4× stats (mean/median/q1/q3 → 256-D)…")
                        saved_alpha4 = stream_save_alpha_embeddings_four(
                            img=img_alpha,
                            gdf=self._gdf.copy(),
                            out_dir=out_dir,
                            tile_scale=tile_scale,
                            simplify_m=simplify_m,
                            chunk_size=chunk_size,
                            log=self._log_async,
                        )
                        dt4 = time.perf_counter() - t1
                        self._log_async(f"[AEF4] Saved full 4-stats embeddings: {saved_alpha4}  (in {dt4:.1f}s)")

                        # Optional: ensure both CSV and Excel exist post-run (best-effort)
                        _export_alpha_spreadsheets(out_dir, self._log_async)

                except Exception as e:
                    # Don't crash the worker; continue to Galileo if selected
                    self._log_async(f"[AEF] error: {e}")
                finally:
                    # Move progress forward regardless of success (keeps UI fluid)
                    self.after(0, lambda: self.progress.configure(value=72 if gali else 90))

            # ------------------------ Galileo embeddings ------------------------
            if gali:
                # Quick sanity: make sure a file/folder path is present AND exists
                raw_wpath = (self.gali_weights.get() or "").strip()
                wpath = Path(raw_wpath) if raw_wpath else None

                if not wpath or not wpath.exists():
                    self._log_async("[Galileo] Weights path is missing or does not exist.")
                    self.after(0, lambda: messagebox.showwarning(
                        "Galileo",
                        "Select a TorchScript file (encoder.pt) or a folder that contains it."
                    ))
                else:
                    try:
                        t0 = time.perf_counter()
                        self._log_async(
                            f"[Galileo] Encoding polygons with local encoder ({'file' if wpath.is_file() else 'dir'})…"
                        )

                        saved_gali = stream_save_galileo_embeddings(
                            gdf=self._gdf.copy(),
                            out_dir=out_dir,
                            weights_dir=wpath,  # file OR folder
                            device=self.gali_device.get(),  # 'cpu' | 'cuda'
                            year=int(self.ee_year.get()),  # S2 composite year
                            chip_px=128,  # expose in UI if needed
                            log=self._log_async,
                        )
                        dt = time.perf_counter() - t0
                        self._log_async(f"[Galileo] Saved embeddings: {saved_gali}  (in {dt:.1f}s)")
                    except Exception as e:
                        # Common cause: non-TorchScript checkpoint; the encoder logs a helpful message already.
                        self._log_async(f"[Galileo] error: {e}")
                    finally:
                        self.after(0, lambda: self.progress.configure(value=90))

            # -------------------------- final summary ---------------------------
            post_a = _count_files("alpha_*.npy")
            post_a4 = _count_files("alpha4_*.npy")
            post_g = _count_files("gali_*.npy")

            # Build concise summary only for the providers actually requested
            summary_lines = []
            if alpha:
                delta_a = max(0, post_a - pre_a)
                summary_lines.append(f"AlphaEarth (mean 64-D): {saved_alpha_mean} (Δfiles={delta_a})")
                if _alpha4_enabled():
                    delta_a4 = max(0, post_a4 - pre_a4)
                    summary_lines.append(f"AlphaEarth 4× (256-D): {saved_alpha4} (Δfiles={delta_a4})")
            if gali:
                delta_g = max(0, post_g - pre_g)
                summary_lines.append(f"Galileo (local): {saved_gali} (Δfiles={delta_g})")

            if summary_lines:
                self._log_async("[EMB] " + " | ".join(summary_lines))

            # Per-provider hints when nothing was saved
            def _hint_alpha() -> str:
                return (
                    "No AlphaEarth embeddings were saved.\n"
                    "Check: (1) EE sign-in & project access, (2) year/region coverage,\n"
                    "(3) reduceRegions limits — try smaller 'Chunk size' or larger 'tileScale'."
                )

            def _hint_alpha4() -> str:
                return (
                    "No AlphaEarth 4× embeddings were saved.\n"
                    "Make sure the 4× option is enabled and the AlphaEarth image is available for the selected year."
                )

            def _hint_gali() -> str:
                return (
                    "No Galileo embeddings were saved.\n"
                    "Common causes:\n"
                    "  • Provided file is a PyTorch checkpoint (state_dict), not TorchScript → convert to TorchScript.\n"
                    "  • Sentinel-2 chip fetch returned empty tiles (border/sea/heavy clouds) → try a larger chip or another year."
                )

            # Decide which hints to show
            show_hints = []
            if alpha and saved_alpha_mean == 0:
                show_hints.append(_hint_alpha())
            if alpha and _alpha4_enabled() and saved_alpha4 == 0:
                show_hints.append(_hint_alpha4())
            if gali and saved_gali == 0:
                show_hints.append(_hint_gali())

            if (saved_alpha_mean + saved_alpha4 + saved_gali) == 0:
                # Nothing saved across selected providers → unified warning
                self.after(0, lambda: messagebox.showwarning(
                    "No embeddings",
                    "\n\n".join(show_hints) if show_hints else "No embeddings were saved. Check logs for details."
                ))
            else:
                # At least something was saved → informative toast with target folder
                emb_dir = out_dir / "embeddings"
                msg = "Embeddings saved under:\n{}\n\n{}".format(emb_dir, "\n".join(summary_lines))
                self.after(0, lambda m=msg: messagebox.showinfo("Embeddings", m))

        finally:
            # ------------------- always finalize UI state -------------------
            self.after(0, lambda: self.progress.configure(value=100))
            if key == "embed_both":
                self.after(0, lambda: self._set_busy(self.btn_embed_both, key, False))
            elif key == "embed_alpha":
                self.after(0, lambda: self._set_busy(self.btn_embed_alpha, key, False))
            else:
                self.after(0, lambda: self._set_busy(self.btn_embed_gali, key, False))

    #------------------- Polygon-MLP (train / predict) --------------------

    def _provider_mode(self) -> str:

        # -------- 0) Read current UI value safely ---------------------------------
        try:
            raw = str(self.poly_feat_src.get())
        except Exception:
            raw = ""
        raw_norm = raw.strip().lower()

        # -------- 1) Canonical labels we show in the UI (for back-filling) --------
        label_for_mode = {
            "alpha": "AlphaEarth (mean 64-D)",
            "gali": "Galileo",
            "concat": "Concat (Alpha+Galileo)",
            "alpha4": "AlphaEarth (4× stats)",  # note: this uses the real '×' glyph
        }

        # -------- 2) Exact/alias mapping (covers common variants and typos) -------
        aliases = {
            # Alpha (mean 64-D)
            "alpha": "alpha",
            "alphaearth": "alpha",
            "alphaearth (mean 64-d)": "alpha",
            "alphaearth (mean 64-dim)": "alpha",
            "alphaearth (mean 64)": "alpha",

            # Galileo
            "galileo": "gali",
            "galileo (local)": "gali",
            "gali": "gali",

            # Concat
            "concat": "concat",
            "concat (alpha+galileo)": "concat",
            "a+g": "concat",
            "alpha+galileo": "concat",
            "contact": "concat",  # common typo

            # Alpha 4× (256-D)
            "alpha4": "alpha4",
            "alphaearth (4× stats)": "alpha4",  # multiplication sign
            "alphaearth (4x stats)": "alpha4",  # ASCII x
            "alphaearth 4x": "alpha4",
            "alphaearth 4×": "alpha4",
            "alphaearth (quartiles)": "alpha4",
        }

        mode = aliases.get(raw_norm)

        # -------- 3) Heuristic fallback by substrings (when alias misses) ---------
        if mode is None and raw_norm:
            txt = raw_norm
            if "concat" in txt or "alpha+gal" in txt or "+gal" in txt:
                mode = "concat"
            elif "gal" in txt:
                mode = "gali"
            elif "4×" in txt or "4x" in txt or "stats" in txt or "quartile" in txt:
                mode = "alpha4"
            elif "alpha" in txt:
                mode = "alpha"

        # -------- 4) Final default when empty/unknown -----------------------------
        if mode is None:
            # Prefer alpha4 when the pipeline is enabled; otherwise alpha
            try:
                prefer_alpha4 = bool(self.use_alpha4.get())
            except Exception:
                prefer_alpha4 = False
            mode = "alpha4" if prefer_alpha4 else "alpha"

            # Back-fill the combobox so the UI shows a valid, self-consistent label
            try:
                current_label = self.poly_feat_src.get()
            except Exception:
                current_label = ""
            target_label = label_for_mode.get(mode, "")
            if current_label != target_label and target_label:
                try:
                    self.poly_feat_src.set(target_label)
                    # Optional: let the rest of the UI react to the change if you bound handlers
                    # (we don't synthesize an actual virtual event here to avoid recursion)
                except Exception:
                    pass

            # Optional: log why we defaulted
            try:
                self._log_async(f"[UI] 'Features' selection was empty/unknown; defaulted to {mode} ({target_label}).")
            except Exception:
                pass

        # -------- 5) Safety net: ensure the return is one of the 4 known modes ---
        if mode not in {"alpha", "gali", "concat", "alpha4"}:
            # As a last resort, choose alpha and fix the UI label
            mode = "alpha"
            try:
                self.poly_feat_src.set(label_for_mode["alpha"])
            except Exception:
                pass
            try:
                self._log_async("[UI] Normalized features mode fell back to 'alpha'.")
            except Exception:
                pass

        return mode

    @staticmethod
    def _count_embeddings(out_dir: Path, provider: str) -> int:
        """
        Count embedding files for sanity checks before training.
        provider ∈ {"alpha", "gali"}.
        """
        emb_dir = Path(out_dir) / "embeddings"
        pat = "alpha_*.npy" if provider == "alpha" else "gali_*.npy"
        return len(list(emb_dir.glob(pat)))

    def train_polygon_mlp_btn(self):

        # --- 0) Quick guards before spawning a thread ---------------------------------
        if self._gdf is None or len(self._gdf) == 0:
            return self.after(0, lambda: messagebox.showwarning("No polygons", "Load polygons first."))

        mode = self._provider_mode().strip().lower()  # "alpha" | "gali" | "concat" | "alpha4"
        if mode not in {"alpha", "gali", "concat", "alpha4"}:
            return self.after(0, lambda: messagebox.showerror("Config", f"Unsupported feature source: {mode!r}"))

        key = f"train_poly_{mode}"
        # If a same job is already running, don't start another one.
        if key in getattr(self, "_running_jobs", set()):
            return

        self._set_busy(self.btn_poly_train, key, True, text_busy="Training…")

        # --- 1) Background worker ------------------------------------------------------
        def worker():
            import os
            import time
            import json
            import random
            import traceback
            from pathlib import Path

            t0 = time.perf_counter()
            try:
                out_dir = Path(self.out_dir.get() or DEFAULT_OUTDIR)
                models_dir = out_dir / "models"
                emb_dir = out_dir / "embeddings"
                models_dir.mkdir(parents=True, exist_ok=True)

                # ---- 1a) Parse numeric UI params safely (with clamps) ------------------
                # Year is clamped to AlphaEarth coverage (~2017–2024)
                try:
                    year = int(self.ee_year.get())
                except Exception:
                    year = 2024
                year = max(2017, min(2024, year))

                # NDVI gate used to derive binary labels (veg vs non-veg)
                try:
                    ndvi_thr = float(self.ndvi_thresh.get())
                except Exception:
                    ndvi_thr = 0.30
                ndvi_thr = max(-1.0, min(1.0, ndvi_thr))

                # Tiling scale used for EE tabular ops during label building
                try:
                    tile_scale = int(self.ee_tilescale.get())
                except Exception:
                    tile_scale = 4
                tile_scale = max(1, min(16, tile_scale))

                self._log_async(
                    f"[Poly-MLP] Preflight — mode={mode} | year={year} | NDVI≥{ndvi_thr:.2f} | tileScale={tile_scale} | "
                    f"polygons={len(self._gdf)}"
                )

                # ---- 1b) Output folder writability check ------------------------------
                try:
                    probe = models_dir / ".write_test.tmp"
                    probe.write_text("ok", encoding="utf-8")
                    probe.unlink(missing_ok=True)
                except Exception as e:
                    return self.after(0, lambda: messagebox.showerror(
                        "Outputs not writeable",
                        f"Cannot write to:\n{models_dir}\n\n{e}"
                    ))

                # ---- 1c) Embedding existence preflight --------------------------------
                # For 'concat', BOTH alpha and gali must be present.
                def _count_embeddings(_provider: str) -> int:
                    return self._count_embeddings(out_dir, _provider)

                if mode == "alpha":
                    if _count_embeddings("alpha") == 0:
                        return self.after(0, lambda: messagebox.showwarning(
                            "No embeddings",
                            f"No AlphaEarth embeddings found in:\n{emb_dir}\n\n"
                            "Compute embeddings first, then train."
                        ))
                elif mode == "gali":
                    if _count_embeddings("gali") == 0:
                        return self.after(0, lambda: messagebox.showwarning(
                            "No embeddings",
                            f"No Galileo embeddings found in:\n{emb_dir}\n\n"
                            "Compute embeddings first, then train."
                        ))
                elif mode == "concat":
                    ca, cg = _count_embeddings("alpha"), _count_embeddings("gali")
                    if ca == 0 or cg == 0:
                        return self.after(0, lambda: messagebox.showwarning(
                            "No embeddings (Concat)",
                            "Concat requires BOTH AlphaEarth and Galileo embeddings.\n"
                            f"Found in {emb_dir} → Alpha={ca}, Galileo={cg}"
                        ))
                else:  # "alpha4"
                    # alpha4 uses 256-D vectors produced by the 'AlphaEarth (4× stats)' pipeline.
                    # Accept either pre-concatenated alpha4_*.npy or all four parts.
                    have_alpha4_full = any(emb_dir.glob("alpha4_*.npy"))
                    have_parts = (
                            any(emb_dir.glob("alpha4_mean_*.npy")) and
                            any(emb_dir.glob("alpha4_mid_*.npy")) and
                            any(emb_dir.glob("alpha4_q1_*.npy")) and
                            any(emb_dir.glob("alpha4_q3_*.npy"))
                    )
                    if not (have_alpha4_full or have_parts):
                        return self.after(0, lambda: messagebox.showwarning(
                            "No embeddings (Alpha4)",
                            "No AlphaEarth 4× embeddings found.\n"
                            "Enable 'Also compute 4× stats (alpha4)' and run the AlphaEarth embedding step first."
                        ))

                # ---- 1d) EE init for NDVI label computation ---------------------------
                self._log_async("[Poly-MLP] Initializing Earth Engine (for NDVI labels)…")
                ok = ee_initialize(
                    self.ee_project.get().strip(),
                    self.sa_email.get().strip() or None,
                    self.sa_json.get().strip() or None,
                    interactive_fallback=False,
                    log=self._log_async
                )
                if not ok:
                    return self.after(0, lambda: messagebox.showerror(
                        "Earth Engine",
                        "EE is not initialized. Click 'Sign in (Google)' or 'Test EE (cached)' and retry."
                    ))

                # ---- 1e) Resolve model/scaler output paths by provider mode ------------
                if mode == "alpha4":
                    mp = models_dir / "mlp_poly_alpha4.keras"
                    sp = models_dir / "scaler_poly_alpha4.pkl"
                else:
                    mp = models_dir / f"mlp_poly_{mode}.keras"
                    sp = models_dir / f"scaler_poly_{mode}.pkl"

                # Reflect paths back to the UI (so the Predict action picks them up)
                self.poly_model_path.set(str(mp))
                self.poly_scaler_path.set(str(sp))

                # ---- 1f) Reproducibility seeds (best-effort) ---------------------------
                try:
                    import numpy as _np
                    random.seed(42)
                    _np.random.seed(42)
                    # TF seed if available (won't crash if TF not installed)
                    try:
                        import tensorflow as _tf
                        try:
                            _tf.keras.utils.set_random_seed(42)  # TF >= 2.9
                        except Exception:
                            _tf.random.set_seed(42)
                        if bool(self.tf_use_gpu.get()):
                            # Informative GPU presence message (does not enforce placement)
                            gpus = _tf.config.list_physical_devices('GPU')
                            if not gpus:
                                self._log_async("[Poly-MLP] GPU requested but TensorFlow did not detect a GPU.")
                            else:
                                self._log_async(f"[Poly-MLP] TF sees GPU(s): {len(gpus)}")
                    except Exception:
                        pass
                except Exception:
                    pass

                # ---- 1g) Launch training -------------------------------------------------
                self._log_async(
                    f"[Poly-MLP] Training with mode={mode} | year={year} | NDVI≥{ndvi_thr:.2f} | tileScale={tile_scale}"
                )
                # Keep the bar somewhere mid-way while heavy work runs
                try:
                    self.progress.configure(value=60)
                except Exception:
                    pass

                # Core training routine (implemented elsewhere in your codebase)
                train_polygon_mlp(
                    gdf=self._gdf.copy(),
                    out_dir=out_dir,
                    year=year,
                    ndvi_thr=ndvi_thr,
                    provider_mode=mode,
                    use_gpu_tf=bool(self.tf_use_gpu.get()),
                    tile_scale=tile_scale,
                    model_path=mp,
                    scaler_path=sp,
                    log=self._log_async,
                )

                # ---- 1h) Save a small meta alongside the model (useful for inference) ----
                try:
                    meta = {
                        "provider_mode": mode,
                        "year": int(year),
                        "ndvi_threshold": float(ndvi_thr),
                        "tile_scale": int(tile_scale),
                        "model_path": str(mp),
                        "scaler_path": str(sp),
                        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    }
                    (mp.parent / (mp.name + ".trainmeta.json")).write_text(json.dumps(meta, indent=2), encoding="utf-8")
                except Exception as e:
                    self._log_async(f"[Poly-MLP] Could not write train meta: {e}")

                # ---- 1i) Done -----------------------------------------------------------
                elapsed = time.perf_counter() - t0
                self._log_async(f"[Poly-MLP] Training complete in {elapsed:.1f}s → {mp.name}")
                self.after(0, lambda: messagebox.showinfo(
                    "Polygon-MLP",
                    f"Training finished.\n\nModel:   {mp}\nScaler:  {sp}"
                ))

            except Exception as e:
                # Log the root cause and display it to the user (with traceback to log)
                tb = traceback.format_exc(limit=5)
                self._log_async(f"[Poly-MLP] error: {e}\n{tb}")
                self.after(0, lambda m=str(e): messagebox.showerror("Polygon-MLP", m))
            finally:
                # Always release busy state
                self.after(0, lambda: self._set_busy(self.btn_poly_train, key, False))
                self.after(0, lambda: self.progress.configure(value=100))

        # --- 2) Launch background worker -------------------------------------------
        threading.Thread(target=worker, daemon=True).start()

    def predict_polygon_mlp_btn(self):
        """
        Predict polygon labels with a trained polygon-level MLP and paint the map.
        Saves both GeoJSON and GPKG for downstream GIS workflows.
        """
        if self._gdf is None or len(self._gdf) == 0:
            return self.after(0, lambda: messagebox.showwarning("No polygons", "Load polygons first."))

        mode = self._provider_mode()
        key = f"pred_poly_{mode}"
        self._set_busy(self.btn_poly_predict, key, True)

        def worker():
            try:
                mp = Path(self.poly_model_path.get())
                sp = Path(self.poly_scaler_path.get())

                if not mp.exists() or not sp.exists():
                    return self.after(0, lambda: messagebox.showwarning(
                        "No model",
                        f"Model or scaler not found:\n{mp}\n{sp}\n\nTrain Polygon-MLP ({mode}) first."
                    ))

                g = predict_polygons_with_polygon_mlp(
                    self._gdf.copy(),
                    Path(self.out_dir.get()),
                    provider_mode=mode,
                    model_path=mp,
                    scaler_path=sp,
                    log=self._log_async,
                )


                # Build a compact vegetation mask dataframe from page-2 output
                veg_df = g[["poly_id", "final_label", "p_veg_mean"]].copy()
                veg_df["veg_flag"] = veg_df["final_label"].astype(str).eq("vegetation")

                # Keep it in memory for page 3 (no NDVI recompute needed)
                self._veg_mask_df = veg_df

                # Persist on disk too (helps across sessions)
                try:
                    out_dir = Path(self.out_dir.get())
                    out_dir.mkdir(parents=True, exist_ok=True)
                    veg_csv = out_dir / "veg_mask_last.csv"
                    veg_df.to_csv(veg_csv, index=False)
                    self._log_async(f"[PAGE-2] Vegetation mask saved: {veg_csv.name}")
                except Exception as e:
                    self._log_async(f"[PAGE-2] Could not save veg mask CSV: {e}")

                cnt = g["final_label"].value_counts().to_dict()
                self._counts_var.set(
                    f"Final — vegetation: {cnt.get('vegetation', 0)} | "
                    f"non-vegetation: {cnt.get('non_vegetation', 0)} | "
                    f"unknown: {cnt.get('unknown', 0)}"
                )
                self._log_async(self._counts_var.get())

                out_dir = Path(self.out_dir.get())
                ts = pd.Timestamp.utcnow().strftime("%Y%m%dT%H%M%SZ")
                try:
                    g.to_file(out_dir / f"final_poly_{mode}_{ts}.geojson", driver="GeoJSON")
                    g.to_file(out_dir / f"final_poly_{mode}_{ts}.gpkg", driver="GPKG")
                    self._log_async(f"[SAVE] final_poly_{mode}_{ts}.* written to {out_dir}")
                except Exception as e:
                    self._log_async(f"[SAVE] {e}")

                self._draw_gdf(g, color_field="final_label")

            except Exception as e:
                self._log_async(f"[Poly-PRED] error: {e}")
            finally:
                self.after(0, lambda: self._set_busy(self.btn_poly_predict, key, False))

        threading.Thread(target=worker, daemon=True).start()

# --------------------------- Entry ---------------------------
if __name__ == "__main__":
    App().mainloop()
