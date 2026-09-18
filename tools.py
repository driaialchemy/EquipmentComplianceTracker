import math
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
from scipy.signal import welch


# ----------------------------
# Existing "compliance" tools
# (kept simple; your agents expect these names)
# ----------------------------
HIGH_RISK_R2_THRESHOLD = 20.0
HIGH_RISK_GDPR_THRESHOLD = 30.0
HIGH_ARBITER_RATINGS = {"high", "critical"}


def classify_compliance_tier(
    r2_score: float = 0.0,
    gdpr_score: float = 0.0,
    arbiter_rating: Optional[str] = None,
) -> Dict[str, Any]:
    rating = str(arbiter_rating or "").strip().lower()
    high = (
        float(r2_score or 0) >= HIGH_RISK_R2_THRESHOLD
        or float(gdpr_score or 0) >= HIGH_RISK_GDPR_THRESHOLD
        or rating in HIGH_ARBITER_RATINGS
    )
    if high:
        return {
            "risk_tier": "high",
            "review_route": "human_review",
            "disposition": "breach",
        }
    return {
        "risk_tier": "low",
        "review_route": "auto_logged",
        "disposition": "drift",
    }


def evaluate_r2v3_risk(device_dict: Dict[str, Any]) -> Dict[str, Any]:
    flags = device_dict.get("r2v3_flags") or {}
    score = 0
    for _, v in flags.items():
        if v:
            score += 10
    result = {"r2v3_flag_count": int(sum(1 for v in flags.values() if v)), "r2v3_risk_score": float(score)}
    result.update(classify_compliance_tier(r2_score=score))
    return result


def evaluate_gdpr_flags(device_dict: Dict[str, Any]) -> Dict[str, Any]:
    flags = device_dict.get("gdpr_flags") or {}
    risk = 0
    for _, v in flags.items():
        if v:
            risk += 15
    result = {"gdpr_flag_count": int(sum(1 for v in flags.values() if v)), "gdpr_risk_score": float(risk)}
    result.update(classify_compliance_tier(gdpr_score=risk))
    return result


def evaluate_gdpr_violation_patterns(device_dict: Dict[str, Any]) -> Dict[str, Any]:
    # placeholder: treat "residual_data_risk" field if present
    r = device_dict.get("residual_data_risk")
    if r is None:
        return {"pattern": "unknown", "signal": "no residual_data_risk provided"}
    return {"pattern": "residual-data-risk", "signal": float(r)}


def evaluate_audit_trail(device_dict: Dict[str, Any]) -> Dict[str, Any]:
    trail = device_dict.get("audit_trail") or {}
    steps = trail.get("steps") if isinstance(trail, dict) else None
    n = len(steps) if isinstance(steps, list) else 0
    return {"audit_steps_count": int(n), "audit_trail_present": bool(trail)}


def evaluate_brand_model_risk(device_dict: Dict[str, Any]) -> Dict[str, Any]:
    brand = (device_dict.get("brand") or "").lower()
    model = (device_dict.get("model") or "").lower()
    # synthetic heuristic
    baseline = 10.0
    if "garmin" in brand:
        baseline = 12.0
    if "ultra" in model:
        baseline += 5.0
    return {"brand_model_baseline_risk": float(baseline)}


def evaluate_sensor_health(device_dict: Dict[str, Any]) -> Dict[str, Any]:
    sensor = device_dict.get("sensor") or ""
    grade = device_dict.get("sensor_grade") or "B"
    return {"sensor": sensor, "grade": grade}


def evaluate_device_health(device_dict: Dict[str, Any]) -> Dict[str, Any]:
    battery = device_dict.get("battery_health_pct")
    if battery is None:
        return {"battery_health_pct": None, "health_band": "unknown"}
    b = float(battery)
    band = "good" if b >= 85 else ("ok" if b >= 70 else "poor")
    return {"battery_health_pct": b, "health_band": band}


def evaluate_risk_profile(device_dict: Dict[str, Any]) -> Dict[str, Any]:
    # simple roll-up
    return {"profile": "demo", "note": "Replace with real scoring later."}


# ----------------------------
# FFT / Spectral Tool (real)
# ----------------------------
def run_fft_welch(
    samples: List[float],
    sample_rate_hz: float,
    nperseg: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Welch PSD from raw samples.
    Returns dominant frequency, PSD summary, and bandpower metrics.
    """
    x = np.asarray(samples, dtype=float)
    if x.size < 8:
        raise ValueError("Not enough samples for FFT/Welch analysis.")

    fs = float(sample_rate_hz)
    if fs <= 0:
        raise ValueError("sample_rate_hz must be > 0")

    if nperseg is None:
        nperseg = min(256, x.size)

    freqs, psd = welch(x, fs=fs, nperseg=nperseg)

    # Dominant frequency (ignore DC)
    if freqs.size < 2:
        dom = float(freqs[0]) if freqs.size == 1 else 0.0
    else:
        idx0 = 1
        dom_idx = int(np.argmax(psd[idx0:]) + idx0)
        dom = float(freqs[dom_idx])

    def bandpower(f_lo: float, f_hi: float) -> float:
        mask = (freqs >= f_lo) & (freqs < f_hi)
        if not np.any(mask):
            return 0.0
        return float(np.trapz(psd[mask], freqs[mask]))

    bp_0_3 = bandpower(0.0, 3.0)
    bp_3_10 = bandpower(3.0, 10.0)

    noise_floor = float(np.median(psd)) if psd.size else 0.0

    return {
        "fft_method": "welch",
        "sample_rate_hz": fs,
        "n_samples": int(x.size),
        "nperseg": int(nperseg),
        "dominant_freq_hz": dom,
        "bandpower_0_3hz": bp_0_3,
        "bandpower_3_10hz": bp_3_10,
        "noise_floor": noise_floor,
        # Keep PSD summarized (not full arrays) for logging cleanliness
        "psd_summary": {
            "freq_min": float(freqs.min()) if freqs.size else 0.0,
            "freq_max": float(freqs.max()) if freqs.size else 0.0,
            "psd_min": float(psd.min()) if psd.size else 0.0,
            "psd_max": float(psd.max()) if psd.size else 0.0,
        },
    }
