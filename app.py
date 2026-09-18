import json
import os
from pathlib import Path

import pandas as pd
import streamlit as st
import numpy as np
from scipy.signal import welch

from db_access import (
    compute_arbiter_risk_distribution,
    count_devices,
    fetch_audit_events,
    fetch_audit_chain,
    fetch_recent_runs,
    fetch_recent_runs_with_event_counts,
    fetch_runs_for_device,
    get_device_payload,
    init_db_schema,
    list_devices,
)
from db_persist import persist_audit_result, persist_run
from orchestrator import Orchestrator

BASE_DIR = Path(__file__).resolve().parent

# ------------------------------------------------------------------------------
# 1. SETUP & SIDEBAR (Must be at the very top)
# ------------------------------------------------------------------------------
st.set_page_config(page_title="Device Passport", page_icon="🛡️", layout="wide")

st.sidebar.title("Device Passport")
st.sidebar.markdown("---")
role = st.sidebar.selectbox("Select Role", ["Operator", "Engineering", "Executive"])
st.sidebar.caption(f"Logged in as: **{role}**")


# ------------------------------------------------------------------------------
# 2. HELPER FUNCTIONS (Formatting Logic)
# ------------------------------------------------------------------------------
def format_flags(title: str, data):
    """Parse dict-like flag payloads and render readable bullets."""
    st.markdown(f"#### {title}")

    # Handle stringified JSON if database returns strings.
    if isinstance(data, str):
        try:
            data = json.loads(data)
        except Exception:
            st.error(f"Could not parse data for {title}")
            return

    if not data or not isinstance(data, dict):
        st.info("No flags set.")
        return

    for key, val in data.items():
        label = key.replace("_", " ").title()
        if val is True:
            icon, text = "✅", "Yes"
        elif val is False:
            icon, text = "❌", "No"
        else:
            icon, text = "🔹", str(val)
        st.markdown(f"{icon} **{label}**: {text}")


def device_registry_dataframe(devices: list[dict]) -> pd.DataFrame:
    rows = []
    for device in devices:
        payload = device.get("device_payload") or {}
        rows.append(
            {
                "device_id": device.get("device_id"),
                "brand": payload.get("brand"),
                "model": payload.get("model"),
                "region": payload.get("region"),
                "firmware_version": payload.get("firmware_version"),
                "purchase_date": payload.get("purchase_date"),
                "serial_number": payload.get("serial_number"),
            }
        )
    return pd.DataFrame(rows)


def device_posture_dataframe(devices: list[dict], runs: pd.DataFrame | None = None) -> pd.DataFrame:
    run_counts = {}
    if runs is not None and not runs.empty and "device_id" in runs.columns:
        run_counts = runs["device_id"].value_counts().to_dict()

    rows = []
    for device in devices:
        payload = device.get("device_payload") or {}
        gdpr_flags = payload.get("gdpr_flags") or {}
        r2v3_flags = payload.get("r2v3_flags") or {}

        gdpr_risks = sum(
            1
            for key in ("residual_identifiers_found", "account_tokens_present", "paired_phone_id_present")
            if gdpr_flags.get(key) is True
        )
        r2v3_risks = sum(
            1
            for key in ("tamper_seal_broken", "mdm_profile_present", "residual_personal_data_risk", "unofficial_firmware")
            if r2v3_flags.get(key) is True
        )
        if r2v3_flags.get("factory_reset_verified") is False:
            r2v3_risks += 1

        risk_signals = gdpr_risks + r2v3_risks
        if risk_signals >= 4:
            risk_band = "High"
        elif risk_signals >= 2:
            risk_band = "Medium"
        elif risk_signals == 1:
            risk_band = "Watch"
        else:
            risk_band = "Low"

        rows.append(
            {
                "device_id": device.get("device_id"),
                "brand": payload.get("brand"),
                "model": payload.get("model"),
                "region": payload.get("region"),
                "risk_band": risk_band,
                "risk_signals": risk_signals,
                "gdpr_risk_signals": gdpr_risks,
                "r2v3_risk_signals": r2v3_risks,
                "audit_runs": int(run_counts.get(device.get("device_id"), 0)),
                "factory_reset_verified": r2v3_flags.get("factory_reset_verified"),
                "sensor_suite": payload.get("sensor_suite"),
            }
        )
    return pd.DataFrame(rows)


def data_quality_dataframe(devices: list[dict]) -> pd.DataFrame:
    required_payload_fields = ["brand", "model", "serial_number", "firmware_version", "region"]
    rows = []
    for device in devices:
        payload = device.get("device_payload") or {}
        missing = [
            field
            for field in required_payload_fields
            if payload.get(field) in (None, "", [], {})
        ]
        rows.append(
            {
                "device_id": device.get("device_id"),
                "missing_fields": ", ".join(missing) if missing else "None",
                "has_gdpr_flags": bool(payload.get("gdpr_flags")),
                "has_r2v3_flags": bool(payload.get("r2v3_flags")),
                "has_serial": bool(payload.get("serial_number")),
            }
        )
    return pd.DataFrame(rows)


def audit_run_dataframe(runs: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    if runs.empty:
        return pd.DataFrame()

    final_events = {}
    if not events.empty:
        for _, row in events[events["event_type"] == "FINAL"].iterrows():
            final_events.setdefault(row.get("run_id"), row.get("payload_json") or {})

    rows = []
    for _, run in runs.iterrows():
        final_payload = final_events.get(run.get("run_id"), {})
        rows.append(
            {
                "run_id": run.get("run_id"),
                "device_id": run.get("device_id"),
                "created_at": run.get("created_at"),
                "structured_events": int(run.get("audit_events", 0) or 0),
                "arbiter_risk_rating": final_payload.get("arbiter_risk_rating", "Not captured"),
                "final_uncertainty": final_payload.get("final_uncertainty"),
                "status": "Structured" if final_payload else "Run record only",
            }
        )
    return pd.DataFrame(rows)


def audit_artifacts_dataframe() -> pd.DataFrame:
    log_dir = Path("audit_logs")
    if not log_dir.exists():
        return pd.DataFrame(columns=["artifact", "size_kb", "modified"])

    rows = []
    for path in sorted(log_dir.glob("*.txt"), key=lambda item: item.stat().st_mtime, reverse=True):
        stat = path.stat()
        rows.append(
            {
                "artifact": path.name,
                "size_kb": round(stat.st_size / 1024, 1),
                "modified": pd.to_datetime(stat.st_mtime, unit="s"),
            }
        )
    return pd.DataFrame(rows)


def latest_final_payload(events: pd.DataFrame) -> dict:
    if events.empty:
        return {}
    finals = events[events["event_type"] == "FINAL"]
    if finals.empty:
        return {}
    payload = finals.iloc[0].get("payload_json")
    return payload if isinstance(payload, dict) else {}


def seed_workbook_path() -> Path:
    raw_path = os.getenv("DP_SEED_XLSX", "seed/devicepassport_catalog.xlsx")
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return BASE_DIR / path


@st.cache_data(show_spinner=False)
def load_seed_sheet(seed_path: str, sheet_name: str) -> pd.DataFrame:
    path = Path(seed_path)
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_excel(path, sheet_name=sheet_name)
    except Exception:
        return pd.DataFrame()


def load_battery_history() -> pd.DataFrame:
    df = load_seed_sheet(str(seed_workbook_path()), "battery_history")
    if df.empty:
        return df
    df = df.copy()
    df["device_id"] = df["device_id"].astype(str)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["battery_health_pct"] = pd.to_numeric(df["battery_health_pct"], errors="coerce")
    return df.dropna(subset=["date", "battery_health_pct"])


def load_sensor_timeseries() -> pd.DataFrame:
    df = load_seed_sheet(str(seed_workbook_path()), "sensor_timeseries")
    if df.empty:
        return df
    df = df.copy()
    df["device_id"] = df["device_id"].astype(str)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return df.dropna(subset=["timestamp", "value"])


def device_label(device: dict) -> str:
    payload = device.get("device_payload") or {}
    return " | ".join(
        str(part)
        for part in (
            device.get("device_id"),
            payload.get("brand", "Unknown"),
            payload.get("model", "Unknown"),
        )
        if part
    )


def select_device_id(devices: list[dict], label: str, key: str) -> str:
    device_map = {device_label(device): device["device_id"] for device in devices}
    selected_label = st.selectbox(label, list(device_map.keys()), key=key)
    return device_map[selected_label]


def missing_llm_keys() -> list[str]:
    return [key for key in ("OPENAI_API_KEY", "GEMINI_API_KEY") if not os.getenv(key)]


def run_and_persist_audit(device_id: str, payload: dict) -> dict:
    orch = Orchestrator()
    result = orch.run_device_audit(payload)
    persist_run(result["run_id"], device_id)
    persist_audit_result(result, device_id)
    from governance_logger import require_approval
    require_approval("Device-Lifecycle-Intelligence-Platform-DLIP-", "Agent completed successfully", {
        "device_id": device_id,
        "run_id": result.get("run_id"),
        "result": result
    })
    return result


def audit_lens_options(role_name: str) -> dict[str, str]:
    if role_name == "Engineering":
        return {
            "Sensor degradation deep dive": "Prioritize sensor health, uncertainty, traceability, and diagnostic evidence.",
            "R2v3 repair readiness": "Prioritize factory reset, firmware, tamper, MDM, and resale readiness signals.",
            "Privacy data remanence": "Prioritize GDPR erasure, token residue, paired phone IDs, and privacy remediation.",
            "Audit-chain traceability": "Prioritize structured events, agent disagreement, verifier output, and reproducibility.",
        }
    if role_name == "Executive":
        return {
            "Board risk brief": "Summarize business exposure, audit confidence, and decision implications.",
            "Portfolio compliance exposure": "Prioritize regulatory, privacy, and compliance posture.",
            "Revenue recovery view": "Prioritize repair, resale, discard, and operational impact.",
            "Escalation watchlist": "Prioritize devices likely to require leadership attention.",
        }
    return {
        "Intake compliance audit": "Check whether this device is ready for normal processing.",
        "Exception review": "Focus on unusual flags that may need manual handling.",
        "Release re-check": "Confirm the device is still safe after remediation or repair.",
    }


def render_lens_guidance(role_name: str, lens: str, payload: dict) -> None:
    options = audit_lens_options(role_name)
    st.caption(options.get(lens, "Run the selected AI analysis lens."))

    if role_name == "Engineering":
        r2v3_flags = payload.get("r2v3_flags") or {}
        gdpr_flags = payload.get("gdpr_flags") or {}
        col1, col2, col3 = st.columns(3)
        col1.metric("R2v3 Flags", sum(1 for value in r2v3_flags.values() if value is True))
        col2.metric("GDPR Flags", sum(1 for value in gdpr_flags.values() if value is True))
        col3.metric("Sensor Suite", payload.get("sensor_suite", "Unknown"))
    elif role_name == "Executive":
        brand_model = f"{payload.get('brand', 'Unknown')} {payload.get('model', '')}".strip()
        col1, col2, col3 = st.columns(3)
        col1.metric("Selected Asset", brand_model or "Unknown")
        col2.metric("Region", payload.get("region", "Unknown"))
        col3.metric("Firmware", payload.get("firmware_version", "Unknown"))


def render_role_audit_result(result: dict, role_name: str, lens: str) -> None:
    details = result.get("details", {}) if isinstance(result, dict) else {}
    arbiter = details.get("arb", {}) if isinstance(details, dict) else {}
    meta = details.get("meta", {}) if isinstance(details, dict) else {}
    risk_rating = arbiter.get("risk_rating", "Not captured") if isinstance(arbiter, dict) else "Not captured"
    uncertainty = result.get("final_uncertainty", 0)

    st.success(f"AI analysis complete. Run ID: {result.get('run_id', 'unknown')}")

    if role_name == "Executive":
        c1, c2, c3 = st.columns(3)
        c1.metric("Board Risk Rating", risk_rating)
        c2.metric("Confidence Gap", f"{float(uncertainty):.2f}" if isinstance(uncertainty, (int, float)) else "Unknown")
        c3.metric("Analysis Lens", lens)
        st.markdown("#### Executive Consensus")
        st.info(result.get("final_narrative", "No narrative generated."))
        st.caption("Use this view for business risk, compliance exposure, and escalation decisions.")
        return

    if role_name == "Engineering":
        c1, c2, c3 = st.columns(3)
        c1.metric("Arbiter Risk", risk_rating)
        c2.metric("Final Uncertainty", f"{float(uncertainty):.2f}" if isinstance(uncertainty, (int, float)) else "Unknown")
        c3.metric("Lens", lens)
        st.markdown("#### Engineering Analysis")
        st.info(result.get("final_narrative", "No narrative generated."))

        tab_summary, tab_agents, tab_trace = st.tabs(["Consensus", "Agent Outputs", "Trace Payload"])
        with tab_summary:
            if isinstance(arbiter, dict) and arbiter:
                st.json(arbiter)
            if isinstance(meta, dict) and meta:
                st.markdown("##### Meta Reviewer")
                st.json(meta)
        with tab_agents:
            r1 = details.get("r1", {}) if isinstance(details, dict) else {}
            r2 = details.get("r2", {}) if isinstance(details, dict) else {}
            for round_name, round_payload in (("Round 1", r1), ("Round 2", r2)):
                st.markdown(f"##### {round_name}")
                if isinstance(round_payload, dict) and round_payload:
                    for agent_name, agent_data in round_payload.items():
                        with st.expander(f"{round_name}: {agent_name}"):
                            st.json(agent_data)
                else:
                    st.info(f"No {round_name.lower()} output captured.")
        with tab_trace:
            st.json(details)
        return

    st.markdown("### Final Consensus Narrative")
    st.info(result.get("final_narrative", "No narrative generated."))
    st.metric("Uncertainty Score", f"{float(uncertainty):.2f}" if isinstance(uncertainty, (int, float)) else "Unknown")

    st.markdown("### Agent Opinions")
    r1 = (
        result.get("audit_record", {}).get("round1", {})
        or details.get("round1", {})
        or details.get("r1", {})
    )
    if not isinstance(r1, dict) or not r1:
        st.info("No agent opinions were returned for this run.")
    for agent_name, agent_data in (r1.items() if isinstance(r1, dict) else []):
        if not isinstance(agent_data, dict):
            continue
        with st.expander(f"Agent: {agent_name}"):
            st.markdown(agent_data.get("text", "No text provided."))
            st.caption("Raw Tool Outputs:")
            st.json(agent_data.get("tool_outputs", {}))


def render_audit_action_panel(
    role_name: str,
    device_id: str,
    payload: dict,
    lens_key: str,
    button_key: str,
    result_key: str,
) -> None:
    lens_map = audit_lens_options(role_name)
    lens = st.selectbox(f"{role_name} AI analysis lens", list(lens_map.keys()), key=lens_key)
    render_lens_guidance(role_name, lens, payload)

    keys = missing_llm_keys()
    if keys:
        st.warning(
            "Set these environment variables in .env before running an AI audit: "
            + ", ".join(keys)
        )

    if st.button(f"Run {role_name} AI Analysis", type="primary", disabled=bool(keys), key=button_key):
        with st.spinner("Agents are negotiating and producing a role-specific analysis..."):
            try:
                st.session_state[result_key] = run_and_persist_audit(device_id, payload)
            except Exception as exc:
                st.error(f"Audit failed: {exc}")
                return

    result = st.session_state.get(result_key)
    if result:
        render_role_audit_result(result, role_name, lens)


def forecast_battery_state(history: pd.DataFrame, horizon_days: int, method: str) -> tuple[pd.DataFrame, dict]:
    if history.empty or len(history) < 2:
        return pd.DataFrame(), {"reason": "At least two battery observations are required."}

    df = history.sort_values("date").copy()
    df["x"] = (df["date"] - df["date"].min()).dt.days.astype(float)
    x = df["x"].to_numpy(dtype=float)
    y = df["battery_health_pct"].to_numpy(dtype=float)
    future_x = np.arange(x.max() + 1, x.max() + horizon_days + 1, dtype=float)
    future_dates = df["date"].max() + pd.to_timedelta(np.arange(1, horizon_days + 1), unit="D")

    if "recent-average" in method.lower():
        recent_window = min(21, len(y) - 1)
        recent_slope = float(np.median(np.diff(y[-recent_window:]))) if recent_window > 1 else float(y[-1] - y[-2])
        forecast = y[-1] + np.arange(1, horizon_days + 1) * recent_slope
        fitted = y[0] + np.arange(len(y)) * recent_slope
        method_note = "Recent median daily change. Useful as a conservative baseline."
    else:
        coef = np.polyfit(x, y, deg=1)
        fitted = coef[0] * x + coef[1]
        forecast = coef[0] * future_x + coef[1]
        method_note = "Linear degradation trend. Best baseline for battery health because battery decline is usually monotonic."

        if "fourier" in method.lower() and len(y) >= 30:
            residual = y - fitted
            centered_x = x - x.min()
            step = float(np.median(np.diff(np.sort(centered_x)))) if len(centered_x) > 2 else 1.0
            frequencies = np.fft.rfftfreq(len(residual), d=max(step, 1.0))
            spectrum = np.abs(np.fft.rfft(residual))
            candidate_idx = np.argsort(spectrum[1:])[-2:] + 1 if len(spectrum) > 2 else []

            columns = [np.ones_like(centered_x), centered_x]
            future_centered_x = future_x - x.min()
            future_columns = [np.ones_like(future_centered_x), future_centered_x]

            for idx in candidate_idx:
                freq = frequencies[idx]
                if freq <= 0:
                    continue
                columns.extend([np.sin(2 * np.pi * freq * centered_x), np.cos(2 * np.pi * freq * centered_x)])
                future_columns.extend([
                    np.sin(2 * np.pi * freq * future_centered_x),
                    np.cos(2 * np.pi * freq * future_centered_x),
                ])

            design = np.vstack(columns).T
            future_design = np.vstack(future_columns).T
            fit_coef, *_ = np.linalg.lstsq(design, y, rcond=None)
            fitted = design @ fit_coef
            forecast = future_design @ fit_coef
            method_note = "Hybrid trend plus Fourier residual cycles. Use when history has repeated maintenance or charging patterns."

    residual_sigma = float(np.std(y - fitted)) if len(y) > 2 else 1.0
    forecast = np.clip(forecast, 0, 100)
    lower = np.clip(forecast - 1.96 * residual_sigma, 0, 100)
    upper = np.clip(forecast + 1.96 * residual_sigma, 0, 100)

    actual_df = pd.DataFrame({"date": df["date"], "actual": y, "forecast": np.nan, "lower": np.nan, "upper": np.nan})
    forecast_df = pd.DataFrame({"date": future_dates, "actual": np.nan, "forecast": forecast, "lower": lower, "upper": upper})
    combined = pd.concat([actual_df, forecast_df], ignore_index=True)
    slope = float((forecast[-1] - y[-1]) / horizon_days) if horizon_days else 0.0
    projected = float(forecast[-1])
    band = "Healthy" if projected >= 85 else ("Watch" if projected >= 75 else ("Service Soon" if projected >= 65 else "Retire/Parts Risk"))

    return combined, {
        "projected_battery": projected,
        "daily_slope": slope,
        "residual_sigma": residual_sigma,
        "future_band": band,
        "method_note": method_note,
    }


def compute_sensor_spectrum(sensor_df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    if sensor_df.empty or len(sensor_df) < 16:
        return pd.DataFrame(), {"reason": "At least 16 sensor samples are required."}

    df = sensor_df.sort_values("timestamp").copy()
    values = df["value"].to_numpy(dtype=float)
    values = values - np.nanmean(values)
    timestamps = pd.to_datetime(df["timestamp"])
    dt_seconds = np.median(np.diff(timestamps.values).astype("timedelta64[ms]").astype(float)) / 1000.0
    sample_rate = 1.0 / dt_seconds if dt_seconds and dt_seconds > 0 else 1.0

    freqs, psd = welch(values, fs=sample_rate, nperseg=min(256, len(values)))
    spectrum_df = pd.DataFrame({"frequency_hz": freqs, "power": psd})
    non_dc = spectrum_df[spectrum_df["frequency_hz"] > 0]
    if non_dc.empty:
        dominant_freq = 0.0
        peak_power = 0.0
    else:
        peak_row = non_dc.loc[non_dc["power"].idxmax()]
        dominant_freq = float(peak_row["frequency_hz"])
        peak_power = float(peak_row["power"])
    noise_floor = float(np.median(psd)) if len(psd) else 0.0
    snr = peak_power / (noise_floor + 1e-9)
    stability = "Stable" if snr >= 8 else ("Watch" if snr >= 3 else "Noisy")
    return spectrum_df, {
        "sample_rate_hz": sample_rate,
        "dominant_freq_hz": dominant_freq,
        "noise_floor": noise_floor,
        "peak_to_noise": snr,
        "stability": stability,
    }


def render_future_state_lab(device_id: str, key_prefix: str, compact: bool = False) -> None:
    battery_df = load_battery_history()
    sensor_df = load_sensor_timeseries()
    battery_device = battery_df[battery_df["device_id"] == str(device_id)] if not battery_df.empty else pd.DataFrame()
    sensor_device = sensor_df[sensor_df["device_id"] == str(device_id)] if not sensor_df.empty else pd.DataFrame()

    if battery_device.empty and sensor_device.empty:
        st.info("No battery or sensor history found for this device in the seed workbook.")
        return

    st.caption(
        "Recommended approach: use linear degradation for battery future state, then use Fourier/Welch spectral analysis "
        "to explain periodic sensor noise or instability. Fourier is excellent for signal patterns; battery state still needs a trend model."
    )

    horizon = st.selectbox("Forecast horizon", [30, 60, 90, 180], index=2, key=f"{key_prefix}_horizon")
    method = st.selectbox(
        "Battery forecast method",
        [
            "Recommended: linear degradation trend",
            "Hybrid: linear trend + Fourier residual cycles",
            "Conservative: recent-average slope",
        ],
        key=f"{key_prefix}_method",
    )

    chart_options = ["Battery forecast", "Sensor waveform", "Sensor FFT/Welch spectrum", "Combined future-state summary"]
    chart_choice = st.selectbox("Chart to generate", chart_options, key=f"{key_prefix}_chart")

    if not battery_device.empty:
        forecast_df, summary = forecast_battery_state(battery_device, int(horizon), method)
        if forecast_df.empty:
            st.info(summary.get("reason", "Could not build battery forecast."))
        else:
            c1, c2, c3 = st.columns(3)
            c1.metric(f"Projected Battery ({horizon}d)", f"{summary['projected_battery']:.1f}%")
            c2.metric("Daily Change", f"{summary['daily_slope']:.3f}%")
            c3.metric("Future Band", summary["future_band"])
            st.caption(summary["method_note"])

            if chart_choice in {"Battery forecast", "Combined future-state summary"}:
                chart_df = forecast_df.set_index("date")[["actual", "forecast", "lower", "upper"]]
                st.line_chart(chart_df)

    if not sensor_device.empty and chart_choice in {"Sensor waveform", "Sensor FFT/Welch spectrum", "Combined future-state summary"}:
        sensor_options = sorted(sensor_device["sensor"].dropna().astype(str).unique().tolist())
        selected_sensor = st.selectbox("Sensor stream", sensor_options, key=f"{key_prefix}_sensor")
        selected_sensor_df = sensor_device[sensor_device["sensor"].astype(str) == selected_sensor].sort_values("timestamp")

        if chart_choice in {"Sensor waveform", "Combined future-state summary"}:
            st.markdown("#### Sensor Time Series")
            st.line_chart(selected_sensor_df.tail(600).set_index("timestamp")["value"])

        if chart_choice in {"Sensor FFT/Welch spectrum", "Combined future-state summary"}:
            spectrum_df, spectrum_summary = compute_sensor_spectrum(selected_sensor_df)
            if spectrum_df.empty:
                st.info(spectrum_summary.get("reason", "Could not build sensor spectrum."))
            else:
                c1, c2, c3 = st.columns(3)
                c1.metric("Dominant Frequency", f"{spectrum_summary['dominant_freq_hz']:.4f} Hz")
                c2.metric("Peak/Noise", f"{spectrum_summary['peak_to_noise']:.1f}x")
                c3.metric("Sensor Stability", spectrum_summary["stability"])
                st.line_chart(spectrum_df.set_index("frequency_hz")["power"])


# ------------------------------------------------------------------------------
# 3. DATABASE CONNECTION
# ------------------------------------------------------------------------------
try:
    init_db_schema()
except Exception as e:
    st.error(f"Database Connection Error: {e}")
    st.stop()


# ------------------------------------------------------------------------------
# 4. VIEW LOGIC
# ------------------------------------------------------------------------------

# === OPERATOR VIEW ===
if role == "Operator":
    st.title("🎛️ Operator Dashboard")

    devices = list_devices(limit=50)
    if not devices:
        st.warning("Database empty. Run the seed script.")
        st.stop()

    dev_map = {
        f"{d['device_id']} | {d.get('device_payload', {}).get('brand', 'Unknown')}": d["device_id"]
        for d in devices
    }
    selected_label = st.selectbox("Select Device to Inspect", list(dev_map.keys()))
    device_id = dev_map[selected_label]

    payload = get_device_payload(device_id)

    tab1, tab2, tab3, tab4 = st.tabs(["📋 Overview (Readable)", "🤖 Run Audit", "🔒 Audit Chain", "📈 Future State"])

    with tab1:
        st.subheader(f"Device: {device_id}")

        c1, c2 = st.columns(2)

        with c1:
            format_flags("GDPR & Privacy Status", payload.get("gdpr_flags", {}))

        with c2:
            format_flags("R2v3 Compliance", payload.get("r2v3_flags", {}))

        st.divider()
        with st.expander("🛠️ View Raw JSON Data (Debug Only)"):
            st.json(payload)

    with tab2:
        st.subheader("Multi-Agent Compliance Audit")
        render_audit_action_panel(
            "Operator",
            device_id,
            payload,
            lens_key="operator_audit_lens",
            button_key="operator_run_audit",
            result_key=f"operator_audit_result_{device_id}",
        )

    with tab3:
        st.subheader("Immutable Ledger")
        runs = fetch_runs_for_device(device_id)

        if not runs.empty:
            run_id = st.selectbox("Select Past Run", runs["run_id"].unique())
            chain = fetch_audit_chain(run_id)

            if not chain.empty:
                for _, row in chain.iterrows():
                    etype = row.get("event_type")
                    ts = row.get("created_at")
                    raw_payload = row.get("payload_json")

                    if isinstance(raw_payload, str):
                        try:
                            payload_json = json.loads(raw_payload)
                        except Exception:
                            payload_json = raw_payload
                    else:
                        payload_json = raw_payload

                    with st.expander(f"{ts} | {etype}"):
                        st.write(payload_json)
            else:
                st.info("No events found for this run.")
        else:
            st.info("No audit history for this device.")

    with tab4:
        st.subheader("Watch Future-State Preview")
        render_future_state_lab(device_id, key_prefix=f"operator_future_{device_id}", compact=True)

# === ENGINEERING VIEW ===
elif role == "Engineering":
    st.title("🛠️ Engineering Console")
    st.info("Operational view for registry health, data quality, audit coverage, and traceability.")

    devices = list_devices(limit=1000)
    runs = fetch_recent_runs_with_event_counts(limit=200)
    events = fetch_audit_events(limit=500)
    posture = device_posture_dataframe(devices, runs)
    quality = data_quality_dataframe(devices)
    audit_runs = audit_run_dataframe(runs, events)
    artifacts = audit_artifacts_dataframe()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Registry Devices", len(devices))
    c2.metric("Audit Runs", 0 if runs.empty else len(runs))
    c3.metric("Structured Events", 0 if events.empty else len(events))
    c4.metric("Local Audit Files", 0 if artifacts.empty else len(artifacts))

    engineering_device_id = None
    engineering_payload = {}
    if devices:
        st.markdown("#### Engineering Workbench Target")
        engineering_device_id = select_device_id(
            devices,
            "Select device for engineering AI analysis and forecasting",
            key="engineering_device_select",
        )
        engineering_payload = get_device_payload(engineering_device_id)

    tab_registry, tab_quality, tab_ai, tab_forecast, tab_audits, tab_artifacts = st.tabs(
        ["Registry", "Data Quality", "AI Analysis", "Forecast Lab", "Audit Operations", "Artifacts"]
    )

    with tab_registry:
        st.subheader("Device Registry And Risk Signals")
        if posture.empty:
            st.warning("No devices found in device_registry. Run the seed script.")
        else:
            st.dataframe(posture, use_container_width=True)
            brand_summary = (
                posture.groupby("brand", dropna=False)["risk_signals"]
                .sum()
                .reset_index()
                .sort_values("risk_signals", ascending=False)
            )
            st.markdown("#### Risk Signals By Brand")
            st.bar_chart(brand_summary.set_index("brand"))

    with tab_quality:
        st.subheader("Registry Completeness")
        if quality.empty:
            st.info("No registry rows to validate.")
        else:
            missing_rows = quality[quality["missing_fields"] != "None"]
            q1, q2, q3 = st.columns(3)
            q1.metric("Rows Missing Required Fields", len(missing_rows))
            q2.metric("Rows With GDPR Flags", int(quality["has_gdpr_flags"].sum()))
            q3.metric("Rows With R2v3 Flags", int(quality["has_r2v3_flags"].sum()))
            st.dataframe(quality, use_container_width=True)

    with tab_ai:
        st.subheader("Engineering Audit And AI Analysis")
        if not engineering_device_id:
            st.info("No devices found in device_registry. Run the seed script.")
        else:
            st.caption("Engineering lenses expose diagnostic evidence, agent disagreement, and trace payloads.")
            render_audit_action_panel(
                "Engineering",
                engineering_device_id,
                engineering_payload,
                lens_key="engineering_audit_lens",
                button_key="engineering_run_audit",
                result_key=f"engineering_audit_result_{engineering_device_id}",
            )

    with tab_forecast:
        st.subheader("Watch Future-State Forecast Lab")
        if not engineering_device_id:
            st.info("No devices found in device_registry. Run the seed script.")
        else:
            render_future_state_lab(engineering_device_id, key_prefix=f"engineering_forecast_{engineering_device_id}")

    with tab_audits:
        st.subheader("Audit Run Health")
        if audit_runs.empty:
            st.info("No audit runs yet. Start an audit from the Operator view.")
        else:
            st.dataframe(audit_runs, use_container_width=True)
            if "status" in audit_runs:
                status_counts = audit_runs["status"].value_counts().rename_axis("status").reset_index(name="count")
                st.markdown("#### Structured Capture Status")
                st.bar_chart(status_counts.set_index("status"))

        if not events.empty:
            st.markdown("#### Recent Structured Audit Events")
            event_preview = events[["created_at", "run_id", "event_type"]].head(25)
            st.dataframe(event_preview, use_container_width=True)

    with tab_artifacts:
        st.subheader("Local Audit Artifacts")
        st.caption("Plain-text hash-chained audit files written by the orchestrator.")
        if artifacts.empty:
            st.info("No local audit files found.")
        else:
            st.dataframe(artifacts, use_container_width=True)


# === EXECUTIVE VIEW ===
elif role == "Executive":
    st.title("📊 Executive Dashboard")
    st.caption("Portfolio-level compliance posture and audit coverage.")

    devices = list_devices(limit=1000)
    runs = fetch_recent_runs_with_event_counts(limit=500)
    events = fetch_audit_events(limit=1000)
    posture = device_posture_dataframe(devices, runs)
    audit_runs = audit_run_dataframe(runs, events)
    risk_dist = compute_arbiter_risk_distribution(days=30)
    latest_final = latest_final_payload(events)

    total_devices = len(devices)
    audited_devices = 0 if runs.empty else runs["device_id"].nunique()
    audit_coverage = 0 if total_devices == 0 else int((audited_devices / total_devices) * 100)
    open_risk_signals = 0 if posture.empty else int(posture["risk_signals"].sum())
    high_or_medium = 0 if posture.empty else int(posture["risk_band"].isin(["High", "Medium"]).sum())

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Devices In Registry", total_devices)
    c2.metric("Audit Coverage", f"{audit_coverage}%")
    c3.metric("Audit Runs", 0 if runs.empty else len(runs))
    c4.metric("Open Risk Signals", open_risk_signals)

    executive_device_id = None
    executive_payload = {}
    if devices:
        st.markdown("#### Executive Review Target")
        executive_device_id = select_device_id(
            devices,
            "Select device for executive AI analysis",
            key="executive_device_select",
        )
        executive_payload = get_device_payload(executive_device_id)

        with st.expander("Executive AI Analysis", expanded=False):
            render_audit_action_panel(
                "Executive",
                executive_device_id,
                executive_payload,
                lens_key="executive_audit_lens",
                button_key="executive_run_audit",
                result_key=f"executive_audit_result_{executive_device_id}",
            )

        with st.expander("Future-State Snapshot", expanded=False):
            render_future_state_lab(executive_device_id, key_prefix=f"executive_future_{executive_device_id}", compact=True)

    left, right = st.columns(2)
    with left:
        st.subheader("Portfolio Risk Bands")
        if posture.empty:
            st.info("No device registry data available.")
        else:
            band_counts = posture["risk_band"].value_counts().rename_axis("risk_band").reset_index(name="devices")
            st.bar_chart(band_counts.set_index("risk_band"))

    with right:
        st.subheader("Risk Signals By Brand")
        if posture.empty:
            st.info("No device registry data available.")
        else:
            brand_risk = (
                posture.groupby("brand", dropna=False)["risk_signals"]
                .sum()
                .reset_index()
                .sort_values("risk_signals", ascending=False)
            )
            st.bar_chart(brand_risk.set_index("brand"))

    st.subheader("Audit Outcomes")
    if not risk_dist.empty:
        st.bar_chart(risk_dist.set_index("arbiter_risk_rating"))
    elif not audit_runs.empty:
        st.info("Audit runs exist, but older runs did not capture structured arbiter outcomes. Future audits will populate this chart.")
    else:
        st.info("No audit runs yet. Run an audit from the Operator view to populate outcomes.")

    if latest_final:
        st.subheader("Latest Board Consensus")
        latest_cols = st.columns(3)
        latest_cols[0].metric("Risk Rating", latest_final.get("arbiter_risk_rating", "Unknown"))
        uncertainty = latest_final.get("final_uncertainty")
        latest_cols[1].metric("Uncertainty", f"{uncertainty:.2f}" if isinstance(uncertainty, (int, float)) else "Unknown")
        latest_cols[2].metric("High/Medium Devices", high_or_medium)
        st.info(latest_final.get("final_narrative", "No final narrative captured."))

    st.subheader("Devices Needing Attention")
    if posture.empty:
        st.info("No device registry data available.")
    else:
        attention = posture[posture["risk_signals"] > 0].sort_values(
            ["risk_signals", "audit_runs"], ascending=[False, True]
        )
        if attention.empty:
            st.success("No open registry risk signals across the device portfolio.")
        else:
            st.dataframe(
                attention[["device_id", "brand", "model", "risk_band", "risk_signals", "audit_runs"]],
                use_container_width=True,
            )

    st.subheader("Recent Audit Activity")
    if audit_runs.empty:
        st.info("No audit run activity yet.")
    else:
        st.dataframe(audit_runs.head(20), use_container_width=True)
