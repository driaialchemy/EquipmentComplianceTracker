"""
demo_spreadsheet_app.py

Spreadsheet-driven UI for the Device Passport Multi-Agent Audit.

Features:
- Uses sample_data/device_data.xlsx by default, or user upload (.xlsx/.csv).
- Single-device audit: runs full multi-agent pipeline via Orchestrator.
- Batch audits: run multi-agent audits over multiple devices and show a risk dashboard.
- Sensor / FFT visualization: accelerometer time series + FFT from sample_data/sensor_timeseries.xlsx.

No JSON, no dict paste.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import pandas as pd
import streamlit as st

from orchestrator import Orchestrator
from tools import evaluate_r2v3_risk  # for simple risk band on dashboard


# --------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------

BASE_DIR = Path(__file__).parent
DEFAULT_SPREADSHEET = BASE_DIR / "sample_data" / "device_data.xlsx"
SENSOR_TIMESERIES = BASE_DIR / "sample_data" / "sensor_timeseries.xlsx"


# --------------------------------------------------------------------
# Spreadsheet loader
# --------------------------------------------------------------------

def load_spreadsheet(file_or_path) -> pd.DataFrame:
    """
    Load a spreadsheet (uploaded file or file path) into a DataFrame.
    Supports .xlsx, .xls, .csv.
    """
    # Case 1: Path / string
    if isinstance(file_or_path, (str, Path)):
        p = Path(file_or_path)
        if not p.exists():
            raise FileNotFoundError(f"File not found: {p}")
        if p.suffix.lower() == ".csv":
            return pd.read_csv(p)
        elif p.suffix.lower() in {".xlsx", ".xls"}:
            return pd.read_excel(p)
        else:
            raise ValueError(f"Unsupported file type: {p.suffix}")

    # Case 2: Streamlit UploadedFile
    uploaded = file_or_path
    name = uploaded.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(uploaded)
    elif name.endswith(".xlsx") or name.endswith(".xls"):
        return pd.read_excel(uploaded)
    else:
        raise ValueError(f"Unsupported uploaded file type: {name}")


# --------------------------------------------------------------------
# device_dict mapping
# --------------------------------------------------------------------

def row_to_device_dict(row: pd.Series) -> Dict[str, Any]:
    """
    Map a device_data row to the device_dict shape expected
    by the existing multi-agent stack.
    """
    return {
        "device_id": str(row.get("device_id", "")),
        "brand": row.get("brand"),
        "model": row.get("model"),

        # R2v3 / risk score
        "r2v3_risk_score": row.get("r2v3_risk_score"),

        # GDPR / notes
        "gdpr_flags": row.get("gdpr_flags"),
        "overall_recommendation": row.get("overall_recommendation"),
        "notes": row.get("notes"),

        # Sensor grades
        "sensor_accel_grade": row.get("sensor_accel_grade"),
        "sensor_ppg_grade": row.get("sensor_ppg_grade"),
        "sensor_eda_grade": row.get("sensor_eda_grade"),
        "sensor_hrm_grade": row.get("sensor_hrm_grade"),
    }


def parse_risk_from_narrative(text: str) -> str:
    """
    Best-effort extraction of a RISK RATING line from the fused narrative.
    Falls back to 'Unknown' if not found.
    """
    if not isinstance(text, str):
        return "Unknown"

    for line in text.splitlines():
        up = line.upper().strip()
        if up.startswith("RISK RATING:"):
            try:
                return line.split(":", 1)[1].strip()
            except Exception:
                return "Unknown"
    return "Unknown"


# --------------------------------------------------------------------
# Sensor / FFT utilities (for visualization tab)
# --------------------------------------------------------------------

def load_sensor_timeseries() -> pd.DataFrame | None:
    if not SENSOR_TIMESERIES.exists():
        return None
    try:
        return pd.read_excel(SENSOR_TIMESERIES)
    except Exception:
        return None


def get_accel_timeseries(df_ts: pd.DataFrame, device_id: str) -> pd.DataFrame:
    """
    Filter accelerometer samples for a given device_id.
    """
    df = df_ts.copy()
    df["device_id"] = df["device_id"].astype(str)
    sub = df[
        (df["device_id"] == str(device_id))
        & (df["sensor"].astype(str).str.lower() == "accelerometer")
    ]
    if sub.empty:
        return sub

    sub = sub.sort_values("timestamp")
    sub["timestamp"] = pd.to_datetime(sub["timestamp"])
    return sub


def compute_fft(values: np.ndarray, timestamps: pd.Series) -> pd.DataFrame:
    """
    Compute a simple FFT magnitude spectrum for visualization.
    """
    if len(values) < 32:
        return pd.DataFrame(columns=["frequency_hz", "magnitude"])

    timestamps = pd.to_datetime(timestamps)
    dt_ms = np.median(
        np.diff(timestamps.values).astype("timedelta64[ms]").astype(float)
    )
    dt = dt_ms / 1000.0 if dt_ms > 0 else 0.01  # fallback
    fs = 1.0 / dt

    N = len(values)
    freq = np.fft.rfftfreq(N, d=1.0 / fs)
    fft_vals = np.abs(np.fft.rfft(values))

    return pd.DataFrame(
        {
            "frequency_hz": freq,
            "magnitude": fft_vals,
        }
    )


# --------------------------------------------------------------------
# Streamlit UI
# --------------------------------------------------------------------

def main():
    st.set_page_config(
        page_title="Device Passport – Spreadsheet Multi-Agent UI",
        layout="wide",
    )

    st.title("Device Passport – Spreadsheet-Driven Multi-Agent Audit")

    st.markdown(
        """
This UI replaces JSON/dict input with **spreadsheet-driven** device passports.

Back-end stack (unchanged):

- R2v3 agent (risk-minimizing, evidence-bound)
- GDPR agent (privacy-first, evidence-bound)
- Sensor Health agent (degradation-focused)
- Round 2 adversarial debate
- Chain-of-Verification (CoVe)
- Blind Arbiter
- Meta-Reviewer (reflection + cognitive verification)
- Append-only hash-chained audit logs
        """
    )

    # Upload or use default spreadsheet
    uploaded_file = st.file_uploader(
        "Upload a device spreadsheet (.xlsx, .xls, .csv), or leave empty to use sample_data/device_data.xlsx",
        type=["xlsx", "xls", "csv"],
    )

    df = None
    current_source = None
    error_msg = None

    if uploaded_file is not None:
        try:
            df = load_spreadsheet(uploaded_file)
            current_source = f"Uploaded file: {uploaded_file.name}"
        except Exception as exc:
            error_msg = f"Error loading uploaded file: {exc}"
    else:
        try:
            df = load_spreadsheet(DEFAULT_SPREADSHEET)
            current_source = f"Built-in spreadsheet: {DEFAULT_SPREADSHEET.name}"
        except Exception as exc:
            error_msg = (
                f"Could not load default spreadsheet '{DEFAULT_SPREADSHEET.name}': {exc}\n\n"
                "Place device_data.xlsx in sample_data/, or upload a file above."
            )

    if error_msg:
        st.error(error_msg)
        return

    st.success(f"Currently using: {current_source}")

    if "device_id" not in df.columns:
        st.error("Spreadsheet must include a 'device_id' column.")
        return

    # Normalize device_id to string for selection
    df["device_id"] = df["device_id"].astype(str)

    # Tabs: single audit, batch dashboard, sensor/FFT
    tab_single, tab_batch, tab_sensor = st.tabs(
        ["Single-Device Audit", "Batch Audits & Risk Dashboard", "Sensor / FFT Visualization"]
    )

    # ----------------------------------------------------------------
    # TAB 1: Single-Device Audit (full agentic arbiter pipeline)
    # ----------------------------------------------------------------
    with tab_single:
        st.subheader("Single-Device Multi-Agent Audit")

        st.caption("Preview of device data (first 80 rows):")
        st.dataframe(df.head(80), use_container_width=True)

        device_ids = df["device_id"].tolist()
        default_id = device_ids[0] if device_ids else None

        selected_id = st.selectbox(
            "Select a device_id to audit",
            options=device_ids,
            index=0 if default_id is not None else None,
        )

        if st.button("Run Multi-Agent Audit for Selected Device"):
            row = df[df["device_id"] == selected_id]
            if row.empty:
                st.error("Selected device_id not found in spreadsheet.")
            else:
                device_row = row.iloc[0]
                device_dict = row_to_device_dict(device_row)

                st.info("Running full audit with agents, CoVe, arbiter, and meta-review…")
                with st.spinner("Agents debating, verifying, arbitrating…"):
                    try:
                        orch = Orchestrator()
                        result = orch.run_device_audit(device_dict)
                    except Exception as exc:
                        st.error(f"Audit failed: {exc}")
                        return

                final_narrative = result["final_narrative"]
                final_uncertainty = result["final_uncertainty"]
                run_id = result["run_id"]

                st.markdown("### Final Audit Narrative")
                st.write(final_narrative)

                st.markdown("### Final Uncertainty Score")
                st.write(f"{final_uncertainty:.3f}")

                st.markdown("### Run ID")
                st.code(run_id)

                # Download audit log if available
                log_text = None
                for root, _, files in os.walk("audit_logs"):
                    for f in files:
                        if f.startswith(run_id):
                            with open(Path(root) / f, "r", encoding="utf-8") as lf:
                                log_text = lf.read()
                            break

                if log_text:
                    st.download_button(
                        label="Download Full Audit Log",
                        data=log_text,
                        file_name=f"{run_id}.txt",
                        mime="text/plain",
                    )
                else:
                    st.warning("Audit log file not found (this should be rare).")

    # ----------------------------------------------------------------
    # TAB 2: Batch Audits & Risk Dashboard
    # ----------------------------------------------------------------
    with tab_batch:
        st.subheader("Batch Audits & Risk Dashboard")

        device_ids = df["device_id"].tolist()
        default_multi = device_ids[:5]

        selected_ids = st.multiselect(
            "Select one or more device_ids for batch audit",
            options=device_ids,
            default=default_multi,
        )

        st.caption(
            "Note: Each device runs the full multi-agent pipeline. "
            "Large selections will be slower and consume more LLM calls."
        )

        if st.button("Run Batch Multi-Agent Audits"):
            if not selected_ids:
                st.warning("Select at least one device_id.")
            else:
                results = []
                orch = Orchestrator()

                for did in selected_ids:
                    row = df[df["device_id"] == did]
                    if row.empty:
                        continue
                    device_row = row.iloc[0]
                    device_dict = row_to_device_dict(device_row)

                    with st.spinner(f"Auditing device {did}…"):
                        try:
                            out = orch.run_device_audit(device_dict)
                        except Exception as exc:
                            st.error(f"Audit failed for {did}: {exc}")
                            continue

                    narrative = out["final_narrative"]
                    uncertainty = out["final_uncertainty"]
                    risk_from_text = parse_risk_from_narrative(narrative)

                    # Optional: simple numeric band from spreadsheet r2v3 score
                    r2_score = device_row.get("r2v3_risk_score")
                    r2_struct = evaluate_r2v3_risk(r2_score)
                    r2_band = r2_struct.get("risk_level", "unknown")

                    results.append(
                        {
                            "device_id": did,
                            "brand": device_row.get("brand"),
                            "model": device_row.get("model"),
                            "r2v3_score": r2_score,
                            "r2v3_band": r2_band,
                            "agentic_risk_rating": risk_from_text,
                            "final_uncertainty": uncertainty,
                        }
                    )

                if not results:
                    st.warning("No successful audits were completed.")
                else:
                    res_df = pd.DataFrame(results)

                    st.markdown("### Batch Audit Summary")
                    st.dataframe(res_df, use_container_width=True)

                    st.markdown("### Uncertainty Distribution (Lower = Better)")
                    chart_df = res_df.set_index("device_id")[["final_uncertainty"]]
                    st.bar_chart(chart_df)

    # ----------------------------------------------------------------
    # TAB 3: Sensor / FFT Visualization
    # ----------------------------------------------------------------
    with tab_sensor:
        st.subheader("Sensor / FFT Visualization (Accelerometer)")

        ts_df = load_sensor_timeseries()
        if ts_df is None:
            st.warning(
                "sample_data/sensor_timeseries.xlsx not found or could not be loaded.\n"
                f"Expected path: {SENSOR_TIMESERIES}"
            )
        else:
            # device IDs present in timeseries
            ts_df["device_id"] = ts_df["device_id"].astype(str)
            ts_device_ids = sorted(ts_df["device_id"].unique().tolist())

            if not ts_device_ids:
                st.warning("No device_ids found in sample_data/sensor_timeseries.xlsx.")
            else:
                selected_ts_id = st.selectbox(
                    "Select device_id for sensor visualization (accelerometer)",
                    options=ts_device_ids,
                )

                sub = get_accel_timeseries(ts_df, selected_ts_id)
                if sub.empty:
                    st.warning(
                        f"No accelerometer samples found for device_id={selected_ts_id}."
                    )
                else:
                    st.markdown("### Time Series (Accelerometer)")

                    ts_plot = sub[["timestamp", "value"]].set_index("timestamp")
                    st.line_chart(ts_plot)

                    st.markdown("### FFT Magnitude Spectrum (Accelerometer)")

                    fft_df = compute_fft(sub["value"].astype(float).values, sub["timestamp"])
                    if fft_df.empty:
                        st.info("Not enough samples for FFT (need at least 32).")
                    else:
                        # Focus on a reasonable frequency range for visual sanity
                        fft_view = fft_df[fft_df["frequency_hz"] <= fft_df["frequency_hz"].max()]
                        fft_view = fft_view.set_index("frequency_hz")
                        st.line_chart(fft_view)


if __name__ == "__main__":
    main()
