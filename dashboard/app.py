from __future__ import annotations

import json
import sys 
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import plotly.graph_objects as go
import streamlit as st
import torch

from dashboard.helpers import (
    CONFIDENCE_THRESHOLD,
    DEMO_SCENARIOS,
    DEFAULT_WINDOW_SIZE,
    build_forecast_probability_timeline,
    confidence_status,
    format_non_benign_forecast_index,
    build_forward_looking_alert,
    encoder_from_checkpoint,
    load_csv_bytes,
    prepare_records,
    raw_window_summary,
    select_demo_scenario,
    summarize_observed_progression,
    validate_probability_matrix,
    window_stage,
)
from src.evaluation.baseline_comparison import (
    format_baseline_comparison_table,
    load_baseline_comparison,
)
from src.explainability.explanation import build_reference_state, explain_forecast
from src.explainability.mitre_mapping import get_mitre_mapping
from src.forecasting.rollout import rollout_future
from src.forecasting.world_model import load_world_model
from src.preprocessing.feature_engineering import load_network_csv

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_PATH = ROOT / "models" / "network_attack_world_model.pt"
SAMPLE_PATH = ROOT / "data" / "sample" / "sample_network_traffic.csv"
EVALUATION_PATH = ROOT / "models" / "world_model_evaluation_results.json"
BASELINE_COMPARISON_PATH = ROOT / "models" / "baseline_comparison_results.json"
ROLLOUT_STEPS = 5

st.set_page_config(page_title="ThreatCast AI | SIH26153", page_icon=":material/security:", layout="wide")


@st.cache_resource(show_spinner="Loading saved world model…")
def cached_model(checkpoint_path: str, modified_ns: int):
    del modified_ns  # cache key invalidates the resource when the checkpoint changes
    return load_world_model(checkpoint_path, device="cpu")


@st.cache_data(show_spinner=False)
def cached_sample(path: str, modified_ns: int):
    del modified_ns
    return load_network_csv(path)


@st.cache_data(show_spinner=False)
def cached_evaluation(path: str, modified_ns: int) -> dict[str, Any]:
    del modified_ns
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


@st.cache_data(show_spinner=False)
def cached_baseline_comparison(path: str, modified_ns: int) -> dict[str, Any]:
    del modified_ns
    return load_baseline_comparison(path)


def _fmt(value: Any, digits: int = 2) -> str:
    return "—" if value is None else f"{float(value):,.{digits}f}"


def _probability_chart(
    probabilities: np.ndarray,
    class_names: tuple[str, ...],
    step_labels: tuple[str, ...],
) -> go.Figure:
    fig = go.Figure()
    for index, stage in enumerate(class_names):
        fig.add_trace(go.Scatter(x=step_labels, y=probabilities[:, index] * 100, mode="lines+markers", name=display_stage(stage)))
    fig.update_layout(
        title=None, xaxis_title="Forecast point",
        yaxis_title="Probability (%)", yaxis=dict(range=[0, 100]),
        xaxis=dict(categoryorder="array", categoryarray=list(step_labels), tickfont=dict(size=10)),
        legend=dict(
            title_text="", orientation="h", x=0.5, xanchor="center",
            y=-0.42, yanchor="top", font=dict(size=10),
        ),
        margin=dict(l=10, r=10, t=15, b=120), height=430,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#0c1927",
        font=dict(color="#dce8f2", family="Arial, sans-serif"),
        xaxis_gridcolor="#20384c", yaxis_gridcolor="#20384c",
        colorway=["#55c4dc", "#76a8ff", "#b995ff", "#f0b45b", "#7dd3a7"],
    )
    return fig


def _analyze(records, model, encoder, start, scenario_selection=None):
    ordered, features, labels = prepare_records(records, encoder, DEFAULT_WINDOW_SIZE)
    max_start = len(ordered) - DEFAULT_WINDOW_SIZE
    if start < 0 or start > max_start:
        raise ValueError("Selected sequence start is outside the available observations.")
    sequence = features[start : start + DEFAULT_WINDOW_SIZE]
    current_stage = window_stage(labels, start, DEFAULT_WINDOW_SIZE)
    observed_progression = summarize_observed_progression(labels[start : start + DEFAULT_WINDOW_SIZE])
    with torch.inference_mode():
        _, stage_logits = model(torch.as_tensor(sequence, dtype=torch.float32).unsqueeze(0))
    class_names = tuple(model.class_names)
    one_step_probabilities = torch.softmax(stage_logits, dim=-1)[0].detach().cpu().numpy()
    # Use the existing world-model API for its first step and autoregressive rollout.
    rollout = rollout_future(
        model, sequence, steps=ROLLOUT_STEPS, device="cpu",
        confidence_threshold=CONFIDENCE_THRESHOLD, class_names=class_names,
    )
    matrix = validate_probability_matrix(rollout.stage_probabilities, len(class_names))
    first_stage = class_names[int(np.argmax(one_step_probabilities))]
    first_confidence = float(np.max(one_step_probabilities))
    one_step_map = {name: float(one_step_probabilities[i]) for i, name in enumerate(class_names)}

    sample_records = cached_sample(str(SAMPLE_PATH), SAMPLE_PATH.stat().st_mtime_ns)
    reference_features = encoder.transform(sample_records)
    train_end = max(1, int(len(sample_records) * 0.70))
    reference_state = build_reference_state(reference_features[:train_end], model.feature_names)
    evidence = explain_forecast(
        sequence, first_stage, one_step_map, model.feature_names, reference_state,
        top_n=5, confidence_threshold=CONFIDENCE_THRESHOLD,
    )
    mitre = get_mitre_mapping(first_stage)
    return {
        "records": ordered, "features": features, "labels": labels,
        "sequence": sequence, "start": start, "current_stage": current_stage,
        "prediction": {"stage": first_stage, "confidence": first_confidence, "probabilities": one_step_map},
        "rollout": rollout, "matrix": matrix, "class_names": class_names,
        "evidence": evidence, "mitre": mitre,
        "scenario_selection": scenario_selection,
        "observed_progression": observed_progression,
        "telemetry": raw_window_summary(ordered[start : start + DEFAULT_WINDOW_SIZE]),
        "reference_note": f"Reference state uses the built-in synthetic training region ({train_end} observations).",
    }


from dashboard.ui_components import (
    display_stage,
    inject_custom_css,
    render_detail_cards,
    render_forecast_rail,
    render_header,
    render_stage_progression,
    render_summary_cards,
)

inject_custom_css()

with st.sidebar:
    st.markdown("## ThreatCast AI")
    st.caption("SOC analysis controls")
    st.badge("OFFLINE CSV ANALYSIS", icon=":material/database:", color="blue")
    st.markdown("#### DATA SOURCE")
    source = st.radio(
        "Choose traffic source",
        ("Built-in Synthetic Demo", "Upload CSV"),
        index=0,
        key="traffic_source",
    )
    uploaded = None
    scenario = None
    if source == "Upload CSV":
        st.markdown("#### UPLOAD")
        uploaded = st.file_uploader("Network traffic CSV", type=["csv"], key="network_csv")
    else:
        st.markdown("#### SCENARIO")
        scenario = st.selectbox("Demo scenario", DEMO_SCENARIOS, index=0, key="demo_scenario")
    st.caption("CSV schema: timestamp and protocol required; attack_stage optional. No data is downloaded automatically.")

if not CHECKPOINT_PATH.is_file():
    render_header("MODEL CHECKPOINT UNAVAILABLE")
    st.error(f"Saved world-model checkpoint not found: {CHECKPOINT_PATH}")
    st.stop()

try:
    model = cached_model(str(CHECKPOINT_PATH), CHECKPOINT_PATH.stat().st_mtime_ns)
    encoder_state = getattr(model, "feature_encoder_state", None)
    if not encoder_state:
        raise ValueError("The saved model checkpoint has no fitted feature encoder.")
    encoder = encoder_from_checkpoint(encoder_state)
except Exception as exc:
    render_header("MODEL CHECKPOINT UNAVAILABLE")
    st.error(f"Could not load the saved world model: {exc}")
    st.stop()

render_header()

records = None
records_error = None
try:
    if source == "Built-in Synthetic Demo":
        if not SAMPLE_PATH.is_file():
            raise FileNotFoundError(f"Built-in synthetic CSV not found: {SAMPLE_PATH}")
        records = cached_sample(str(SAMPLE_PATH), SAMPLE_PATH.stat().st_mtime_ns)
    elif uploaded is not None:
        records = load_csv_bytes(uploaded.getvalue())
except (ValueError, FileNotFoundError, UnicodeError, OSError) as exc:
    records_error = str(exc)

selected_start = 0
run = False
scenario_selection = None
if records is not None:
    try:
        ordered_for_control, _, _ = prepare_records(records, encoder, DEFAULT_WINDOW_SIZE)
        max_start = len(ordered_for_control) - DEFAULT_WINDOW_SIZE
        if source == "Built-in Synthetic Demo":
            scenario_selection = select_demo_scenario(records, scenario, DEFAULT_WINDOW_SIZE)
            suggested_start = scenario_selection["window_start"]
        else:
            suggested_start = max_start
        selection_context = (source, scenario if source == "Built-in Synthetic Demo" else None)
        if st.session_state.get("_scenario_selection_context") != selection_context:
            st.session_state["sequence_start_row"] = suggested_start
            st.session_state["_scenario_selection_context"] = selection_context
        with st.sidebar:
            st.markdown("#### SEQUENCE CONTROL")
            selected_start = st.slider(
                "Sequence start row", min_value=0, max_value=max_start,
                help="Select the 20-observation context used for the next-stage forecast.",
                key="sequence_start_row",
            )
            run = st.button("ANALYZE TRAFFIC", type="primary", width="stretch", key="analyze_traffic")
            st.caption(f"Selected window · rows {selected_start + 1}–{selected_start + DEFAULT_WINDOW_SIZE} of {len(ordered_for_control):,}")
    except ValueError as exc:
        records_error = str(exc)
if records is None:
    with st.sidebar:
        st.markdown("#### SEQUENCE CONTROL")
        st.caption("Select a source with at least 20 compatible observations.")
        st.button("ANALYZE TRAFFIC", type="primary", disabled=True, width="stretch", key="analyze_traffic_disabled")

if records_error:
    st.error(f"Unable to load this input: {records_error}")

with st.sidebar:
    st.divider()
    st.markdown("#### SYSTEM HEALTH")
    st.badge("MODEL CHECKPOINT LOADED", icon=":material/check_circle:", color="green")
    if records is not None and not records_error:
        st.badge("INPUT READY", icon=":material/check_circle:", color="green")
    elif records_error:
        st.badge("INPUT NEEDS ATTENTION", icon=":material/warning:", color="orange")
    else:
        st.badge("AWAITING CSV", icon=":material/pending:", color="gray")

# Do not keep showing a previous forecast after changing to an empty/invalid source.
if records is None and st.session_state.get("analysis_source") != source:
    st.session_state.pop("analysis", None)
    st.session_state.pop("analysis_source", None)

previous_analysis = st.session_state.get("analysis")
source_changed = st.session_state.get("analysis_source") != source
selection_changed = previous_analysis is not None and previous_analysis.get("start") != selected_start
scenario_changed = (
    previous_analysis is not None
    and source == "Built-in Synthetic Demo"
    and (previous_analysis.get("scenario_selection") or {}).get("scenario") != scenario
)
should_analyze = bool(run and records is not None) or bool(
    previous_analysis is not None and records is not None and (source_changed or selection_changed or scenario_changed)
)
if should_analyze:
    try:
        st.session_state["analysis"] = _analyze(records, model, encoder, selected_start, scenario_selection)
        st.session_state["analysis_source"] = source
    except (ValueError, FileNotFoundError, UnicodeError, OSError) as exc:
        st.error(f"Unable to analyze this input: {exc}")
    except Exception as exc:
        st.error(f"Analysis could not be completed ({type(exc).__name__}): {exc}")

analysis = st.session_state.get("analysis")
if analysis is None:
    with st.container(border=True):
        if records is not None and not records_error:
            st.badge("READY FOR ANALYSIS", icon=":material/check_circle:", color="green")
            st.subheader("Selected input is ready")
            st.write("The saved model and selected traffic window are loaded. Select **ANALYZE TRAFFIC** in the sidebar to generate the forecast.")
        else:
            st.badge("AWAITING INPUT", icon=":material/pending:", color="orange")
            st.subheader("Start an offline analysis")
            st.write("Choose a development scenario or upload a compatible network-traffic CSV, then select **ANALYZE TRAFFIC** in the sidebar.")
        st.caption("The dashboard does not monitor a live network. Forecast values appear only after analysis runs with the saved model.")
    st.stop()

result = analysis
prediction = result["prediction"]
rollout = result["rollout"]
class_names = result["class_names"]
observed = result["observed_progression"]
forecast_timeline = build_forecast_probability_timeline(
    prediction["probabilities"], rollout.stage_probabilities, class_names
)
security_alert = build_forward_looking_alert(
    current_observed_stage=observed["current_observed_stage"],
    forecasted_stage=prediction["stage"],
    model_confidence=prediction["confidence"],
    forecast_horizon=len(rollout.future_steps),
)

st.subheader("Executive SOC summary")
render_summary_cards(
    observed["current_observed_stage"], prediction["stage"], prediction["confidence"],
    format_non_benign_forecast_index(forecast_timeline[0]["non_benign_forecast_index"]),
    len(rollout.future_steps),
)
st.caption("Derived as 1 − P(BENIGN); this is a model-derived index, not a calibrated probability that an attack will occur.")
confidence = confidence_status(prediction["confidence"])
if confidence == "LOW_CONFIDENCE":
    st.warning("LOW CONFIDENCE — additional telemetry recommended.")

with st.container(border=True):
    st.markdown("### Forward-Looking Security Alert")
    alert_col, actions_col = st.columns([1.4, 1], vertical_alignment="top")
    with alert_col:
        if security_alert["low_confidence"]:
            st.warning(security_alert["headline"])
        elif security_alert["is_benign_forecast"]:
            st.success(security_alert["headline"])
        else:
            st.info(security_alert["headline"])
        render_detail_cards((
            ("CURRENT OBSERVED STAGE", display_stage(security_alert["current_observed_stage"])),
            ("PREDICTED NEXT STAGE", display_stage(security_alert["forecasted_next_stage"])),
            ("MODEL CONFIDENCE", f"{security_alert['model_confidence']:.1%}"),
            ("FORECAST HORIZON", f"{security_alert['forecast_horizon']} steps"),
        ))
        st.write(security_alert["explanation"])
        if security_alert["low_confidence"]:
            st.warning(security_alert["low_confidence_message"])
    with actions_col:
        st.markdown("#### Recommended analyst actions")
        for action in security_alert["recommended_actions"]:
            st.markdown(f"- {action}")
    st.caption(security_alert["disclaimer"])

with st.container(border=True):
    st.markdown("### Selected data context")
    label_available = any(str(label).strip().upper() not in {"", "UNKNOWN"} for label in result["labels"])
    render_detail_cards((
        ("DATA SOURCE", st.session_state.get("analysis_source", "Traffic data")),
        ("OBSERVATIONS", f"{len(result['records']):,}"),
        ("WINDOW", f"Rows {selected_start + 1}–{selected_start + DEFAULT_WINDOW_SIZE}"),
        ("LABEL STATUS", "Available" if label_available else "Unlabeled"),
    ))

with st.container(border=True):
    render_stage_progression(observed, prediction["stage"], prediction["confidence"])

st.subheader("AI FUTURE-STATE SIMULATION")
st.caption("Current selected window model output → five autoregressive forecast steps. The first node is the model's next-stage output for the selected window, not an observed label.")
render_forecast_rail(forecast_timeline)
st.caption("Confidence is the maximum class score. The Non-Benign Forecast Index is derived only from 1 − P(BENIGN); neither is a calibrated probability that an attack will occur.")

with st.container(border=True):
    st.markdown("### FORECAST STAGE PROBABILITY TIMELINE")
    st.caption("Model class probabilities across the forecast horizon. The model repeatedly forecasts the next state using its previous prediction.")
    timeline_matrix = np.asarray(
        [[row["probabilities"][stage] for stage in class_names] for row in forecast_timeline],
        dtype=np.float64,
    )
    timeline_labels = tuple(row["step"] for row in forecast_timeline)
    st.plotly_chart(
        _probability_chart(timeline_matrix, class_names, timeline_labels),
        width="stretch", theme="streamlit",
    )

with st.expander("Input, scenario & telemetry details", icon=":material/query_stats:"):
    scenario_result = result.get("scenario_selection")
    if scenario_result:
        st.markdown(f"**Demo scenario · {scenario_result['scenario']}**")
        st.write(scenario_result["description"])
        stage_summary = " · ".join(f"{stage}: {count}" for stage, count in scenario_result["stage_counts"].items())
        st.caption(
            f"Selected context rows {scenario_result['window_start'] + 1}–{scenario_result['window_end'] + 1} · "
            f"observed labels: {stage_summary}."
        )
        st.caption("Demo scenarios select traffic windows from the development dataset. Forecasts are generated by the trained world model.")
    else:
        st.markdown("**Uploaded CSV analysis**")
        st.caption("Forecasting the selected observation window from the uploaded traffic data.")
    telemetry = result["telemetry"]
    telemetry_items = (
        ("Packets · sum", telemetry["packet_count"]),
        ("Bytes · sum", telemetry["byte_count"]),
        ("Duration · mean", telemetry["duration"]),
        ("TTL · mean", telemetry["ttl"]),
        ("TCP window · mean", telemetry["tcp_window_size"]),
        ("Inter-arrival · mean", telemetry["inter_arrival_mean"]),
        ("Payload size · mean", telemetry["payload_size_mean"]),
    )
    telemetry_cols = st.columns(4)
    for column, (label, value) in zip(telemetry_cols, telemetry_items):
        column.metric(label, _fmt(value))
    st.caption(f"Source ports: {', '.join(telemetry['src_ports'][:8]) or '—'} · Destination ports: {', '.join(telemetry['dst_ports'][:8]) or '—'}")

evidence_col, mitre_col = st.columns([1.2, 1])
with evidence_col:
    with st.container(border=True):
        st.markdown("### MODEL EVIDENCE")
        st.caption("Reference-State Feature Deviation")
        st.caption(result["evidence"]["interpretation_note"])
        for item in result["evidence"]["evidence"]:
            row_cols = st.columns([1.4, 0.8, 0.8], vertical_alignment="center")
            row_cols[0].markdown(f"**{item['feature']}**")
            row_cols[1].badge(str(item["direction"]).upper(), color="blue")
            row_cols[2].caption(f"Deviation · {item['normalized_deviation']:.3f}")
            st.caption(item["description"])
        if not result["evidence"]["evidence"]:
            st.info("No feature deviations were identified against the reference state.")
        st.caption(result["reference_note"])
        st.caption("This evidence is based on deviation from the reference state and is not causal attribution.")

with mitre_col:
    with st.container(border=True):
        st.markdown("### MITRE ATT&CK CONTEXT")
        mapping = result["mitre"]
        if mapping["tactic"]:
            st.caption("TACTIC")
            st.markdown(f"**{mapping['tactic']}** · {mapping['tactic_id']}")
            for technique in mapping["potential_techniques"]:
                st.caption("TECHNIQUE · TECHNIQUE ID")
                st.markdown(f"[{technique['name']} ({technique['id']})]({technique['url']})")
                st.caption(f"CONTEXT · {technique['reason']}")
        else:
            st.badge("BENIGN", color="green")
            st.write("No suggested ATT&CK technique is associated with the BENIGN forecast in this prototype.")
        st.caption("Contextual mapping only — the traffic data does not by itself prove that the technique occurred.")
        st.caption(mapping["disclaimer"])

with st.container(border=True):
    st.markdown("### MODEL EVALUATION")
    st.caption("Synthetic development-data evaluation")
    try:
        evaluation = cached_evaluation(str(EVALUATION_PATH), EVALUATION_PATH.stat().st_mtime_ns)
        if evaluation.get("synthetic_demo_only") is True:
            st.warning("Synthetic development evaluation only · not a public benchmark · not real-world performance.")
        else:
            st.info("Evaluation scope is recorded in the saved result file; review its scope before interpreting metrics.")
        st.caption(f"Scope: {evaluation.get('evaluation_scope', 'not specified')} · checkpoint dataset: {getattr(model, 'dataset_type', 'synthetic development')}")
        eval_cols = st.columns(4)
        values = (
            ("Top-1 stage accuracy", evaluation.get("top1_stage_accuracy"), ".1%"),
            ("Transition accuracy", evaluation.get("transition_accuracy"), ".1%"),
            ("Trajectory exact match", evaluation.get("trajectory_exact_match_rate"), ".1%"),
            ("Autoregressive state MSE", evaluation.get("autoregressive_test_state_mse"), ".4f"),
        )
        for col, (label, value, fmt) in zip(eval_cols, values):
            col.metric(label, "—" if value is None else format(float(value), fmt))
        st.caption("Saved held-out chronological synthetic measurements; no metrics are computed from uploaded traffic.")
    except (FileNotFoundError, OSError, ValueError) as exc:
        st.info(f"Saved evaluation results are unavailable: {exc}")

    st.markdown("#### Temporal world model vs Logistic Regression baseline")
    try:
        comparison = cached_baseline_comparison(
            str(BASELINE_COMPARISON_PATH), BASELINE_COMPARISON_PATH.stat().st_mtime_ns
        )
        st.caption(comparison["evaluation_label"])
        header_cols = st.columns([1.2, 1, 1])
        header_cols[0].markdown("**Metric**")
        header_cols[1].markdown("**Temporal World Model**")
        header_cols[2].markdown("**Logistic Regression Baseline**")
        for row in format_baseline_comparison_table(comparison):
            cells = st.columns([1.2, 1, 1])
            cells[0].write(row["Metric"])
            cells[1].write(row["Temporal World Model"])
            cells[2].write(row["NumPy Logistic Regression"])
        st.caption("The Logistic Regression baseline uses one normalized observation, while the temporal world model uses the preceding 20 observations. Both are evaluated on the same chronological held-out targets.")
        st.caption("Synthetic development-data evaluation — not real-world benchmark performance.")
    except (FileNotFoundError, OSError, ValueError) as exc:
        st.info(f"Saved baseline comparison is unavailable: {exc}")

with st.container(border=True):
    st.markdown("#### Technical limitations")
    st.caption("This prototype analyzes uploaded CSV data or a built-in synthetic development dataset. It does not provide real-time monitoring or automatic traffic blocking.")
    st.caption("Forecast confidence is a model output, not a calibrated probability that an attack will occur.")
    st.caption("Synthetic development results should not be interpreted as benchmark performance. ATT&CK references are contextual and do not establish that a technique occurred.")

st.caption("ThreatCast AI  ·  SIH26153 • AI-Based Network Attack Forecasting")
