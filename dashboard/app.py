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

st.set_page_config(page_title="SIH26153 | Network Forecasting", page_icon="🛡️", layout="wide")


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
        fig.add_trace(go.Scatter(x=step_labels, y=probabilities[:, index] * 100, mode="lines+markers", name=stage))
    fig.update_layout(
        title="Stage probability distribution by forecast point", xaxis_title="Forecast point",
        yaxis_title="Probability (%)", yaxis=dict(range=[0, 100]),
        xaxis=dict(categoryorder="array", categoryarray=list(step_labels)),
        legend_title="Attack stage", margin=dict(l=10, r=10, t=55, b=10), height=350,
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


st.title("SIH26153 - AI Network Attack Forecasting")
st.caption("Defensive SOC prototype · CPU inference · Synthetic development model")

with st.sidebar:
    st.header("Traffic source")
    source = st.radio("Choose input", ("Built-in synthetic demo", "Upload CSV"), index=0)
    uploaded = None
    scenario = None
    if source == "Upload CSV":
        uploaded = st.file_uploader("Network traffic CSV", type=["csv"])
    else:
        scenario = st.selectbox("Demo Scenario", DEMO_SCENARIOS, index=0)
    run = st.button("Analyze traffic", type="primary", disabled=(source == "Upload CSV" and uploaded is None))
    st.caption("Required CSV columns: timestamp and protocol. attack_stage is optional; missing numeric flow features are imputed.")

# Section guide remains visible before analysis and becomes populated after Analyze.
if not CHECKPOINT_PATH.is_file():
    st.error(f"Saved world-model checkpoint not found: {CHECKPOINT_PATH}")
    st.stop()

try:
    model = cached_model(str(CHECKPOINT_PATH), CHECKPOINT_PATH.stat().st_mtime_ns)
    encoder_state = getattr(model, "feature_encoder_state", None)
    if not encoder_state:
        raise ValueError("The saved model checkpoint has no fitted feature encoder.")
    encoder = encoder_from_checkpoint(encoder_state)
except Exception as exc:
    st.error(f"Could not load the saved world model: {exc}")
    st.stop()

records = None
records_error = None
try:
    if source == "Built-in synthetic demo":
        if not SAMPLE_PATH.is_file():
            raise FileNotFoundError(f"Built-in synthetic CSV not found: {SAMPLE_PATH}")
        records = cached_sample(str(SAMPLE_PATH), SAMPLE_PATH.stat().st_mtime_ns)
    elif uploaded is not None:
        records = load_csv_bytes(uploaded.getvalue())
except (ValueError, FileNotFoundError, UnicodeError, OSError) as exc:
    records_error = str(exc)

selected_start = 0
scenario_selection = None
if records is not None:
    try:
        ordered_for_control, _, _ = prepare_records(records, encoder, DEFAULT_WINDOW_SIZE)
        max_start = len(ordered_for_control) - DEFAULT_WINDOW_SIZE
        if source == "Built-in synthetic demo":
            scenario_selection = select_demo_scenario(records, scenario, DEFAULT_WINDOW_SIZE)
            suggested_start = scenario_selection["window_start"]
        else:
            suggested_start = max_start
        selection_context = (source, scenario if source == "Built-in synthetic demo" else None)
        if st.session_state.get("_scenario_selection_context") != selection_context:
            st.session_state["sequence_start_row"] = suggested_start
            st.session_state["_scenario_selection_context"] = selection_context
        selected_start = st.sidebar.slider(
            "Sequence start row", min_value=0, max_value=max_start,
            help="Choose the 20-observation context used for the next-stage forecast. Scenario selection sets the initial row.",
            key="sequence_start_row",
        )
    except ValueError as exc:
        records_error = str(exc)

if records_error:
    st.error(f"Unable to load this CSV: {records_error}")

# Do not keep showing a previous forecast after the user switches to an empty
# or invalid upload source.
if records is None and st.session_state.get("analysis_source") != source:
    st.session_state.pop("analysis", None)
    st.session_state.pop("analysis_source", None)

previous_analysis = st.session_state.get("analysis")
source_changed = st.session_state.get("analysis_source") != source
selection_changed = previous_analysis is not None and previous_analysis.get("start") != selected_start
scenario_changed = (
    previous_analysis is not None
    and source == "Built-in synthetic demo"
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
        st.error(f"Unable to analyze this CSV: {exc}")
    except Exception as exc:
        st.error(f"Analysis failed: {type(exc).__name__}: {exc}")

analysis = st.session_state.get("analysis")
if analysis is None:
    st.info("Choose the built-in synthetic demo or upload a compatible CSV, then select **Analyze traffic**. No data is downloaded automatically.")
    for title in (
        "Dataset Upload", "Current Network State", "Predicted Next Attack Stage",
        "5-Step Attack Trajectory", "Attack Probability Timeline", "Forecast Evidence",
        "MITRE ATT&CK Context", "Model and Evaluation",
    ):
        with st.container(border=True):
            st.subheader(title)
            st.caption("Run an analysis to populate this panel.")
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

st.success(f"Analysis ready · {st.session_state.get('analysis_source', 'traffic data')} · {len(result['records']):,} observations · {result['features'].shape[1]} model features")
scenario_result = result.get("scenario_selection")
with st.container(border=True):
    if scenario_result:
        st.subheader(f"Demo Scenario · {scenario_result['scenario']}")
        st.markdown("**Scenario Description**")
        st.write(scenario_result["description"])
        stage_summary = " · ".join(f"{stage}: {count}" for stage, count in scenario_result["stage_counts"].items())
        st.caption(
            f"Selected model context: rows {scenario_result['window_start'] + 1}–{scenario_result['window_end'] + 1}. "
            f"Actual labels in context: {stage_summary}."
        )
    else:
        st.subheader("Uploaded CSV Scenario")
        st.write("Forecasting the selected 20-observation context from the uploaded traffic data.")
    st.caption("Demo scenarios select traffic windows from the development dataset. Forecasts are generated by the trained world model.")

with st.container(border=True):
    st.subheader("Attack Progression")
    if observed["observed_stages"]:
        st.markdown(" → ".join(observed["observed_stages"]))
        st.caption("Observed from traffic window · ordered from the actual stage labels.")
    else:
        st.info("No recognized attack stage is labeled in this traffic window.")
    if observed["benign_observed"]:
        st.caption("Observed from traffic window: BENIGN. It is shown separately, not as an attack stage.")
    if observed["complete_attack_progression"]:
        st.caption("All four attack stages are present in this window in the observed order.")
    else:
        st.info("This selected traffic window does not contain the complete attack progression.")

    current_col, forecast_col = st.columns(2)
    with current_col:
        st.markdown("**CURRENT OBSERVED STAGE**")
        st.markdown(f"### {observed['current_observed_stage']}")
    with forecast_col:
        st.markdown("**AI FORECAST**")
        st.markdown(f"### {prediction['stage']}")
        st.write(f"Confidence: **{prediction['confidence']:.1%}**")
    st.caption("The observed stage comes from the selected rows' labels. The AI forecast and confidence come from the existing saved world model.")

with st.container(border=True):
    st.subheader("Forward-Looking Security Alert")
    st.info(security_alert["headline"])
    alert_columns = st.columns(4)
    alert_columns[0].metric("CURRENT OBSERVED STAGE", security_alert["current_observed_stage"])
    alert_columns[1].metric("FORECASTED NEXT STAGE", security_alert["forecasted_next_stage"])
    alert_columns[2].metric("MODEL CONFIDENCE", f"{security_alert['model_confidence']:.1%}")
    alert_columns[3].metric("FORECAST HORIZON", f"{security_alert['forecast_horizon']} steps")
    st.write(security_alert["explanation"])
    if security_alert["low_confidence"]:
        st.warning(security_alert["low_confidence_message"])
    st.markdown("**Recommended Analyst Action**")
    for action in security_alert["recommended_actions"]:
        st.markdown(f"- {action}")
    st.caption(security_alert["disclaimer"])

metric_columns = st.columns(4)
with metric_columns[0]:
    st.metric("Current observed stage", result["current_stage"])
with metric_columns[1]:
    st.metric("Next predicted stage", prediction["stage"])
with metric_columns[2]:
    st.metric("Forecast confidence", f"{prediction['confidence']:.1%}")
with metric_columns[3]:
    status = confidence_status(prediction["confidence"])
    st.metric("Confidence check", status, delta="Threshold 0.60", delta_color="off")
if status == "LOW_CONFIDENCE":
    st.warning("LOW_CONFIDENCE: treat this forecast as uncertain; confidence is not a calibrated incident probability.")

left, right = st.columns([1, 1])
with left:
    with st.container(border=True):
        st.subheader("1 · Dataset Upload")
        st.write(f"Source: **{st.session_state.get('analysis_source', 'traffic data')}**")
        st.write(f"Rows: **{len(result['records']):,}** · Sequence length: **{DEFAULT_WINDOW_SIZE}** · Features: **{result['features'].shape[1]}**")
        st.caption("CSV upload uses the Phase 2 schema validator and the model checkpoint’s training-fitted encoder. Missing numeric inputs are safely median-imputed.")
with right:
    with st.container(border=True):
        st.subheader("2 · Current Network State")
        telemetry = result["telemetry"]
        cols = st.columns(4)
        for col, key, label in zip(cols, ("packet_count", "byte_count", "duration", "ttl"), ("Packets · sum", "Bytes · sum", "Duration · mean", "TTL · mean")):
            col.metric(label, _fmt(telemetry[key]))
        cols2 = st.columns(4)
        for col, key, label in zip(cols2, ("tcp_window_size", "inter_arrival_mean", "payload_size_mean", "dst_ports"), ("TCP window · mean", "Inter-arrival · mean", "Payload size · mean", "Destination ports")):
            val = telemetry[key]
            col.metric(label, ", ".join(val[:4]) if isinstance(val, list) and val else (_fmt(val) if not isinstance(val, list) else "—"))
        st.caption(f"Source ports in selected window: {', '.join(telemetry['src_ports'][:8]) or '—'}")

left, right = st.columns([1, 1])
with left:
    with st.container(border=True):
        st.subheader("3 · Predicted Next Attack Stage")
        st.markdown(f"### {prediction['stage']}")
        st.write(f"Confidence: **{prediction['confidence']:.1%}**")
        st.caption("This is a synthetic-development model output, not a real-world detection or calibrated probability.")
        st.write("Class probabilities")
        for stage in class_names:
            st.progress(prediction["probabilities"][stage], text=f"{stage}: {prediction['probabilities'][stage]:.1%}")
with right:
    with st.container(border=True):
        st.subheader("4 · 5-Step Attack Trajectory")
        for item in rollout.future_steps:
            suffix = " · LOW_CONFIDENCE" if item["confidence_status"] == "LOW_CONFIDENCE" else ""
            st.write(f"**Step {item['step']}: {item['predicted_stage']}** · {item['confidence']:.1%}{suffix}")
        st.caption("Each predicted network state is fed into the next rollout step (autoregressive forecast).")

left, right = st.columns([1, 1])
with left:
    with st.container(border=True):
        st.subheader("5 · Forecast Stage Probability Timeline")
        st.caption(
            "The model repeatedly forecasts the next state using its previous prediction, "
            "producing an autoregressive view of possible future progression."
        )
        timeline_columns = st.columns(6)
        for column, row in zip(timeline_columns, forecast_timeline):
            column.metric(
                row["step"], row["predicted_stage"],
                delta=f"Confidence {row['confidence']:.1%}", delta_color="off",
            )
            column.caption(
                "Non-Benign Forecast Index\n"
                + format_non_benign_forecast_index(row["non_benign_forecast_index"])
            )
        st.caption(
            "Derived as 1 − P(BENIGN); this is a model-derived index, not a calibrated probability that an attack will occur."
        )
        timeline_matrix = np.asarray(
            [[row["probabilities"][stage] for stage in class_names] for row in forecast_timeline],
            dtype=np.float64,
        )
        timeline_labels = tuple(row["step"] for row in forecast_timeline)
        st.plotly_chart(
            _probability_chart(timeline_matrix, class_names, timeline_labels),
            width="stretch", theme="streamlit",
        )
with right:
    with st.container(border=True):
        st.subheader("6 · Attack Trajectory")
        st.write(" → ".join(rollout.predicted_stages))
        st.write("Step confidence and status")
        for item in rollout.future_steps:
            st.write(f"Step {item['step']}: {item['predicted_stage']} · {item['confidence']:.1%} · {item['confidence_status']}")

left, right = st.columns([1, 1])
with left:
    with st.container(border=True):
        st.subheader("7 · Explainable AI · Feature Evidence")
        st.caption(result["evidence"]["interpretation_note"])
        st.caption(result["reference_note"])
        for item in result["evidence"]["evidence"]:
            st.write(f"**{item['feature']}** · {item['direction']} · deviation {item['normalized_deviation']:.3f}")
            st.caption(item["description"])
        if not result["evidence"]["evidence"]:
            st.info("No feature deviations were identified against the reference state.")
        st.caption("Evidence describes deviations from a reference, not causal attribution or model-gradient importance.")
with right:
    with st.container(border=True):
        st.subheader("8 · MITRE ATT&CK Context")
        mapping = result["mitre"]
        if mapping["tactic"]:
            st.write(f"**{mapping['tactic']}** ({mapping['tactic_id']})")
            for technique in mapping["potential_techniques"]:
                st.markdown(f"[{technique['name']} ({technique['id']})]({technique['url']})")
                st.caption(technique["reason"])
        else:
            st.write("BENIGN has no suggested ATT&CK technique in this prototype.")
        st.caption(mapping["disclaimer"])

with st.container(border=True):
    st.subheader("Model Evaluation & Baseline Comparison")
    try:
        evaluation = cached_evaluation(str(EVALUATION_PATH), EVALUATION_PATH.stat().st_mtime_ns)
        if evaluation.get("synthetic_demo_only") is True:
            st.warning("Synthetic development evaluation only · not a public benchmark · not real-world performance.")
        else:
            st.info("Evaluation scope is recorded in the saved result file; review its scope before interpreting metrics.")
        st.caption(f"Scope: {evaluation.get('evaluation_scope', 'not specified')}")
        st.caption(f"Checkpoint dataset type: {getattr(model, 'dataset_type', 'synthetic development')}")
        eval_cols = st.columns(4)
        values = (
            ("Top-1 stage accuracy", evaluation.get("top1_stage_accuracy"), ".1%"),
            ("Transition accuracy", evaluation.get("transition_accuracy"), ".1%"),
            ("Trajectory exact match", evaluation.get("trajectory_exact_match_rate"), ".1%"),
            ("Autoregressive state MSE", evaluation.get("autoregressive_test_state_mse"), ".4f"),
        )
        for col, (label, value, fmt) in zip(eval_cols, values):
            col.metric(label, "—" if value is None else format(float(value), fmt))
        st.caption("These are the saved held-out chronological synthetic evaluation results; no metrics are computed from the uploaded data.")
    except FileNotFoundError:
        st.info("Saved evaluation results are unavailable. No metrics are being inferred or fabricated.")

    st.markdown("**Saved baseline comparison**")
    try:
        comparison = cached_baseline_comparison(
            str(BASELINE_COMPARISON_PATH), BASELINE_COMPARISON_PATH.stat().st_mtime_ns
        )
        st.caption(comparison["evaluation_label"])
        headers = st.columns([1.2, 1, 1])
        headers[0].markdown("**Metric**")
        headers[1].markdown("**NumPy Logistic Regression**")
        headers[2].markdown("**Temporal World Model**")
        for row in format_baseline_comparison_table(comparison):
            cells = st.columns([1.2, 1, 1])
            cells[0].write(row["Metric"])
            cells[1].write(row["NumPy Logistic Regression"])
            cells[2].write(row["Temporal World Model"])
        st.caption(
            "The Logistic Regression baseline uses one normalized observation, while the temporal world model uses the preceding 20 observations. "
            "Both are evaluated on the same chronological held-out targets."
        )
        st.caption(
            "These results are development-data measurements and should not be interpreted as real-world benchmark performance."
        )
    except (FileNotFoundError, OSError, ValueError) as exc:
        st.info(f"Saved baseline comparison is unavailable: {exc}")
