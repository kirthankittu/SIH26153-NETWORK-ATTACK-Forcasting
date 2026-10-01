"""Presentation-only components for the ThreatCast Streamlit dashboard."""

from __future__ import annotations

from collections.abc import Mapping
from html import escape

import streamlit as st


def inject_custom_css() -> None:
    """Apply the dashboard's restrained SOC visual system."""
    st.markdown(
        """<style>
        :root {
            color-scheme: dark;
            --tc-bg: #07111d;
            --tc-panel: #0c1927;
            --tc-panel-raised: #102235;
            --tc-border: #20384c;
            --tc-accent: #45b8d2;
            --tc-text: #e8f0f7;
            --tc-muted: #9aafc1;
        }
        .stApp { background: var(--tc-bg); color: var(--tc-text); }
        .block-container { max-width: 1500px; padding-top: 1.35rem; padding-bottom: 2rem; }
        [data-testid="stHeader"] { background: rgba(7, 17, 29, 0.88); }
        [data-testid="stSidebar"] { background: #091522; border-right: 1px solid var(--tc-border); }
        [data-testid="stVerticalBlockBorderWrapper"] {
            background: rgba(12, 25, 39, 0.82);
            border-color: var(--tc-border) !important;
            border-radius: 12px;
        }
        [data-testid="stMetric"] {
            background: var(--tc-panel);
            border: 1px solid var(--tc-border);
            border-radius: 10px;
            padding: 0.85rem 0.95rem;
            min-height: 104px;
        }
        [data-testid="stMetricLabel"] p { color: var(--tc-muted); font-size: 0.76rem; letter-spacing: 0.055em; }
        [data-testid="stMetricValue"] { color: var(--tc-text); font-weight: 650; }
        [data-testid="stMetricDelta"] { color: var(--tc-muted); }
        .stButton button[kind="primary"] {
            background: #087e9d; border: 1px solid #35b7d5; color: white;
            border-radius: 8px; font-weight: 700; letter-spacing: 0.04em;
        }
        .stButton button[kind="primary"]:hover { background: #096b86; border-color: #67d0e5; }
        h1 { letter-spacing: -0.035em; font-weight: 750; }
        h2, h3 { letter-spacing: -0.02em; }
        a { color: #70cbe0 !important; }
        [data-testid="stCaptionContainer"] { color: var(--tc-muted); }
        .tc-summary-grid {
            display: grid; grid-template-columns: 1fr 1.3fr 1fr 1.18fr .82fr;
            gap: 10px; margin: .25rem 0 .5rem;
        }
        .tc-kpi {
            min-width: 0; min-height: 104px; box-sizing: border-box;
            background: linear-gradient(155deg, #102235, #0b1825 72%);
            border: 1px solid #263f54; border-top: 2px solid #33566f;
            border-radius: 7px; padding: 15px 14px 12px;
        }
        .tc-kpi-primary { background: linear-gradient(155deg, #102c3d, #0b1b29 74%); border-top-color: var(--tc-accent); }
        .tc-kpi-label { color: #9aafc1; font-size: .66rem; line-height: 1.35; letter-spacing: .075em; font-weight: 700; overflow-wrap: anywhere; }
        .tc-kpi-value { color: #edf5fb; font-size: 1.24rem; line-height: 1.25; font-weight: 720; margin-top: 12px; overflow-wrap: anywhere; }
        .tc-kpi-primary .tc-kpi-value { color: #71d4e7; font-size: 1.48rem; }
        .tc-kpi-note { color: #879daf; font-size: .72rem; line-height: 1.35; margin-top: 4px; }
        .tc-detail-grid { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 8px; margin: 8px 0; }
        .tc-detail-card { min-width: 0; box-sizing: border-box; padding: 10px 11px; background: #0d1b29; border: 1px solid #253e52; border-radius: 6px; }
        .tc-detail-label { color: #91a7ba; font-size: .64rem; line-height: 1.3; letter-spacing: .065em; font-weight: 700; overflow-wrap: anywhere; }
        .tc-detail-value { color: #eaf2f8; font-size: .92rem; line-height: 1.3; font-weight: 700; margin-top: 5px; overflow-wrap: anywhere; }
        .tc-forecast-rail { display: grid; grid-template-columns: repeat(6, minmax(0, 1fr)); gap: 0; align-items: stretch; margin: 10px 0 14px; }
        .tc-forecast-node {
            position: relative; min-width: 0; box-sizing: border-box; text-align: center;
            background: #0c1a28; border: 1px solid #294156; border-radius: 7px;
            padding: 13px 8px 12px; margin: 0 6px;
        }
        .tc-forecast-node:not(:last-child)::after {
            content: ""; position: absolute; z-index: 2; right: -13px; top: 50%;
            width: 12px; border-top: 1px solid #69bdd0;
        }
        .tc-forecast-node:not(:last-child)::before {
            content: "›"; position: absolute; z-index: 3; right: -14px; top: calc(50% - 11px);
            color: #69bdd0; font-size: 1.15rem; font-weight: 700;
        }
        .tc-forecast-current { background: #102a38; border-color: #378197; }
        .tc-forecast-future { border-top: 2px solid #356278; }
        .tc-forecast-step { color: #8fa8bb; font-size: .68rem; font-weight: 750; letter-spacing: .08em; }
        .tc-forecast-stage { color: #eaf2f8; font-size: .91rem; line-height: 1.25; font-weight: 700; margin: 12px 0 9px; overflow-wrap: anywhere; }
        .tc-forecast-detail { color: #9aafc1; font-size: .7rem; line-height: 1.55; }
        .tc-forecast-index { color: #72cde0; font-weight: 700; }
        .tc-attack-track { display: flex; flex-wrap: wrap; align-items: center; gap: 8px; margin: 12px 0; }
        .tc-attack-node { min-width: 132px; background: #0d1c2a; border: 1px solid #2c465a; border-radius: 7px; padding: 11px 12px; }
        .tc-attack-node-complete { border-top: 2px solid #4b9b79; }
        .tc-attack-node-current { background: #102a38; border: 1px solid #4bb1c8; border-top: 2px solid #4bb1c8; }
        .tc-attack-stage { color: #e8f0f7; font-size: .78rem; font-weight: 720; line-height: 1.25; }
        .tc-attack-state { color: #9aafc1; font-size: .64rem; letter-spacing: .07em; margin-top: 5px; }
        .tc-attack-link { color: #648195; font-size: 1rem; }
        .tc-attack-forecast { border: 1px solid #3d7386; border-top: 2px solid var(--tc-accent); background: #102331; }
        .tc-current-benign { display: inline-block; margin: 4px 0 8px; padding: 7px 10px; color: #a9e5c1; background: #10261f; border: 1px solid #2d624b; border-radius: 6px; font-size: .75rem; font-weight: 700; }
        @media (max-width: 850px) {
            .block-container { padding-left: 1rem; padding-right: 1rem; }
            .tc-summary-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
            .tc-kpi-primary { grid-column: span 2; }
            .tc-detail-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
            .tc-forecast-rail { grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 8px; }
            .tc-forecast-node { margin: 0; }
            .tc-forecast-node:not(:last-child)::after { display: none; }
            .tc-forecast-node:not(:last-child)::before { display: none; }
            .tc-attack-node { min-width: 0; }
        }
        @media (max-width: 520px) {
            .tc-summary-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
            .tc-kpi-value { font-size: 1.05rem; }
            .tc-kpi-primary .tc-kpi-value { font-size: 1.25rem; }
            .tc-forecast-rail { grid-template-columns: repeat(2, minmax(0, 1fr)); }
            .tc-forecast-node:nth-child(even)::after { display: none; }
        }
        </style>""",
        unsafe_allow_html=True,
    )


def render_header(engine_status: str = "ANALYSIS ENGINE ONLINE") -> None:
    """Render the product identity and non-real-time operating context."""
    with st.container(border=True):
        title_col, status_col = st.columns([2.7, 1.4], vertical_alignment="center")
        with title_col:
            st.title("ThreatCast AI")
            st.markdown("**AI-Based Network Attack Forecasting & Future-State Simulation**")
            st.caption("NETWORK SECURITY  ·  TEMPORAL AI ENGINE  ·  5-STEP FORECASTING")
        with status_col:
            status_color = "green" if engine_status == "ANALYSIS ENGINE ONLINE" else "orange"
            st.badge(engine_status, icon=":material/check_circle:" if status_color == "green" else ":material/warning:", color=status_color)
            st.caption("Offline CSV / demo analysis")
            st.badge("SIH26153 • PROTOTYPE", color="blue")


def display_stage(stage: object) -> str:
    """Format a known model or observed stage label for presentation."""
    normalized = str(stage).strip().upper()
    labels = {
        "BENIGN": "BENIGN",
        "RECONNAISSANCE": "RECONNAISSANCE",
        "INITIAL_ACCESS": "INITIAL ACCESS",
        "LATERAL_MOVEMENT": "LATERAL MOVEMENT",
        "COMMAND_AND_CONTROL": "COMMAND & CONTROL",
        "UNKNOWN": "UNKNOWN",
    }
    return labels.get(normalized, escape(str(stage).strip() or "UNKNOWN"))


def render_summary_cards(
    current_stage: object,
    forecast_stage: object,
    confidence: float,
    non_benign_index: str,
    horizon: int,
) -> None:
    """Render compact responsive KPIs without metric-value truncation."""
    cards = (
        ("CURRENT STATE", display_stage(current_stage), "Observed in selected window", ""),
        ("FORECAST · NEXT STAGE", display_stage(forecast_stage), "Saved model output", "primary"),
        ("FORECAST CONFIDENCE", f"{confidence:.1%}", "Maximum class score", ""),
        ("NON-BENIGN FORECAST INDEX", non_benign_index, "Derived from 1 − P(BENIGN)", ""),
        ("FORECAST HORIZON", f"{horizon} STEPS", "Autoregressive", ""),
    )
    markup = ['<div class="tc-summary-grid">']
    for label, value, note, emphasis in cards:
        card_class = "tc-kpi tc-kpi-primary" if emphasis else "tc-kpi"
        markup.append(
            f'<div class="{card_class}"><div class="tc-kpi-label">{escape(label)}</div>'
            f'<div class="tc-kpi-value">{escape(value)}</div>'
            f'<div class="tc-kpi-note">{escape(note)}</div></div>'
        )
    markup.append("</div>")
    st.markdown("".join(markup), unsafe_allow_html=True)


def render_detail_cards(cards: tuple[tuple[str, str], ...]) -> None:
    """Render concise supporting values with wrapping rather than metric truncation."""
    markup = ['<div class="tc-detail-grid">']
    for label, value in cards:
        markup.append(
            '<div class="tc-detail-card">'
            f'<div class="tc-detail-label">{escape(label)}</div>'
            f'<div class="tc-detail-value">{escape(value)}</div></div>'
        )
    markup.append("</div>")
    st.markdown("".join(markup), unsafe_allow_html=True)


def render_forecast_rail(timeline: list[Mapping[str, object]]) -> None:
    """Render the current-window model output and five rollout points as a rail."""
    nodes = ['<div class="tc-forecast-rail">']
    for index, row in enumerate(timeline):
        is_current = index == 0
        step = "CURRENT WINDOW" if is_current else f"+{index}"
        status = "NEXT-STAGE OUTPUT" if is_current else "AUTOREGRESSIVE"
        node_class = "tc-forecast-node tc-forecast-current" if is_current else "tc-forecast-node tc-forecast-future"
        stage = escape(display_stage(row["predicted_stage"]))
        confidence = escape(f"{float(row['confidence']):.1%}")
        index_value = escape(f"{float(row['non_benign_forecast_index']):.2%}")
        nodes.append(
            f'<div class="{node_class}"><div class="tc-forecast-step">{step}</div>'
            f'<div class="tc-forecast-stage">{stage}</div>'
            f'<div class="tc-forecast-detail">{status}<br>Confidence · {confidence}'
            f'<br>Non-Benign index · <span class="tc-forecast-index">{index_value}</span></div></div>'
        )
    nodes.append("</div>")
    st.markdown("".join(nodes), unsafe_allow_html=True)


def render_stage_progression(
    observed: Mapping[str, object],
    forecast_stage: str,
    forecast_confidence: float,
) -> None:
    """Display only helper-derived observed attack stages and the model forecast."""
    st.subheader("Attack progression")
    stages = list(observed["observed_stages"])
    current = str(observed["current_observed_stage"])

    if observed["benign_observed"]:
        benign_state = "CURRENT" if current == "BENIGN" else "OBSERVED"
        st.markdown(f'<div class="tc-current-benign">BENIGN · {benign_state}</div>', unsafe_allow_html=True)
        st.caption("Benign observations are shown separately from the attack-stage chain.")

    if stages:
        nodes = ['<div class="tc-attack-track">']
        for index, stage in enumerate(stages):
            is_current = stage == current
            node_class = "tc-attack-node tc-attack-node-current" if is_current else "tc-attack-node tc-attack-node-complete"
            state = "CURRENT" if is_current else "OBSERVED"
            marker = "●" if is_current else "✓"
            nodes.append(
                f'<div class="{node_class}"><div class="tc-attack-stage">{marker} {escape(display_stage(stage))}</div>'
                f'<div class="tc-attack-state">{state}</div></div>'
            )
            if index < len(stages) - 1:
                nodes.append('<div class="tc-attack-link">→</div>')
        nodes.append('<div class="tc-attack-link">→</div>')
        nodes.append(
            f'<div class="tc-attack-node tc-attack-forecast"><div class="tc-attack-stage">'
            f'{escape(display_stage(forecast_stage))}</div><div class="tc-attack-state">AI FORECAST · {forecast_confidence:.1%}</div></div>'
        )
        nodes.append("</div>")
        st.markdown("".join(nodes), unsafe_allow_html=True)
        st.caption("Observed stages follow the selected window's labels. The forecast is the saved model's next-stage output.")
    else:
        current_label = current if current not in {"", "UNKNOWN"} else "UNKNOWN / UNLABELED"
        if observed["benign_observed"]:
            st.caption("No attack stage is labeled in this window. The current BENIGN label is shown separately above.")
            st.markdown(
                '<div class="tc-attack-track">'
                f'<div class="tc-attack-node tc-attack-forecast"><div class="tc-attack-stage">{escape(display_stage(forecast_stage))}</div><div class="tc-attack-state">AI FORECAST · {forecast_confidence:.1%}</div></div>'
                '</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="tc-attack-track">'
                f'<div class="tc-attack-node tc-attack-node-current"><div class="tc-attack-stage">{escape(display_stage(current_label))}</div><div class="tc-attack-state">CURRENT · UNKNOWN LABEL</div></div>'
                '<div class="tc-attack-link">→</div>'
                f'<div class="tc-attack-node tc-attack-forecast"><div class="tc-attack-stage">{escape(display_stage(forecast_stage))}</div><div class="tc-attack-state">AI FORECAST · {forecast_confidence:.1%}</div></div>'
                '</div>',
                unsafe_allow_html=True,
            )
            st.caption("No recognized attack stage is labeled in this window; unknown labels are not mapped to an attack stage.")

    if observed["complete_attack_progression"]:
        st.caption("The complete four-stage attack progression is present in the observed window.")
    else:
        st.caption("Only observed stages are shown; stages absent from this window are not inferred or displayed.")
