# SIH26153 — AI-Based Network Attack Forecasting

## Project objective

Build a defensive prototype that ingests network traffic from CSV, converts it into temporal network states, forecasts likely next attack stages and short K-step trajectories, and presents the results in a Streamlit dashboard. The project is a forecasting and analysis tool; it does not perform offensive actions.

## Architecture overview

- `data/raw/`: user-provided source traffic data.
- `data/processed/`: derived, prepared data.
- `data/sample/`: optional manually supplied example data; no data is downloaded automatically.
- `src/preprocessing/`: CSV validation, checkpoint-compatible feature encoding, and temporal windows.
- `src/forecasting/`: LSTM/world model inference and autoregressive K-step rollout.
- `src/evaluation/`: forecast metrics and saved synthetic evaluation results.
- `src/explainability/`: descriptive feature evidence and contextual ATT&CK mappings.
- `src/utils/`: shared configuration and utilities.
- `models/`: model artifacts when a model is trained.
- `dashboard/`: Streamlit SOC-style interface for synthetic demo and user CSV inference.
- `notebooks/`: exploration and experiments.
- `tests/`: future automated checks.

## Setup (Windows 11, PowerShell)

Python 3.12 or newer and Git should be installed. From the project directory:

```powershell
py -3.12 -m venv .venv
```

If `.venv` already exists, keep it and activate it instead of recreating it:

```powershell
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, use the current-process policy just for this terminal:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

Install project requirements:

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

To run the dashboard:

```powershell
streamlit run dashboard/app.py
```

For the Phase 6 demo on port 8765:

```powershell
streamlit run dashboard/app.py --server.port 8765
```

Choose **Built-in synthetic demo** or upload a UTF-8 CSV, then select **Analyze traffic**. The required CSV fields are `timestamp` and `protocol`; `attack_stage` is optional and is used only as the observed stage label. The supported flow columns are `src_ip`, `dst_ip`, `src_port`, `dst_port`, `packet_count`, `byte_count`, `duration`, `tcp_flags`, `ttl`, `tcp_window_size`, `inter_arrival_mean`, and `payload_size_mean`. Missing numerical values are imputed by the training-fitted encoder; unseen categorical values use its unknown-category code. At least 20 observations are needed.

Alternatively, double-click `run_dashboard.bat` from Explorer or run it in PowerShell.

PyShark is a Python wrapper. Live packet capture or some capture-file operations may also require Wireshark/TShark installed on the machine. The initial prototype does not capture traffic; CSV is the planned initial input.

## Current implementation status

The project includes CSV preprocessing, a multi-episode synthetic dataset generator, a next-stage LSTM, a multi-task world model with autoregressive rollout, forecast metrics, feature evidence explanations, contextual ATT&CK mappings, and a Streamlit dashboard. The dashboard loads `models/network_attack_world_model.pt` on CPU, uses the encoder saved with that checkpoint, forecasts one next stage and a 5-step autoregressive trajectory, and presents stage probabilities, descriptive feature deviations, and contextual ATT&CK links. The saved results in `models/world_model_evaluation_results.json` are held-out chronological synthetic-demo results only; they are not public-benchmark or real-world results. Upload metrics are not calculated. No external datasets are downloaded automatically.

## Planned ML components

1. Validate dashboard behavior against approved public benchmark data when available.
2. Improve calibration and compare temporal model variants.
3. Expand upload validation and operational testing with representative labeled and unlabeled traffic exports.

## Explainability

The explainability helpers provide **driving feature analysis** by comparing the latest state (or each predicted state in a rollout) with a reference state built from training vectors. Numeric deviations are differences in Phase 2 min-max feature space. Encoded categorical features use a mode reference and are described as `changed` or `stable`, not as ordered values. Evidence is ranked by the relative size of these deviations. A configurable tolerance marks small numeric changes stable.

This is forecast feature evidence associated with the model input/output. It is **not causal attribution**, does not describe model gradients, and does not prove that a feature caused a predicted stage. The default confidence threshold is 0.60; forecasts below it are explicitly flagged `LOW_CONFIDENCE`. Confidence is the largest model stage probability, not a calibrated probability that an attack is occurring.

Run the held-out demonstration with:

```powershell
python -m src.explainability.demo
```

## MITRE ATT&CK Context

The stage-to-tactic/technique mapping is centralized in `src/explainability/mitre_mapping.py` for future dashboard use. It uses official Enterprise ATT&CK names and identifiers for Reconnaissance (TA0043), Initial Access (TA0001), Lateral Movement (TA0008), and Command and Control (TA0011), with contextual techniques where applicable. `BENIGN` maps to no suggested technique. Official references include [Reconnaissance](https://attack.mitre.org/tactics/TA0043/), [Initial Access](https://attack.mitre.org/tactics/TA0001/), [Lateral Movement](https://attack.mitre.org/tactics/TA0008/), [Command and Control](https://attack.mitre.org/tactics/TA0011/), [Active Scanning T1595](https://attack.mitre.org/techniques/T1595/), [Exploit Public-Facing Application T1190](https://attack.mitre.org/techniques/T1190/), [Remote Services T1021](https://attack.mitre.org/techniques/T1021/), and [Application Layer Protocol T1071](https://attack.mitre.org/techniques/T1071/).

These mappings are **potentially relevant ATT&CK context** associated with a forecast stage. Flow features and model outputs do not prove that a tactic or technique occurred. The technique rationales explain the limited relationship to this prototype's synthetic behavior and explicitly note missing evidence.
