"""Evaluation summary for the dashboard.

Three parts, each present only if it can be computed:
  nowcast   statistics of the detections in the current data (always available)
  vs_truth  hit rate, false alarms and latency against a truth catalog (needs KNOXIS_TRUTH_FILE)
  batch     whatever the offline evaluation job wrote to metrics.json (forecast PR-AUC, baselines, CIs)
"""
from __future__ import annotations

import json
from typing import Optional

import numpy as np
import pandas as pd

from api.config import METRICS_FILE_PATH, SUSTAIN_MINUTES, TRUTH_FILE_PATH
from api.snapshot import Snapshot
from api.timeutil import utc

MATCH_PAD_BEFORE = pd.Timedelta(minutes=5)
MATCH_DEFAULT_LENGTH = pd.Timedelta(hours=2)


def _nowcast_stats(snap: Snapshot) -> Optional[dict]:
    if snap.nowcast is None or snap.df.empty:
        return None
    df, res = snap.df, snap.nowcast
    hours = (df.index[-1] - df.index[0]).total_seconds() / 3600
    lvl = res["alert_level"].astype(str)
    cat = res.get("flare_catalog")
    n = 0 if cat is None else len(cat)
    out = {
        "data_hours": round(hours, 2),
        "detections": n,
        "detections_per_day": round(n / hours * 24, 2) if hours > 0 else None,
        "fraction_of_time_at_alert": round(float((lvl == "ALERT").mean()), 4),
        "fraction_of_time_at_watch_or_warning": round(float(lvl.isin(["WATCH", "WARNING"]).mean()), 4),
        "built_in_confirmation_delay_s": SUSTAIN_MINUTES * 60,
    }
    if n:
        classes = cat["goes_class"].dropna().astype(str).str[0].value_counts().to_dict()
        out["detected_classes"] = ", ".join(f"{k}: {v}" for k, v in sorted(classes.items()))
    return out


def _vs_truth(snap: Snapshot) -> Optional[dict]:
    if not TRUTH_FILE_PATH.exists() or snap.nowcast is None or snap.df.empty:
        return None
    truth = pd.read_csv(TRUTH_FILE_PATH)
    if "start_time" not in truth.columns:
        return {"error": f"{TRUTH_FILE_PATH.name} needs a start_time column"}
    for c in ("start_time", "peak_time", "end_time"):
        if c in truth.columns:
            truth[c] = pd.to_datetime(truth[c], utc=True, errors="coerce")
    lo, hi = utc(snap.df.index[0]), utc(snap.df.index[-1])
    truth = truth[(truth["start_time"] >= lo) & (truth["start_time"] <= hi)].reset_index(drop=True)

    cat = snap.nowcast.get("flare_catalog")
    det = pd.DataFrame(columns=["detection_time", "goes_class"]) if cat is None else cat.copy()
    if len(det):
        det["detection_time"] = det["detection_time"].map(utc)

    def window(row):
        end = row.get("end_time")
        if end is None or pd.isna(end):
            end = row["start_time"] + MATCH_DEFAULT_LENGTH
        return row["start_time"] - MATCH_PAD_BEFORE, end

    hits, latencies, letter_ok, letter_n = 0, [], 0, 0
    matched_det: set[int] = set()
    for _, ev in truth.iterrows():
        a, b = window(ev)
        m = det[(det["detection_time"] >= a) & (det["detection_time"] <= b)] if len(det) else det
        if len(m):
            hits += 1
            matched_det.update(m.index.tolist())
            first = m.sort_values("detection_time").iloc[0]
            latencies.append((first["detection_time"] - ev["start_time"]).total_seconds())
            if "goes_class" in truth.columns and isinstance(first.get("goes_class"), str) and isinstance(ev.get("goes_class"), str):
                letter_n += 1
                letter_ok += int(first["goes_class"][0].upper() == ev["goes_class"][0].upper())
    n_truth, n_det = len(truth), len(det)
    false_alarms = n_det - len(matched_det)
    return {
        "truth_events_in_span": n_truth,
        "detections": n_det,
        "hit_rate": round(hits / n_truth, 3) if n_truth else None,
        "missed_events": n_truth - hits,
        "false_alarms": false_alarms,
        "false_alarm_ratio": round(false_alarms / n_det, 3) if n_det else None,
        "median_detection_latency_s": round(float(np.median(latencies)), 1) if latencies else None,
        "class_letter_agreement": round(letter_ok / letter_n, 3) if letter_n else None,
        "note": "A detection matches a truth event if it falls between 5 min before the event start and its end "
                "(or 2 h after the start when end_time is absent).",
    }


def compute_evaluation(snap: Snapshot) -> dict:
    out: dict = {"nowcast": _nowcast_stats(snap)}
    try:
        out["vs_truth"] = _vs_truth(snap)
    except Exception as exc:
        out["vs_truth"] = {"error": f"{type(exc).__name__}: {exc}"}
    batch = None
    if METRICS_FILE_PATH.exists():
        try:
            batch = json.loads(METRICS_FILE_PATH.read_text())
        except Exception as exc:
            batch = {"error": f"metrics file unreadable: {exc}"}
    out["batch"] = batch
    notes = []
    if out["vs_truth"] is None:
        notes.append(f"No truth catalog at {TRUTH_FILE_PATH}; hit rate, false alarms and latency are not computed.")
    if batch is None:
        notes.append(f"No metrics file at {METRICS_FILE_PATH}; forecast skill against baselines is not shown.")
    out["notes"] = notes
    return out
