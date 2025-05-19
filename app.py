from __future__ import annotations

import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import mimetypes
import numpy as np
import streamlit as st
import supervision as sv
import torch
from sort.sort import Sort  # Ensure "sort" is installed in requirements.txt
from torchvision.ops import box_iou
from ultralytics import YOLO

# ────────────────────────────── UI CONFIG ──────────────────────────────
st.set_page_config("🚗 Vehicle Detection & Speed Estimation", layout="wide")

st.title("🚗 Vehicle Detection & Speed Estimation")

with st.sidebar:
    st.header("⚙️ Settings")
    TARGET_SIZE: int = st.slider("YOLO input size (px)", 320, 1280, 640, 160)
    TARGET_FPS: int = st.slider("Target FPS in output video", 5, 30, 15, 1)
    SPEED_LIMIT: int = st.slider("Overspeed threshold (km/h)", 10, 120, 20, 5)
    CONF_THRES: float = st.slider("Confidence threshold", 0.05, 0.9, 0.25, 0.05)
    IOU_THRES: float = st.slider("Polygon IoU threshold", 0.0, 1.0, 0.3, 0.05)
    st.markdown("---")
    st.caption("Upload an MP4 clip. 📹 Processing happens completely in‑browser on “Streamlit Cloud”.")

uploaded_video = st.file_uploader("Video file", type=["mp4"], accept_multiple_files=False)

# ────────────────────────────── HELPERS ────────────────────────────────
TEMP_DIR = Path(tempfile.gettempdir()) / f"vehicledemo_{st.session_state.get('run_id', os.urandom(4).hex())}"
TEMP_DIR.mkdir(exist_ok=True, parents=True)

@st.cache_resource(show_spinner=False)
def load_model() -> Tuple[YOLO, torch.device]:
    """Load YOLO model once per session and move to the best device."""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = YOLO("best.pt").to(device)
    # keep only vehicle‑like classes to speed up inference (7: truck, 8: bus)
    model.overrides["classes"] = [7, 8]
    return model, device

model, device = load_model()
CLASS_NAMES = model.model.names  # type: ignore[attr-defined]
KEEP_IDS = set(model.overrides.get("classes", []))


def cleanup_temp() -> None:
    """Delete all temp files created in this session."""
    for f in TEMP_DIR.glob("*"):
        try:
            f.unlink()
        except Exception:
            pass


def seconds_to_kmh(seconds: float, dist_m: float = 100.0) -> float:
    """Convert travel time for *dist_m* meters to km/h."""
    return (dist_m / seconds) * 3.6 if seconds > 0 else 0.0


def best_match(tbox: List[int], dboxes: List[List[int]]) -> Tuple[int, float] | None:
    if not dboxes:
        return None
    ious = box_iou(torch.tensor([tbox]), torch.tensor(dboxes))[0]
    max_val, idx = torch.max(ious, 0)
    return (int(idx), float(max_val)) if max_val > 0 else None


# ────────────────────────────── MAIN FLOW ──────────────────────────────
if uploaded_video:
    mime, _ = mimetypes.guess_type(uploaded_video.name)
    if mime != "video/mp4":
        st.error("Please upload a valid MP4 file.")
        st.stop()

    # Save uploaded file to temp dir
    in_path = TEMP_DIR / Path(uploaded_video.name).name
    with in_path.open("wb") as f:
        f.write(uploaded_video.read())

    # Prepare output path
    out_path = TEMP_DIR / f"processed_{in_path.stem}.mp4"

    # Tracker
    tracker = Sort(max_age=60, min_hits=3, iou_threshold=IOU_THRES)

    # ROI & speed lines (relative ratios, independent of resolution)
    ROI_POLY_REL = np.array([[0.42, 0.30], [0.55, 0.30], [0.65, 0.73], [0.27, 0.73]])
    LINE_A_REL, LINE_B_REL = 0.55, 0.82  # y positions as fraction of H

    cap = cv2.VideoCapture(str(in_path))
    ret, frame = cap.read()
    if not ret:
        st.error("Cannot read the video file.")
        st.stop()

    H, W = frame.shape[:2]
    ROI_POLY = np.round(ROI_POLY_REL * np.array([[W, H]])).astype(int).reshape(-1, 2)
    poly_zone = sv.PolygonZone(polygon=ROI_POLY)

    fps_in: float = cap.get(cv2.CAP_PROP_FPS) or 30.0
    skip_N = max(1, int(round(fps_in / TARGET_FPS)))
    out_fps = fps_in / skip_N

    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), out_fps, (W, H))

    entry_times: Dict[int, float] = {}
    speeds: Dict[int, float] = {}
    id_map: Dict[int, int] = {}

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 1)
    processed = 0
    prog = st.progress(0.0, text="Processing… 0.0% | ETA –")

    def update_progress(curr_idx: int) -> None:
        pct = curr_idx / total_frames
        eta_frames = total_frames - curr_idx
        eta_sec = eta_frames / fps_in if fps_in else 0
        prog.progress(pct, text=f"Processing… {pct*100:4.1f}% | ETA {eta_sec:4.0f}s")

    idx = 0
    while ret:
        if idx % skip_N == 0:
            ts_video = idx / fps_in  # timestamp in the video itself

            # crop to ROI square to reduce inference cost
            rx1, ry1, rx2, ry2 = cv2.boundingRect(ROI_POLY)
            side = max(rx2 - rx1, ry2 - ry1)
            cx, cy = rx1 + (rx2 - rx1) // 2, ry1 + (ry2 - ry1) // 2
            roi = frame[max(0, cy - side // 2): min(H, cy + side // 2),
                        max(0, cx - side // 2): min(W, cx + side // 2)]
            scale = TARGET_SIZE / roi.shape[0]
            resized = cv2.resize(roi, None, fx=scale, fy=scale)

            results = model(resized, verbose=False, device=device)[0]
            dets = sv.Detections.from_ultralytics(results)
            mask = (np.isin(dets.class_id, list(KEEP_IDS)) & (dets.confidence > CONF_THRES))
            dets = dets[mask]
            for i in range(len(dets)):
                dets.xyxy[i] = [int(d / scale + (cx - side // 2 if n % 2 == 0 else cy - side // 2))
                                for n, d in enumerate(dets.xyxy[i])]
            dets = dets[poly_zone.trigger(dets)]

            det_sort = np.hstack((dets.xyxy, dets.confidence.reshape(-1, 1))) if len(dets) else np.empty((0, 5))
            tracks = tracker.update(det_sort)

            for tr in tracks.astype(int):
                x1, y1, x2, y2, tid = tr
                match = best_match([x1, y1, x2, y2], dets.xyxy)
                cid, conf = (int(dets.class_id[match[0]]), float(dets.confidence[match[0]])) if match else (None, 0.0)
                label = CLASS_NAMES[cid] if cid is not None else "Unknown"
                disp_id = id_map.setdefault(tid, len(id_map) + 1)

                cy2 = y2
                lineA_y, lineB_y = int(LINE_A_REL * H), int(LINE_B_REL * H)
                if abs(cy2 - lineA_y) <= 3 and tid not in entry_times:
                    entry_times[tid] = ts_video
                elif tid in entry_times and tid not in speeds and abs(cy2 - lineB_y) <= 3:
                    dt = ts_video - entry_times[tid]
                    speeds[tid] = round(seconds_to_kmh(dt), 2)

                col = (0, 0, 255) if speeds.get(tid, 0) > SPEED_LIMIT else (255, 255, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), col, 2)
                cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                txt = f"ID {disp_id}"
                if tid in speeds:
                    txt += f" | {speeds[tid]} km/h"
                    if speeds[tid] > SPEED_LIMIT:
                        cv2.putText(frame, "OVERSPEEDING", (x1, y1 - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                cv2.putText(frame, txt, (x1, y1 - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Draw ROI & speed lines
            cv2.polylines(frame, [ROI_POLY], True, (0, 255, 0), 2)
            cv2.line(frame, (0, int(LINE_A_REL * H)), (W, int(LINE_A_REL * H)), (0, 255, 0), 2)
            cv2.line(frame, (0, int(LINE_B_REL * H)), (W, int(LINE_B_REL * H)), (0, 0, 255), 2)

            writer.write(frame)
            processed += 1
            update_progress(idx)

        ret, frame = cap.read()
        idx += 1

    cap.release()
    writer.release()

    prog.empty()
    st.success(f"✅ Completed – wrote {processed} frames @ {out_fps:.1f} FPS")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original video")
        st.video(str(in_path))
    with col2:
        st.subheader("Processed video")
        st.video(str(out_path))

    st.download_button("📥 Download processed MP4", data=open(out_path, "rb"), file_name=out_path.name, mime="video/mp4")

    with st.expander("🧹 Clean up temp files"):
        if st.button("Delete temp files now"):
            cleanup_temp()
            st.success("Temporary files removed.")
