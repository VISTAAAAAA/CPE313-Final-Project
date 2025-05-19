# vehicle_speed_app.py  ── version without the device= bug
import streamlit as st
import cv2, numpy as np, torch, tempfile, os, mimetypes, supervision as sv
from pathlib import Path
from datetime import datetime
from ultralytics import YOLO
from sort.sort import Sort
from torchvision.ops import box_iou

# ───────────────────────────── UI options ────────────────────────────
st.set_page_config(page_title="Vehicle Detection & Speed Estimation",
                   layout="wide")
st.title("🚗 Vehicle Detection & Speed Estimation")

TARGET_SIZE  = st.sidebar.slider("YOLO input size (px)", 320, 1280, 640, 160)
TARGET_FPS   = st.sidebar.slider("Target FPS in output video", 5, 30, 15, 1)
SPEED_LIMIT  = st.sidebar.slider("Overspeed threshold (km/h)", 10, 120, 20, 5)

uploaded_video = st.file_uploader("Upload an MP4 video",
                                  type=["mp4"],
                                  accept_multiple_files=False)

# ───────────────────── helper: delete old temp files ──────────────────
def cleanup_files():
    for f in list(st.session_state.get("tmpfiles", set())):
        Path(f).unlink(missing_ok=True)
        st.session_state["tmpfiles"].discard(f)

# ───────────────────────────── main flow ──────────────────────────────
if uploaded_video:
    mime, _ = mimetypes.guess_type(uploaded_video.name)
    if mime != "video/mp4":
        st.error("Not a valid MP4 file."); st.stop()

    cleanup_files()                           # start fresh each run

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_in:
        tmp_in.write(uploaded_video.read())
        in_path = tmp_in.name
    fd, out_path = tempfile.mkstemp(suffix=".mp4"); os.close(fd)
    st.session_state.setdefault("tmpfiles", set()).update({in_path, out_path})

    # ───────────────────────── model / tracker ────────────────────────
    model   = YOLO("best.pt")                 # FIX ① no device arg here
    tracker = Sort(max_age=60, min_hits=3, iou_threshold=0.3)

    KEEP_IDS     = {7, 8};  CONF_THRES = 0.05
    CLASS_NAMES  = model.model.names
    FACTOR_KM    = 3.6;  DIST_METERS  = 100
    POINT_A_Y, POINT_B_Y, LINE_TOL = 1000, 1400, 10

    poly_zone = sv.PolygonZone(
        polygon=np.array([[1600, 800], [2100, 800], [2500, 1900], [1000, 1900]])
    )

    cap = cv2.VideoCapture(in_path)
    ok, frame = cap.read()
    if not ok: st.error("Cannot read video."); st.stop()

    fps_in  = cap.get(cv2.CAP_PROP_FPS) or 30
    skip_N  = max(1, int(round(fps_in / TARGET_FPS)))
    out_fps = fps_in / skip_N
    H, W = frame.shape[:2]
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"),
                             out_fps, (W, H))

    entry_times, speeds, id_map = {}, {}, {}
    rx1, ry1, rx2, ry2 = cv2.boundingRect(poly_zone.polygon)
    side = max(rx2 - rx1, ry2 - ry1)
    cx, cy = rx1 + (rx2 - rx1)//2, ry1 + (ry2 - ry1)//2
    ROI = (max(0, cx - side//2), max(0, cy - side//2),
           min(W-1, cx + side//2), min(H-1, cy + side//2))

    def scale_bbox(b,s,dx,dy):
        x1,y1,x2,y2=b; return [int(x1/s+dx),int(y1/s+dy),int(x2/s+dx),int(y2/s+dy)]
    def best_match(tbox, dboxes):
        if len(dboxes)==0: return None
        ious = box_iou(torch.tensor([tbox]), torch.tensor(dboxes))[0]
        m,i = torch.max(ious,0); return (int(i),float(m)) if m>0 else None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 1)
    processed = idx = 0
    prog = st.progress(0.0,"Processing…")

    while ok:
        if idx % skip_N == 0:
            crop = frame[ROI[1]:ROI[3], ROI[0]:ROI[2]]
            scale = TARGET_SIZE / crop.shape[0]
            resized = cv2.resize(crop,None,fx=scale,fy=scale)

            # FIX ② pass device each call
            results = model(resized, verbose=False,
                            device=0 if torch.cuda.is_available() else "cpu")[0]

            dets = sv.Detections.from_ultralytics(results)
            mask = (np.isin(dets.class_id, list(KEEP_IDS))
                    & (dets.confidence > CONF_THRES)); dets = dets[mask]
            for i in range(len(dets)):
                dets.xyxy[i] = scale_bbox(dets.xyxy[i], scale, ROI[0], ROI[1])
            dets = dets[poly_zone.trigger(dets)]

            det_sort = (np.hstack((dets.xyxy, dets.confidence.reshape(-1,1)))
                        if len(dets) else np.empty((0,5)))
            tracks = tracker.update(det_sort)

            for tr in tracks.astype(int):
                x1,y1,x2,y2,tid=tr
                match = best_match([x1,y1,x2,y2], dets.xyxy)
                if match:
                    d_idx,_ = match; cid=int(dets.class_id[d_idx])
                    conf=float(dets.confidence[d_idx]); label=CLASS_NAMES[cid]
                else: label,conf="Unknown",0.0
                disp_id=id_map.setdefault(tid,len(id_map)+1)
                cy2=y2
                if abs(cy2-POINT_A_Y)<=LINE_TOL and tid not in entry_times:
                    entry_times[tid]=datetime.now()
                elif tid in entry_times and tid not in speeds and \
                     abs(cy2-POINT_B_Y)<=LINE_TOL:
                    dt=(datetime.now()-entry_times[tid]).total_seconds()
                    if dt>0: speeds[tid]=round(DIST_METERS/dt*FACTOR_KM,2)

                col=(0,0,255) if speeds.get(tid,0)>SPEED_LIMIT else (255,255,0)
                cv2.rectangle(frame,(x1,y1),(x2,y2),col,2)
                cv2.putText(frame,f"{label} {conf:.2f}",(x1,y1-40),
                            cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2)
                txt=f"ID {disp_id}"
                if tid in speeds:
                    txt+=f" | {speeds[tid]} km/h"
                    if speeds[tid]>SPEED_LIMIT:
                        cv2.putText(frame,"OVERSPEEDING",(x1,y1-60),
                                    cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,0,255),2)
                cv2.putText(frame,txt,(x1,y1-20),
                            cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2)

            cv2.polylines(frame,[poly_zone.polygon],True,(0,255,0),2)
            cv2.line(frame,(0,POINT_A_Y),(W,POINT_A_Y),(0,255,0),2)
            cv2.line(frame,(0,POINT_B_Y),(W,POINT_B_Y),(0,0,255),2)

            writer.write(frame); processed += 1
            prog.progress(idx/total_frames)
        ok, frame = cap.read(); idx += 1

    cap.release(); writer.release(); prog.empty()

    st.success(f"Done – wrote {processed} frames @ {out_fps:.1f} FPS")
    st.video(out_path)                              # FIX ③ plays correctly
    st.download_button("Download processed MP4",
                       data=open(out_path,"rb"),
                       file_name="processed.mp4",
                       mime="video/mp4")

if st.session_state.get("tmpfiles"):
    if st.button("Delete temporary files"):
        cleanup_files(); st.success("Temporary files removed.")
