import streamlit as st
import cv2, numpy as np, torch, supervision as sv
from datetime import datetime
from ultralytics import YOLO
from sort.sort import Sort
from torchvision.ops import box_iou

st.set_page_config(page_title="Realtime Vehicle Speed",
                   layout="wide")
st.title("🚦 Realtime Vehicle Detection & Speed Estimation")

# ───────────────────── sidebar knobs ─────────────────────
TARGET_SIZE  = st.sidebar.slider("YOLO input (px)", 320, 1280, 640, 160)
SPEED_LIMIT  = st.sidebar.slider("Overspeed limit (km/h)", 10, 20, 40, 60)
CONF_THRES   = st.sidebar.slider("Confidence ≥", 0.01, 0.50, 0.05, 0.01)

start_button = st.sidebar.button("▶ Start / restart camera")

# ───────────────────── build model once ──────────────────
import streamlit as st

# Pick the decorator that exists in the current install
if hasattr(st, "singleton"):
    singleton = st.singleton
elif hasattr(st, "cache_resource"):
    singleton = st.cache_resource
else:                             # very old Streamlit
    singleton = st.experimental_singleton


@singleton
def load_model():
    return YOLO("best.pt")


model   = load_model()
tracker = Sort(max_age=30, min_hits=3, iou_threshold=0.3)

CLASS_NAMES = model.model.names
KEEP_IDS    = {7, 8}                       # truck, bus
FACTOR_KM   = 3.6
DIST_METERS = 100
POINT_A_Y, POINT_B_Y = 200, 300           # adapt to your camera
LINE_TOL    = 8
poly_zone   = sv.PolygonZone(
    polygon=np.array([[150,100],[400,100],[400,400],[150,400]]),
)

entry_times, speeds, id_map = {}, {}, {}

# ─────────────────── helper functions ────────────────────
def scale_bbox(box, s, dx, dy):
    x1,y1,x2,y2 = box
    return [int(x1/s+dx), int(y1/s+dy), int(x2/s+dx), int(y2/s+dy)]


def best_match(tbox, dboxes):
    if len(dboxes) == 0: return None
    ious = box_iou(torch.tensor([tbox]), torch.tensor(dboxes))[0]
    m,i = torch.max(ious,0)
    return (int(i), float(m)) if m>0 else None


# ───────────────────── main realtime loop ─────────────────
if start_button:
    cap = cv2.VideoCapture(0)  # ← Directly set to camera index 0 (default camera)
    if not cap.isOpened():
        st.error("Cannot open camera"); st.stop()

    frame_placeholder = st.empty()        # where we draw each frame
    st.write("Press **R** in the frame window to reload, **Q** to quit.")

    while True:
        ok, frame = cap.read()
        if not ok: break
        H, W = frame.shape[:2]

        # keep a square ROI around polygon (faster)
        rx1, ry1, rx2, ry2 = cv2.boundingRect(poly_zone.polygon)
        side = max(rx2-rx1, ry2-ry1)
        cx, cy = rx1+(rx2-rx1)//2, ry1+(ry2-ry1)//2
        ROI = (max(0,cx-side//2), max(0,cy-side//2),
               min(W-1,cx+side//2), min(H-1,cy+side//2))

        # ──────────────── your ORIGINAL pipeline ────────────────
        # ↳ copy the heavy detection/tracking/drawing code here
        crop = frame[ROI[1]:ROI[3], ROI[0]:ROI[2]]
        scale = TARGET_SIZE / crop.shape[0]
        resized = cv2.resize(crop,None,fx=scale,fy=scale)

        results = model(resized, verbose=False,
                        device=0 if torch.cuda.is_available() else "cpu")[0]
        dets = sv.Detections.from_ultralytics(results)
        mask = (np.isin(dets.class_id, list(KEEP_IDS)) &
                (dets.confidence > CONF_THRES)); dets = dets[mask]
        for i in range(len(dets)):
            dets.xyxy[i] = scale_bbox(dets.xyxy[i], scale, ROI[0], ROI[1])
        dets = dets[poly_zone.trigger(dets)]
        det_sort = (np.hstack((dets.xyxy, dets.confidence.reshape(-1,1)))
                    if len(dets) else np.empty((0,5)))
        tracks = tracker.update(det_sort)

        for tr in tracks.astype(int):
            x1,y1,x2,y2,tid = tr
            match = best_match([x1,y1,x2,y2], dets.xyxy)
            if match:
                d_idx,_ = match
                cid  = int(dets.class_id[d_idx])
                conf = float(dets.confidence[d_idx])
                label= CLASS_NAMES[cid]
            else:
                label,conf = "Unknown",0.0

            disp_id = id_map.setdefault(tid, len(id_map)+1)
            cy2 = y2
            if abs(cy2-POINT_A_Y)<=LINE_TOL and tid not in entry_times:
                entry_times[tid]= datetime.now()
            elif tid in entry_times and tid not in speeds and \
                 abs(cy2-POINT_B_Y)<=LINE_TOL:
                dt = (datetime.now()-entry_times[tid]).total_seconds()
                if dt>0:
                    speeds[tid] = round(DIST_METERS/dt*FACTOR_KM,2)

            col = (0,0,255) if speeds.get(tid,0)>SPEED_LIMIT else (255,255,0)
            cv2.rectangle(frame,(x1,y1),(x2,y2),col,2)
            cv2.putText(frame,f"{label} {conf:.2f}",(x1,y1-35),
                        cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0),2)
            txt = f"ID {disp_id}"
            if tid in speeds:
                txt += f"|{speeds[tid]} km/h"
                if speeds[tid]>SPEED_LIMIT:
                    cv2.putText(frame,"OVERSPEED",(x1,y1-55),
                                cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,0,255),2)
            cv2.putText(frame,txt,(x1,y1-15),
                        cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0),2)

        cv2.polylines(frame,[poly_zone.polygon],True,(0,255,0),2)
        cv2.line(frame,(0,POINT_A_Y),(W,POINT_A_Y),(0,255,0),2)
        cv2.line(frame,(0,POINT_B_Y),(W,POINT_B_Y),(0,0,255),2)
        # ───────────────── end of your pipeline ─────────────────

        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(frame_bgr, channels="RGB", use_column_width=True)

        # graceful exit when user presses Q in the OpenCV window
        if cv2.waitKey(1) & 0xFF in (ord('q'), ord('Q')):
            break

    cap.release()
    cv2.destroyAllWindows()
