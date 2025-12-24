import cv2
import numpy as np
import mediapipe as mp
import csv
from collections import deque
from scipy.signal import butter, filtfilt, detrend

import pandas as pd
import matplotlib.pyplot as plt


VIDEO_PATH = r"E:\s1\s1\vid_s1_T1.avi"
EDA_PATH   = r"E:\s1\s1\eda_s1_T1.csv"

OUT_CSV   = r"E:\s1\output_1_fused_4hz_t1.csv"
OUT_VIDEO = r"E:\s1\output_1_fused_product_view_t1.mp4"


DRAW_FACE_CONTOURS = True   # set False to remove face contour lines
DRAW_RPPG_ROI      = False  # set True to draw ROI polygon

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

LEFT_EYE  = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]


def _pt(lm, idx, w, h):
    return np.array([lm[idx].x * w, lm[idx].y * h], dtype=np.float32)


def eye_aspect_ratio(lm, eye_idx, w, h):
    p1 = _pt(lm, eye_idx[0], w, h)
    p2 = _pt(lm, eye_idx[1], w, h)
    p3 = _pt(lm, eye_idx[2], w, h)
    p4 = _pt(lm, eye_idx[3], w, h)
    p5 = _pt(lm, eye_idx[4], w, h)
    p6 = _pt(lm, eye_idx[5], w, h)
    return (np.linalg.norm(p2 - p6) + np.linalg.norm(p3 - p5)) / (2.0 * np.linalg.norm(p1 - p4) + 1e-6)


def bandpass_filter(x, fs, low=0.7, high=4.0, order=3):
    nyq = 0.5 * fs
    if high >= nyq:
        high = nyq - 1e-3
    b, a = butter(order, [low/nyq, high/nyq], btype="band")
    return filtfilt(b, a, x)


def estimate_bpm(sig, fs):
    if len(sig) < int(fs * 5):
        return None
    sig = detrend(sig)
    sig = (sig - np.mean(sig)) / (np.std(sig) + 1e-6)
    sig = bandpass_filter(sig, fs, low=0.7, high=4.0, order=3)
    freqs = np.fft.rfftfreq(len(sig), 1/fs)
    spec = np.abs(np.fft.rfft(sig))
    mask = (freqs >= 0.7) & (freqs <= 4.0)
    if not np.any(mask):
        return None
    return float(freqs[mask][np.argmax(spec[mask])] * 60.0)


def roi_mean_green(frame_bgr, pts):
    mask = np.zeros(frame_bgr.shape[:2], dtype=np.uint8)
    cv2.fillConvexPoly(mask, pts.astype(np.int32), 255)
    mean_bgr = cv2.mean(frame_bgr, mask=mask)[:3]
    return float(mean_bgr[1])


def clamp01(x): return max(0.0, min(1.0, float(x)))
def clamp100(x): return max(0.0, min(100.0, float(x)))


def percentile_baseline(values, p=20):
    if len(values) == 0:
        return None
    return float(np.percentile(np.array(values, dtype=np.float32), p))


def load_eda_empatica_or_plain(path, default_fs=4.0):

    raw = pd.read_csv(path, header=None)
    try:
        v0 = float(raw.iloc[0, 0])
        if v0 > 1e9 and len(raw) >= 3:
            fs = float(raw.iloc[1, 0])
            eda = raw.iloc[2:, 0].astype(float).to_numpy()
            t_sec = np.arange(len(eda)) / fs
            return pd.DataFrame({"t_sec": t_sec, "eda": eda})
    except Exception:
        pass

    try:
        dfh = pd.read_csv(path)
        cols = [c.lower() for c in dfh.columns]

        if "t_sec" in cols and "eda" in cols:
            return dfh.rename(columns={
                dfh.columns[cols.index("t_sec")]: "t_sec",
                dfh.columns[cols.index("eda")]: "eda"
            })[["t_sec", "eda"]]
        for name in ["eda", "gsr", "electrodermal", "skinconductance"]:
            if name in cols:
                eda_col = dfh.columns[cols.index(name)]
                eda = dfh[eda_col].astype(float).to_numpy()
                t_sec = np.arange(len(eda)) / default_fs
                return pd.DataFrame({"t_sec": t_sec, "eda": eda})
    except Exception:
        pass

    eda = raw.iloc[:, 0].astype(float).to_numpy()
    t_sec = np.arange(len(eda)) / default_fs
    return pd.DataFrame({"t_sec": t_sec, "eda": eda})


def draw_timeseries_graph(img, x0, y0, w, h, series, vmin=None, vmax=None, label="", value_text=""):
    cv2.rectangle(img, (x0, y0), (x0 + w, y0 + h), (30, 30, 30), -1)
    cv2.rectangle(img, (x0, y0), (x0 + w, y0 + h), (120, 120, 120), 1)

    if label:
        cv2.putText(img, label, (x0 + 6, y0 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 230, 230), 1)
    if value_text:
        cv2.putText(img, value_text, (x0 + 6, y0 + h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 230, 230), 1)

    if series is None or len(series) < 2:
        return

    raw = np.array(series, dtype=np.float32)
    finite = raw[np.isfinite(raw)]
    if len(finite) < 2:
        return

    if vmin is None: vmin = float(np.min(finite))
    if vmax is None: vmax = float(np.max(finite))
    if abs(vmax - vmin) < 1e-6:
        vmax = vmin + 1e-6

    n = len(raw)
    pts = []
    for i in range(n):
        val = raw[i]
        if not np.isfinite(val):
            pts.append(None)
            continue
        xn = i / (n - 1)
        yn = (val - vmin) / (vmax - vmin)
        px = int(x0 + 6 + xn * (w - 12))
        py = int(y0 + h - 10 - yn * (h - 28))
        pts.append((px, py))

    prev = None
    for p in pts:
        if p is None:
            prev = None
            continue
        if prev is not None:
            cv2.line(img, prev, p, (200, 200, 200), 2)
        prev = p


cap = cv2.VideoCapture(0)
fps = cap.get(cv2.CAP_PROP_FPS)
if fps <= 1:
    fps = 30.0

ok, first_frame = cap.read()
if not ok:
    raise RuntimeError("Could not read video. Check VIDEO_PATH.")
H0, W0 = first_frame.shape[:2]
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

GRAPH_H = 170
OUT_H = H0 + GRAPH_H
OUT_W = W0

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer_vid = cv2.VideoWriter(OUT_VIDEO, fourcc, fps, (OUT_W, OUT_H))

EDA_FS = 4.0
SAVE_INTERVAL = 1.0 / EDA_FS

BASELINE_SKIP_SEC = 15.0
BASELINE_WINDOW_SEC = 40.0
BASELINE_PERCENTILE = 20.0

RPPG_WINDOW_SEC = 10
rgb_buf = deque(maxlen=int(fps * RPPG_WINDOW_SEC))

BPM_SMOOTH_ALPHA = 0.05
BPM_MIN = 45
BPM_MAX = 160
STRESS_HR_RANGE = 45.0
CALIB_EAR_SEC = 20.0
EAR_DROP_RATIO = 0.75
CONSEC_FRAMES = 2
BLINK_RANGE = 20.0
W_HR = 0.75
W_BLINK = 0.25
STRESS_SMOOTH_A = 0.10
MIN_ROI_AREA_FRAC = 0.005
HIST_SEC = 30.0
hist_len = int(HIST_SEC * EDA_FS)
hist_bpm = deque([np.nan]*hist_len, maxlen=hist_len)
hist_blink = deque([np.nan]*hist_len, maxlen=hist_len)
hist_stress = deque([np.nan]*hist_len, maxlen=hist_len)

closed_frames = 0
blink_times = []
blink_count = 0

baseline_hr_vals = []
baseline_hr = None

baseline_blink_vals = []
baseline_blink = None

baseline_ear_vals = []
baseline_ear = None

bpm_smooth = None
stress_smooth = None
last_save_t = -1e9

f = open(OUT_CSV, "w", newline="")
writer_csv = csv.writer(f)
writer_csv.writerow(["t_sec", "stress_index", "bpm", "blink_rate", "baseline_hr", "baseline_blink"])

with mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as face_mesh:

    frame_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        t_sec = frame_idx / fps
        frame_idx += 1

        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = face_mesh.process(rgb)

        ear = None
        blink_rate = 0.0
        bpm = None
        ear_thresh = 0.21
        ear_calibrating = (t_sec < CALIB_EAR_SEC)
        hr_baseline_collect = (BASELINE_SKIP_SEC <= t_sec < (BASELINE_SKIP_SEC + BASELINE_WINDOW_SEC))

        have_face = False
        roi_ok = False

        if result.multi_face_landmarks:
            have_face = True
            face_landmarks = result.multi_face_landmarks[0]
            lm = face_landmarks.landmark

            ear = (eye_aspect_ratio(lm, LEFT_EYE, w, h) +
                   eye_aspect_ratio(lm, RIGHT_EYE, w, h)) / 2.0

            if ear_calibrating and ear is not None:
                baseline_ear_vals.append(ear)
                if len(baseline_ear_vals) > 30:
                    baseline_ear = float(np.median(baseline_ear_vals))

            ear_thresh = (baseline_ear * EAR_DROP_RATIO) if baseline_ear is not None else 0.21

            if ear is not None and ear < ear_thresh:
                closed_frames += 1
            else:
                if closed_frames >= CONSEC_FRAMES:
                    blink_count += 1
                    blink_times.append(t_sec)
                closed_frames = 0

            blink_times[:] = [t for t in blink_times if t_sec - t <= 60.0]
            blink_rate = float(len(blink_times))
            roi_ids = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397]
            pts = np.array([[lm[i].x * w, lm[i].y * h] for i in roi_ids], dtype=np.float32)

            area = abs(cv2.contourArea(pts.astype(np.int32)))
            roi_ok = (area / (w * h)) >= MIN_ROI_AREA_FRAC

            if DRAW_RPPG_ROI:
                cv2.polylines(frame, [pts.astype(np.int32)], True, (255, 255, 255), 1)

            if roi_ok:
                g = roi_mean_green(frame, pts)
                rgb_buf.append(g)
            if roi_ok and len(rgb_buf) >= int(fps * 5):
                bpm = estimate_bpm(np.array(rgb_buf, dtype=np.float32), fps)

                if bpm is not None and (bpm < BPM_MIN or bpm > BPM_MAX):
                    bpm = None
                if bpm is not None:
                    bpm_smooth = bpm if bpm_smooth is None else (
                        (1.0 - BPM_SMOOTH_ALPHA) * bpm_smooth + BPM_SMOOTH_ALPHA * bpm
                    )
            if DRAW_FACE_CONTOURS:
                mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_styles.get_default_face_mesh_contours_style()
                )

        if hr_baseline_collect and bpm_smooth is not None:
            baseline_hr_vals.append(bpm_smooth)
            baseline_hr = percentile_baseline(baseline_hr_vals, BASELINE_PERCENTILE)

        if hr_baseline_collect and have_face:
            baseline_blink_vals.append(blink_rate)
            baseline_blink = percentile_baseline(baseline_blink_vals, 50.0)  # median baseline blink

        hr_score = None
        blink_score = None

        if baseline_hr is not None and bpm_smooth is not None:
            hr_score = clamp01((bpm_smooth - baseline_hr) / STRESS_HR_RANGE)

        if baseline_blink is not None and have_face:
            blink_score = clamp01((blink_rate - baseline_blink) / BLINK_RANGE)

        stress_now = None
        if hr_score is not None and blink_score is not None:
            fused = clamp01(W_HR * hr_score + W_BLINK * blink_score)
            stress_now = clamp100(fused * 100.0)
        elif hr_score is not None:
            stress_now = clamp100(hr_score * 100.0)

        if stress_now is not None:
            stress_smooth = stress_now if stress_smooth is None else (
                (1.0 - STRESS_SMOOTH_A) * stress_smooth + STRESS_SMOOTH_A * stress_now
            )
        else:
            stress_smooth = None
        if (t_sec - last_save_t) >= SAVE_INTERVAL:
            row_stress = stress_smooth if stress_smooth is not None else np.nan
            row_bpm    = bpm_smooth if bpm_smooth is not None else np.nan
            row_blink  = blink_rate if have_face else np.nan

            writer_csv.writerow([
                round(t_sec, 2),
                row_stress,
                row_bpm,
                row_blink,
                baseline_hr if baseline_hr is not None else np.nan,
                baseline_blink if baseline_blink is not None else np.nan
            ])
            last_save_t = t_sec

            hist_bpm.append(row_bpm)
            hist_blink.append(row_blink)
            hist_stress.append(row_stress)


        cv2.putText(frame, f"t: {t_sec:.1f}s", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        cv2.putText(frame, f"EAR: {ear:.3f}" if ear is not None else "EAR: --", (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        cv2.putText(frame, f"EARthr: {ear_thresh:.3f}", (20, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        cv2.putText(frame, f"Blinks: {blink_count}  Rate: {blink_rate:.0f}/min", (20, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

        if bpm_smooth is None:
            cv2.putText(frame, "HR: estimating...", (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        else:
            cv2.putText(frame, f"HR: {bpm_smooth:.1f} BPM", (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

        if baseline_hr is None:
            if t_sec < BASELINE_SKIP_SEC:
                cv2.putText(frame, f"Baseline HR: waiting ({int(BASELINE_SKIP_SEC - t_sec)}s)", (20, 180),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            else:
                cv2.putText(frame, "Baseline HR: collecting...", (20, 180),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        else:
            cv2.putText(frame, f"Baseline HR (p{int(BASELINE_PERCENTILE)}): {baseline_hr:.1f}", (20, 180),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

        if baseline_blink is not None:
            cv2.putText(frame, f"Baseline Blink: {baseline_blink:.0f}/min", (20, 210),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

        if stress_smooth is not None:
            cv2.putText(frame, f"Stress (fused): {stress_smooth:.1f}/100", (20, 240),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        else:
            cv2.putText(frame, "Stress (fused): --", (20, 240),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

        out = np.zeros((OUT_H, OUT_W, 3), dtype=np.uint8)
        out[:H0, :W0] = frame

        panel = out[H0:, :]
        panel[:] = (15, 15, 15)

        pad = 10
        gw = (OUT_W - pad * 4) // 3
        gh = GRAPH_H - pad * 2
        y0 = pad
        x1 = pad
        x2 = pad * 2 + gw
        x3 = pad * 3 + gw * 2

        bpm_text = "--" if bpm_smooth is None else f"{bpm_smooth:.1f} BPM"
        blink_text = "--" if not have_face else f"{blink_rate:.0f}/min"
        stress_text = "--" if stress_smooth is None else f"{stress_smooth:.1f}/100"

        draw_timeseries_graph(panel, x1, y0, gw, gh, list(hist_bpm),
                              vmin=50, vmax=160, label="BPM", value_text=bpm_text)
        draw_timeseries_graph(panel, x2, y0, gw, gh, list(hist_blink),
                              vmin=0, vmax=60, label="Blink/min", value_text=blink_text)
        draw_timeseries_graph(panel, x3, y0, gw, gh, list(hist_stress),
                              vmin=0, vmax=100, label="Stress", value_text=stress_text)

        cv2.imshow("Stress Detector (rPPG + Blink) - Product View", out)
        writer_vid.write(out)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
writer_vid.release()
f.close()
cv2.destroyAllWindows()

print("Saved CSV:", OUT_CSV)
print("Saved overlay video:", OUT_VIDEO)

df = pd.read_csv(OUT_CSV)

plt.figure(figsize=(12, 4))
plt.plot(df["t_sec"], df["stress_index"], linewidth=2, label="Stress_Index (fused)")
plt.xlabel("Time (s)")
plt.ylabel("Stress (0-100)")
plt.title("Stress Index over time")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 4))
plt.plot(df["t_sec"], df["bpm"], linewidth=2, label="BPM")
plt.xlabel("Time (s)")
plt.ylabel("BPM")
plt.title("Estimated Heart Rate (rPPG) over time")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 4))
plt.plot(df["t_sec"], df["blink_rate"], linewidth=2, label="Blink/min")
plt.xlabel("Time (s)")
plt.ylabel("Blinks per minute")
plt.title("Blink rate over time")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

gt = load_eda_empatica_or_plain(EDA_PATH, default_fs=4.0)

eda_interp = np.interp(df["t_sec"].to_numpy(),
                       gt["t_sec"].to_numpy(),
                       gt["eda"].to_numpy())

eda_norm = (eda_interp - np.nanmean(eda_interp)) / (np.nanstd(eda_interp) + 1e-6)

plt.figure(figsize=(12, 5))
plt.plot(df["t_sec"], df["stress_index"], linewidth=2, label="Stress_Index (yours)")
plt.plot(df["t_sec"], eda_norm * 20 + 50, linewidth=2, label="EDA GT (z-scored, scaled)")
plt.xlabel("Time (seconds)")
plt.title("Stress_Index vs Ground Truth EDA (UBFC-Phys)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
