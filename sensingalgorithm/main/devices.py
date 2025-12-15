import cv2 as cv
import numpy as np
import mediapipe as mp
import time
import os
import csv
from datetime import datetime
import serial


# --------------------------------------------------- arduino data
SERIAL_PORT = "COM11"
BAUD_RATE = 9600
TRIGGER_CMD = b"Temp10\n"
TRIGGER_EVERY_SEC = 2.0
# --------------------------------------------------- ml params
ML_MIN_SAMPLES = 80
ML_LEARN_RATE = 0.08
ML_L2 = 1e-4
OPTIMIZE_EVERY_SEC = 5.0
NOTIFY_COOLDOWN_SEC = 6.0
PROB_LOW_THRESH = 0.55
SUGGEST_TOL = {"TempC": 0.5, "Sound": 25.0, "Light": 2.0,}

# --------------------------------------------------- eye tracker params
GRID_POINTS = 9 
CENTER_BOX     = (0.30, 0.70)
CENTER_BOX_V   = (0.25, 0.75)
MAX_YAW_DEG    = 25
MAX_PITCH_DEG  = 25
SMOOTH_FRAMES  = 20
MIN_EYE_OPEN   = 0.3
L_EYE_LEFT  = 33
L_EYE_RIGHT = 133
L_EYE_TOP   = 159
L_EYE_BOT   = 145

R_EYE_LEFT  = 362
R_EYE_RIGHT = 263
R_EYE_TOP   = 386
R_EYE_BOT   = 374

L_IRIS = [468, 469, 470, 471]
R_IRIS = [473, 474, 475, 476]

mp_face = mp.solutions.face_mesh

def iris_center(landmarks, ids, w, h):
    pts = np.array([[landmarks[i].x * w, landmarks[i].y * h] for i in ids], dtype=np.float32)
    return pts.mean(axis=0)

def norm_ratio(val, a, b):
    denom = max(1e-6, (b - a))
    return float(np.clip((val - a) / denom, 0.0, 1.0))

def eye_openness(top, bot, left, right):
    h = abs(top[1] - bot[1])
    w = abs(right[0] - left[0])
    return float(h / max(w, 1e-6))

def head_pose_angles(landmarks, w, h):
    model_points = np.array([
        (0.0,   0.0,   0.0),
        (0.0,  -63.6, -12.5),
        (-43.3, 32.7, -26.0),
        (43.3,  32.7, -26.0),
        (-28.9,-28.9, -24.1),
        (28.9, -28.9, -24.1), 
    ], dtype=np.float64)

    idxs = [1, 152, 33, 263, 61, 291]

    image_points = np.array(
        [(landmarks[i].x * w, landmarks[i].y * h) for i in idxs],
        dtype=np.float64
    )

    if image_points.shape != (6, 2) or model_points.shape != (6, 3):
        return 0.0, 0.0
    if not np.isfinite(image_points).all():
        return 0.0, 0.0

    focal_length = float(w)
    center = (w / 2.0, h / 2.0)
    camera_matrix = np.array([
        [focal_length, 0.0, center[0]],
        [0.0, focal_length, center[1]],
        [0.0, 0.0, 1.0]
    ], dtype=np.float64)
    dist_coeffs = np.zeros((4, 1), dtype=np.float64)

    ok, rvec, tvec = cv.solvePnP(
        model_points,
        image_points,
        camera_matrix,
        dist_coeffs,
        flags=cv.SOLVEPNP_ITERATIVE
    )
    if not ok:
        return 0.0, 0.0

    rot_mat, _ = cv.Rodrigues(rvec)
    sy = np.sqrt(rot_mat[0, 0]**2 + rot_mat[1, 0]**2)
    pitch = np.degrees(np.arctan2(-rot_mat[2, 0], sy))
    yaw   = np.degrees(np.arctan2(rot_mat[1, 0], rot_mat[0, 0]))
    return float(yaw), float(pitch)


def decide_on_screen(left_ratios, right_ratios, yaw_deg, pitch_deg, use_vertical=True):
    h_ratio = (left_ratios[0] + right_ratios[0]) / 2.0
    v_ratio = (left_ratios[1] + right_ratios[1]) / 2.0

    centered_h = (CENTER_BOX[0] <= h_ratio <= CENTER_BOX[1])
    centered_v = (CENTER_BOX_V[0] <= v_ratio <= CENTER_BOX_V[1]) if use_vertical else True
    head_ok    = (abs(yaw_deg) <= MAX_YAW_DEG) and (abs(pitch_deg) <= MAX_PITCH_DEG)

    return (centered_h and centered_v and head_ok), h_ratio, v_ratio


def round2(x):
    if x is None:
        return None
    try:
        return round(float(x), 2)
    except:
        return None


# read arduino data
def read_arduino_row(ser):
    try:
        raw = ser.readline().decode(errors="ignore").strip()
        if not raw:
            return None

        # skip headers
        if any(c.isalpha() for c in raw):
            return None

        parts = raw.split(",")
        vals = []
        for i in range(9):
            if i < len(parts):
                try:
                    vals.append(float(parts[i]))
                except:
                    vals.append(None)
            else:
                vals.append(None)

        if all(v is None for v in vals):
            return None
        return vals
    except:
        return None


# ml regression
class RunningStandardizer:
    def __init__(self, d):
        self.n = 0
        self.mean = np.zeros(d, dtype=np.float64)
        self.M2 = np.zeros(d, dtype=np.float64)

    def update(self, x):
        x = np.asarray(x, dtype=np.float64)
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self.M2 += delta * delta2

    def transform(self, x):
        x = np.asarray(x, dtype=np.float64)
        if self.n < 2:
            return x
        var = self.M2 / max(1, (self.n - 1))
        std = np.sqrt(np.maximum(var, 1e-6))
        return (x - self.mean) / std

class OnlineLogReg:
    def __init__(self, d, lr=0.05, l2=1e-4):
        self.w = np.zeros(d + 1, dtype=np.float64) 
        self.lr = lr
        self.l2 = l2
        self.std = RunningStandardizer(d)

    def _sigmoid(self, z):
        z = np.clip(z, -30, 30)
        return 1.0 / (1.0 + np.exp(-z))

    def predict_proba(self, x):
        x = np.asarray(x, dtype=np.float64)
        xz = self.std.transform(x)
        xb = np.concatenate(([1.0], xz))
        return float(self._sigmoid(np.dot(self.w, xb)))

    def update(self, x, y):

        x = np.asarray(x, dtype=np.float64)
        self.std.update(x)
        xz = self.std.transform(x)
        xb = np.concatenate(([1.0], xz))
        p = self._sigmoid(np.dot(self.w, xb))
        grad = (p - y) * xb + self.l2 * self.w
        self.w -= self.lr * grad
        return float(p)


def main():
    cap = cv.VideoCapture(0)

    # get arduino stream
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0.05)
    time.sleep(2.0)
    ser.reset_input_buffer()
    ser.write(TRIGGER_CMD)
    ser.flush()
    last_trigger = time.time()

    # writing csv
    combined_path = "combined_dataset.csv"
    new_file = not os.path.exists(combined_path)
    combined_log = open(combined_path, "a", newline="")
    combined_writer = csv.writer(combined_log)

    if new_file:
        combined_writer.writerow([
            "timestamp",
            "eye_status",
            "yaw", "pitch",
            "iris_h", "iris_v",
            "TempC", "AccelX", "AccelY", "AccelZ",
            "GyroX", "GyroY", "GyroZ",
            "Sound", "Light",
            "p_focus_pred",
            "Temp_opt", "Sound_opt", "Light_opt"
        ])

    status_hist = []
    last_status = "INIT"
    font = cv.FONT_HERSHEY_SIMPLEX

    latest_arduino_vals = [None] * 9

    # ml
    model = OnlineLogReg(d=3, lr=ML_LEARN_RATE, l2=ML_L2)
    ml_samples = 0
    last_opt_time = 0.0
    last_notify_time = 0.0
    opt_values = None  
    obs_min = np.array([np.inf, np.inf, np.inf], dtype=np.float64)
    obs_max = np.array([-np.inf, -np.inf, -np.inf], dtype=np.float64)

    with mp_face.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as face_mesh:

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            now_t = time.time()
            if now_t - last_trigger >= TRIGGER_EVERY_SEC:
                ser.write(TRIGGER_CMD)
                ser.flush()
                last_trigger = now_t

            # read arduino data
            arduino_vals = read_arduino_row(ser)
            if arduino_vals is not None:
                latest_arduino_vals = arduino_vals

            # eye tracker
            h, w = frame.shape[:2]
            rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            res = face_mesh.process(rgb)

            have_face = False
            on_screen = False
            yaw_deg = 0.0
            pitch_deg = 0.0
            h_ratio = 0.5
            v_ratio = 0.5

            if res.multi_face_landmarks:
                have_face = True
                lms = res.multi_face_landmarks[0].landmark

                lc = iris_center(lms, L_IRIS, w, h)
                rc = iris_center(lms, R_IRIS, w, h)

                l_left  = np.array([lms[L_EYE_LEFT ].x * w, lms[L_EYE_LEFT ].y * h])
                l_right = np.array([lms[L_EYE_RIGHT].x * w, lms[L_EYE_RIGHT].y * h])
                l_top   = np.array([lms[L_EYE_TOP  ].x * w, lms[L_EYE_TOP  ].y * h])
                l_bot   = np.array([lms[L_EYE_BOT  ].x * w, lms[L_EYE_BOT  ].y * h])

                r_left  = np.array([lms[R_EYE_LEFT ].x * w, lms[R_EYE_LEFT ].y * h])
                r_right = np.array([lms[R_EYE_RIGHT].x * w, lms[R_EYE_RIGHT].y * h])
                r_top   = np.array([lms[R_EYE_TOP  ].x * w, lms[R_EYE_TOP  ].y * h])
                r_bot   = np.array([lms[R_EYE_BOT  ].x * w, lms[R_EYE_BOT  ].y * h])

                l_h = norm_ratio(lc[0], l_left[0], l_right[0])
                l_v = norm_ratio(lc[1], l_top[1],  l_bot[1])
                r_h = norm_ratio(rc[0], r_left[0], r_right[0])
                r_v = norm_ratio(rc[1], r_top[1],  r_bot[1])

                yaw_deg, pitch_deg = head_pose_angles(lms, w, h)

                l_open = eye_openness(l_top, l_bot, l_left, l_right)
                r_open = eye_openness(r_top, r_bot, r_left, r_right)
                use_vertical = ((l_open + r_open) / 2.0) >= MIN_EYE_OPEN

                on_screen, h_ratio, v_ratio = decide_on_screen(
                    (l_h, l_v), (r_h, r_v), yaw_deg, pitch_deg, use_vertical=use_vertical
                )

                color = (0, 200, 0) if on_screen else (0, 0, 255)
                for p in [lc, rc]:
                    cv.circle(frame, (int(p[0]), int(p[1])), 2, color, -1)
                cv.putText(frame, f"yaw={yaw_deg:+.0f} pitch={pitch_deg:+.0f}", (20, 80),
                           font, 0.6, (0, 0, 0), 1, cv.LINE_AA)

            # smoothing
            status_hist.append(on_screen)
            if len(status_hist) > SMOOTH_FRAMES:
                status_hist.pop(0)
            majority = status_hist.count(True) >= (len(status_hist) / 2.0)
            last_status = "ON" if majority else "OFF"
            y_focus = 1 if last_status == "ON" else 0


            TempC = latest_arduino_vals[0] if latest_arduino_vals else None
            Sound = latest_arduino_vals[7] if latest_arduino_vals else None
            Light = latest_arduino_vals[8] if latest_arduino_vals else None

            p_pred = None


            if have_face and (TempC is not None) and (Sound is not None) and (Light is not None):
                x_env = np.array([TempC, Sound, Light], dtype=np.float64)
                p_pred = model.update(x_env, y_focus)
                ml_samples += 1

                obs_min = np.minimum(obs_min, x_env)
                obs_max = np.maximum(obs_max, x_env)

            # using ml to find optimal values
            if (ml_samples >= ML_MIN_SAMPLES) and (now_t - last_opt_time >= OPTIMIZE_EVERY_SEC):
                last_opt_time = now_t

                lo = obs_min.copy()
                hi = obs_max.copy()
                for i in range(3):
                    if not np.isfinite(lo[i]) or not np.isfinite(hi[i]) or abs(hi[i] - lo[i]) < 1e-6:
                        lo[i] = lo[i] if np.isfinite(lo[i]) else 0.0
                        hi[i] = hi[i] if np.isfinite(hi[i]) else lo[i] + 1.0
                        if abs(hi[i] - lo[i]) < 1e-6:
                            hi[i] = lo[i] + 1.0

                t_grid = np.linspace(lo[0], hi[0], GRID_POINTS)
                s_grid = np.linspace(lo[1], hi[1], GRID_POINTS)
                l_grid = np.linspace(lo[2], hi[2], GRID_POINTS)

                best_p = -1.0
                best_x = None

                for t in t_grid:
                    for s in s_grid:
                        for li in l_grid:
                            p = model.predict_proba([t, s, li])
                            if p > best_p:
                                best_p = p
                                best_x = (t, s, li)

                opt_values = best_x 

                print( f"optimal vals: Temp={best_x[0]:.2f}, Sound={best_x[1]:.2f}, Light={best_x[2]:.2f} "
                    f"(pred focus p={best_p:.2f}, samples={ml_samples})")

            # sending console notifications
            if (opt_values is not None) and (TempC is not None) and (Sound is not None) and (Light is not None):
                if p_pred is None:
                    p_cur = model.predict_proba([TempC, Sound, Light]) if ml_samples >= 2 else None
                else:
                    p_cur = float(p_pred)
                if (p_cur is not None) and (now_t - last_notify_time >= NOTIFY_COOLDOWN_SEC):
                    if (p_cur < PROB_LOW_THRESH) or (last_status == "OFF"):
                        t_opt, s_opt, l_opt = opt_values
                        msgs = []

                        if abs(TempC - t_opt) > SUGGEST_TOL["TempC"]:
                            direction = "increase" if TempC < t_opt else "decrease"
                            msgs.append(f"Temp: {direction} toward {t_opt:.2f} (now {TempC:.2f})")

                        if abs(Sound - s_opt) > SUGGEST_TOL["Sound"]:
                            direction = "increase" if Sound < s_opt else "decrease"
                            msgs.append(f"Sound: {direction} toward {s_opt:.2f} (now {Sound:.2f})")

                        if abs(Light - l_opt) > SUGGEST_TOL["Light"]:
                            direction = "increase" if Light < l_opt else "decrease"
                            msgs.append(f"Light: {direction} toward {l_opt:.2f} (now {Light:.2f})")

                        if msgs:
                            last_notify_time = now_t
                            print(f"[ADJUST] focus={last_status} p={p_cur:.2f} | " + " | ".join(msgs))

            # writing log data
            if have_face and (latest_arduino_vals is not None):
                ts = datetime.now().strftime("%H%M%S")
                t_opt = opt_values[0] if opt_values else None
                s_opt = opt_values[1] if opt_values else None
                l_opt = opt_values[2] if opt_values else None

                combined_writer.writerow([
                    ts,
                    last_status,
                    round2(yaw_deg),
                    round2(pitch_deg),
                    round2(h_ratio),
                    round2(v_ratio),
                    *[round2(v) for v in latest_arduino_vals],
                    round2(p_pred) if p_pred is not None else None,
                    round2(t_opt), round2(s_opt), round2(l_opt)
                ])
                combined_log.flush()

            # opencv overlay
            cv.putText(
                frame,
                f"Status: {last_status}",
                (20, 45),
                font,
                0.8,
                (0, 200, 0) if majority else (0, 0, 255),
                2,
                cv.LINE_AA
            )

            if opt_values is not None and TempC is not None and Sound is not None and Light is not None:
                cv.putText(frame, f"Env: T={TempC:.2f} S={Sound:.2f} L={Light:.2f}", (20, 140),
                           font, 0.55, (0, 0, 0), 1, cv.LINE_AA)
                cv.putText(frame, f"Opt: T={opt_values[0]:.2f} S={opt_values[1]:.2f} L={opt_values[2]:.2f}", (20, 165),
                           font, 0.55, (0, 0, 0), 1, cv.LINE_AA)

            cv.imshow("Eye + Arduino + Focus Optimizer", frame)
            if cv.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    combined_log.close()
    ser.close()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()
