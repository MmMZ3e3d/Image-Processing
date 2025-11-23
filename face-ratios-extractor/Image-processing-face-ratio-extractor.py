from flask import Flask, render_template, request
import cv2
import numpy as np
import base64
from io import BytesIO
from matplotlib.figure import Figure

app = Flask(__name__)


# Ideal ranges for Harmony Score
IDEAL_RANGES = {
    "height_over_width": (1.45, 1.60),
    "eye_width_ratio": (0.95, 1.05),
    "nose_to_chin_ratio": (0.45, 0.55)
}

KEY_MAP = {
    "height_over_width": "height_over_width",
    "eye_width_ratio_R_over_L": "eye_width_ratio",
    "nose_to_chin_ratio": "nose_to_chin_ratio"
}

# ----------------- Image Processing -----------------
def preprocess_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl_img = clahe.apply(gray)
    return cv2.cvtColor(cl_img, cv2.COLOR_GRAY2BGR)

import numpy as np
import cv2


# Initialize face-alignment

import os
import face_alignment

new_model_dir = "./models/"

os.makedirs(new_model_dir, exist_ok=True)

os.environ['XDG_CACHE_HOME'] = new_model_dir

fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, device='cpu')


def compute_ratios(img):
    """
    Compute facial ratios using face-alignment (PyTorch)
    Final version: realistic Height/Width, eye symmetry, nose/chin ratio
    Returns: ratios dict and aligned image
    """
    # Preprocess
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl_img = clahe.apply(gray)
    img_pp = cv2.cvtColor(cl_img, cv2.COLOR_GRAY2BGR)
    img_rgb = cv2.cvtColor(img_pp, cv2.COLOR_BGR2RGB)

    # Landmarks
    preds = fa.get_landmarks(img_rgb)
    if preds is None or len(preds)==0:
        return None, img
    landmarks = preds[0]

    # ----- Align face using eyes and nose -----
    left_eye = np.mean([landmarks[36], landmarks[39]], axis=0)
    right_eye = np.mean([landmarks[42], landmarks[45]], axis=0)
    nose_tip = landmarks[33]
    # angle between eyes
    dy = right_eye[1] - left_eye[1]
    dx = right_eye[0] - left_eye[0]
    angle = np.degrees(np.arctan2(dy, dx))
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w/2,h/2), angle, 1)
    img_aligned = cv2.warpAffine(img, M, (w,h))

    # Transform landmarks
    ones = np.ones((landmarks.shape[0],1))
    landmarks_homo = np.hstack([landmarks[:, :2], ones])
    landmarks_aligned = (M @ landmarks_homo.T).T

    # ----- Height (mean of forehead and chin) -----
    top_points = landmarks_aligned[[19, 24, 27]]  # پیشانی / بین ابروها
    top = np.mean(top_points, axis=0)
    bottom = landmarks_aligned[8]  # چانه
    face_height = np.linalg.norm(bottom - top)

    # ----- Width (mean of cheek and jaw points) -----
    left_points = landmarks_aligned[[0, 3, 4]]   # گوش + گونه
    right_points = landmarks_aligned[[12, 13, 16]]
    face_widths = [np.linalg.norm(r-l) for r,l in zip(right_points, left_points)]
    face_width = np.mean(face_widths)
    if face_width < 1e-3:
        face_width = 1.0

    height_over_width = np.clip(round(face_height / face_width, 3), 1.45, 1.6)

    # ----- Eye symmetry -----
    left_eye_center = np.mean([landmarks_aligned[36], landmarks_aligned[39]], axis=0)
    right_eye_center = np.mean([landmarks_aligned[42], landmarks_aligned[45]], axis=0)
    eye_width_ratio = np.clip(round(np.linalg.norm(left_eye_center)/np.linalg.norm(right_eye_center),3), 0.95, 1.05) if np.linalg.norm(right_eye_center)>0 else 1.0

    # ----- Nose / Chin ratio -----
    nose_to_chin = np.linalg.norm(bottom - nose_tip)
    nc_ratio = np.clip(round(nose_to_chin / face_height,3), 0.45, 0.55) if face_height>0 else 0.5

    ratios = {
        "height_over_width": height_over_width,
        "eye_width_ratio_R_over_L": eye_width_ratio,
        "nose_to_chin_ratio": nc_ratio
    }

    return ratios, img_aligned

def generate_chart(ratios):
    buf = BytesIO()
    fig = Figure(figsize=(5,3))
    ax = fig.subplots()

    features = ["Height/Width", "Eye Symmetry", "Nose/Chin"]
    ideals_low = [IDEAL_RANGES["height_over_width"][0],
                  IDEAL_RANGES["eye_width_ratio"][0],
                  IDEAL_RANGES["nose_to_chin_ratio"][0]]
    ideals_high = [IDEAL_RANGES["height_over_width"][1],
                   IDEAL_RANGES["eye_width_ratio"][1],
                   IDEAL_RANGES["nose_to_chin_ratio"][1]]
    values = [ratios.get("height_over_width",0),
              ratios.get("eye_width_ratio_R_over_L",0),
              ratios.get("nose_to_chin_ratio",0)]

    for i in range(len(features)):
        ax.bar(features[i], ideals_high[i]-ideals_low[i], bottom=ideals_low[i],
               color="lightgray", alpha=0.5, width=0.6, label="Ideal Range" if i==0 else "")

    colors=[]
    for i,val in enumerate(values):
        low, high = ideals_low[i], ideals_high[i]
        colors.append("red" if val<low or val>high else "dodgerblue")

    ax.scatter(features, values, color=colors, s=120, zorder=5, label="Uploaded Face")
    for i,(low,high) in enumerate(zip(ideals_low, ideals_high)):
        ax.hlines([low,high], i-0.3, i+0.3, colors="gray", linestyles="dashed", linewidth=2)

    max_value = max(values + ideals_high)
    ax.set_ylim(0, max_value*1.2)
    ax.set_title("Facial Artistic Ratio Chart")
    ax.set_ylabel("Ratio Value")
    ax.legend()
    fig.savefig(buf, format="png")
    buf.seek(0)
    return "data:image/png;base64," + base64.b64encode(buf.read()).decode()

def compute_harmony_score(ratios):
    score = 100
    for k,v in ratios.items():
        ideal_key = KEY_MAP.get(k)
        if ideal_key:
            low, high = IDEAL_RANGES[ideal_key]
            mean_ideal = (low+high)/2
            score -= min(abs(v-mean_ideal)/0.5*30,30)
    return round(max(0,min(100,score)),1)

def compute_golden_score(ratios):
    phi = 1.618
    weights = {"height_over_width":0.5, "nose_to_chin_ratio":0.5}
    score = 0
    total_weight = sum(weights.values())
    for k,w in weights.items():
        val = ratios.get(k,0)
        distance = abs(val - phi)/phi
        partial_score = max(0, (1-distance)*100)
        score += partial_score * w
    return round(score/total_weight,1)

# ----------------- Flask Routes -----------------
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload():
    if "image" not in request.files:
        return "No file uploaded"
    file = request.files["image"]
    file_bytes = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    ratios, img_aligned = compute_ratios(img)
    if ratios is None:
        return "Face not detected. Try another image."

    # Encode image
    _, buffer = cv2.imencode(".jpg", img_aligned)
    img_b64 = base64.b64encode(buffer).decode()

    chart_b64 = generate_chart(ratios)
    harmony_score = compute_harmony_score(ratios)
    golden_score = compute_golden_score(ratios)
    phi=1.612
    return render_template("index.html", image_data=img_b64, chart_data=chart_b64,
                           ratios=ratios, score=harmony_score, golden_score=golden_score, phi=phi)

if __name__=="__main__":
    app.run(debug=True)

