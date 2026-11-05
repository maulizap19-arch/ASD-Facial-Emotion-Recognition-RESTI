import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from torchvision import models
from torchvision import transforms

from PIL import Image

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="Klasifikasi Emosi Wajah Anak ASD",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================================================
# CUSTOM CSS
# =====================================================

st.markdown(
    """
    <style>
    .main {
        background-color: #fafafa;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    .app-header {
        background: linear-gradient(135deg, #6C63FF 0%, #4834d4 100%);
        padding: 2rem 2rem;
        border-radius: 16px;
        color: white;
        margin-bottom: 1.5rem;
    }

    .app-header h1 {
        margin: 0;
        font-size: 2rem;
        font-weight: 700;
    }

    .app-header p {
        margin: 0.4rem 0 0 0;
        opacity: 0.9;
        font-size: 0.95rem;
    }

    .img-card {
        background: white;
        border-radius: 14px;
        padding: 1rem;
        box-shadow: 0 2px 10px rgba(0,0,0,0.06);
        text-align: center;
    }

    .img-card p {
        font-weight: 600;
        color: #444;
        margin-bottom: 0.5rem;
    }

    .img-card div[data-testid="stImage"] {
        display: flex;
        justify-content: center;
        align-items: center;
        height: 260px;
        overflow: hidden;
        border-radius: 10px;
        background: #f0f0f0;
    }

    .img-card div[data-testid="stImage"] img {
        height: 260px;
        width: 100%;
        object-fit: cover;
    }

    .chart-card {
        background: white;
        border-radius: 14px;
        padding: 1rem;
        box-shadow: 0 2px 10px rgba(0,0,0,0.06);
        display: flex;
        justify-content: center;
    }

    .result-card {
        background: white;
        border-radius: 14px;
        padding: 1.5rem;
        box-shadow: 0 2px 10px rgba(0,0,0,0.06);
    }

    div[data-testid="stMetric"] {
        background: white;
        border-radius: 12px;
        padding: 1rem;
        box-shadow: 0 2px 10px rgba(0,0,0,0.06);
    }

    .stFileUploader {
        background: white;
        border-radius: 14px;
        padding: 1rem;
        box-shadow: 0 2px 10px rgba(0,0,0,0.06);
    }
    </style>
    """,
    unsafe_allow_html=True
)

# =====================================================
# LOAD MODEL
# =====================================================

@st.cache_resource
def load_model():

    model = models.resnet18(
        weights=None
    )

    num_features = model.fc.in_features

    model.fc = nn.Linear(
        num_features,
        2
    )

    model.load_state_dict(
        torch.load(
            "resnet18_best.pth",
            map_location="cpu"
        )
    )

    model.eval()

    return model


model = load_model()

# =====================================================
# MEDIAPIPE
# =====================================================

mp_face_mesh = mp.solutions.face_mesh

face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5
)

# =====================================================
# LANDMARK REGIONS
# =====================================================

LEFT_EYE = [
    33,133,160,159,158,157,173,
    246,161,163,144,145,153,154,155
]

RIGHT_EYE = [
    362,263,387,386,385,384,398,
    466,388,390,373,374,380,381,382
]

LEFT_EYEBROW = [
    70,63,105,66,107,55,65,52
]

RIGHT_EYEBROW = [
    336,296,334,293,300,285,295,282
]

NOSE = [
    168,6,197,195,5,4,
    1,19,94,2,98,327
]

MOUTH = [
    61,146,91,181,84,17,
    314,405,321,375,291,
    308,324,318,402,317,
    14,87,178,88,95
]

# =====================================================
# FACE CROP
# SESUAI NOTEBOOK
# =====================================================

def crop_face(image, margin=35):

    rgb = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    results = face_mesh.process(rgb)

    if not results.multi_face_landmarks:
        return None

    h, w, _ = image.shape

    landmarks = (
        results
        .multi_face_landmarks[0]
        .landmark
    )

    x_coords = [
        int(lm.x * w)
        for lm in landmarks
    ]

    y_coords = [
        int(lm.y * h)
        for lm in landmarks
    ]

    x_min = max(
        0,
        min(x_coords) - margin
    )

    x_max = min(
        w,
        max(x_coords) + margin
    )

    y_min = max(
        0,
        min(y_coords) - margin
    )

    y_max = min(
        h,
        max(y_coords) + margin
    )

    cropped = image[
        y_min:y_max,
        x_min:x_max
    ]

    if cropped.size == 0:
        return None

    return cropped

# =====================================================
# LANDMARK POINTS
# =====================================================

def get_points(
    indices,
    landmarks,
    width,
    height
):

    points = []

    for idx in indices:

        x = int(
            landmarks[idx].x * width
        )

        y = int(
            landmarks[idx].y * height
        )

        points.append(
            [x, y]
        )

    return np.array(
        points,
        dtype=np.int32
    )

# =====================================================
# FACE SEGMENTATION
# SESUAI NOTEBOOK
# =====================================================

def segment_face_regions(image):

    orig_h, orig_w = image.shape[:2]

    if min(orig_h, orig_w) < 250:

        image = cv2.resize(
            image,
            None,
            fx=2,
            fy=2,
            interpolation=cv2.INTER_CUBIC
        )

    rgb = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    results = face_mesh.process(rgb)

    if not results.multi_face_landmarks:
        return None

    h, w, _ = image.shape

    landmarks = (
        results
        .multi_face_landmarks[0]
        .landmark
    )

    mask = np.zeros(
        (h, w),
        dtype=np.uint8
    )

    regions = [

        LEFT_EYE,
        RIGHT_EYE,

        LEFT_EYEBROW,
        RIGHT_EYEBROW,

        NOSE,
        MOUTH
    ]

    for region in regions:

        pts = get_points(
            region,
            landmarks,
            w,
            h
        )

        hull = cv2.convexHull(
            pts
        )

        cv2.fillConvexPoly(
            mask,
            hull,
            255
        )

    kernel = np.ones(
        (25, 25),
        np.uint8
    )

    mask = cv2.dilate(
        mask,
        kernel,
        iterations=1
    )

    segmented = cv2.bitwise_and(
        image,
        image,
        mask=mask
    )

    return segmented

# =====================================================
# TRANSFORM
# =====================================================

transform = transforms.Compose([

    transforms.Resize(
        (224, 224)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

])

classes = [
    "joy",
    "sadness"
]

# =====================================================
# SIDEBAR
# =====================================================

with st.sidebar:

    st.markdown("### ℹ️ Deskripsi Sistem")

    st.write(
        "Sistem ini digunakan untuk mengklasifikasikan ekspresi "
        "emosi (senang / sedih) pada wajah anak dengan ASD "
        "menggunakan **ResNet18** dan segmentasi landmark "
        "wajah dari **MediaPipe**."
    )

    st.markdown("---")

    st.markdown("### ⚙️ Cara Pakai")

    st.write(
        "1. Upload foto wajah (JPG/JPEG/PNG).\n"
        "2. Sistem melakukan deteksi, cropping, dan segmentasi wajah secara otomatis.\n"
        "3. Model ResNet18 mengklasifikasikan emosi wajah.\n"
        "4. Hasil prediksi emosi dan nilai confidence ditampilkan.\n"
        "5. Lihat visualisasi emosi pada ruang Valence-Arousal. "
    )

    st.markdown("---")

    st.caption(f"MediaPipe v{mp.__version__}")
    st.caption("Model: ResNet18 + Facial Landmark Segmentation")

# =====================================================
# HEADER
# =====================================================

st.markdown(
    """
    <div class="app-header">
        <h1>Klasifikasi Emosi Wajah Anak ASD</h1>
        <p>Deteksi ekspresi senang / sedih berbasis ResNet18 dan segmentasi landmark wajah MediaPipe</p>
    </div>
    """,
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader(
    "📤 Upload Gambar Wajah",
    type=["jpg", "jpeg", "png"]
)

# =====================================================
# PROCESS IMAGE
# =====================================================

if uploaded_file is not None:

    image = Image.open(
        uploaded_file
    ).convert("RGB")

    image_np = np.array(
        image
    )

    image_bgr = cv2.cvtColor(
        image_np,
        cv2.COLOR_RGB2BGR
    )

    with st.spinner("Mendeteksi & memproses wajah..."):

        # -------------------------
        # Face Crop
        # -------------------------

        cropped_face = crop_face(
            image_bgr
        )

        if cropped_face is None:

            st.error(
                "❌ Wajah tidak terdeteksi. Coba unggah foto lain dengan wajah yang lebih jelas."
            )

            st.stop()

        # -------------------------
        # Face Segmentation
        # -------------------------

        # Memanggil fungsi segment_face_regions() untuk melakukan
        # segmentasi wajah pada hasil cropping.
        # Fungsi ini akan mempertahankan area penting wajah
        # (alis, mata, hidung, dan mulut) menggunakan landmark MediaPipe.
        segmented = segment_face_regions(
            cropped_face
        )
        # Jika landmark wajah tidak berhasil dideteksi,
        # maka proses dihentikan dan sistem menampilkan pesan error.
        if segmented is None:

            st.error(
                "❌ Landmark wajah tidak terdeteksi."
            )

            st.stop()

        # -------------------------
        # Convert RGB (konversi warna BGR -> RGB)
        # -------------------------

        # OpenCV membaca gambar dalam format BGR,
        # sedangkan Streamlit menampilkan gambar dalam format RGB.
        # Oleh karena itu dilakukan konversi warna.

        cropped_rgb = cv2.cvtColor(
            cropped_face,
            cv2.COLOR_BGR2RGB
        )

        segmented_rgb = cv2.cvtColor(
            segmented,
            cv2.COLOR_BGR2RGB
        )

    # -------------------------
    # Display images side by side
    # -------------------------

    st.markdown("### 🖼️ Tahapan Praproses")

    # Membagi tampilan menjadi 3 kolom agar gambar
    # ditampilkan berdampingan.

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown('<div class="img-card">', unsafe_allow_html=True)
        st.markdown("<p>Gambar Asli</p>", unsafe_allow_html=True)
        st.image(image, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="img-card">', unsafe_allow_html=True)
        st.markdown("<p>Hasil Crop Wajah</p>", unsafe_allow_html=True)
        st.image(cropped_rgb, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col3:
        st.markdown('<div class="img-card">', unsafe_allow_html=True)
        st.markdown("<p>Hasil Face Segmentation</p>", unsafe_allow_html=True)
        st.image(segmented_rgb, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # -------------------------
    # Prediction
    # -------------------------

    segmented_pil = Image.fromarray(
        segmented_rgb
    )

    input_tensor = transform(
        segmented_pil
    ).unsqueeze(0)

    with torch.no_grad():

        outputs = model(
            input_tensor
        )

        probabilities = torch.softmax(
            outputs,
            dim=1
        )

        _, pred = torch.max(
            outputs,
            1
        )

        emotion = classes[
            pred.item()
        ]

        confidence = (
            probabilities[0][pred.item()]
            .item()
            * 100
        )

    # =========================
    # HASIL PREDIKSI
    # =========================

    va_mapping = {

        "joy": {
            "valence": 0.8,
            "arousal": 0.6
        },

        "sadness": {
            "valence": -0.7,
            "arousal": -0.4
        }

    }

    valence = va_mapping[emotion]["valence"]
    arousal = va_mapping[emotion]["arousal"]

    st.markdown("### 🎯 Hasil Prediksi")

    res_col1, res_col2, res_col3 = st.columns(3)

    with res_col1:
        st.metric(
            label="Emosi Terdeteksi",
            value=f"{emotion.capitalize()}"
        )

    with res_col2:
        st.metric(
            label="Confidence",
            value=f"{confidence:.2f}%"
        )

    with res_col3:
        st.metric(
            label="Valence / Arousal",
            value=f"{valence:+.1f} / {arousal:+.1f}"
        )

    st.progress(min(int(confidence), 100), text=f"Tingkat Keyakinan Model: {confidence:.2f}%")

    st.markdown("<br>", unsafe_allow_html=True)

    # -------------------------
    # Tabs: Tabel & Grafik
    # -------------------------

    tab1, tab2 = st.tabs(["📋 Tabel Hasil", "📈 Grafik Valence-Arousal"])

    with tab1:

        st.markdown('<div class="result-card">', unsafe_allow_html=True)

        st.table({

            "Parameter": [
                "Emotion",
                "Valence",
                "Arousal"
            ],

            "Value": [
                emotion,
                f"{valence:+.1f}",
                f"{arousal:+.1f}"
            ]

        })

        st.markdown('</div>', unsafe_allow_html=True)

    with tab2:

        chart_col1, chart_col2, chart_col3 = st.columns([1, 2, 1])

        with chart_col2:

            st.markdown('<div class="chart-card">', unsafe_allow_html=True)

            fig, ax = plt.subplots(figsize=(3.2, 3.2), dpi=150)

            fig.patch.set_alpha(0.0)
            ax.set_facecolor("white")

            ax.axhline(y=0, linewidth=1, color="#999")
            ax.axvline(x=0, linewidth=1, color="#999")

            ax.scatter(
                valence,
                arousal,
                s=150,
                color="#6C63FF",
                edgecolors="white",
                linewidths=1.2,
                zorder=3
            )

            ax.annotate(
                f"{emotion}",
                (valence, arousal),
                textcoords="offset points",
                xytext=(8, 8),
                fontsize=9,
                fontweight="bold"
            )

            ax.set_xlim(-1, 1)
            ax.set_ylim(-1, 1)

            ax.set_xlabel("Valence", fontsize=9)
            ax.set_ylabel("Arousal", fontsize=9)

            ax.set_title("Valence-Arousal", fontsize=10)

            ax.tick_params(labelsize=8)

            ax.grid(True, alpha=0.3)

            fig.tight_layout()

            st.pyplot(fig, use_container_width=False)

            st.markdown('</div>', unsafe_allow_html=True)

else:

    st.info("⬆️ Silakan upload gambar wajah untuk memulai analisis emosi.")