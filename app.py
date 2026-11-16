import streamlit as st
import tensorflow as tf
import numpy as np
import json
import cv2
import av
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase

st.title("Diagnóstico de Enfermedades en Hojas")
st.write("Centra una hoja dentro del recuadro. Si no se detecta una hoja, se muestra Desconocido.")


@st.cache_resource
def load_resources():
    model = tf.keras.models.load_model("resultados/mejor_fase2.keras")
    with open("resultados/clases.json") as f:
        classes = json.load(f)
    return model, classes


model, class_names = load_resources()

UMBRAL_CONFIANZA = 50.0
UMBRAL_VERDE = 0.05
UMBRAL_BORDES = 0.002


class LeafProcessor(VideoProcessorBase):
    def __init__(self):
        self.clase = "Desconocido"
        self.conf = 0.0
        self.frames = 0

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        h, w = img.shape[:2]

        x1, y1 = int(w * 0.2), int(h * 0.2)
        x2, y2 = int(w * 0.8), int(h * 0.8)

        self.frames += 1
        if self.frames % 15 == 0:
            crop = img[y1:y2, x1:x2]
            crop = cv2.resize(crop, (224, 224))
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            x = rgb.astype("float32")[None, ...]
            preds = model.predict(x, verbose=0)[0]
            idx = int(np.argmax(preds))
            conf = float(preds[idx] * 100)

            hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
            verde = (
                (hsv[..., 0] >= 35)
                & (hsv[..., 0] <= 85)
                & (hsv[..., 1] > 40)
                & (hsv[..., 2] > 40)
            ).mean()
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            bordes = float((cv2.Canny(gray, 50, 150) > 0).mean())

            if (
                conf < UMBRAL_CONFIANZA
                or verde < UMBRAL_VERDE
                or bordes < UMBRAL_BORDES
            ):
                self.clase = "Desconocido"
            else:
                self.clase = class_names[idx]
            self.conf = conf

        if self.clase == "Desconocido":
            label = "Hoja: Desconocido"
            color = (0, 0, 255)
        else:
            label = f"Hoja: {self.clase} {self.conf:.1f}%"
            color = (0, 255, 0)

        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
        cv2.putText(
            img,
            label,
            (x1, max(y1 - 10, 25)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 0),
            4,
            cv2.LINE_AA,
        )
        cv2.putText(
            img,
            label,
            (x1, max(y1 - 10, 25)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2,
            cv2.LINE_AA,
        )
        return av.VideoFrame.from_ndarray(img, format="bgr24")


webrtc_streamer(
    key="leaf",
    video_processor_factory=LeafProcessor,
    media_stream_constraints={
        "video": {"facingMode": "environment"},
        "audio": False,
    },
    async_processing=True,
)
