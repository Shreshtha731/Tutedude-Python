"""
Exhibition-mode Streamlit dashboard: same recognition -> stabilizer ->
sentence pipeline as main.py, rendered as a shareable web dashboard with
big, readable panels for a booth/demo table.

Run: streamlit run dashboard.py
"""
import time
import cv2
import streamlit as st

import config as cfg
from recognition_interface import MockRecognitionEngine
from stabilizer import SignStabilizer, SentenceBuilder
from tts_module import TTSModule

st.set_page_config(page_title="Sign Language Recognition - Live Demo", layout="wide")


@st.cache_resource
def get_pipeline():
    return {
        "engine": MockRecognitionEngine(),
        "stabilizer": SignStabilizer(),
        "sentence": SentenceBuilder(),
        "tts": TTSModule(),
    }


def main():
    pipeline = get_pipeline()
    engine = pipeline["engine"]
    stabilizer = pipeline["stabilizer"]
    sentence = pipeline["sentence"]
    tts = pipeline["tts"]

    st.title("🤟 Sign Language Recognition — Live Exhibition Dashboard")

    col_video, col_info = st.columns([2, 1])
    with col_video:
        video_slot = st.empty()
    with col_info:
        sign_slot = st.empty()
        conf_slot = st.empty()
        sentence_slot = st.empty()
        st.divider()
        run = st.toggle("Run camera", value=True)
        if st.button("🔊 Speak sentence"):
            tts.speak(sentence.text())
        if st.button("🗑️ Clear sentence"):
            sentence.words = []

    if not run:
        st.info("Toggle 'Run camera' on to start the live feed.")
        return

    cap = cv2.VideoCapture(cfg.CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.FRAME_HEIGHT)

    while run:
        ok, frame = cap.read()
        if not ok:
            st.error("Could not read from camera.")
            break
        frame = cv2.flip(frame, 1)

        label, confidence = engine.predict(frame)
        confirmed = stabilizer.push(label, confidence)
        if confirmed:
            sentence.add(confirmed)
            if sentence.consume_speak_request():
                tts.speak(sentence.text())

        video_slot.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")
        sign_slot.markdown(f"### Current sign: `{label}`")
        conf_slot.progress(min(1.0, max(0.0, confidence)), text=f"Confidence: {confidence*100:.0f}%")
        sentence_slot.markdown(f"### Sentence\n> {sentence.text() or '_(empty)_'}")

        time.sleep(0.03)

    cap.release()


if __name__ == "__main__":
    main()
