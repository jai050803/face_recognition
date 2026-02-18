import streamlit as st
import cv2
import numpy as np
import pickle
import tempfile
import os
import time
from collections import defaultdict
import pandas as pd
import csv
from insightface.app import FaceAnalysis
import onnxruntime

# -------------------- Page config --------------------
st.set_page_config(page_title="Faculty Face Recognition", layout="wide")

# -------------------- Utility functions --------------------
def cosine_similarity(a, b):
    return np.dot(a, b)

def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH

    boxAArea = (boxA[2]-boxA[0]) * (boxA[3]-boxA[1])
    boxBArea = (boxB[2]-boxB[0]) * (boxB[3]-boxB[1])

    union = boxAArea + boxBArea - interArea
    if union == 0:
        return 0
    return interArea / union

class SimpleTracker:
    def __init__(self, iou_th=0.3, max_age=30, confirm_frames=3, lost_frames=3):
        self.tracks = {}
        self.next_id = 0
        self.iou_th = iou_th
        self.max_age = max_age
        self.confirm_frames = confirm_frames
        self.lost_frames = lost_frames

    def iou(self, a, b):
        return iou(a, b)

    def update(self, detections):
        # Age tracks & remove expired
        for tid in list(self.tracks.keys()):
            t = self.tracks[tid]
            t['age'] += 1
            if t['age'] > self.max_age:
                del self.tracks[tid]
                continue
            # If confirmed and lost streak too long, drop confirmation
            if t['confirmed_name'] != "Unknown" and t['lost_streak'] >= self.lost_frames:
                t['confirmed_name'] = "Unknown"
                t['lost_streak'] = 0

        # Match detections to existing tracks
        used = set()
        for tid, t in self.tracks.items():
            best_iou = 0
            best_idx = -1
            for i, (bbox, name, score) in enumerate(detections):
                if i in used: continue
                iou_val = self.iou(t['bbox'], bbox)
                if iou_val > best_iou and iou_val > self.iou_th:
                    best_iou = iou_val
                    best_idx = i
            if best_idx >= 0:
                bbox, name, score = detections[best_idx]
                used.add(best_idx)
                t['bbox'] = bbox
                t['age'] = 0
                t['votes'].append((name, score))
                if len(t['votes']) > 10: t['votes'].pop(0)

                # Update streaks based on match with confirmed name
                if t['confirmed_name'] != "Unknown":
                    if name == t['confirmed_name'] and score >= SOFT_TH:
                        t['lost_streak'] = 0
                        t['pos_streak'] += 1
                    else:
                        t['lost_streak'] += 1
                        t['pos_streak'] = 0
                else:
                    # No confirmed name – check if we can confirm now
                    recent = t['votes'][-self.confirm_frames:]
                    name_counts = defaultdict(int)
                    for n, s in recent:
                        if s >= STRICT_TH:
                            name_counts[n] += 1
                    if name_counts:
                        best_candidate = max(name_counts.items(), key=lambda x: x[1])
                        if best_candidate[1] >= self.confirm_frames:
                            t['confirmed_name'] = best_candidate[0]
                            t['pos_streak'] = best_candidate[1]
                            t['lost_streak'] = 0
            else:
                # No detection matched – increment lost streak if confirmed
                if t['confirmed_name'] != "Unknown":
                    t['lost_streak'] += 1

        # Create new tracks for unmatched detections
        for i, (bbox, name, score) in enumerate(detections):
            if i not in used:
                self.tracks[self.next_id] = {
                    'bbox': bbox,
                    'votes': [(name, score)],
                    'age': 0,
                    'confirmed_name': "Unknown",
                    'pos_streak': 0,
                    'lost_streak': 0
                }
                self.next_id += 1

        return self.tracks

STRICT_TH = 0.40
SOFT_TH = 0.30

def match_face(face_emb, faculty_embeddings):
    best_name = "Unknown"
    best_score = -1
    for name, ref_emb in faculty_embeddings.items():
        score = cosine_similarity(face_emb, ref_emb)
        if score > best_score:
            best_score = score
            best_name = name

    if best_score >= STRICT_TH:
        return best_name, best_score, "strong"
    elif best_score >= SOFT_TH:
        return best_name, best_score, "weak"
    else:
        return "Unknown", best_score, "no"

def final_identity(votes):
    strong = [v for v in votes if v[1] >= STRICT_TH]
    weak = [v for v in votes if v[1] >= SOFT_TH]

    if len(strong) >= 2:
        return max(strong, key=lambda x: x[1])[0]
    if len(weak) >= 4:
        return max(weak, key=lambda x: x[1])[0]
    return "Unknown"

# -------------------- Load models (cached) --------------------
@st.cache_resource
def load_face_analysis():
    app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=0, det_size=(1024, 1024))
    return app

# -------------------- Embeddings tab --------------------
def embeddings_tab():
    st.header("Faculty Embeddings Information")
    uploaded_emb = st.file_uploader("Upload embeddings.pkl", type=["pkl"])
    if uploaded_emb is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pkl") as tmp:
            tmp.write(uploaded_emb.read())
            tmp_path = tmp.name
        with open(tmp_path, "rb") as f:
            faculty_embeddings = pickle.load(f)
        st.success(f"Loaded embeddings for {len(faculty_embeddings)} faculty members.")
        st.write("Faculty names:")
        for name in faculty_embeddings.keys():
            st.write(f"- {name}")
        # Store in session state for later use
        st.session_state["faculty_embeddings"] = faculty_embeddings
        st.session_state["embeddings_loaded"] = True
        # Clean up temp file
        os.unlink(tmp_path)
    else:
        st.info("Please upload your embeddings.pkl file.")

# -------------------- Video processing tab --------------------
def video_tab():
    st.header("Video Face Recognition")
    if "faculty_embeddings" not in st.session_state:
        st.warning("Please upload embeddings.pkl in the first tab first.")
        return

    faculty_embeddings = st.session_state["faculty_embeddings"]

    uploaded_video = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov", "mkv"])
    if uploaded_video is not None:
        # Save uploaded video to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_video:
            tmp_video.write(uploaded_video.read())
            video_path = tmp_video.name

        # Show video details
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = total_frames / fps
        cap.release()

        st.subheader("Video Details")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("FPS", f"{fps:.2f}")
        col2.metric("Total Frames", total_frames)
        col3.metric("Resolution", f"{width}x{height}")
        col4.metric("Duration", f"{duration:.2f} sec")

        # Sampling logic based on duration
        if duration > 300:  # longer than 5 minutes
            process_every_n_frames = int(12 * fps)
            st.info(f"Long video: processing every {process_every_n_frames} frames (≈5 frames per minute)")
        else:
            process_every_n_frames = int(fps)
            st.info(f"Short video: processing every {process_every_n_frames} frames (≈1 fps)")

        # Safety
        if process_every_n_frames > total_frames:
            process_every_n_frames = total_frames

        if st.button("Process Video"):
            # Initialize face analysis
            app = load_face_analysis()

            # Prepare output video path
            output_video_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name

            # Try different codecs (avc1 is widely supported)
            fourcc = cv2.VideoWriter_fourcc(*'avc1')  # H.264
            out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
            if not out.isOpened():
                # Fallback to mp4v
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

            # Tracker and logs
            tracker = SimpleTracker()
            presence_log = defaultdict(lambda: {"first": None, "last": None})

            # Processing loop
            cap = cv2.VideoCapture(video_path)
            frame_number = 0
            progress_bar = st.progress(0, text="Processing video...")
            status_text = st.empty()

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_number += 1
                progress = frame_number / total_frames
                progress_bar.progress(progress, text=f"Processing frame {frame_number}/{total_frames}")

                # Only run heavy processing on selected frames
                if frame_number % process_every_n_frames == 0:
                    # ROI: top 70%
                    roi_h = int(0.7 * height)
                    roi = frame[0:roi_h, :]

                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    rgb_roi = rgb[0:roi_h, :]

                    faces = app.get(rgb_roi)

                    detections = []
                    for face in faces:
                        x1, y1, x2, y2 = map(int, face.bbox)
                        emb = face.normed_embedding
                        name, score, strength = match_face(emb, faculty_embeddings)
                        bbox_full = (x1, y1, x2, y2)
                        detections.append((bbox_full, name, score))

                    tracks = tracker.update(detections)

                    # Draw tracks
                    # Inside the frame processing loop
                    for tid, t in tracks.items():
                        bbox = t['bbox']
                        identity = t['confirmed_name'] if t['confirmed_name'] != "Unknown" else "?"
                        color = (0, 255, 0) if t['confirmed_name'] != "Unknown" else (0, 0, 255)
                        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
                        cv2.putText(frame, identity, (bbox[0], bbox[1]-10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                        
                        # Update presence log only when identity is confirmed
                        if t['confirmed_name'] != "Unknown":
                            timestamp = frame_number / fps
                            name = t['confirmed_name']
                            if presence_log[name]["first"] is None:
                                presence_log[name]["first"] = timestamp
                            presence_log[name]["last"] = timestamp
                        # Presence log
                        if identity != "Unknown":
                            timestamp_sec = frame_number / fps
                            if presence_log[identity]["first"] is None:
                                presence_log[identity]["first"] = timestamp_sec
                            presence_log[identity]["last"] = timestamp_sec

                # Write frame to output video
                out.write(frame)

            cap.release()
            out.release()
            progress_bar.empty()
            status_text.success("Processing complete!")

            # Check if output video was created successfully
            if os.path.getsize(output_video_path) == 0:
                st.error("Output video file is empty. Codec may not be supported.")
            else:
                st.subheader("Annotated Video")
                # Read video file as bytes for display
                with open(output_video_path, "rb") as f:
                    video_bytes = f.read()
                st.video(video_bytes)

                # Also provide download button
                st.download_button(
                    label="Download Annotated Video",
                    data=video_bytes,
                    file_name="annotated_output.mp4",
                    mime="video/mp4"
                )

            # Prepare presence log table
            log_data = []
            for name, times in presence_log.items():
                first = times["first"]
                last = times["last"]
                duration_sec = last - first if first is not None and last is not None else 0
                log_data.append({
                    "Name": name,
                    "First Seen (sec)": f"{first:.2f}" if first else "N/A",
                    "Last Seen (sec)": f"{last:.2f}" if last else "N/A",
                    "Duration (sec)": f"{duration_sec:.2f}"
                })
            df = pd.DataFrame(log_data)
            st.subheader("Presence Log")
            st.dataframe(df)

            # Download log as CSV
            csv_buffer = tempfile.NamedTemporaryFile(delete=False, suffix=".csv").name
            df.to_csv(csv_buffer, index=False)
            with open(csv_buffer, "rb") as f:
                st.download_button(
                    label="Download Presence Log as CSV",
                    data=f,
                    file_name="presence_log.csv",
                    mime="text/csv"
                )
            os.unlink(csv_buffer)

            # Cleanup temp files
            os.unlink(video_path)
            os.unlink(output_video_path)

# -------------------- Main app --------------------
tab1, tab2 = st.tabs(["📁 Embeddings Info", "🎥 Video Processing"])

with tab1:
    embeddings_tab()

with tab2:
    video_tab()