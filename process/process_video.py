import os
import cv2
import numpy as np
from ultralytics import YOLO

# DEBUG FLAGS
DEBUG = True
DEBUG_CLASSIFIER = True
DEBUG_CROP = True
DEBUG_TRACK = True
DEBUG_TEAM = True

COLOR_PLAYER_YOLO = (180, 80, 80)
COLOR_GOALKEEPER_YOLO = (120, 220, 120)
COLOR_REFEREE = (200, 200, 200)
COLOR_BALL = (255, 255, 0)

COLOR_TEAM_A = (0, 0, 255)
COLOR_TEAM_B = (0, 255, 0)

def parse_team_classifier_label(label_str):
    label_str = label_str.strip().lower()
    if "team a" in label_str:
        return "A"
    if "team b" in label_str:
        return "B"
    return None

def process_video(model, classifier, local_video, original_name):

    print("\n==============================")
    print("   DEBUG MODE: TeamClassifier")
    print("==============================\n")

    cap = cv2.VideoCapture(local_video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output_path = "debug_output.mp4"
    writer = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps if fps > 0 else 30,
        (w, h)
    )

    frame_index = 0
    team_memory = {}
    EMA_ALPHA = 0.25
    CONF_THRESHOLD = 0.40

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model.track(frame, persist=True)
        detections = results[0].boxes

        print(f"\n--- FRAME {frame_index} ---")

        for box in detections:
            cls = int(box.cls[0])
            base_label = model.names[cls].lower()

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            w_box = x2 - x1
            h_box = y2 - y1

            track_id = int(box.id[0]) if box.id is not None else None

            print(f"\nObject: {base_label} | track_id={track_id}")
            print(f"Box: {x1},{y1},{x2},{y2}")

            # REFEREE
            if base_label == "referee":
                print("→ YOLO: Referee")
                continue

            # BALL
            if base_label == "ball":
                print("→ YOLO: Ball")
                continue

            # PLAYER / GOALKEEPER
            role = base_label
            team = None

            # -----------------------------
            # TEAM CLASSIFIER DEBUG
            # -----------------------------
            if classifier is not None:
                crop = frame[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]

                if DEBUG_CROP:
                    print(f"Crop shape: {crop.shape}, empty={crop.size == 0}")

                if crop.size == 0:
                    print("⚠️ Crop vazio → classificador ignorado")
                else:
                    cls_results = classifier.predict(crop, verbose=False)
                    cls_boxes = cls_results[0].boxes

                    print("Classifier raw results:", cls_results)

                    if cls_boxes is None or len(cls_boxes) == 0:
                        print("⚠️ Classifier devolveu 0 boxes")
                    else:
                        cbox = cls_boxes[0]
                        ccls = int(cbox.cls[0])
                        raw_label = classifier.names[ccls]
                        conf = float(cbox.conf[0]) if cbox.conf is not None else 1.0

                        print(f"Classifier label: {raw_label} | conf={conf}")

                        team_detected = parse_team_classifier_label(raw_label)
                        print(f"Team detected: {team_detected}")

                        if team_detected is None:
                            print("⚠️ parse_team_classifier_label devolveu None")
                        else:
                            team = team_detected

            # -----------------------------
            # FINAL LABEL DEBUG
            # -----------------------------
            if team == "A":
                final_label = f"{role.capitalize()} Team A"
                print("✔ FINAL: Team A")
            elif team == "B":
                final_label = f"{role.capitalize()} Team B"
                print("✔ FINAL: Team B")
            else:
                final_label = role.capitalize()
                print("❌ FINAL: YOLO fallback (sem equipa)")

        writer.write(frame)
        frame_index += 1

    cap.release()
    writer.release()

    print("\n==============================")
    print("   DEBUG COMPLETO")
    print("==============================\n")

    return output_path
