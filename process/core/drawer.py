import cv2
import numpy as np

COLOR_PLAYER_YOLO = (180, 80, 80)
COLOR_GOALKEEPER_YOLO = (120, 220, 120)
COLOR_REFEREE = (200, 200, 200)
COLOR_BALL = (255, 255, 0)

COLOR_TEAM_A = (0, 0, 255)
COLOR_TEAM_B = (0, 255, 0)

def draw_triangle(frame, x, y, color):
    pts = np.array([[x, y], [x - 14, y - 22], [x + 14, y - 22]], np.int32)
    cv2.polylines(frame, [pts], True, (0, 0, 0), 2)
    cv2.fillPoly(frame, [pts], color)

def draw_label(frame, x, y, label):
    cv2.putText(frame, label, (x - 35, y - 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3)
    cv2.putText(frame, label, (x - 35, y - 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

def draw_ball(frame, x, y, w_box, h_box):
    radius = int(max(w_box, h_box) / 2)
    cx = x + w_box // 2
    cy = y + h_box // 2
    cv2.circle(frame, (cx, cy), radius, (0, 0, 0), 3)
    cv2.circle(frame, (cx, cy), radius, COLOR_BALL, 2)

def draw_entity(frame, role, team, x1, y1, x2, y2):
    w_box = x2 - x1
    h_box = y2 - y1
    cx = x1 + w_box // 2

    if role == "referee":
        color = COLOR_REFEREE
        label = "Referee"
    elif role == "ball":
        draw_ball(frame, x1, y1, w_box, h_box)
        return f"ball - {x1} {y1} {x2} {y2}"
    else:
        if team == "A":
            color = COLOR_TEAM_A
            label = f"{role.capitalize()} Team A"
        elif team == "B":
            color = COLOR_TEAM_B
            label = f"{role.capitalize()} Team B"
        else:
            color = COLOR_PLAYER_YOLO if role == "player" else COLOR_GOALKEEPER_YOLO
            label = role.capitalize()

    draw_triangle(frame, cx, y1, color)
    draw_label(frame, cx, y1, label)
    return f"{label} - {x1} {y1} {x2} {y2}"
