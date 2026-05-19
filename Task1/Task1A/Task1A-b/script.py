import cv2
import numpy as np


def analyze_video(video_path):

    result = {
        "top_wall_hits": 0,
        "bottom_wall_hits": 0,
        "left_wall_hits": 0,
        "right_wall_hits": 0
    }

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error opening video")
        return result

    lower_green = np.array([40, 80, 80])
    upper_green = np.array([85, 255, 255])

    WIDTH  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    HEIGHT = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))


    left_collision   = False
    right_collision  = False
    top_collision    = False
    bottom_collision = False

    wall_threshold = 50

    prev_cx, prev_cy = None, None

    while True:

        ret, frame = cap.read()
        if not ret:
            break

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        mask = cv2.inRange(hsv, lower_green, upper_green)

        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,   kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel)

        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        cx, cy = prev_cx, prev_cy          # fall back to last known position

        if contours:
            largest = max(contours, key=cv2.contourArea)

            if cv2.contourArea(largest) >= 200:

                M = cv2.moments(largest)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    prev_cx, prev_cy = cx, cy  # update last known position

        # skip frame if we have no position at all
        if cx is None or cy is None:
            continue

        # --- LEFT WALL ---
        if cx <= wall_threshold:
            if not left_collision:
                result["left_wall_hits"] += 1
                left_collision = True
        else:
            left_collision = False

        # --- RIGHT WALL ---
        if cx >= WIDTH - wall_threshold:
            if not right_collision:
                result["right_wall_hits"] += 1
                right_collision = True
        else:
            right_collision = False

        # --- TOP WALL ---
        if cy <= wall_threshold:
            if not top_collision:
                result["top_wall_hits"] += 1
                top_collision = True
        else:
            top_collision = False

        # --- BOTTOM WALL ---
        if cy >= HEIGHT - wall_threshold:
            if not bottom_collision:
                result["bottom_wall_hits"] += 1
                bottom_collision = True
        else:
            bottom_collision = False

    cap.release()
    return result