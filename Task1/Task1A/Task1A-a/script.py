import cv2
import numpy as np


def analyze_arena(input_image):

    image = cv2.imread(input_image)

    if image is None:
        print("Error loading image.")
        return {}

    result = {
        "arena_size": None,
        "start": None,
        "goal": None,
        "special_cells": {}
    }

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)
    coords = cv2.findNonZero(thresh)
    x, y, w, h = cv2.boundingRect(coords)
    arena = image[y:y+h, x:x+w]

    # Detect arena size 
    arena_gray = cv2.cvtColor(arena, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(arena_gray, 50, 150)

    def find_grid_lines(projection, min_gap=5):
        threshold = np.max(projection) * 0.1
        above = projection > threshold
        lines = []
        in_line = False
        start_idx = 0
        for i, val in enumerate(above):
            if val and not in_line:
                in_line = True
                start_idx = i
            elif not val and in_line:
                in_line = False
                center = (start_idx + i) // 2
                if not lines or (center - lines[-1]) > min_gap:
                    lines.append(center)
        if in_line:
            lines.append((start_idx + len(above)) // 2)
        return lines

    h_lines = find_grid_lines(np.sum(edges, axis=1))
    v_lines = find_grid_lines(np.sum(edges, axis=0))

    possible_sizes = [6, 8, 10, 12]

    def nearest_valid(n):
        return min(possible_sizes, key=lambda s: abs(s - n))

    detected_rows = nearest_valid(len(h_lines) - 1) if len(h_lines) > 1 else 8
    detected_cols = nearest_valid(len(v_lines) - 1) if len(v_lines) > 1 else 8
    arena_size = nearest_valid(max(detected_rows, detected_cols))
    result["arena_size"] = arena_size

    hsv = cv2.cvtColor(arena, cv2.COLOR_BGR2HSV)
    cell_h = arena.shape[0] / arena_size
    cell_w = arena.shape[1] / arena_size

    solid_color_ranges = {
        "RED":    [(np.array([0, 200, 200]),   np.array([5, 255, 255])),
                   (np.array([175, 200, 200]), np.array([180, 255, 255]))],
        "GREEN":  [(np.array([55, 200, 200]),  np.array([65, 255, 255]))],
        "BLUE":   [(np.array([115, 200, 200]), np.array([125, 255, 255]))],
        "ORANGE": [(np.array([14, 200, 200]),  np.array([24, 255, 255]))],
    }

    color_to_label = {
        "RED": "DANGER", "GREEN": "SAFE",
        "BLUE": "REFUEL", "ORANGE": "SLOW",
    }

    cyan_lo   = np.array([85, 150, 150])
    cyan_hi   = np.array([100, 255, 255])
    yellow_lo = np.array([25, 150, 150])
    yellow_hi = np.array([35, 255, 255])

    for row in range(arena_size):
        for col in range(arena_size):
            y0 = int(row * cell_h)
            y1 = int((row + 1) * cell_h)
            x0 = int(col * cell_w)
            x1 = int((col + 1) * cell_w)

            cell = hsv[y0:y1, x0:x1]
            if cell.size == 0:
                continue

            letter = chr(ord('A') + col)
            number = arena_size - row
            coord = f"{letter}{number}"

            cyan_count   = np.sum(cv2.inRange(cell, cyan_lo, cyan_hi) > 0)
            yellow_count = np.sum(cv2.inRange(cell, yellow_lo, yellow_hi) > 0)

            if cyan_count > 100:
                result["goal"] = coord
                continue
            if yellow_count > 100:
                result["start"] = coord
                continue

            ch, cw = cell.shape[:2]
            mh, mw = int(ch * 0.25), int(cw * 0.25)
            patch = cell[mh:ch-mh, mw:cw-mw]
            total = patch.shape[0] * patch.shape[1]

            scores = {}
            for color_name, ranges in solid_color_ranges.items():
                mask = None
                for (lo, hi) in ranges:
                    m = cv2.inRange(patch, lo, hi)
                    mask = m if mask is None else cv2.bitwise_or(mask, m)
                scores[color_name] = np.sum(mask > 0)

            best = max(scores, key=scores.get)
            if total > 0 and scores[best] / total > 0.40:
                result["special_cells"][coord] = color_to_label[best]

    sorted_cells = dict(
        sorted(
            result["special_cells"].items(), key=lambda item: (item[0][0], int(item[0][1:])))
            )

    result["special_cells"] = sorted_cells

    return result