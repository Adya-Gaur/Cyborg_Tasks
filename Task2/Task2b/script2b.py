import cv2
import numpy as np
import os
 
def map_arena():
    """
    Task 2B: Perspective Transformation and Coordinate Mapping
    """
    result = {
        "corner_points_detected": [],
        "robot_pixel_coord": [],
        "robot_real_world_coord": []
    }
 
    # ==========================================
    # STEP 1: Corner Detection (Color Tracking)
    # ==========================================
 
    # Use absolute path relative to this script's location
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    image_path = os.path.join(BASE_DIR, "test_images", "angled_arena.png")
    image = cv2.imread(image_path)
 
    if image is None:
        print(f"Error: Could not load image from {image_path}")
        return result
 
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
 
    # HSV ranges for each corner color
    color_masks = {
        "red":    [
            cv2.inRange(hsv, np.array([0,   120,  70]),  np.array([10,  255, 255])),
            cv2.inRange(hsv, np.array([170, 120,  70]),  np.array([180, 255, 255]))
        ],
        "green":  [cv2.inRange(hsv, np.array([40,  100, 70]),  np.array([80,  255, 255]))],
        "blue":   [cv2.inRange(hsv, np.array([100, 150, 70]),  np.array([130, 255, 255]))],
        "yellow": [cv2.inRange(hsv, np.array([20,  100, 100]), np.array([35,  255, 255]))]
    }
 
    def get_centroid(masks):
        """Combine masks, find largest contour, return centroid (cx, cy)."""
        combined = masks[0]
        for m in masks[1:]:
            combined = cv2.bitwise_or(combined, m)
 
        # Morphological cleanup to remove noise
        kernel = np.ones((5, 5), np.uint8)
        combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN,  kernel)
        combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
 
        contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
 
        largest = max(contours, key=cv2.contourArea)
        M = cv2.moments(largest)
        if M["m00"] == 0:
            return None
 
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        return [cx, cy]
 
    # Detect centroids in order: TL(Red), TR(Green), BR(Blue), BL(Yellow)
    red_center    = get_centroid(color_masks["red"])
    green_center  = get_centroid(color_masks["green"])
    blue_center   = get_centroid(color_masks["blue"])
    yellow_center = get_centroid(color_masks["yellow"])
 
    result["corner_points_detected"] = [
        red_center,    # Top-Left
        green_center,  # Top-Right
        blue_center,   # Bottom-Right
        yellow_center  # Bottom-Left
    ]
 
    # ==========================================
    # STEP 2: Perspective Transformation
    # ==========================================
 
    src_pts = np.array(result["corner_points_detected"], dtype=np.float32)
 
    dst_pts = np.array([
        [0,   0  ],   # Top-Left
        [499, 0  ],   # Top-Right
        [499, 499],   # Bottom-Right
        [0,   499]    # Bottom-Left
    ], dtype=np.float32)
 
    matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped  = cv2.warpPerspective(image, matrix, (500, 500))
 
    # ==========================================
    # STEP 3: Robot Detection on Warped Arena
    # ==========================================
 
    aruco_dict   = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    aruco_params = cv2.aruco.DetectorParameters()
    detector     = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)
 
    corners, ids, _ = detector.detectMarkers(warped)
 
    if ids is not None:
        for i, marker_id in enumerate(ids.flatten()):
            if marker_id == 1:
                c = corners[i][0]   # shape (4, 2)
                cx = int(np.mean(c[:, 0]))
                cy = int(np.mean(c[:, 1]))
                result["robot_pixel_coord"] = [cx, cy]
                break
 
    # ==========================================
    # STEP 4: Real-World Coordinate Conversion
    # ==========================================
    # 500 pixels  →  200 cm   =>   1 pixel = 200/500 = 0.4 cm
 
    if result["robot_pixel_coord"]:
        px, py = result["robot_pixel_coord"]
        x_cm = round(px * (200 / 500), 2)
        y_cm = round(py * (200 / 500), 2)
        result["robot_real_world_coord"] = [x_cm, y_cm]
 
    return result
 
 
if __name__ == "__main__":
    output = map_arena()
    print("Task 2B Output:")
    print(output)
 