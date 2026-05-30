
import cv2
import numpy as np
import glob
 
def localize_bot():
    """
    Task 2A: Camera Calibration and ArUco Pose Estimation
    """
    # Initialize the output dictionary with exact keys required by the evaluator
    result = {
        "camera_matrix_trace": 0.0,
        "markers": {}
    }
 
    # ==========================================
    # STEP 1: Camera Calibration
    # ==========================================
    # Checkerboard inner corners: 9 wide x 6 tall, each square = 2.5 cm
    GRID_W, GRID_H = 9, 6
    SQUARE_SIZE = 2.5  # cm
 
    # Real-world 3D object points for one checkerboard image
    # Shape: (9*6, 3) — z=0 since the board is flat
    objp = np.zeros((GRID_W * GRID_H, 3), np.float32)
    objp[:, :2] = np.mgrid[0:GRID_W, 0:GRID_H].T.reshape(-1, 2)
    objp *= SQUARE_SIZE
 
    obj_points = []  # 3D points in real world space
    img_points = []  # 2D points in image plane
    img_size = None
 
    # Read all calibration images
    images = glob.glob("calibration_images/*.jpg") + \
             glob.glob("calibration_images/*.png") + \
             glob.glob("calibration_images/*.jpeg")
 
    for fname in images:
        img = cv2.imread(fname)
        if img is None:
            continue
 
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_size = gray.shape[::-1]  # (width, height)
 
        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (GRID_W, GRID_H), None)
 
        if ret:
            # Refine corner locations to sub-pixel accuracy
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
 
            obj_points.append(objp)
            img_points.append(corners_refined)
 
    # Calibrate the camera
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        obj_points, img_points, img_size, None, None
    )
 
    # Trace = sum of main diagonal elements of the 3x3 camera matrix
    trace_value = float(np.trace(mtx))
    result["camera_matrix_trace"] = round(trace_value, 2)
 
 
    # ==========================================
    # STEP 2: Image Undistortion
    # ==========================================
    # Try both possible filenames/paths
    test_img = None
    for path in ["test_images/test_arena.png",
                 "test_images/test_arena.jpg",
                 "test_arena.png",
                 "test_arena.jpg"]:
        test_img = cv2.imread(path)
        if test_img is not None:
            break
 
    if test_img is None:
        print("ERROR: Could not load test image. Check path.")
        return result
 
    h, w = test_img.shape[:2]
 
    # Compute optimal new camera matrix to retain full image after undistortion
    new_mtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
 
    # Undistort
    undistorted = cv2.undistort(test_img, mtx, dist, None, new_mtx)
 
    # Crop to valid ROI (optional but cleaner)
    x, y, rw, rh = roi
    if rw > 0 and rh > 0:
        undistorted = undistorted[y:y+rh, x:x+rw]
        # Adjust camera matrix for the crop offset
        new_mtx[0, 2] -= x
        new_mtx[1, 2] -= y
 
 
    # ==========================================
    # STEP 3: ArUco Detection & Pose Estimation
    # ==========================================
    MARKER_SIZE = 5.0  # cm
 
    # Real-world 3D corners of a single ArUco marker (centred at origin, z=0)
    half = MARKER_SIZE / 2.0
    marker_obj_points = np.array([
        [-half,  half, 0],
        [ half,  half, 0],
        [ half, -half, 0],
        [-half, -half, 0]
    ], dtype=np.float32)
 
    # Load the 4x4_50 ArUco dictionary
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    aruco_params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)
 
    # Detect markers in the undistorted image
    corners, ids, rejected = detector.detectMarkers(undistorted)
 
    if ids is not None:
        for i, marker_id in enumerate(ids.flatten()):
            # corners[i] shape: (1, 4, 2) — the 4 corner pixel coordinates
            img_corners = corners[i].reshape(4, 2).astype(np.float32)
 
            # Estimate pose using solvePnP
            success, rvec, tvec = cv2.solvePnP(
                marker_obj_points,
                img_corners,
                new_mtx,
                None  # distortion already removed
            )
 
            if success:
                # tvec is in cm (same units as marker_obj_points)
                x_offset = round(float(tvec[0][0]), 1)
                distance_z = round(float(tvec[2][0]), 1)
 
                key = f"id_{marker_id}"
                result["markers"][key] = {
                    "distance_z": distance_z,
                    "x_offset": x_offset
                }
 
    # ==========================================
    # SORT MARKERS BY ARUCO ID (descending)
    # ==========================================
    result["markers"] = dict(
        sorted(
            result["markers"].items(),
            key=lambda item: int(item[0].split("_")[1]),
            reverse=True
        )
    )
 
    # ==========================================
    # RETURN FINAL OUTPUT
    # ==========================================
    return result
 
 
if __name__ == "__main__":
    output = localize_bot()
    print("Task 2A Output:")
    print(output)
 