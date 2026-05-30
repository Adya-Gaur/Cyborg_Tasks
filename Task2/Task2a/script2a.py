
import cv2
import numpy as np
import glob
 
def localize_bot():
    result = {"camera_matrix_trace": 0.0,
              "markers": {}}
 
    GRID_W, GRID_H = 9, 6
    SQUARE_SIZE = 2.5  
 
    objp = np.zeros((GRID_W * GRID_H, 3), np.float32)
    objp[:, :2] = np.mgrid[0:GRID_W, 0:GRID_H].T.reshape(-1, 2)
    objp *= SQUARE_SIZE
 
    obj_points = []  
    img_points = []  
    img_size = None
 
    images = glob.glob("calibration_images/*.jpg") + \
             glob.glob("calibration_images/*.png") + \
             glob.glob("calibration_images/*.jpeg")
 
    for fname in images:
        img = cv2.imread(fname)
        if img is None:
            continue
 
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_size = gray.shape[::-1]  #(W, H)
 
        ret, corners = cv2.findChessboardCorners(gray, (GRID_W, GRID_H), None)
 
        if ret:
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
 
            obj_points.append(objp)
            img_points.append(corners_refined)
 
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        obj_points, img_points, img_size, None, None
    )
 
    trace_value = float(np.trace(mtx))
    result["camera_matrix_trace"] = round(trace_value, 2)
 
    test_img = None
    for path in ["test_images/test_arena.png",
                 "test_arena.png"]:
        test_img = cv2.imread(path)
        if test_img is not None:
            break
 
    if test_img is None:
        print("ERROR: Could not load test image. Check path.")
        return result
 
    h, w = test_img.shape[:2]
 
    new_mtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
 
    undistorted = cv2.undistort(test_img, mtx, dist, None, new_mtx)
 
    x, y, rw, rh = roi
    if rw > 0 and rh > 0:
        undistorted = undistorted[y:y+rh, x:x+rw]
        
        new_mtx[0, 2] -= x
        new_mtx[1, 2] -= y
 
    MARKER_SIZE = 5.0 
 
    half = MARKER_SIZE / 2.0
    marker_obj_points = np.array([[-half,  half, 0],
                                  [ half,  half, 0],
                                  [ half, -half, 0],
                                  [-half, -half, 0]],
                                    dtype=np.float32)
 
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    aruco_params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

    corners, ids, rejected = detector.detectMarkers(undistorted)
 
    if ids is not None:
        for i, marker_id in enumerate(ids.flatten()):
            img_corners = corners[i].reshape(4, 2).astype(np.float32)
 
            success, rvec, tvec = cv2.solvePnP(marker_obj_points, img_corners, new_mtx, None)
 
            if success:
                x_offset = round(float(tvec[0][0]), 1)
                distance_z = round(float(tvec[2][0]), 1)
 
                key = f"id_{marker_id}"
                result["markers"][key] = {"distance_z": distance_z,
                                          "x_offset": x_offset}
 
    result["markers"] = dict(sorted(result["markers"].items(),key=lambda item: int(item[0].split("_")[1]),reverse=True))

    return result
 
if __name__ == "__main__":
    output = localize_bot()
    print("Task 2A Output:")
    print(output)
 