import cv2
import numpy as np
from flask import Flask, request, render_template, send_file
import os

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
RESULT_FOLDER = 'results'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

def sift_match_and_homography(img1, img2):
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)
    
    flann = cv2.FlannBasedMatcher(dict(algorithm=6, table_number=6, key_size=12, multi_probe_level=1), dict(checks=50))
    matches = flann.knnMatch(des1, des2, k=2)
    
    good_matches = [m for m, n in matches if m.distance < 0.75 * n.distance]
    pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches])
    
    H, _ = cv2.findHomography(pts1, pts2, cv2.RANSAC, 5.0)
    return H

def stitch_images(left, middle, right):
    out_h, out_w = 1312, 1608
    padded_middle = cv2.copyMakeBorder(middle, 0, max(0, out_h - middle.shape[0]), 0, max(0, out_w - middle.shape[1]), cv2.BORDER_CONSTANT, value=0)
    
    H_lm = sift_match_and_homography(left, middle)
    warp_left = cv2.warpPerspective(left, H_lm, (padded_middle.shape[1], padded_middle.shape[0]))
    
    H_rm = sift_match_and_homography(right, padded_middle)
    warp_right = cv2.warpPerspective(right, H_rm, (padded_middle.shape[1], padded_middle.shape[0]))
    
    blended = cv2.addWeighted(padded_middle, 0.5, warp_left, 0.5, 0)
    blended = cv2.addWeighted(blended, 0.5, warp_right, 0.5, 0)
    
    kernel = np.ones((5,5), np.uint8)
    blended = cv2.morphologyEx(blended, cv2.MORPH_CLOSE, kernel)
    
    result_path = os.path.join(RESULT_FOLDER, 'stitched.jpg')
    cv2.imwrite(result_path, blended)
    return result_path

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        files = request.files.getlist('images')
        if len(files) != 3:
            return "Please upload exactly 3 images."
        
        file_paths = []
        for file in files:
            path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(path)
            file_paths.append(path)
        
        left = cv2.imread(file_paths[0])
        middle = cv2.imread(file_paths[1])
        right = cv2.imread(file_paths[2])
        
        stitched_path = stitch_images(left, middle, right)
        return send_file(stitched_path, mimetype='image/jpeg')
    
    return '''
    <form method="post" enctype="multipart/form-data">
        <input type="file" name="images" multiple required>
        <input type="submit" value="Upload and Stitch">
    </form>
    '''

if __name__ == '__main__':
    app.run(debug=True)