from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import io
import base64
import random
import traceback

# --- Computer Vision Imports ---
from skimage import io as skimage_io
from skimage.color import rgb2gray, rgb2hsv
from skimage.feature import canny
from skimage.transform import probabilistic_hough_line
from skimage.measure import find_contours
from skimage.morphology import skeletonize
from scipy.cluster.hierarchy import fclusterdata
from scipy.spatial.distance import cdist

app = Flask(__name__)
# Apply a more robust CORS configuration
CORS(app, resources={r"/grade": {"origins": "*"}})

def isolate_writing(image_array, mode='ink'):
    """
    Smarter isolation using HSV color space.
    'gray' for guide letters, 'ink' for student writing.
    """
    # Ensure image is RGB before converting to HSV
    if len(image_array.shape) < 3 or image_array.shape[2] < 3:
        grayscale_image = image_array if len(image_array.shape) == 2 else rgb2gray(image_array)
        thresh = 0.8 # Assume a light background
        if mode == 'ink': # Assume ink is darker than the threshold
            binary_image = grayscale_image < thresh
        else: # Assume gray is lighter than ink but darker than background
             binary_image = (grayscale_image < 0.9) & (grayscale_image > 0.4)
        return (~binary_image * 255).astype(np.uint8) # Return as a mask (0=writing, 255=bg)


    hsv_image = rgb2hsv(image_array)
    mask = np.ones_like(hsv_image[:, :, 0], dtype=np.uint8) * 255
    
    gray_saturation_max = 0.2
    gray_value_min = 0.3
    gray_value_max = 0.85
    
    ink_value_max = 0.45
    ink_saturation_min = 0.1

    for r in range(hsv_image.shape[0]):
        for c in range(hsv_image.shape[1]):
            s = hsv_image[r, c, 1]
            v = hsv_image[r, c, 2]
            
            is_gray = (s < gray_saturation_max and gray_value_min < v < gray_value_max)
            is_ink = (v < ink_value_max and s > ink_saturation_min)

            if mode == 'gray' and is_gray:
                mask[r, c] = 0
            
            if mode == 'ink' and is_ink:
                mask[r, c] = 0

    return mask

def analyze_image_properties(image_stream, mode):
    """
    Analyzes image for cursive metrics. Now more robust with checks for empty data.
    """
    image_bytes = image_stream.read()
    
    # --- ROBUSTNESS FIX: Convert ALL images to a standard RGB format first ---
    pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image_array = np.array(pil_image)
    
    isolated_mask = isolate_writing(image_array, mode=mode)
    binary_image = isolated_mask < 255

    # --- SKELETON FOR SHAPE ANALYSIS ---
    skeleton = skeletonize(binary_image)
    skeleton_points = np.argwhere(skeleton)
    if skeleton_points.ndim == 2 and skeleton_points.shape[1] >= 2:
        skeleton_points = skeleton_points[:, ::-1]
    else:
        skeleton_points = np.array([])

    average_slant = 0
    if np.any(binary_image):
        edges = canny(isolated_mask, sigma=2.0)
        lines = probabilistic_hough_line(edges, threshold=10, line_length=20, line_gap=3)
        
        angles = []
        if lines is not None:
            angles = [np.rad2deg(np.arctan2(p1[1] - p0[1], p1[0] - p0[0])) 
                      for p0, p1 in lines if p1[0] != p0[0] 
                      and 20 < abs(np.rad2deg(np.arctan2(p1[1] - p0[1], p1[0] - p0[0]))) < 85]
        if angles:
            average_slant = np.median(angles)

    largest_gap, open_loop_location = 0, None
    contours = find_contours(binary_image, 0.5)
    if contours:
        for contour in contours:
            if len(contour) > 50:
                start_point, end_point = contour[0], contour[-1]
                gap = np.linalg.norm(start_point - end_point)
                if gap > largest_gap:
                    largest_gap = gap
                    open_loop_location = ((start_point[1] + end_point[1]) / 2, (start_point[0] + end_point[0]) / 2)
    loop_closure_score = max(0.0, 1.0 - (largest_gap / 15.0))

    y_coords, x_coords = np.where(binary_image)
    top_line_y, bottom_line_y = 0, 0
    baseline_adherence = 1.0 
    all_ink_points = []
    
    if len(y_coords) > 10:
        top_line_y = np.percentile(y_coords, 5)
        bottom_line_y = np.percentile(y_coords, 95)
        all_ink_points = list(zip(x_coords, y_coords))
        
        if (bottom_line_y - top_line_y) > 10:
             y_values_bottom = [p[1] for p in all_ink_points if p[1] > (top_line_y + (bottom_line_y - top_line_y) * 0.8)]
             if y_values_bottom:
                 baseline_std = np.std(y_values_bottom)
                 baseline_adherence = max(0, 1.0 - (baseline_std / 5.0))

    return {
        'average_slant': average_slant,
        'loop_closure': loop_closure_score,
        'has_open_loop': largest_gap > 5,
        'open_loop_location': open_loop_location,
        'baseline_adherence': baseline_adherence,
        'top_line_y': top_line_y,
        'bottom_line_y': bottom_line_y,
        'all_ink_points': all_ink_points,
        'skeleton_points': skeleton_points,
        'original_image_bytes': image_bytes
    }

def draw_mistakes_on_image(student_metrics, reference_metrics):
    """Draws specific, clustered annotations for mistakes."""
    image_bytes = student_metrics['original_image_bytes']
    image = Image.open(io.BytesIO(image_bytes)).convert("RGBA")
    draw = ImageDraw.Draw(image)
    mistake_found = False

    ideal_top = reference_metrics.get('top_line_y', 0)
    ideal_bottom = reference_metrics.get('bottom_line_y', 0)
    
    if ideal_top > 0 and ideal_bottom > 0 and (ideal_bottom > ideal_top):
        # Draw green guide lines from the reference
        draw.line([(0, ideal_top), (image.width, ideal_top)], fill=(0, 200, 0, 100), width=2)
        draw.line([(0, ideal_bottom), (image.width, ideal_bottom)], fill=(0, 200, 0, 100), width=2)

        mistake_points = [p for p in student_metrics.get('all_ink_points', []) if p[1] < ideal_top or p[1] > ideal_bottom]

        if len(mistake_points) > 5:
            mistake_found = True
            # Cluster points to draw fewer, more meaningful circles
            clusters = fclusterdata(mistake_points, 30.0, criterion='distance')
            for cluster_id in np.unique(clusters):
                points_in_cluster = [mistake_points[i] for i, c_id in enumerate(clusters) if c_id == cluster_id]
                if not points_in_cluster: continue
                
                center_x = np.mean([p[0] for p in points_in_cluster])
                center_y = np.mean([p[1] for p in points_in_cluster])
                
                draw.ellipse([(center_x - 10, center_y - 10), (center_x + 10, center_y + 10)], outline=(255, 0, 0, 220), width=3)

    student_skeleton = student_metrics.get('skeleton_points', np.array([]))
    ref_skeleton = reference_metrics.get('skeleton_points', np.array([]))
    
    if ref_skeleton.size > 10 and student_skeleton.size > 10:
        distances = cdist(student_skeleton, ref_skeleton).min(axis=1)
        mistake_indices = np.where(distances > 15)[0] # Increased threshold for more significant errors
        shape_mistake_points = student_skeleton[mistake_indices]

        if len(shape_mistake_points) > 5:
            mistake_found = True
            clusters = fclusterdata(shape_mistake_points, 35.0, criterion='distance')
            for cluster_id in np.unique(clusters):
                points_in_cluster = [shape_mistake_points[i] for i, c_id in enumerate(clusters) if c_id == cluster_id]
                if not points_in_cluster: continue
                center_x = np.mean([p[0] for p in points_in_cluster])
                center_y = np.mean([p[1] for p in points_in_cluster])
                draw.ellipse([(center_x - 12, center_y - 12), (center_x + 12, center_y + 12)], outline=(128, 0, 128, 220), width=3)

    if student_metrics.get('has_open_loop') and student_metrics.get('open_loop_location'):
        mistake_found = True
        x, y = student_metrics['open_loop_location']
        draw.ellipse([(x - 12, y - 12), (x + 12, y + 12)], outline=(255, 165, 0, 220), width=4)

    if not mistake_found:
        font = ImageFont.load_default()
        draw.text((10, 10), "Good formation!", fill=(0, 128, 0, 255), font=font)

    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return img_str


@app.route('/grade', methods=['POST'])
def grade_cursive():
    if 'reference' not in request.files or 'student' not in request.files:
        return jsonify({'error': 'Missing image file'}), 400

    student_file = request.files['student']
    reference_file = request.files['reference']
    
    student_stream = io.BytesIO(student_file.read())
    reference_stream = io.BytesIO(reference_file.read())

    try:
        reference_metrics = analyze_image_properties(reference_stream, mode='gray')
        student_metrics = analyze_image_properties(student_stream, mode='ink')

        slant_diff = abs(student_metrics['average_slant'] - reference_metrics['average_slant']) if reference_metrics.get('average_slant') != 0 else 0
        slant_score = max(0, 40 - (slant_diff * 2))
        loop_score = student_metrics.get('loop_closure', 0) * 30
        
        student_skeleton = student_metrics.get('skeleton_points', np.array([]))
        ref_skeleton = reference_metrics.get('skeleton_points', np.array([]))
        shape_score = 0
        if ref_skeleton.size > 10 and student_skeleton.size > 10:
            distances = cdist(student_skeleton, ref_skeleton).min(axis=1)
            avg_distance = np.mean(distances)
            shape_score = max(0, 30 * (1 - (avg_distance / 12.0)))

        score = int(slant_score + loop_score + shape_score)
        
        feedback = []
        if reference_metrics.get('average_slant') == 0:
             feedback.append({'text': 'Could not detect guide letters in the reference image.', 'type': 'bad'})
        elif slant_diff < 5:
            feedback.append({'text': 'Excellent slant consistency!', 'type': 'good'})
        else:
            feedback.append({'text': 'The slant is different from the reference.', 'type': 'bad'})

        if student_metrics.get('has_open_loop'):
            feedback.append({'text': 'An open loop was detected (circled in orange).', 'type': 'bad'})
        
        if shape_score < 20:
            feedback.append({'text': 'Shape or baseline errors detected (circled).', 'type': 'bad'})
        else:
            feedback.append({'text': 'Excellent overall shape and baseline!', 'type': 'good'})

        if not feedback:
            feedback.append({'text': 'Great work!', 'type': 'good'})

        marked_up_image_b64 = draw_mistakes_on_image(student_metrics, reference_metrics)
        
        return jsonify({
            'score': min(100, max(10, score)),
            'feedback': feedback,
            'marked_up_image': marked_up_image_b64
        })

    except Exception as e:
        print(f"An error occurred during analysis: {e}")
        traceback.print_exc()
        return jsonify({'error': "Analysis failed. Ensure the images are clear, well-lit, and the student's ink is a dark color (black or blue)."}), 500


if __name__ == '__main__':
    # Makes the server visible on your local network and runs on port 5001
    app.run(host='0.0.0.0', port=5001, debug=True)

