import os
import re
import math
import numpy as np
import json
from datetime import datetime
from flask import Flask, request, jsonify
from ultralytics import YOLO
import cv2
 
# Define Flask app
app = Flask(__name__)
 
# Directories for saving videos, frames, and labels
UPLOAD_FOLDER = 'uploaded_videos'
FRAMES_FOLDER = 'saved_frames'
LABELS_FOLDER = 'saved_labels'
RESULTS_FOLDER = 'results'
 
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(FRAMES_FOLDER, exist_ok=True)
os.makedirs(LABELS_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)
 
# Load YOLOv8 model
model = YOLO("best.pt")
 
# Function to load detection results from YOLOv8 text files
def load_detections(file_path):
    detections = []
    with open(file_path, 'r') as file:
        lines = file.read().strip().splitlines()
        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 6:
                class_id = int(parts[0])
                conf = float(parts[1])
                xmin = float(parts[2])
                ymin = float(parts[3])
                xmax = float(parts[4])
                ymax = float(parts[5])
                x_center = (xmin + xmax) / 2
                y_center = (ymin + ymax) / 2
                detections.append({
                    'class_id': class_id,
                    'confidence': conf,
                    'xmin': xmin,
                    'ymin': ymin,
                    'xmax': xmax,
                    'ymax': ymax,
                    'x_center': x_center,
                    'y_center': y_center
                })
    return detections
 
# Function to calculate the distance between two points
def calculate_distance(point1, point2):
    return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)
 
# Function to calculate the center of the egg clutch
def calculate_egg_clutch_center(egg_positions):
    if not egg_positions:
        return None
    avg_x = sum(egg['x_center'] for egg in egg_positions) / len(egg_positions)
    avg_y = sum(egg['y_center'] for egg in egg_positions) / len(egg_positions)
    return (avg_x, avg_y)
 
# Function to calculate the bounding box for the entire egg clutch
def calculate_egg_clutch_bounding_box(egg_positions):
    if not egg_positions:
        return None
    min_x = min(egg['xmin'] for egg in egg_positions)
    min_y = min(egg['ymin'] for egg in egg_positions)
    max_x = max(egg['xmax'] for egg in egg_positions)
    max_y = max(egg['ymax'] for egg in egg_positions)
    return min_x, min_y, max_x, max_y
 
# Function to calculate the incubation radius based on the egg clutch bounding box
def calculate_incubation_radius(egg_positions):
    if not egg_positions:
        return 0
    min_x, min_y, max_x, max_y = calculate_egg_clutch_bounding_box(egg_positions)
    clutch_width = max_x - min_x
    clutch_height = max_y - min_y
 
    # Calculate diagonal of the bounding box to use as incubation radius
    return math.sqrt(clutch_width ** 2 + clutch_height ** 2) / 2
 
def classify_behavior(detections, class_names, radius_buffer=0.7, previous_behavior=None, min_frames_for_absent=5, consecutive_absent_frames=0):
    has_egg = any(det['class_id'] == class_names.index('egg') for det in detections)
    birds = [det for det in detections if det['class_id'] == class_names.index('bird')]
 
    # If no eggs and no birds, check how long this has been happening
    if not has_egg and not birds:
        consecutive_absent_frames += 1
        if consecutive_absent_frames >= min_frames_for_absent:
            return 'Absent'
        else:
            return previous_behavior  # Retain the last valid behavior longer
 
    # Reset consecutive_absent_frames when behavior changes
    consecutive_absent_frames = 0
 
    # If eggs are present
    if has_egg:
        egg_positions = [det for det in detections if det['class_id'] == class_names.index('egg')]
        egg_clutch_center = calculate_egg_clutch_center(egg_positions)
        incubation_radius = calculate_incubation_radius(egg_positions) * radius_buffer
 
        # If both eggs and birds are present
        if birds:
            for bird in birds:
                distance_to_clutch = calculate_distance(egg_clutch_center, (bird['x_center'], bird['y_center']))
                if distance_to_clutch <= incubation_radius:
                    return 'Incubation'
            return 'Attention'
        else:
            return 'Unattended'
 
    return previous_behavior if previous_behavior else 'Absent'
 
# Function to remove outliers using the IQR method
def remove_outliers(data):
    if len(data) < 4:
        return data# Not enough data to have meaningful outliers
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    return [x for x in data if lower_bound <= x <= upper_bound]
 
# Function to smooth the egg counts
def smooth_egg_counts(egg_counts, window_size=5):
    smoothed_counts = []
    for i in range(len(egg_counts)):
        window = egg_counts[max(i - window_size // 2, 0):min(i + window_size // 2 + 1, len(egg_counts))]
        smoothed_counts.append(round(np.mean(window)))
    return smoothed_counts
 
# Function to get the most consistent egg count
def get_consistent_egg_count(egg_counts):
    count_threshold = len(egg_counts) * 0.1
    count_frequency = {}
    for count in egg_counts:
        if count not in count_frequency:
            count_frequency[count] = 0
        count_frequency[count] += 1
    consistent_count = max(count_frequency, key=lambda x: count_frequency[x] if count_frequency[x] >= count_threshold else 0)
    return consistent_count
 
def calculate_reference_incubation_radius(detections_dir, class_names, num_initial_frames=5):
    total_incubation_radius = 0
    frame_count = 0
 
    for frame_index, file in enumerate(sorted(os.listdir(detections_dir), key=lambda x: int(re.search(r'\d+', x).group()))):
        if file.endswith('.txt') and frame_index < num_initial_frames:
            file_path = os.path.join(detections_dir, file)
            detections = load_detections(file_path)
            egg_positions = [det for det in detections if det['class_id'] == class_names.index('egg')]
 
            # Calculate incubation radius if eggs are present
            if egg_positions:
                incubation_radius = calculate_incubation_radius(egg_positions)
                total_incubation_radius += incubation_radius
                frame_count += 1
 
    # Calculate the average incubation radius from the initial frames
    if frame_count > 0:
        return total_incubation_radius / frame_count
    else:
        return None # No eggs detected in initial frames
 
# Sort files based on numeric value in the filename to ensure correct sequence
def numerical_sort(value):
    numbers = re.findall(r'\d+', value)
    return int(numbers[0]) if numbers else 0
 
# Function to process all frames and classify behaviors with timestamps
def process_frames(detections_dir, frame_rate, class_names):
    # First, calculate the reference incubation radius using the initial frames
    reference_incubation_radius = calculate_reference_incubation_radius(detections_dir, class_names)
 
    results = {}
    durations = {'Incubation': 0, 'Attention': 0, 'Unattended': 0, 'Absent': 0}
    frame_duration = 1 / frame_rate
    behavior_timestamps = {'Incubation': [], 'Attention': [], 'Unattended': [], 'Absent': []}
 
    egg_counts = []
    hatchling_counts = []
 
    previous_behavior = None
    start_frame = 0
 
    # Sort files numerically to ensure correct processing order
    sorted_files = sorted(os.listdir(detections_dir), key=numerical_sort)
 
    for frame_index, file in enumerate(sorted_files):
        if file.endswith('.txt'):
            file_path = os.path.join(detections_dir, file)
            detections = load_detections(file_path)
            behavior = classify_behavior(detections, class_names, reference_incubation_radius, previous_behavior)
 
            # Record egg and hatchling counts
            egg_count = sum(1 for det in detections if det['class_id'] == class_names.index('egg'))
            hatchling_count = sum(1 for det in detections if det['class_id'] == class_names.index('babybird'))
 
            egg_counts.append(egg_count)
            hatchling_counts.append(hatchling_count)
 
            # If the behavior changes, record the previous behavior's start and end times
            if behavior != previous_behavior and previous_behavior is not None:
                end_frame = frame_index - 1
                start_time = start_frame * frame_duration
                end_time = end_frame * frame_duration
                behavior_timestamps[previous_behavior].append(f"{start_time:.2f}-{end_time:.2f}")
                durations[previous_behavior] += (end_frame - start_frame + 1) * frame_duration
                start_frame = frame_index
 
            previous_behavior = behavior
            results[file] = behavior
 
    # Record the last behavior segment
    if previous_behavior is not None:
        end_frame = frame_index
        start_time = start_frame * frame_duration
        end_time = end_frame * frame_duration
        behavior_timestamps[previous_behavior].append(f"{start_time:.2f}-{end_time:.2f}")
        durations[previous_behavior] += (end_frame - start_frame + 1) * frame_duration
 
    # Apply smoothing to the egg counts
    egg_counts = smooth_egg_counts(egg_counts)
 
    # Remove outliers and calculate the most consistent egg count
    egg_counts = remove_outliers(egg_counts)
    consistent_egg_count = get_consistent_egg_count(egg_counts)
 
    # Get the maximum hatchling count instead of removing outliers and calculating the median
    hatchling_max = max(hatchling_counts) if hatchling_counts else 0
 
    return results, durations, behavior_timestamps, consistent_egg_count, hatchling_max
 
# Function to find the next available results file name
def get_next_results_filename(directory, base_name="results", extension=".json"):
    # Find all files in the directory that match the naming pattern
    pattern = re.compile(rf"{base_name}(\d+){extension}$")
    existing_files = [f for f in os.listdir(directory) if pattern.match(f)]
 
    # Extract the numbers from existing files and determine the next available number
    if existing_files:
        numbers = [int(pattern.match(f).group(1)) for f in existing_files]
        next_number = max(numbers) + 1
    else:
        next_number = 1
 
    # Return the next file name
    return os.path.join(directory, f"{base_name}{next_number}{extension}")
 
# Function to save results to JSON
def save_to_json(results, behavior_durations, behavior_timestamps, egg_median, hatchling_max, total_duration, percentages, output_file):
    data = {
        "results": results,
        "behavior_durations": behavior_durations,
        "behavior_timestamps": behavior_timestamps,
        "consistent_egg_count": egg_median,
        "hatchling_max": hatchling_max,
        "total_duration": total_duration,
        "percentages": percentages,
        "timestamp": datetime.now().isoformat()
    }
    with open(output_file, 'w') as file:
        json.dump(data, file, indent=4)
 
@app.route('/process_video', methods=['POST'])
def process_video():
    if 'video' not in request.files:
        return jsonify({"error": "No video file uploaded"}), 400
    video = request.files['video']
    video_path = os.path.join(UPLOAD_FOLDER, video.filename)
    video.save(video_path)
 
    # Run YOLO prediction and save outputs directly with normalized coordinates
    results = model.predict(source=video_path, save_txt=True, save_conf=True)

    # Move YOLO's saved label files to LABELS_FOLDER
    output_dir = os.path.join(model.predictor.save_dir, "labels")
    for file_name in os.listdir(output_dir):
        os.rename(os.path.join(output_dir, file_name), os.path.join(LABELS_FOLDER, file_name))


    class_names = ['babybird', 'bird', 'egg']
    frame_rate = 25
 
    reference_incubation_radius = calculate_reference_incubation_radius(LABELS_FOLDER, class_names)
    results, behavior_durations, behavior_timestamps, consistent_egg_count, hatchling_max = process_frames(
        LABELS_FOLDER, frame_rate, class_names)
 
    total_duration = sum(behavior_durations.values())
    percentages = {behavior: round((duration / total_duration) * 100, 2) for behavior, duration in behavior_durations.items()} if total_duration > 0 else {behavior: 0 for behavior in behavior_durations}
   
    output_file = get_next_results_filename(RESULTS_FOLDER)
    save_to_json(results, behavior_durations, behavior_timestamps, consistent_egg_count, hatchling_max, total_duration, percentages, output_file)
   
    return jsonify({
        "message": "Video processed successfully",
        "results_file": output_file
    })
 
if __name__ == '__main__':
    app.run(debug=True)
 