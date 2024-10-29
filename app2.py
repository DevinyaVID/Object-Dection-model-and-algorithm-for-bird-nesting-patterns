from flask import  Flask,request, jsonify
import os
import json
 
app = Flask(__name__)
 
 
# Function to load counts from stored files within a selected range of videos
def load_counts(directory, start_video=None, end_video=None):
    egg_counts = []
    hatchling_counts = []
    is_in_range = False
 
    for file in sorted(os.listdir(directory)):
        if file.endswith('.json'):
            if file == start_video:
                is_in_range = True
            if is_in_range:
                file_path = os.path.join(directory, file)
                with open(file_path, 'r') as f:
                    data = json.load(f)  # Assuming the file contains a single dictionary
                    egg_counts.append(data.get('consistent_egg_count', 0))
                    hatchling_counts.append(data.get('hatchling_max', 0))
            if file == end_video:
                break
 
    return egg_counts, hatchling_counts
 
# Function to calculate hatching success rate using maximum counts
def calculate_hatching_success_rate(egg_counts, hatchling_counts):
    if len(egg_counts) == 0:
        return 0, 0, 0  # Handle case where there are no egg counts
    if len(hatchling_counts) == 0:
        return 0, 0, 0  # Handle case where there are no hatchling counts
 
    # Calculate the maximum counts
    egg_max = max(egg_counts)
    hatchling_max = max(hatchling_counts)
 
    # Calculate the hatching success rate
    if egg_max > 0:
        success_rate = (hatchling_max / egg_max) * 100
    else:
        success_rate = 0  # To avoid division by zero
 
    return egg_max, hatchling_max, round(success_rate, 2)
 
 
@app.route('/success', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400
 
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
 
    if file and file.filename.endswith('.json'):
        # Save the uploaded file to a temporary directory
        temp_directory = 'temp_uploads'
        os.makedirs(temp_directory, exist_ok=True)
        file_path = os.path.join(temp_directory, file.filename)
        file.save(file_path)
 
        # Load the counts from the uploaded file
        egg_counts, hatchling_counts = load_counts(temp_directory, start_video=file.filename, end_video=file.filename)
 
        # Calculate the hatching success rate using the maximum counts
        egg_max, hatchling_max, success_rate = calculate_hatching_success_rate(egg_counts, hatchling_counts)
 
        # Clean up the temporary file
        os.remove(file_path)
 
        # Return the results
        return jsonify({
            'egg_max': egg_max,
            'hatchling_max': hatchling_max,
            'success_rate': success_rate
        })
 
    return jsonify({'error': 'Invalid file format, only JSON files are allowed'}), 400
 
 
if __name__ == '__main__':
    app.run(debug=True, port=4000)