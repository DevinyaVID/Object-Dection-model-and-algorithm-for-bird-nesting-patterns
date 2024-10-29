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
    if 'files' not in request.files:
        return jsonify({'error': 'No files part in the request'}), 400

    files = request.files.getlist('files')
    # Log the list of uploaded files
    print(f"Uploaded files: {[file.filename for file in files]}")
    
    if len(files) == 0:
        return jsonify({'error': 'No selected files'}), 400

    if all(file.filename.endswith('.json') for file in files):
        # Save the uploaded files to a temporary directory
        temp_directory = 'temp_uploads'
        os.makedirs(temp_directory, exist_ok=True)
        file_paths = []
        for file in files:
            file_path = os.path.join(temp_directory, file.filename)
            file.save(file_path)
            file_paths.append(file_path)

        # Load the counts from the first and last uploaded files
        start_video = files[0].filename
        end_video = files[-1].filename
        egg_counts, hatchling_counts = load_counts(temp_directory, start_video=start_video, end_video=end_video)

        # Calculate the hatching success rate using the maximum counts
        egg_max, hatchling_max, success_rate = calculate_hatching_success_rate(egg_counts, hatchling_counts)

        # Clean up the temporary files
        for file_path in file_paths:
            os.remove(file_path)

        # Return the results
        return jsonify({
            'egg_max': egg_max,
            'hatchling_max': hatchling_max,
            'success_rate': success_rate
        })
    else:
        return jsonify({'error': 'Invalid file format, only JSON files are allowed'}), 400
 
 
if __name__ == '__main__':
    app.run(debug=True, port=4000)