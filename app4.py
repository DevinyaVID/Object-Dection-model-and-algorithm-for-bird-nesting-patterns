from flask import Flask, request, jsonify
import os
import json

app = Flask(__name__)

# Function to load counts from stored files within a selected range of videos
def load_counts(directory, file_list):
    egg_counts = []
    hatchling_counts = []
    
    for file in file_list:
        if file.endswith('.json'):
            file_path = os.path.join(directory, file)
            with open(file_path, 'r') as f:
                data = json.load(f)  # Assuming the file contains a single dictionary
                egg_counts.append(data.get('consistent_egg_count', 0))
                hatchling_counts.append(data.get('hatchling_max', 0))

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
def upload_files():
    if 'files' not in request.files:
        return jsonify({'error': 'No files part in the request'}), 400

    files = request.files.getlist('files')
    if not files:
        return jsonify({'error': 'No selected files'}), 400

    temp_directory = 'temp_uploads'
    os.makedirs(temp_directory, exist_ok=True)

    # Save files to the temporary directory and prepare list of filenames
    file_list = []
    for file in files:
        if file and file.filename.endswith('.json'):
            file_path = os.path.join(temp_directory, file.filename)
            file.save(file_path)
            file_list.append(file.filename)

    # Load the counts from all uploaded files
    egg_counts, hatchling_counts = load_counts(temp_directory, file_list)

    # Calculate the hatching success rate using the maximum counts
    egg_max, hatchling_max, success_rate = calculate_hatching_success_rate(egg_counts, hatchling_counts)

    # Clean up the temporary files
    for file_name in file_list:
        os.remove(os.path.join(temp_directory, file_name))

    # Return the results
    return jsonify({
        'egg_max': egg_max,
        'hatchling_max': hatchling_max,
        'success_rate': success_rate
    })

if __name__ == '__main__':
    app.run(debug=True, port=4000)
