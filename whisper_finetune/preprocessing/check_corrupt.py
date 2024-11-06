import soundfile as sf
import os

def check_wav_files(directory):
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.wav'):
                file_path = os.path.join(root, file)
                try:
                    with sf.SoundFile(file_path) as f:
                        # If the file is valid, do nothing
                        pass
                except Exception as e:
                    print(f"Error with file {file_path}: {e}")
                    try:
                        os.remove(file_path)
                        print(f"Deleted corrupted file: {file_path}")
                    except Exception as delete_error:
                        print(f"Error deleting file {file_path}: {delete_error}")

# Example usage
check_wav_files('/Users/elizzy/Desktop/Speechlink/data/audio_data')

import json
import os
import soundfile as sf

def check_and_remove_invalid_entries(json_file_path, base_audio_dir):
    # Read the JSON file
    with open(json_file_path, 'r') as f:
        data = json.load(f)

    # Filter out entries with missing or corrupted audio files
    filtered_data = []
    for entry in data:
        audio_path = os.path.join(base_audio_dir, entry['audio_path'])
        if os.path.exists(audio_path):
            try:
                with sf.SoundFile(audio_path) as f:
                    # If the file is valid, add to filtered data
                    filtered_data.append(entry)
            except Exception as e:
                print(f"Error with file {audio_path}: {e}")
                print(f"File is corrupted and removed from JSON: {audio_path}")
        else:
            print(f"File not found and removed from JSON: {audio_path}")

    # Save the filtered data back to the JSON file
    with open(json_file_path, 'w') as f:
        json.dump(filtered_data, f, indent=4)

# Example usage
json_file_path = "/Users/elizzy/Desktop/Speechlink/data/val_data.json"
base_audio_dir = "/Users/elizzy/Desktop/Speechlink/data"
check_and_remove_invalid_entries(json_file_path, base_audio_dir)