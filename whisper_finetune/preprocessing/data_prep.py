import argparse
import json
import os
from datasets import Dataset, Audio, Value

# Argument parsing
parser = argparse.ArgumentParser(description='Preliminary data preparation script before Whisper Fine-tuning.')
parser.add_argument('--source_data_file', type=str, required=True, help='Path to the JSON file containing the audio paths and sentences.')
parser.add_argument('--output_data_dir', type=str, required=False, default='op_data_dir', help='Output data directory path.')
parser.add_argument('--base_audio_dir', type=str, required=True, help='Base directory for audio files.')

args = parser.parse_args()

# Read the JSON file
with open(args.source_data_file, 'r') as f:
    data = json.load(f)

# Extract and update audio paths and sentences
audio_paths = [os.path.join(args.base_audio_dir, entry['audio_path']) for entry in data]
sentences = [entry['prompt'] for entry in data]

# Create a dataset
audio_dataset = Dataset.from_dict({"audio": audio_paths, "sentence": sentences})

# Cast columns to appropriate types
audio_dataset = audio_dataset.cast_column("audio", Audio(sampling_rate=16_000))
audio_dataset = audio_dataset.cast_column("sentence", Value("string"))

# Save the dataset to disk
audio_dataset.save_to_disk(args.output_data_dir)
print('Data preparation done')
