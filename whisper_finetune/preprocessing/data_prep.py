import argparse
import json
from datasets import Dataset, Audio, Value

# Argument parsing
parser = argparse.ArgumentParser(description='Preliminary data preparation script before Whisper Fine-tuning.')
parser.add_argument('--source_data_file', type=str, required=True, help='Path to the JSON file containing the audio paths.')
parser.add_argument('--output_data_dir', type=str, required=False, default='op_data_dir', help='Output data directory path.')

args = parser.parse_args()

# Read the JSON file
with open(args.source_data_file, 'r') as f:
    data = json.load(f)

# Extract audio paths
audio_paths = [entry['audio_path'] for entry in data]

# Create a dataset
audio_dataset = Dataset.from_dict({"audio": audio_paths})

# Cast columns to appropriate types
audio_dataset = audio_dataset.cast_column("audio", Audio(sampling_rate=16_000))

# Save the dataset to disk
audio_dataset.save_to_disk(args.output_data_dir)
print('Data preparation done')
