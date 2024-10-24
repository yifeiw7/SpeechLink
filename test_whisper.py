import os
import torch
import string
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from jiwer import wer

# Set device and data type depending on whether CUDA (GPU) is available
device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

# Model identifier for Whisper small model
model_id = "openai/whisper-medium"

# Load the model with specific options for memory and device efficiency
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
)
model.to(device)

# Load the processor for tokenizing and feature extraction
processor = AutoProcessor.from_pretrained(model_id)

# Create a pipeline for automatic speech recognition (ASR)
pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    torch_dtype=torch_dtype,
    device=device,
    return_timestamps=True  # Enable timestamps in the results
)

# Function to load transcription from a text file
def load_transcript(txt_file):
    with open(txt_file, 'r') as file:
        return file.read().strip()

# Function to clean text (lowercase and remove punctuation except spaces)
def clean_text(text):
    text = text.lower()
    return text.translate(str.maketrans('', '', string.punctuation.replace(" ", "")))

# Function to evaluate a pair of .wav and .txt files
def evaluate_wav_and_txt(wav_file, txt_file):
    try:
        # Load the ground truth transcription
        ground_truth = load_transcript(txt_file)
        ground_truth_cleaned = clean_text(ground_truth)

        # Transcribe the .wav file using Whisper
        result = pipe(wav_file)
        predicted_transcript = result["text"].strip()
        predicted_transcript_cleaned = clean_text(predicted_transcript)

        # Calculate Word Error Rate (WER)
        error_rate = wer(ground_truth_cleaned, predicted_transcript_cleaned)

        return {
            "wav_file": wav_file,
            "ground_truth": ground_truth,
            "predicted": predicted_transcript,
            "WER": error_rate
        }

    except ValueError as e:
        if "Malformed soundfile" in str(e):
            print(f"Skipping {wav_file} due to malformed audio file.")
            return None
        else:
            raise e

# Directory paths
# F01 S1
# wav_dir = "F/F01/Session1/wav_headMic"
# txt_dir = "F/F01/Session1/prompts"

# F03 S3
wav_dir = "F/F03/Session3/wav_headMic"
txt_dir = "F/F03/Session3/prompts"

# F04 S1
# wav_dir = "F/F04/Session1/wav_arrayMic"
# txt_dir = "F/F04/Session1/prompts"

# F04 S2
# wav_dir = "F/F04/Session2/wav_arrayMic"
# txt_dir = "F/F04/Session2/prompts"


# List of .wav files to evaluate
wav_files = [f for f in os.listdir(wav_dir) if f.endswith('.wav')]

# Initialize variables to track accuracy
total_wer = 0
valid_samples = 0
total_prompts_tested = 0

# Evaluate each file
for wav_file in wav_files:
    # Extract the base name (without extension) to find corresponding .txt file
    base_name = os.path.splitext(wav_file)[0]
    txt_file = os.path.join(txt_dir, f"{base_name}.txt")

    # Full path to the .wav file
    wav_file_path = os.path.join(wav_dir, wav_file)

    # Evaluate the pair of .wav and .txt
    result = evaluate_wav_and_txt(wav_file_path, txt_file)

    # Skip if the result is None (e.g., due to malformed audio)
    if result is None:
        continue

    # Check if the error rate is valid (less than or equal to 1)
    if result["WER"] <= 1:
        total_wer += result["WER"]
        valid_samples += 1
        total_prompts_tested += 1

    # Calculate total accuracy so far
    total_accuracy = (1 - total_wer / valid_samples) * 100 if valid_samples > 0 else 0

    # Print or log the result
    print(f"Evaluating {wav_file}:")
    print(f"Ground Truth: {result['ground_truth']}")
    print(f"Predicted: {result['predicted']}")
    print(f"Word Error Rate (WER): {result['WER']}")
    print(f"Total Accuracy so far: {total_accuracy:.2f}%")
    print(f"Total Prompts Included in Accuracy Calculation: {total_prompts_tested}\n")

# Final accuracy report
print(f"Final Accuracy after evaluating all valid samples: {total_accuracy:.2f}%")
print(f"Total Prompts Included in Accuracy Calculation: {total_prompts_tested}")
