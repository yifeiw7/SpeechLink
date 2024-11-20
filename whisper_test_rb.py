import os
import argparse
import torchaudio
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from evaluate import load
import nltk

# Download NLTK tokenizer for BLEU
nltk.download("punkt")

# Argument parser for specifying output directory
parser = argparse.ArgumentParser()
parser.add_argument("--output_dir", type=str, default="./results/", help="Directory to save validation outputs")
args = parser.parse_args()

# Ensure the output directory exists
os.makedirs(args.output_dir, exist_ok=True)

# Load the fine-tuned model and processor
try:
    model = WhisperForConditionalGeneration.from_pretrained("./checkpoints/")
    processor = WhisperProcessor.from_pretrained("./checkpoints/")
except Exception as e:
    print(f"Error loading model or processor: {e}")
    exit(1)

model.eval()
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
model.to(device)

# Function to preprocess and generate predictions
def generate_predictions(batch):
    try:
        audio_path = "data/" + batch["audio_path"]
        audio, sample_rate = torchaudio.load(audio_path)
        audio = audio.squeeze().numpy()

        # Resample if needed
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
            audio = resampler(torch.tensor(audio)).numpy()

        # Generate predictions
        input_features = processor(audio, sampling_rate=16000, return_tensors="pt").input_features
        input_features = input_features.to(device)

        with torch.no_grad():
            generated_ids = model.generate(input_features)
        predicted_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        return predicted_text
    except Exception as e:
        print(f"Error generating predictions for batch {batch}: {e}")
        return None

# Load validation data
val_data_path = 'data/val_data.json'
try:
    from datasets import Dataset
    validation_dataset = Dataset.from_json(val_data_path)
except Exception as e:
    print(f"Error loading validation data: {e}")
    exit(1)

# Metrics for evaluation
try:
    bleu = load("bleu")
except Exception as e:
    print(f"Error loading BLEU metric: {e}")
    exit(1)

# Validate the model
predictions = []
references = []

for example in validation_dataset:
    try:
        predicted_text = generate_predictions(example)
        if predicted_text is not None:
            predictions.append(predicted_text)
            references.append([example["prompt"]])  # BLEU requires references as a list of lists
    except Exception as e:
        print(f"Error processing example {example}: {e}")
        continue

# Compute BLEU score
try:
    bleu_score = bleu.compute(predictions=predictions, references=references)
except Exception as e:
    print(f"Error computing BLEU score: {e}")
    bleu_score = {"bleu": 0.0}

# Save validation predictions
try:
    with open(os.path.join(args.output_dir, "validation_predictions.txt"), "w") as f:
        for pred, ref in zip(predictions, references):
            try:
                f.write(f"Prediction: {pred}\n")
                f.write(f"Ground Truth: {ref[0]}\n")
                f.write("-" * 50 + "\n")
            except Exception as e:
                print(f"Error writing prediction: {e}")
except Exception as e:
    print(f"Error saving validation predictions: {e}")

# Save metrics
try:
    with open(os.path.join(args.output_dir, "validation_metrics.txt"), "w") as f:
        f.write(f"BLEU Score: {bleu_score['bleu']:.4f}\n")
except Exception as e:
    print(f"Error saving validation metrics: {e}")

print("Validation results saved to", args.output_dir)
