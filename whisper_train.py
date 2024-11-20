import os
import argparse
from transformers import WhisperProcessor, WhisperForConditionalGeneration, TrainingArguments, Trainer
from datasets import Dataset, DatasetDict, Audio
import torchaudio
import torch

# Argument parser for specifying output directory
parser = argparse.ArgumentParser()
parser.add_argument("--output_dir", type=str, default="./checkpoints/", help="Directory to save model checkpoints")
args = parser.parse_args()

# Load and preprocess data
def load_data():
    # Replace with actual paths to your JSON files
    train_data = 'data/train_data.json'
    val_data = 'data/val_data.json'
    test_data = 'data/test_data.json'

    # Load data and create Hugging Face datasets
    train_dataset = Dataset.from_json(train_data)
    val_dataset = Dataset.from_json(val_data)
    test_dataset = Dataset.from_json(test_data)

    return DatasetDict({
        "train": train_dataset,
        "validation": val_dataset,
        "test": test_dataset
    })

# Load data
dataset = load_data()

# Load the Whisper processor and model
processor = WhisperProcessor.from_pretrained("openai/whisper-small")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

model.to(device)

# Preprocessing function
def preprocess_function(batch):
    audio_path = "data/" + batch["audio_path"]

    # Load the audio data from file
    audio, sample_rate = torchaudio.load(audio_path)
    audio = audio.squeeze().numpy()  # Convert to numpy for processing

    # Resample if needed
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
        audio = resampler(torch.tensor(audio)).numpy()

    # Process audio data and encode the prompt
    batch["input_features"] = processor(audio, sampling_rate=16000, return_tensors="pt").input_features[0]
    batch["labels"] = processor.tokenizer(batch["prompt"]).input_ids  # Encode prompt as labels
    return batch

# Apply preprocessing to the dataset
dataset = dataset.map(preprocess_function, remove_columns=["audio_path", "prompt"], batched=False)

# Define custom data collator to handle padding of variable-length inputs and labels
def data_collator(features):
    # Convert input features to tensors, with padding
    input_features = [torch.tensor(f["input_features"], dtype=torch.float32) for f in features]
    input_features = torch.nn.utils.rnn.pad_sequence(input_features, batch_first=True)

    # Convert labels to tensors and pad them as well
    labels = [torch.tensor(f["labels"], dtype=torch.long) for f in features]
    labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)  # -100 to ignore in loss

    return {"input_features": input_features, "labels": labels}

# Training arguments
training_args = TrainingArguments(
    output_dir=args.output_dir,  # Save checkpoints here
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_dir="./logs",  # Logs directory
    learning_rate=1e-4,
    num_train_epochs=3,
    gradient_accumulation_steps=2,
    fp16=False,
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    tokenizer=processor.feature_extractor,
    data_collator=data_collator,
)

# Start training
trainer.train()

# Save the trained model and processor
model.save_pretrained(args.output_dir)
processor.save_pretrained(args.output_dir)
print(f"Model and processor saved to {args.output_dir}")
