import json
from collections import defaultdict
from sklearn.model_selection import train_test_split

def load_json(filepath):
    with open(filepath, 'r') as file:
        data = json.load(file)
    return data

def save_json(data, filepath):
    with open(filepath, 'w') as file:
        json.dump(data, file, indent=4)

def split_data(data, train_size=0.7, val_size=0.15, test_size=0.15):
    # Create a stratification key based on gender, type, person, and session
    stratify_keys = [(item['gender'], item['type'], item['person'], item['session']) for item in data]
    
    # Split into training and temp (validation + test)
    train_data, temp_data = train_test_split(
        data,
        stratify=stratify_keys,
        test_size=(1 - train_size),
        random_state=42
    )
    
    # Recompute stratification for the remaining data
    temp_stratify_keys = [(item['gender'], item['type'], item['person'], item['session']) for item in temp_data]
    
    # Calculate the proportion of validation vs test from the temp data
    val_ratio = val_size / (val_size + test_size)
    
    # Split the temp data into validation and test sets
    val_data, test_data = train_test_split(
        temp_data,
        stratify=temp_stratify_keys,
        test_size=(1 - val_ratio),
        random_state=42
    )
    
    return train_data, val_data, test_data

def main():
    # Load the data
    filepath = 'data/data.json'  # Replace with the path to your JSON file
    data = load_json(filepath)
    
    # Split the data
    train_data, val_data, test_data = split_data(data)
    
    # Save the results to separate JSON files
    save_json(train_data, 'data/train_data.json')
    save_json(val_data, 'data/val_data.json')
    save_json(test_data, 'data/test_data.json')

    print(f"Training data: {len(train_data)} samples")
    print(f"Validation data: {len(val_data)} samples")
    print(f"Testing data: {len(test_data)} samples")

if __name__ == '__main__':
    main()
