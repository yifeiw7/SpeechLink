import json
from collections import defaultdict


def load_json(filepath):
    with open(filepath, 'r') as file:
        data = json.load(file)
    return data

def count_fields(data):
    # Initialize a defaultdict for counting occurrences of each field
    field_counts = {
        'gender': defaultdict(int),
        'person': defaultdict(int),
        'session': defaultdict(int),
        'type': defaultdict(int)
    }
    
    # Iterate over each item in the data to count occurrences
    for item in data:
        field_counts['gender'][item['gender']] += 1
        field_counts['person'][item['person']] += 1
        field_counts['session'][item['session']] += 1
        field_counts['type'][item['type']] += 1
    
    # Convert defaultdicts to regular dicts for better readability
    return {key: dict(value_counts) for key, value_counts in field_counts.items()}


# Example usage
def main():
    # Load the data (use appropriate file paths)
    data = load_json('data.json')
    train_data = load_json('train_data.json')
    val_data = load_json('val_data.json')
    test_data = load_json('test_data.json')

    # Get counts for each dataset
    data_type_counts = count_fields(data)
    train_type_counts = count_fields(train_data)
    val_type_counts = count_fields(val_data)
    test_type_counts = count_fields(test_data)

    print("All data type counts:", data_type_counts)
    print("Training data type counts:", train_type_counts)
    print("Validation data type counts:", val_type_counts)
    print("Testing data type counts:", test_type_counts)

if __name__ == '__main__':
    main()
