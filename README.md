# SpeechLink

Elderly speech recognition is crucial for improving technology accessibility, especially for those with age-related speech changes or disorders like dysarthria. While automatic speech recognition (ASR) systems are increasingly common, they often struggle with the slower speech, altered pronunciation, and reduced clarity typical of seniors, leading to frustration and isolation. Developing machine learning models specifically for dysarthric speech can enhance communication between seniors and others, helping them better interact with digital devices and maintain social connections. This project aims to create more inclusive technology, improving the well-being and social engagement of elderly individuals.

## Data Processing
We sourced the data from the public [TORGO Database](https://www.cs.toronto.edu/~complingweb/data/TORGO/torgo.html) by University of Toronto. In training this model, only word and sentence prompts are used. The data is split by 70%, 15%, 15% evenly across all fields, summarized in the table below. 

### Data Distribution Summary

| **Category** | **All Data**                     | **Training**                     | **Validation**                   | **Testing**                      |
|--------------|----------------------------------|----------------------------------|----------------------------------|----------------------------------|
| **Gender**   | F: 766, M: 1692                   | F: 537, M: 1183                   | F: 115, M: 254                   | F: 114, M: 255                   |
| **Person**   | 1: 497, 2: 389, 3: 617, 4: 836, 5: 119 | 1: 348, 2: 273, 3: 432, 4: 584, 5: 83 | 1: 74, 2: 58, 3: 93, 4: 126, 5: 18 | 1: 75, 2: 58, 3: 92, 4: 126, 5: 18 |
| **Session**  | 1: 868, 2: 1382, 3: 208           | 1: 607, 2: 967, 3: 146            | 1: 130, 2: 207, 3: 32            | 1: 131, 2: 208, 3: 30            |
| **Type**     | word: 1900, sentence: 558         | word: 1328, sentence: 392         | word: 286, sentence: 83          | word: 286, sentence: 83          |


## Base Model 
The base model is selected as the SOTA text-to-speech model [Whisper by OpenAI](https://github.com/openai/whisper). 


## Finetuning 


## Evaluation 


## User Interface