#!/usr/bin/env python
# coding: utf-8

import json
import random
import os

# Setup the Paths
uniqueId = 1

# Text data paths
RANDOM_CONTINUOUS = "12.14llava/data/textData/random/random-continuous.txt"
SINE_CONTINUOUS = "12.14llava/data/textData/sine/sine-continuous.txt"
SQUARE_CONTINUOUS = "12.14llava/data/textData/square/square-continuous.txt"

RANDOM_RANDOMNESS = "12.14llava/data/textData/random/random-randomness.txt"
SINE_RANDOMNESS = "12.14llava/data/textData/sine/sine-randomness.txt"
SQUARE_RANDOMNESS = "12.14llava/data/textData/square/square-randomness.txt"

RANDOM_SQUARE = "12.14llava/data/textData/random/random-square.txt"
SINE_SQUARE = "12.14llava/data/textData/sine/sine-square.txt"
SQUARE_SQUARE = "12.14llava/data/textData/square/square-square.txt"



data = []

# Random Noise
print("Loading Random Noise text data...")
with open(RANDOM_CONTINUOUS, "r") as file:
    randomContinuous = file.read().splitlines()
with open(RANDOM_RANDOMNESS, "r") as file:
    randomRandomness = file.read().splitlines()
with open(RANDOM_SQUARE, "r") as file:
    randomSquare = file.read().splitlines()

print("Processing NoiseData images...")
noise_dir = "12.14llava/data/stage1_input/input_all/NoiseData"
maxImage = len(os.listdir(noise_dir))
for file in os.listdir(noise_dir):
    json_data = {
        "id": uniqueId,
        "image": f"data/stage1_input/input_all/NoiseData/{file}",
        "conversations": [
            {
                "from": "human",
                "value": "<image>Is the line shown in the graph continuous? Describe the line."
            },
            {
                "from": "gpt",
                "value": f"{random.choice(randomContinuous)}"
            },
            {
                "from": "human",
                "value": "Does the graph contain any random points?"
            },
            {
                "from": "gpt",
                "value": f"{random.choice(randomRandomness)}"
            },
            {
                "from": "human",
                "value": "Does the graph contain sharp corners?"
            },
            {
                "from": "gpt",
                "value": f"{random.choice(randomSquare)}"
            },
        ]
    }
    data.append(json_data)
    uniqueId += 1

# Sine Wave
print("Loading Sine Wave text data...")
with open(SINE_CONTINUOUS, "r") as file:
    sineContinuous = file.read().splitlines()
with open(SINE_RANDOMNESS, "r") as file:
    sineRandomness = file.read().splitlines()
with open(SINE_SQUARE, "r") as file:
    sineSquare = file.read().splitlines()

print("Processing SineData images...")
sine_dir = "12.14llava/data/stage1_input/input_all/SineData"
sineImages = os.listdir(sine_dir)
random.shuffle(sineImages)
sineImages = sineImages[:maxImage]
for file in sineImages:
    json_data = {
        "id": uniqueId,
        "image": f"data/stage1_input/input_all/SineData/{file}",
        "conversations": [
            {
                "from": "human",
                "value": "<image>Is the line shown in the graph continuous? Describe the line."
            },
            {
                "from": "gpt",
                "value": f"{random.choice(sineContinuous)}"
            },
            {
                "from": "human",
                "value": "Does the graph contain any random points?"
            },
            {
                "from": "gpt",
                "value": f"{random.choice(sineRandomness)}"
            },
            {
                "from": "human",
                "value": "Does the graph contain sharp corners?"
            },
            {
                "from": "gpt",
                "value": f"{random.choice(sineSquare)}"
            },
        ]
    }
    data.append(json_data)
    uniqueId += 1

# Square Wave
print("Loading Square Wave text data...")
with open(SQUARE_CONTINUOUS, "r") as file:
    squareContinuous = file.read().splitlines()
with open(SQUARE_RANDOMNESS, "r") as file:
    squareRandomness = file.read().splitlines()
with open(SQUARE_SQUARE, "r") as file:
    squareSquare = file.read().splitlines()

print("Processing SquareData images...")
square_dir = "12.14llava/data/stage1_input/input_all/SquareData"
squareImages = os.listdir(square_dir)
random.shuffle(squareImages)
squareImages = squareImages[:maxImage]
for file in squareImages:
    json_data = {
        "id": uniqueId,
        "image": f"data/stage1_input/input_all/SquareData/{file}",
        "conversations": [
            {
                "from": "human",
                "value": "<image>Is the line shown in the graph continuous? Describe the line."
            },
            {
                "from": "gpt",
                "value": f"{random.choice(squareContinuous)}"
            },
            {
                "from": "human",
                "value": "Does the graph contain any random points?"
            },
            {
                "from": "gpt",
                "value": f"{random.choice(squareRandomness)}"
            },
            {
                "from": "human",
                "value": "Does the graph contain sharp corners?"
            },
            {
                "from": "gpt",
                "value": f"{random.choice(squareSquare)}"
            }
        ]
    }
    data.append(json_data)
    uniqueId += 1

# Output to specified directory
print("Generating final dataset...")
random.shuffle(data)

output_dir = "/data/ilminur/12.14llava/data/"
os.makedirs(output_dir, exist_ok=True)

output_path = os.path.join(output_dir, "fullData.json")
with open(output_path, "w") as outputFile:
    json.dump(data, outputFile, indent=2)

print(f"✅ Generated {len(data)} data entries saved to: {output_path}")
print(f"📊 Breakdown:")
print(f"   - NoiseData: {maxImage}")
print(f"   - SineData:  {len(sineImages)}")
print(f"   - SquareData: {len(squareImages)}")
