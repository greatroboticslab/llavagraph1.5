from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import os
from pathlib import Path


rng = np.random.default_rng(seed=42)
    
 # Configuration
NUM_OF_IMAGES = 5
BASE_PATH = "Old_data/RandomNoise/NoiseTrials-1-Run3.xlsx-7.png"

baseImage = Image.open(BASE_PATH)
    
# Create directories
Path("SyntheticImages/RandomNoise").mkdir(parents=True, exist_ok=True)
Path("SyntheticImages/SineWave").mkdir(parents=True, exist_ok=True)
Path("SyntheticImages/SquareWave").mkdir(parents=True, exist_ok=True)
    
# Time array
t = np.array(range(256))
    
# random noise
px = 1 / plt.rcParams["figure.dpi"]  # pixel in inches
for i in range(NUM_OF_IMAGES):
    fig, ax = plt.subplots(figsize=(700 * px, 300 * px))

    randomNoise = rng.uniform(low=-1, high=1, size=256)

    plt.plot(t, randomNoise, color="#b43ed1")

    plt.axis("off")  # this rows the rectangular frame
    ax.get_xaxis().set_visible(False)  # this removes the ticks and numbers for x axis
    ax.get_yaxis().set_visible(False)

    plt.savefig("temp.png", bbox_inches="tight", pad_inches=0.1)

    chart = Image.open("temp.png")

    new_image = baseImage

    new_image.paste(chart, (80, 200))

    new_image.save(f"SyntheticImages/RandomNoise/{i}.png")


# sine waves
px = 1 / plt.rcParams["figure.dpi"]  # pixel in inches
for i in range(NUM_OF_IMAGES):
    fig, ax = plt.subplots(figsize=(700 * px, 300 * px))

    sineWave = np.sin(rng.uniform(0.05, 0.2) * t)

    plt.plot(t, sineWave, color="#b43ed1")

    plt.axis("off")  # this rows the rectangular frame
    ax.get_xaxis().set_visible(False)  # this removes the ticks and numbers for x axis
    ax.get_yaxis().set_visible(False)

    plt.savefig("temp.png", bbox_inches="tight", pad_inches=0.1)

    chart = Image.open("temp.png")

    new_image = baseImage

    new_image.paste(chart, (80, 200))

    new_image.save(f"SyntheticImages/SineWave/{i}.png")
    
# square waves
for i in range(NUM_OF_IMAGES):
    fig, ax = plt.subplots(figsize=(700 * px, 300 * px))

    sineWave = signal.square(rng.uniform(0.05, 0.2) * t)

    plt.plot(t, sineWave, color="#b43ed1")

    plt.axis("off")  # this rows the rectangular frame
    ax.get_xaxis().set_visible(False)  # this removes the ticks and numbers for x axis
    ax.get_yaxis().set_visible(False)

    plt.savefig("temp.png", bbox_inches="tight", pad_inches=0.1)

    chart = Image.open("temp.png")

    new_image = baseImage

    new_image.paste(chart, (80, 200))

    new_image.save(f"SyntheticImages/SquareWave/{i}.png")
    
    