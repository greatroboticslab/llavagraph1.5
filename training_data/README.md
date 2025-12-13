# 📈 Synthetic Image Generation for Training Data

## Dataset Structure
Input data（Old_data/)

```Shell
training_data/
├── Old_data/                    # Original input images
│   ├── RandomNoise/            # 52 images with random noise signals
│   ├── SineWave/               # 36 images with sinusoidal waveforms  
│   └── SquareWave/             # 53 images with square waveforms
│
└── input_SyntheticImages/            # Generated synthetic images
    ├── RandomNoise/           # 705 Images with random noise overlays
    ├── SineWave/              # 705 Images with sine wave overlays
    └── SquareWave/            # 705 Images with square wave overlays
│
└── output_SyntheticImages/            # Generated synthetic images
    ├── V1/           # first version
    └── V2/              # second version

```

## Synthetic Image Generation Description_inputdata
The generation script is located at: TrainingInputdata_folder.py

To generate synthetic images, run:
```Shell
bash

python TrainingInputdata_folder.py
```

For each original image in your dataset, the script generates three types of waveform variations, each with 5 versions: Random Noise (5 variations), Sine Wave (5 variations), and Square Wave (5 variations), resulting in a total of 15 synthetic images per original image. 
Each generated synthetic image is composed of the following elements: the original instrument panel as the background, a new waveform overlay positioned at coordinates (80, 200), waveform lines in purple (color code #b43ed1), and follows a clear file naming convention: {source}_{original_filename}_{type}{number}.png.



