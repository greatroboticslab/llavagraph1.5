# scripts

## finetune_train.py
Full Training Command

Base Model: LLaVA-v1.5-7B(models_setup/llava-v1.5-7b)

Vision Encoder: CLIP ViT-Large (openai/clip-vit-large-patch14-336)

Fine-tuning Method: LoRA (rank=128, alpha=256)

Training Framework: DeepSpeed ZeRO-2

Hardware: 2x NVIDIA GPUs

## zero2.json
DeepSpeed ZeRO-2 configuration for distributed training


## generate_descriptions.py

This script automates two key steps: it first verifies the operational status of the Gemini service, and then procedes to generate a structured JSON dataset. This is achieved by creating multiple descriptive variations for various graph types and combining them with corresponding image data.

### Features

- **Automated Description Generation**: Uses Gemini API to create 50 variations for each of 9 graph description templates
- **JSON Dataset Creation**: Produces LLaVA-compatible training and test datasets
- **Flexible Image Structure**: Supports organized image directories with separate folders for each graph type
- **API Rate Limiting Protection**: Built-in delays to avoid quota issues
- **Reusable Text Data**: Generated text descriptions can be reused across multiple training runs

### Directory Structure
```
your-project/
├── data/
│ ├── trainData/
│ │ ├── RandomNoise/ # Contains .png images
│ │ ├── SineWave/ # Contains .png images
│ │ └── SquareWave/ # Contains .png images
│ ├── testData/ # Optional test images (same structure as trainData)
│ ├── trainingData.json # Generated training dataset
│ └── testData.json # Generated test dataset (optional)
├── textData/ # Generated text descriptions
│ ├── random/
│ │ ├── random-continuous.txt
│ │ ├── random-randomness.txt
│ │ └── random-square.txt
│ ├── sine/
│ │ ├── sine-continuous.txt
│ │ ├── sine-randomness.txt
│ │ └── sine-square.txt
│ └── square/
│ ├── square-continuous.txt
│ ├── square-randomness.txt
│ └── square-square.txt
└── generate_descriptions.py # This script
```



### Setup Instructions

#### 1. Install Required Packages

```bash
pip install google-generativeai
```

#### 2. Get Google API Key

1. Visit: https://makersuite.google.com/app/apikey
2. Create a new API key
3. Copy your API key

#### 3. Set API Key

You have two options:

**Option A: Environment Variable (Recommended)**
```bash
export GOOGLE_API_KEY='your-api-key-here'
```

**Option B: Pass directly in code**
```python
generator = GraphDescriptionGenerator(api_key="your-api-key-here")
```

#### 4. Run the Script

```bash
python generate_descriptions.py
```

### Configuration Options

#### Template Customization

Edit the `templates` dictionary in the script to modify base sentences:

```python
templates = {
    "random/random-continuous.txt": {
        "sentence": "Your custom sentence here",
        "constraints": "Additional requirements"
    },
    # ... more templates
}
```

#### Number of Variations

Change the number of generated variations (default is 50):

```python
variations = self.generate_variations(
    template_info['sentence'], 
    num_variations=100  # Change this number
)
```

### Advanced Usage

#### Generate Only Specific Wave Types

```python
# Modify the templates dictionary to include only what you need
templates = {
    "sine/sine-continuous.txt": {...},
    "sine/sine-randomness.txt": {...},
}
```

#### Custom JSON Dataset Creation

Uncomment and modify in `main()`:

```python
BASE_DIR = "your_project_base_directory"  # Your project base directory
TRAIN_IMAGE_DIR = f"{BASE_DIR}/data/trainData"
TEST_IMAGE_DIR = f"{BASE_DIR}/data/testData"
OUTPUT_DIR = f"{BASE_DIR}/data"
TEXT_DATA_DIR = "textData"
```

## Error Handling

The script includes error handling for common issues:

1. **Missing API Key**: Clear error message with instructions
2. **Rate Limiting**: Automatic 1-second delay between requests
3. **Missing Files**: Warnings for missing text or image files
4. **Parsing Issues**: Multiple strategies to extract variations

If generation fails:
- Check your API key is valid
- Ensure you have API quota remaining (Gemini has free tier!)
- Check your internet connection
- Verify file paths are correct

### Output Format

#### 1. Text Descriptions (textData/)
- 9 files total: 3 graph types × 3 description aspects
- 50 variations per file: Each file contains 50 different ways to describe that aspect of the graph
- Categories:
    - Continuous: Descriptions about line continuity
    - Randomness: Descriptions about random patterns
    - Square/Corners: Descriptions about sharp corners

#### 2. JSON Datasets (data/)
- trainingData.json: Training dataset in LLaVA conversation format
- testData.json: (Optional) Test dataset with same format


### JSON Dataset
```json
{
  "id": 1,
  "image": "trainData/RandomNoise/image1.png",
  "conversations": [
    {"from": "human", "value": "<image>Is the line shown in the graph continuous? Describe the line."},
    {"from": "gpt", "value": "While the graph is continuous, it has very abrupt changes in measurements."},
    {"from": "human", "value": "Does the graph contain any random points?"},
    {"from": "gpt", "value": "The plot displays an erratic pattern, with no definable arrangement and numerous unpredictable points."},
    {"from": "human", "value": "Does the graph contain sharp corners?"},
    {"from": "gpt", "value": "While the chart does not have acute angles, it does have substantial arbitrary modifications and is not uniform."}
  ]
}
```
