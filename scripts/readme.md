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

## generate_description_ollama.py


### Ollama Setup Guide 

#### What is Ollama?

Ollama is a tool that allows you to run large language models (like LLaMA, Mistral, etc.) locally on your own computer. It's:
-  **100% Free** - No API keys, no subscriptions, no hidden costs
-  **Private** - Your data never leaves your computer
-  **Offline** - Works without internet after initial model download
-  **Unlimited** - No rate limits or usage restrictions

---

### Step-by-Step Installation Guide

#### Step 1: Install Ollama

##### For macOS:

**Option A: Using Homebrew (Recommended)**
```bash
brew install ollama
```

**Option B: Direct Download**
1. Visit https://ollama.com/download
2. Download the macOS installer
3. Open the downloaded file and follow installation instructions

##### For Linux:

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

##### For Windows:

1. Visit https://ollama.com/download
2. Download the Windows installer
3. Run the installer and follow the prompts

---

#### Step 2: Verify Installation

After installation, verify that Ollama is installed correctly:

```bash
# Check if ollama command is available
which ollama

# Should output something like: /usr/local/bin/ollama or /opt/homebrew/bin/ollama
```

If you see a path, Ollama is installed successfully!

---

#### Step 3: Start Ollama Service

Ollama runs as a background service. You need to start it before using:

**Open a new terminal window** and run:

```bash
ollama serve
```

**Important:** Keep this terminal window open! The service needs to keep running.

You should see output like:
```
time=2025-01-23T10:30:00.000-05:00 level=INFO source=images.go:806 msg="total blobs: 0"
time=2025-01-23T10:30:00.000-05:00 level=INFO source=images.go:813 msg="total unused blobs removed: 0"
time=2025-01-23T10:30:00.000-05:00 level=INFO source=routes.go:1172 msg="Listening on 127.0.0.1:11434 (version 0.1.20)"
```

---

#### Step 4: Download a Model

In a **new terminal window** (keep the first one with `ollama serve` running), download a language model:

##### Recommended: LLaMA 3.2 (Small and Fast)

```bash
ollama pull llama3.2
```

This will download approximately 2GB of data. It may take a few minutes depending on your internet speed.

##### Alternative Models:

```bash
# Larger, more capable model (7GB)
ollama pull llama3.1

# Smaller, faster model (1.5GB)
ollama pull llama3.2:1b

# Good for Chinese language support
ollama pull qwen2.5
```

---

#### Step 5: Test Ollama

Verify that everything is working:

```bash
# Test the model
ollama run llama3.2 "Hello, how are you?"
```

If you see a response from the model, congratulations! Ollama is working correctly.

Example output:
```
Hello! I'm doing well, thank you for asking. I'm just a computer program, 
so I don't have feelings or emotions like humans do, but I'm functioning 
properly and ready to help with any questions or tasks you might have. 
How can I assist you today?
```

Press `Ctrl+D` or type `/bye` to exit the chat.

---

#### Step 6: Install Python Package

Install the Ollama Python package to use it in your scripts:

```bash
pip install ollama
```

Or if you're using Python 3 specifically:

```bash
pip3 install ollama
```

---

#### Step 7: Run the Graph Description Generator

Now you're ready to use the script!

**Terminal 1** (Keep Running):
```bash
ollama serve
```

**Terminal 2** (Run the Script):
```bash
python generate_descriptions.py
```

The script will:
1. Check that Ollama is running
2. Check that the model is available
3. Generate 50 variations for each of the 9 templates
4. Save all variations to `textData/` folder

---


### Advanced Configuration

#### Using a Different Model

To use a different model, modify the `main()` function:

```python
# Instead of llama3.2, use llama3.1
generator = GraphDescriptionGenerator(model_name="llama3.1")
```

#### Changing Generation Parameters

You can adjust how the model generates text by modifying the `generate_variations` method to include options:

```python
response = self.client.chat(
    model=self.model,
    messages=[{"role": "user", "content": prompt}],
    options={
        "temperature": 0.7,  # Lower = more consistent, Higher = more creative
        "top_p": 0.9,        # Controls diversity
        "num_predict": 2000  # Max tokens to generate
    }
)
```

---


### Output

1. **Generate Descriptions:**
   - Run the script to create text variations
   - Check the output in `textData/` folder

2. **Create JSON Dataset:**
   - Place your graph images in a folder
   - Uncomment the dataset creation code in `main()`:
   ```python
   generator.create_json_dataset(
       image_folder="path/to/your/images",
       output_file="dataset.json"
   )
   ```

3. **Use the Descriptions:**
   - The generated text files contain 50 variations each
   - Use them for training your graph analysis model
   - Combine with images to create your dataset

---
