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

## generate_description_ollama.py
This script performs automatic status checking of the Ollama service to ensure it is running before proceeding. It provides clear error messages to help users quickly identify and resolve issues, and all functions are well-documented with descriptive docstrings for easy understanding and maintenance.

### Ollama Setup Guide - Complete Instructions

#### What is Ollama?

Ollama is a tool that allows you to run large language models (like LLaMA, Mistral, etc.) locally on your own computer. It's:
- **100% Free** - No API keys, no subscriptions, no hidden costs
- **Private** - Your data never leaves your computer
- **Offline** - Works without internet after initial model download
- **Unlimited** - No rate limits or usage restrictions

---

### Step-by-Step Installation Guide

### Step 1: Install Ollama

#### For macOS:

**Option A: Using Homebrew (Recommended)**
```bash
brew install ollama
```

**Option B: Direct Download**
1. Visit https://ollama.com/download
2. Download the macOS installer
3. Open the downloaded file and follow installation instructions

#### For Linux:

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

#### For Windows:

1. Visit https://ollama.com/download
2. Download the Windows installer
3. Run the installer and follow the prompts

---

### Step 2: Verify Installation

After installation, verify that Ollama is installed correctly:

```bash
# Check if ollama command is available
which ollama

# Should output something like: /usr/local/bin/ollama or /opt/homebrew/bin/ollama
```

If you see a path, Ollama is installed successfully!

---

### Step 3: Start Ollama Service

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

### Step 4: Download a Model

In a **new terminal window** (keep the first one with `ollama serve` running), download a language model:

#### Recommended: LLaMA 3.2 (Small and Fast)

```bash
ollama pull llama3.2
```

This will download approximately 2GB of data. It may take a few minutes depending on your internet speed.

#### Alternative Models:

```bash
# Larger, more capable model (7GB)
ollama pull llama3.1

# Smaller, faster model (1.5GB)
ollama pull llama3.2:1b

# Good for Chinese language support
ollama pull qwen2.5
```

---

### Step 5: Test Ollama

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

### Step 6: Install Python Package

Install the Ollama Python package to use it in your scripts:

```bash
pip install ollama
```

Or if you're using Python 3 specifically:

```bash
pip3 install ollama
```

---

### Step 7: Run the script

Now you're ready to use the script!

**Terminal 1** (Keep Running):
```bash
ollama serve
```

**Terminal 2** (Run the Script):
```bash
python generate_description_ollama.py
```

The script will:
1. Check that Ollama is running
2. Check that the model is available
3. Generate 50 variations for each of the 9 templates
4. Save all variations to `textData/` folder

---

### Understanding the Folder Structure

After running the script, you'll have:

```
your_project/
├── generate_description_ollama.py     # The main script
├── textData/                         # Generated descriptions (output)
│   ├── random/
│   │   ├── random-continuous.txt      # 50 variations
│   │   ├── random-randomness.txt      # 50 variations
│   │   └── random-square.txt          # 50 variations
│   ├── sine/
│   │   ├── sine-continuous.txt        # 50 variations
│   │   ├── sine-randomness.txt        # 50 variations
│   │   └── sine-square.txt            # 50 variations
│   └── square/
│       ├── square-continuous.txt      # 50 variations
│       ├── square-randomness.txt      # 50 variations
│       └── square-square.txt          # 50 variations
└── images/                       # Your graph images (if you have them)
    └── *.png
```

---

### Common Issues and Solutions

### Issue 1: "connection refused" Error

**Problem:** Script can't connect to Ollama

**Solution:**
```bash
# Make sure Ollama is running in another terminal
ollama serve
```

---

### Issue 2: "model not found" Error

**Problem:** The model hasn't been downloaded

**Solution:**
```bash
# Download the model
ollama pull llama3.2

# Verify it's installed
ollama list
```

---

### Issue 3: Ollama Command Not Found

**Problem:** Ollama isn't in your PATH

**Solution:**
```bash
# For macOS with Homebrew
brew install ollama

# Or add to PATH manually (macOS)
export PATH="/usr/local/bin:$PATH"

# Verify
which ollama
```

---

### Issue 4: Generation is Too Slow

**Solutions:**

**Option A:** Use a smaller model
```bash
ollama pull llama3.2:1b  # 1 billion parameter model (faster)
```

Then in the script, change:
```python
generator = GraphDescriptionGenerator(model_name="llama3.2:1b")
```

**Option B:** Check if GPU is being used
```bash
# Ollama automatically uses GPU if available
# Check system resources while running
```

**Option C:** Reduce number of variations
In the script, change from 50 to 25:
```python
variations = self.generate_variations(
    template_info['sentence'], 
    num_variations=25  # Instead of 50
)
```

---

### Advanced Configuration

### Using a Different Model

To use a different model, modify the `main()` function:

```python
# Instead of llama3.2, use llama3.1
generator = GraphDescriptionGenerator(model_name="llama3.1")
```

### Changing Generation Parameters

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

### System Requirements

### Minimum Requirements:
- **RAM:** 8GB
- **Disk Space:** 10GB free (for models)
- **CPU:** Any modern multi-core processor

### Recommended Requirements:
- **RAM:** 16GB or more
- **Disk Space:** 20GB free
- **GPU:** Optional but speeds up generation significantly
  - NVIDIA GPU (CUDA support)
  - Apple Silicon (M1/M2/M3) - works automatically
  - AMD GPU (ROCm support on Linux)

---

### Complete Workflow Example

Here's the complete workflow from start to finish:

### Terminal 1 (Ollama Service):
```bash
# Start the service and keep running
ollama serve
```

### Terminal 2 (Setup and Run):
```bash
# Download model (only needed once)
ollama pull llama3.2

# Test that it works
ollama run llama3.2 "test"
# Type /bye to exit

# Install Python package
pip install ollama

# Run the script
python generate_descriptions.py
```

### Expected Output:
```
======================================================================
Graph Description Generator - Ollama Edition
100% Free, Runs Locally
======================================================================

Checking Ollama installation...

✓ Ollama found at: /opt/homebrew/bin/ollama
✓ Ollama service is running
✓ Available models: ['llama3.2:latest']
✓ ollama Python package is installed

✓ Initialized with model: llama3.2

======================================================================
Generating graph descriptions with Ollama
======================================================================

[1/9] random/random-continuous.txt
   Template: While the graph is continuous, it has very sudden changes...
   Generating variations...
   ✓ Received response from Ollama
   ✓ Parsed 50 variations
   ✓ Saved 50 variations to textData/random/random-continuous.txt

[2/9] random/random-randomness.txt
   ...

======================================================================
Generation complete!
======================================================================

✓ Generated 9 files in textData/

======================================================================
All done! Your graph descriptions are ready to use.
======================================================================
```

---

### Performance Expectations

### Generation Speed:
- **With GPU (Apple M1/M2/M3):** ~10-15 minutes for all 450 variations
- **With GPU (NVIDIA):** ~10-15 minutes for all 450 variations
- **CPU Only:** ~20-30 minutes for all 450 variations

### Model Sizes:
- **llama3.2:1b** - 1.3GB (fastest)
- **llama3.2** - 2GB (recommended)
- **llama3.1** - 4.7GB (highest quality)

---

### Next Steps After Setup

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

## Tips for Best Results

1. **Keep Ollama Running:** Always have `ollama serve` running in a separate terminal
2. **Check Disk Space:** Models can be large, ensure you have enough space
3. **Use Latest Version:** Keep Ollama updated for best performance
4. **Monitor Resources:** Watch CPU/RAM usage during generation
5. **Review Output:** Always check the first few generated files for quality

---

### Getting Help

If you encounter any issues:

1. **Check Ollama Status:**
   ```bash
   ollama list
   ollama ps  # Shows running models
   ```

2. **Restart Ollama:**
   - Stop: Press Ctrl+C in the terminal running `ollama serve`
   - Start: `ollama serve`

3. **Verify Model:**
   ```bash
   ollama run llama3.2 "test"
   ```

4. **Check Logs:**
   - Ollama logs appear in the terminal running `ollama serve`
   - Look for error messages there
