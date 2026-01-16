# original_data

This folder contains the complete pipeline for processing raw experimental data, converting it to physical units, and preparing it for synthetic data generation.

## Issac_data_Aug_27_2024
Original experimental measurements collected from instruments(use as test data)

Format: Raw data files in their native format (not yet processed, source data that needs to be converted to physical units before further processing)


## output_file

Output directory for processed data. Created by process_raw_8.py when "Absolute" mode is selected

Contents: Contains processed data in both PNG (visualizations) and CSV (numeric data) formats

Purpose: Intermediate storage for data that has been converted to physical units


## original_train

Original experimental measurements collected from instruments(use as training data)

Usage: Used as input for synthetic (training)data generation and model training pipelines


## process_raw_8.py

Converts raw collected data to nanometers using a specific conversion formula

Key Features:

```Shell
Implements the conversion algorithm: nm = (D - baseline) * (wavelength / 8) - correction
``` 

Processes files from: original/output/Issac_data_Aug_27_2024/Original_only_txt/

Supports "Absolute" mode for generating calibrated output

<img width="845" height="244" alt="Screenshot 2026-01-07 at 9 33 16 PM" src="https://github.com/user-attachments/assets/0db6427c-f42f-4f20-8dda-2cfb43f38cf3" />

Dependencies: Requires ttkbootstrap and matplotlib packages

For detailed implementation, refer to: https://github.com/greatroboticslab/laserai

## extract_txt_file.py

Extracts original text files from Issac_data_Aug_27_2024/ folder and creates Original_only_txt directory containing extracted text files

Importance: These extracted text files serve as the primary data source for the subsequent processing pipeline

## extract_csv.py

Extracts CSV files specifically from the output_file/ directory

Usage: Since the processed output folder contains both PNG and CSV files, this script isolates CSV files for subsequent synthetic data generation

Output: CSV files ready for augmentation and synthetic data generation

