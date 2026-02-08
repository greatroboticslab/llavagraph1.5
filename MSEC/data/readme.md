# Data Processing


## Issac_data_Aug_27_2024
Original experimental measurements collected from instruments

Format: Raw data files in their native format (not yet processed, source data that needs to be converted to physical units before further processing)


## extract_txt_file.py

Extracts original text files from Issac_data_Aug_27_2024/ folder and creates Original_only_txt directory containing extracted text files

Importance: These extracted text files serve as the primary data source for the subsequent processing pipeline


## process_raw_8.py

Converts raw collected data to nanometers using a specific conversion formula

Key Features:

```Shell
Implements the conversion algorithm: nm = (D - baseline) * (wavelength / 8) - correction
``` 

Processes files from: Issac_data_Aug_27_2024/Original_only_txt/

Supports "Absolute" mode for generating calibrated output

<img width="845" height="244" alt="Screenshot 2026-01-07 at 9 33 16 PM" src="https://github.com/user-attachments/assets/0db6427c-f42f-4f20-8dda-2cfb43f38cf3" />

Dependencies: Requires ttkbootstrap and matplotlib packages

For detailed implementation, refer to: https://github.com/greatroboticslab/laserai


## output_file

Output directory for processed data. Created by process_raw_8.py when "Absolute" mode is selected

Contents: Contains processed data in both PNG (visualizations) and CSV (numeric data) formats

Purpose: Intermediate storage for data that has been converted to physical units


## extract_csv.py

Extracts CSV files specifically from the output_file/ directory

Usage: Since the processed output folder contains both PNG and CSV files, this script isolates CSV files for subsequent synthetic data generation

Output: CSV files ready for augmentation and synthetic data generation


## csv_slice_visualizer.py

Processes displacement data by isolating specific time-series segments for cleaner visualization.

Functionality: Extracts a specific data window (indices 50–200) from input CSV files to bypass initial noise and signal fluctuations.

Usage: Ideal for generating high-quality, standardized plots of "stable" data without the need for manual cropping or synthetic augmentation.

Output: High-resolution PNG visualizations of the selected data slice, saved directly to a dedicated desktop folder.


## ori_Plots

Output folder for Processes displacement data



## data_filtered - Current Run Dataset

The test and training datasets consist of both experimental and synthetic images.

