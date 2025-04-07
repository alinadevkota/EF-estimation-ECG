# EF-estimation-ECG

`EF-estimation-ECG` is a Python-based project aimed at estimating left ventricular ejection fraction (EF) from 12-lead electrocardiogram (ECG) data.

## Features

- **Data Processing Utilities**: Functions for loading, preprocessing, and managing ECG datasets (`data_utils.py`).
- **Model Architectures**: Implementation of deep learning models tailored for ECG data analysis, including ResNet-based models and MLP (`models.py`) and transformer-based models (`ecgformer.py`).
- **Training and Evaluation Pipelines**: Scripts to train models (`main.py`, `main_sklearn.py`) and evaluate their performance (`eval.py`).
- **Dataset Generation**: Tools to format datasets for training and testing (`generate_dataset.py`).
- **Interpretability and Visualization**: Modules to interpret model predictions and visualize ECG data (`interpretability/`, `visualizations/`).

## Installation

To set up the project environment, follow these steps:

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/alinadevkota/EF-estimation-ECG.git
   cd EF-estimation-ECG
2. **Create a Virtual Environment** (optional but recommended):
    ```bash
    python3 -m venv venv
    source venv/bin/activate
3. **Install Required Dependencies:**
    ```bash
    pip install -r requirements.txt
## Usage
### Data Preparation
Ensure your ECG dataset is organized appropriately. The `data_utils.py` module provides functions to assist with data loading and preprocessing. Modify paths and parameters as needed to align with your data storage.

### Training a Model
To train a model using the default settings:

    python main.py

### Evaluating a Model
After training, evaluate the model's performance:

    python eval.py --model_path path/to/saved_model.pth --data_path path/to/test_data/
