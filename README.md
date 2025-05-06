# Linear Regression for Insurance Cost Prediction

This repository contains a Python implementation of a linear regression model trained using gradient descent to predict medical insurance costs based on the `insurance.csv` dataset. The project includes data preprocessing, model training with and without z-score normalization, evaluation using R² scores, and feature importance analysis.

## Table of Contents
- [Project Overview](#project-overview)
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [File Structure](#file-structure)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Project Overview
The goal of this project is to build and evaluate a linear regression model to predict insurance charges based on features such as age, sex, BMI, number of children, smoking status, and region. The model is trained using batch gradient descent, with implementations for both unnormalized and z-score normalized data. The project also analyzes the impact of different learning rates and determines the importance of each feature in predicting insurance costs.

This project fulfills the requirements of an assignment that includes:
1. Data preprocessing and encoding categorical variables.
2. Implementing linear regression with gradient descent.
3. Training the model with different learning rates and visualizing cost vs. iteration.
4. Evaluating model performance using R² scores.
5. Normalizing data using z-score and comparing learning rate effects.
6. Analyzing feature importance using normalized data.

## Features
- **Data Preprocessing**: Encodes categorical variables (`sex`, `smoker`, `region`) and splits data into training and validation sets.
- **Linear Regression Implementation**: Custom functions for prediction, cost computation, gradient computation, and gradient descent.
- **Z-Score Normalization**: Implements normalization using `StandardScaler` to improve model convergence.
- **Model Evaluation**: Computes R² scores for training and validation sets.
- **Feature Importance**: Analyzes the importance of features based on the magnitude of model weights.
- **Visualization**: Plots cost vs. iteration for different learning rates and feature importance.

## Requirements
- Python 3.8 or higher
- Required Python packages:
  - `numpy`
  - `pandas`
  - `matplotlib`
  - `scikit-learn`
  - `seaborn`

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/francescomaxim/multiple-linear-regression.git
   cd multiple-linear-regression
   ```

2. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the required packages:
   ```bash
   pip install numpy pandas matplotlib scikit-learn seaborn
   ```

4. Ensure the `insurance.csv` dataset is in the project root directory. If not, download it from a reliable source (e.g., Kaggle) or use the provided dataset.

## Usage
1. Place the `insurance.csv` dataset in the project root directory.
2. Run the main script to train the model, evaluate performance, and generate visualizations:
   ```bash
   python assig_normalized.py
   ```
3. The script will:
   - Preprocess the data and apply z-score normalization.
   - Train the model with different learning rates and save the cost vs. iteration plot (`cost_vs_iteration_normalized.png`).
   - Train the model with the optimal learning rate (`alpha = 0.1`) and save the detailed cost plot (`cost_vs_iteration_optimal_normalized.png`).
   - Compute and print R² scores for training and validation sets.
   - Analyze and plot feature importance (`feature_importance.png`).

4. Check the `output` directory for generated plots.

## File Structure
```
insurance-cost-prediction/
│
├── assig_normalized.py       # Main script with normalized training and feature importance
├── insurance.csv             # Dataset (not included; must be added)
├── output/                   # Directory for output plots
│   ├── cost_vs_iteration_normalized.png
│   ├── cost_vs_iteration_optimal_normalized.png
│   ├── feature_importance.png
├── README.md                 # Project documentation
```

## Results
- **R² Scores (Normalized)**:
  - Training set: ~0.7512
  - Validation set: ~0.7438
  - These scores are significantly higher than the unnormalized case (~0.099 for training, ~0.141 for validation), due to z-score normalization and increased iterations (1000 vs. 10).

- **Learning Rate Comparison**:
  - **Unnormalized**: Optimal learning rate was `0.001`. Higher rates caused divergence due to varying feature scales.
  - **Normalized**: Optimal learning rate was `0.1`, allowing faster convergence because z-score normalization standardizes feature scales.

- **Feature Importance**:
  - **Most Influential**: `smoker` (smoking status) has the highest impact on insurance costs, reflecting the health risks associated with smoking.
  - **Moderately Influential**: `age` and `bmi` contribute significantly, as older age and higher BMI are linked to increased medical costs.
  - **Less Influential**: `children`, `sex`, and `region` have minimal impact.

## Contributing
Contributions are welcome! Please follow these steps:
1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Make your changes and commit (`git commit -m "Add feature"`).
4. Push to the branch (`git push origin feature-branch`).
5. Open a pull request.
