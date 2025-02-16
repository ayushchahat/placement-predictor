# Placement Predictor

## Overview

Placement Predictor is a machine learning project that predicts whether a college student will get placed based on their IQ level and CGPA. The project involves data preprocessing, model training using Logistic Regression, and deployment using Pickle. The decision boundary is visualized using `mlxtend`.

## Features

- Uses **IQ level** and **CGPA** as input features
- Performs **train-test split** for model validation
- Implements **Logistic Regression** for classification
- **Scales input features** for better model performance
- **Evaluates the model** using accuracy metrics
- **Deploys the trained model** using Pickle
- **Visualizes decision boundary** using `mlxtend`

## Dataset

The dataset contains three columns:

- `cgpa`: Cumulative Grade Point Average of the student
- `iq`: IQ level of the student
- `placement`: Target variable (1 for placed, 0 for not placed)

## Project Workflow

### 0. Preprocessing, EDA & Feature Selection

- Load dataset and check for missing values
- Perform exploratory data analysis (EDA)
- Select relevant features for training

### 1. Extract Input and Output Columns

- Separate features (`cgpa`, `iq`) and target variable (`placement`)

### 2. Scale the Values

- Normalize the feature values using **StandardScaler** from `sklearn`

### 3. Train-Test Split

- Split dataset into **training** and **testing** sets using `train_test_split`

### 4. Train the Model

- Apply **Logistic Regression** to train the classifier

### 5. Evaluate the Model

- Measure model accuracy and performance
- Visualize **decision boundary** using `mlxtend`

### 6. Deploy the Model

- Save the trained model using `pickle`
- Load the model for making predictions

## Installation

To run this project, install the required dependencies:

```bash
pip install numpy pandas scikit-learn matplotlib mlxtend pickle-mixin
# Placement Predictor

## Usage

### Clone the repository:
```bash
git clone https://github.com/yourusername/placement-predictor.git
cd placement-predictor
```

### Run the Python script to train and test the model:
```bash
python train.py
```

### Load the trained model and make predictions:
```python
import pickle
import numpy as np

# Load model
with open('placement_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Predict placement
sample_data = np.array([[6.5, 120]])  # Example: CGPA = 6.5, IQ = 120
prediction = model.predict(sample_data)
print("Placement Prediction:", prediction)
```

## Visualization

### Decision Boundary (Figure 1 - fig1)
This figure shows how the Logistic Regression model separates placed (orange) and non-placed (blue) students.

### Scatter Plot (Figure 2 - fig2)
This plot visualizes the distribution of students based on IQ and CGPA, colored by placement status.

## Author
**Ayush Kumar**

## License
This project is licensed under the MIT License.
