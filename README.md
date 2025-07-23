# Perceptron Classifier on Breast Cancer Dataset

This project implements a simple perceptron from scratch to classify breast cancer as malignant or benign using the Breast Cancer Wisconsin dataset. Only two features are used: `radius_mean` and `texture_mean`.

## Dataset

- [Dataset](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data/data) from kaggle
- File: `breastcancer.csv`
- Features used:

  - `radius_mean`
  - `texture_mean`
- Labels:

  - `M` (Malignant) → 0
  - `B` (Benign) → 1

## Algorithm

- A binary perceptron model is implemented without using any ML libraries.
- The model uses two input features and a step function as the activation.
- Weights and bias are updated using the perceptron learning rule.

## Training

- Initialize weights and bias to 0
- Use a learning rate of 0.01
- Train for 520 epochs over the entire dataset
- On each sample:

  - Calculate the weighted sum (z)
  - Apply step function to get the predicted class
  - Compute the error (true - predicted)
  - Update weights and bias accordingly

## Prediction

- After training, the model accepts new `radius` and `texture` values as input from the user
- It outputs a prediction: "Malignant" or "Benign"

## Visualization

- The data points are plotted using matplotlib with:

  - X-axis: `radius_mean`
  - Y-axis: `texture_mean`
  - Color-coded by class (0 or 1)
- A decision boundary is drawn using the final weights and bias
- The boundary line represents where the perceptron outputs 0.5 (i.e., decision threshold)

## Requirements

- pandas (`pip install pandas`)
- numpy (`pip install numpy`)
- matplotlib (`pip install matplotlib`)

## Usage

- Place the `breastcancer.csv` file in the same directory
- Run the script
- Enter values for radius and texture when prompted
- View the classification result and decision boundary plot

## Demo:
- Graph showing the graph obtained:

<img width="640" height="480" alt="perceptron1" src="https://github.com/user-attachments/assets/38ecf4bb-33c4-4bc2-b59c-2a35fa2218d4" />

- Graph showing the boundary between Benign and Malignant:

<img width="640" height="480" alt="perceptron2" src="https://github.com/user-attachments/assets/6fe3a7b8-9c98-4f06-a8da-40b0be6cb5ab" />

- Prediction sample:

<img width="227" height="68" alt="image" src="https://github.com/user-attachments/assets/b78cf7db-e105-44ba-a206-6ed3d3cdb4ed" />
