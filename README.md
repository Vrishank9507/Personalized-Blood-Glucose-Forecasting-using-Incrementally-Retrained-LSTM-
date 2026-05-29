# Personalized Blood Glucose Forecasting Using Incrementally Retrained LSTM

## Overview

This project implements a personalized blood glucose forecasting system using an Incrementally Retrained Long Short-Term Memory (IS-LSTM) network. The model predicts future blood glucose levels from Continuous Glucose Monitor (CGM) data and adapts to individual glucose patterns through incremental retraining.

The goal is to improve prediction accuracy for people with Type 1 Diabetes while requiring only limited historical data.

---

## Features

* Personalized blood glucose prediction
* Incremental model retraining
* LSTM-based deep learning architecture
* Support for Continuous Glucose Monitor (CGM) data
* Improved forecasting accuracy with limited training data
* Cold-start prediction capability for new users
* Data preprocessing and smoothing techniques

---

## Technologies Used

* Python
* TensorFlow / Keras
* NumPy
* Pandas
* Scikit-Learn
* Matplotlib
* Jupyter Notebook

---

## Dataset

The model uses Continuous Glucose Monitoring (CGM) data collected from diabetes patients.

Input Features:

* Blood glucose readings
* Insulin dosage information
* Carbohydrate intake (optional)

Output:

* Future blood glucose prediction (30-minute and 60-minute horizons)

---

## Model Architecture

1. Data Preprocessing
2. Kalman Filter Smoothing
3. Stacked LSTM Network
4. Incremental Retraining Framework
5. Personalized Prediction Generation

---

## Results

The proposed IS-LSTM model achieved improved glucose prediction accuracy compared to traditional LSTM approaches.

Key Benefits:

* Lower prediction error
* Faster adaptation to individual users
* Better performance with limited historical data
* Clinically safe prediction ranges

---

## Installation

```bash
git clone https://github.com/yourusername/glucose-forecasting-lstm.git

cd glucose-forecasting-lstm

pip install -r requirements.txt
```

---

## Usage

```bash
python train.py
```

For prediction:

```bash
python predict.py
```

---

## Project Structure

```
├── data/
├── models/
├── notebooks/
├── results/
├── train.py
├── predict.py
├── requirements.txt
├── README.md
└── report.pdf
```

---

## Future Improvements

* Real-time glucose monitoring
* Mobile application integration
* Transformer-based forecasting models
* Multi-sensor data fusion
* Integration with Artificial Pancreas Systems

---

## Author

Vrishank Agashe

M.Tech (Computer Science and Engineering)
IIIT Vadodara
