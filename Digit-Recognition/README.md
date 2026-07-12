# Handwritten Digit Recognition

A deep learning application that recognizes handwritten digits from 0 to 9 using a Convolutional Neural Network trained with the MNIST dataset.

The project includes data exploration, image preprocessing, CNN training, model evaluation, prediction visualization, model persistence, and an interactive Streamlit application.

---

## Project Overview

The objective of this project is to build a computer vision model capable of classifying handwritten digits.

The model receives an image containing a digit, preprocesses it into the same format used by MNIST, and returns:

- The predicted digit
- The confidence score
- The probability assigned to every class from 0 to 9
- The processed 28 × 28 image used by the neural network

---

## Demo

The Streamlit application allows users to upload an image containing a handwritten digit.

The application automatically:

1. Converts the image to grayscale
2. Inverts images with a light background
3. Resizes the image to 28 × 28 pixels
4. Normalizes pixel values
5. Sends the image to the trained CNN
6. Displays the prediction and confidence

Example result:

```text
Predicted digit: 7
Confidence: 99.99%
```

---

## Dataset

This project uses the MNIST handwritten digit dataset included with TensorFlow/Keras.

The dataset contains:

- 60,000 training images
- 10,000 test images
- 10 possible classes
- Image dimensions of 28 × 28 pixels
- Grayscale handwritten digits from 0 to 9

---

## Model Architecture

The model is a Convolutional Neural Network composed of:

- Input layer for 28 × 28 grayscale images
- Convolutional layer with 32 filters
- Max pooling layer
- Convolutional layer with 64 filters
- Max pooling layer
- Flatten layer
- Dense layer with 128 neurons
- Dropout layer for regularization
- Softmax output layer with 10 classes

The model was compiled using:

- Optimizer: Adam
- Loss function: Sparse Categorical Crossentropy
- Evaluation metric: Accuracy

---

## Model Performance

The CNN achieved approximately:

- Training accuracy: 99%
- Validation accuracy: 99%
- Test accuracy: 99%

The classification report shows high precision, recall, and F1-score across all ten digit classes.

---

## Technologies Used

- Python
- TensorFlow
- Keras
- NumPy
- Matplotlib
- Scikit-learn
- Pillow
- Streamlit
- Jupyter Notebook

---

## Project Structure

```text
Digit-Recognition/
│
├── data/
│
├── images/
│
├── models/
│   └── digit_recognition_cnn.keras
│
├── notebooks/
│   └── digit_recognition.ipynb
│
├── .gitignore
├── app.py
├── README.md
└── requirements.txt
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/YOUR-USERNAME/machine-learning-projects.git
```

Navigate to the project directory:

```bash
cd machine-learning-projects/Digit-Recognition
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

---

## Run the Application

Start the Streamlit application:

```bash
streamlit run app.py
```

Open the local address displayed in the terminal:

```text
http://localhost:8501
```

Upload a PNG, JPG, or JPEG image containing one handwritten digit.

---

## Image Requirements

For better predictions, the uploaded image should contain:

- One digit only
- A clear and centered digit
- Strong contrast between the digit and background
- No additional text or objects
- A reasonably thick handwritten stroke

The application supports both:

- Black digits on a white background
- White digits on a black background

---

## Notebook Workflow

The Jupyter Notebook includes:

1. Library imports
2. MNIST dataset loading
3. Dataset visualization
4. Pixel-value inspection
5. Image normalization
6. Channel-dimension preparation
7. CNN construction
8. Model compilation
9. Model training
10. Training-history visualization
11. Test-set evaluation
12. Prediction generation
13. Classification report
14. Model export

---

## Future Improvements

Possible improvements include:

- Adding a drawing canvas directly inside the application
- Improving image centering and cropping
- Adding a confusion matrix visualization
- Deploying the application online
- Applying data augmentation
- Comparing the CNN with other architectures

---

## Author

**Gabriel Villarreal**

Computer Science Engineer focused on Artificial Intelligence, Machine Learning, Deep Learning, Natural Language Processing, and Python development.

---

## License

This project is part of the Machine Learning Projects portfolio and is available for educational and portfolio purposes.