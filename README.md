## Neural Network Classifier for MNIST Dataset

### Overview
This project implements a neural network classifier for recognizing hand-written digits using the MNIST dataset. The neural network architecture consists of two hidden layers with ReLU activation functions and a softmax output layer for multi-class classification.

### Files
- **classifier.py**: Contains the implementation of the neural network model, training, and inference functions. It also handles user input for making predictions on custom images.
- **log.txt**: Logs the training progress, including training loss, validation accuracy, and final accuracy on the test set for each epoch.
- **Makefile**: Provides a command to run the classifier python script

### Usage
1. **Dependencies**: Ensure you have Python 3.x, PyTorch, torchvision, and PIL (Python Imaging Library) installed on your system.
2. **Run Classifier**: Navigate to the directory containing `makefile` and `classifier.py` in a command line interface.
3. **Execute Command**: Enter the command `make run` to run `classifier.py`.
4. **Training and Evaluation**: The script will train the model and evaluate its performance on a validation set and then a test set ,log.txt will be ovewritten with current model's training losses and validation accuracy per epoch. Model accuracy per epoch will be displayed until the training process is finished.
5. **Inference**: After training, you will be prompted to enter the path of an image file (.jpg) you want to classify. Enter the relative path or type "exit" to quit the program.
6. **Output**: The program will classify the image and display the predicted label.

### Requirements
- Python 3.x
- PyTorch
- torchvision
- PIL (Python Imaging Library)

### Note
- The project assumes that the MNIST dataset is stored in the "MNIST data/" directory. Ensure the dataset is properly downloaded and accessible to the script.

## Development on Windows 11