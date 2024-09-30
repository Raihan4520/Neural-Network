# Neural Network from Scratch and with TensorFlow & PyTorch

This repository contains three implementations of a 3-layer neural network with the same architecture but built using different approaches: from scratch, with TensorFlow, and with PyTorch. The neural network is trained and evaluated using a **toy dataset** that was generated for binary classification, and the model is optimized using gradient descent.

## Network Architecture

- **Input Layer**: 2 neurons
- **Hidden Layers**: 
  - First hidden layer with 20 neurons
  - Second hidden layer with 10 neurons
- **Output Layer**: 1 neuron (binary classification)
  
## Loss Function and Optimizer

- **Optimizer**: Gradient Descent
- **Loss Function**: Binary Cross-Entropy
- **Epochs**: Trained for 1000 epochs

## Toy Dataset

The toy dataset is generated for training and evaluating the model. It consists of two classes, allowing the network to learn a decision boundary for binary classification. The same dataset is used across all three implementations (scratch, TensorFlow, and PyTorch). Additionally, the **toy dataset is plotted for visualization** to help understand the distribution of data points across classes.

## Files and Implementations

1. **`NN_From_Scratch.ipynb`**:
   - Neural network implementation from scratch without any machine learning libraries.
   - Uses gradient descent to update weights and biases.
   - Visualizes the decision boundary, the binary cross-entropy loss over 1000 epochs, and the toy dataset.

2. **`NN_Using_TensorFlow.ipynb`**:
   - Neural network implemented using **TensorFlow**.
   - The architecture is the same as the scratch version, using TensorFlow's high-level APIs to build and train the model.
   - Plots the decision boundary, the binary cross-entropy loss, and the toy dataset.

3. **`NN_Using_PyTorch.ipynb`**:
   - Neural network implemented using **PyTorch**.
   - Similar to the TensorFlow implementation but built using PyTorch's flexible framework.
   - Includes visualizations of the decision boundary, the loss graph, and the toy dataset.

## Visualizations

Each file contains:
- **Toy Dataset Plot**: Visualizes the distribution of data points used to train the neural network.
- **Binary Cross-Entropy Loss Plot**: Shows how the loss decreases over the 1000 epochs, with loss on the y-axis and epochs on the x-axis.
- **Decision Boundary Plot**: Visualizes how well the neural network classifies data points from the toy dataset.

## How to Run

1. Open the `.ipynb` files in Google Colab or Jupyter Notebook:
   - You can run this project in **Google Colab** by uploading the `.ipynb` files or open them directly by clicking on `Open In Colab`.
   - Alternatively, you can run it locally in **Jupyter Notebook**.

2. To clone the repository:
    ```bash
    git clone https://github.com/Raihan4520/Neural-Network.git
    ```

3. Open any of the `.ipynb` files in your preferred environment (Google Colab or Jupyter Notebook) and run the cells.

## Dependencies and Libraries

- Make sure to install all the dependencies and libraries in order to run the project.
- To install a library locally:
  ```bash
  pip install <library_name>
  ```
- To install a library in google colab:
  ```bash
  !pip install <library_name>
  ```

## Conclusion

This project demonstrates the flexibility of building neural networks from scratch and using popular machine learning libraries like TensorFlow and PyTorch. By comparing the implementations, you can see how much easier it is to implement models with high-level libraries while retaining control and learning the fundamentals by doing it manually.

## Contact

If you have any questions or suggestions, feel free to reach out through the repository's issues or contact me directly.
