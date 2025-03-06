This Jupyter Notebook builds and evaluates a deep learning model to predict the success of charitable donations. The process is structured into three main sections: Data Preprocessing, Model Compilation & Training, and Evaluation.

1. Data Preprocessing
The dataset is loaded from a cloud URL and cleaned by removing non-beneficial columns (EIN and NAME).
The number of unique values in each column is determined to identify potential categorical variables for encoding.
Low-frequency categories in APPLICATION_TYPE and CLASSIFICATION are grouped under "Other" to simplify the dataset.
Categorical variables are converted into numerical format using pd.get_dummies().
The dataset is split into features (X) and target (y), followed by a train-test split.
Feature scaling is applied using StandardScaler to normalize data before feeding it into the model.
2. Model Compilation & Training
A deep neural network (DNN) is defined using TensorFlow/Keras.
The model consists of:
Input layer: Matching the number of features.
Two hidden layers:
First hidden layer with 80 nodes and ReLU activation.
Second hidden layer with 30 nodes and ReLU activation.
Output layer: A single node with a sigmoid activation function to predict binary outcomes.
The model is compiled using the Adam optimizer and binary_crossentropy loss function.
The model is trained for 100 epochs, with accuracy and loss metrics tracked.
3. Model Evaluation
After training, the model is evaluated using the test dataset.
The final accuracy achieved on the test dataset is ~72.6%.
The model is saved in an HDF5 file for future use.
Key Takeaways
The model achieves a decent accuracy (~72.6%) but could be further optimized.
Potential improvements could include hyperparameter tuning, adding more hidden layers, experimenting with dropout regularization, or adjusting the feature selection process.
