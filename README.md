# **Sentiment Analysis of IMDB Movie Reviews using TensorFlow and Google's nnlm-en-dim128-with-normalization/2 model**

This is a machine learning project that performs sentiment analysis on the IMDB movie reviews dataset using natural language processing (NLP) techniques. The model is implemented using TensorFlow and Google's pre-trained embedding model - ```nnlm-en-dim128-with-normalization/2```. The dataset contains 65000 review samples that are used to train and test the model.

### **Problem Statement**
Sentiment analysis is a subfield of NLP that involves analyzing text data to determine the sentiment expressed in it. The objective of this project is to perform sentiment analysis on the IMDB movie reviews dataset and classify them as either positive or negative based on the sentiment expressed in the review.

### **Dataset**
[IMDB Dataset](https://www.tensorflow.org/datasets/catalog/imdb_reviews)
- The IMDB movie reviews dataset is a widely used benchmark dataset for sentiment analysis. It contains 65000 review samples that are labeled as positive or negative based on the sentiment expressed in the review. The dataset is split into training and testing sets, with 80% of the data used for training the model and 20% used for testing.

## **Model Architecture**

- The model used in this project is a neural network that is implemented using TensorFlow. The neural network consists of an embedding layer that is initialized with Google's pre-trained embedding model -```nnlm-en-dim128-with-normalization/2```. The embedding layer is a mapping between the words in the dataset and the vectors in the embedding space. It is followed by a global average pooling layer that computes the average of the embeddings across all the words in the input sentence. The output of the pooling layer is then connected to a dense layer with a sigmoid activation function. The sigmoid activation function produces a probability value between 0 and 1, which indicates the probability that the input review is positive.

- The model is compiled using binary cross-entropy loss and Adam optimizer. Binary cross-entropy loss is used because the output of the model is binary (positive or negative). Adam optimizer is a popular optimization algorithm that is used to update the weights of the model during training.

### Training and Testing
- The dataset is split into training and testing sets, with 80% of the data used for training and 20% used for testing. The model is trained on the training set using a batch size of 512 and 10 epochs. The batch size determines the number of samples that are processed by the model at a time. The number of epochs determines the number of times the model sees the entire dataset during training.

- After training, the model is evaluated on the testing set to determine its performance. The performance of the model is measured using accuracy, precision, recall, and F1-score. Accuracy measures the overall performance of the model, precision measures the proportion of true positives out of all positive predictions, recall measures the proportion of true positives out of all actual positives, and F1-score is the harmonic mean of precision and recall.

#### **Requirements**
The following dependencies are required to run this project:

- ```Python 3.x```
- ```TensorFlow 2.x```
- ```Numpy```
- ```Matplotlib```
- ```tensorflow_hub```
- ```tensorflow_datasets```

**Usage**
To run the model on your local machine, follow these steps:

- Clone the repository to your local machine.
- Download the IMDB movie reviews dataset from the [Tensorflow Datasets](https://www.tensorflow.org/datasets/catalog/imdb_reviews)
- Extract the dataset and place it in the data directory.
- Open the ```sentiment_analysis.ipynb``` notebook in Jupyter Notebook or Google Colab.
- Run the notebook to train and test the model on the IMDB movie reviews dataset.
- The final accuracy score and other performance metrics of the model will be displayed in the notebook.

### **Results**
- The final accuracy score of the model for Training set is 99.91% and Validation set is 97.32%. For the batch of 512, the accuracy stands at 87%.
- Use the ```model.predict()``` function to analyse the input string. Format of input - [".....<String>....."]


