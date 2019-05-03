# ml-price-suggestion
Code for my university individual project that develops a **novel** natural language processing (NLP) method for machine learning feature (ML) extraction for price prediction.


# Motivation and description
Conventional methods such as Bag of Words (BOW) and Mean Word Embeddings (MWE) do not take full advantage of the textual fields accompanying the product, concretely the item title and description. The novel method, **Principal Embeddings (PE)**, uses principal directions of word embeddings of the textual fields as ML features. Random Forest is used as ML algorithm. 

# Installation
You will need Python 3.7 and Jupyter Notebook. It is recommended to install the [Anaconda distribution](https://www.anaconda.com/distribution/) that contains both.

Then run:
```
pip install -r requirements.txt
```
in your terminal while in the main directory (`ml-price_suggestion`), which installs all the dependencies from the `requirements.txt` file.


# Usage
The main logic for the three NLP models is developed in the `lib/language_processing.py` file, which is thoroughly documented. The `models/final` directory contains three Python scripts that run a grid search on the three NLP models to find the best text pre-processing parameters and best Random Forest's hyper-parameters. These are then saved into pickle.

The files use data from the `data` directory. This folder contains a `readme.txt` file which describes how the data can be downloaded. The first approach is recommended, since it downloads both the original and processed files. If the second approach is preferred, the notebook in `feature_engineering/Stem_Word_Embeddings` contains tools to stem the GloVe model and also a script to convert it to the word2vec format.

# Analysis on trained models
The three NLP models, BOW, MWE, and PE, were trained on the ‘Electronics/Video Games & Consoles/Video Gaming Merchandise’ Mercari category. The trained models are stored in the `data/pickle` directory (which needs to be first downloaded as described in `data/readme.txt`). The analysis on the trained models can be found in the `analysis` directory under the relevant name. It also contains analysis on the whole Mercari data set. It was concluded that the training set did not contain enough training data, since the number of features for all the three methods was higher than the number of training samples. Therefore, the model accuracies are not statistically significant. However, the analysis on feature importances suggests that training the models again on larger data set may reverse the results, possibly in benefit of PE that effectively utilises the whole feature space compared to BOW and MWE, where a few features dominate.
