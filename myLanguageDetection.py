import pickle  # To save the trained model and then read it
from typing import re

from keras.models import Sequential
from keras.layers import Dense

from sklearn.preprocessing import MinMaxScaler
from numpy import array
import seaborn as sns  # Create plots
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm._libsvm import predict
import tensorflow as tf

sns.set(style="ticks")

import warnings

warnings.filterwarnings('ignore')
import cv2
import pandas as pd
from tkinter import filedialog

import easyocr
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

import tensorflow as tf
import tensorflow_hub as hub

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Conv2D, MaxPooling1D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import layers
from tensorflow.keras import losses

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding, LSTM
from keras.layers import Conv1D, Flatten, MaxPooling1D
from keras.preprocessing import text

import wandb
from wandb.keras import WandbCallback


def select_image():
    filename = filedialog.askopenfilename(title="select a file",
                                          filetypes=[("all files", "*.*"), ("jpeg image", "*.jpg")])
    img = cv2.imread(filename)
    return img


def ocr_result(image):
    reader = easyocr.Reader(['en', 'es'], gpu=False)
    text = ' '.join(reader.readtext(image, detail=0))
    return text


def training_testing_dt():
    df = pd.read_csv('language_raw_data_sentences.csv', encoding='utf-8')  # Read raw data
    df['text'] = df['text'].astype(str)  # Convert the column "text" from object to a string in order to operate on it
    df['language'] = df['language'].astype(str)

    es_only_char = ['á', 'é', 'í', 'ó', 'ú', 'ü', 'ñ']
    en_only_combos = ['ou', 'ow', 'eigh', 'au', 'aw', 'oo', 'sh', 'th', 'wh', 'ph', 'sl',
                      'sm', 'sts', 'scr', 'spr', 'str']

    df['en_only_char_count'] = df['text'].apply(
        lambda x: sum([any(d_c in a for d_c in en_only_combos) for a in x.split()]))
    df['num_es_only_char'] = df['text'].apply(lambda x: sum([any(v in a for v in es_only_char) for a in x.split()]))

    # split dataset into features and target variable
    feature_cols = list(df.columns)[2:]
    X = df[feature_cols]  # Features
    y = df[['language']]  # Target variable
    # Split dataset into training set and test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)  # 80% train and 20% test

    # Standardize the data
    scaler = StandardScaler()
    # Fit on training set only.
    scaler.fit(X_train)
    # Transform both the training set and the test set.
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    # Make an instance of the model to retain 95% of the variance within the old features.
    pca = PCA(.95)
    pca.fit(X_train)

    # PCA has reduced the number of features to 13
    print('Number of Principal Components = ' + str(pca.n_components_))

    X_train = pca.transform(X_train)
    X_test = pca.transform(X_test)

    dt_clf = DecisionTreeClassifier()  # Create Decision Tree classifer object
    dt_clf = dt_clf.fit(X_train, y_train)  # Fit/Train Decision Tree Classifer on training set

    # Save model to file in the current working directory so that it can be imported and used.
    # I use the pickle library to save the parameters of the trained model
    pkl_file = "decision_tree_model.pkl"
    with open(pkl_file, 'wb') as file:
        pickle.dump(dt_clf, file)

    # Load previously trained model from pickle file
    with open(pkl_file, 'rb') as file:
        dt_clf = pickle.load(file)

    # parameters of the Decision Tree model are shown below and can be further optimized to improve model performance

    y_pred = dt_clf.predict(X_test)  # Predict the response for test dataset

    accuracy_score_dt = accuracy_score(y_test, y_pred)
    print('Accuracy of the Decision Tree Model on the test set = '
          + str(round(accuracy_score_dt * 100, 2)) + "%")
    print('Decision Tree Model Output')
    print(classification_report(y_test, y_pred))


def detect_language_cnn(image):

    text_image = ocr_result(image)

    df = pd.read_csv('language_raw_data_sentences.csv', encoding='utf-8')
    # generate 2d classification dataset
    df['text'] = df['text'].astype(str)  # Convert the column "text" from object to a string in order to operate on it
    df['language'] = df['language'].astype(str)

    X = df[['text']]  # Features
    y = df[['language']]

    tokenizer = text.Tokenizer(num_words=1000)
    tokenizer.fit_on_texts(X)
    X = tokenizer.texts_to_matrix(X)

    X = sequence.pad_sequences(X, maxlen=1000)
    embedding = "https://tfhub.dev/google/tf2-preview/gnews-swivel-20dim/1"
    hub_layer = hub.KerasLayer(embedding, input_shape=[],
                               dtype=tf.string, trainable=True)
    hub_layer(X[:3])

    model = tf.keras.Sequential()
    model.add(hub_layer)
    model.add(tf.keras.layers.Dense(16, activation='relu'))
    model.add(tf.keras.layers.Dense(1))

    # model = Sequential()
    # model.add(Embedding(num_words=1000,
    #                     embedding_dims=10,
    #                     input_length=1000))
    # model.add(Dropout(0.5))
    # model.add(Conv1D(filters=16,
    #                  kernal_size=3,
    #                  padding='valid',
    #                  activation='relu'))
    # model.add(MaxPooling1D())
    # model.add(Conv1D(filters=16,
    #                  kernal_size=3,
    #                  padding='valid',
    #                  activation='relu'))
    # model.add(MaxPooling1D())
    # model.add(Flatten())
    # model.add(Dense(hidden_dims=250, activation='relu'))
    # model.add(Dropout(0.5))
    # model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    model.fit(X, y, batch_size=32, epochs=10)

    # new instance where we do not know the answer
    Xnew = array(text_image)
    # make a prediction
    ynew = model.predict_classes(Xnew)
    # show the inputs and predicted outputs
    print("X=%s, Predicted=%s" % (Xnew[0], ynew[0]))
