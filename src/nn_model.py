import pickle
import tensorflow as tf
from keras import Input
from keras import backend as K
from keras.engine import Model
from keras.layers import Embedding, Reshape, Conv2D, MaxPooling2D, Dropout, Flatten, Dense, LSTM, Bidirectional, merge, \
    Activation, Convolution2D, Softmax
from keras.layers.merge import add, concatenate
from keras.models import load_model
import numpy as np
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report, confusion_matrix
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import Adam
import random

import matplotlib.pyplot as plt


from config import *
from sklearn.metrics.classification import f1_score

def create_no_lstm_bert_model_with_previous_sentence(input_vector_size, learning_rate, output_dimension):
    current_input = Input(shape=(input_vector_size,), name='input')
    previous_input = Input(shape=(input_vector_size,), name='previous_input')

    merged = concatenate([current_input, previous_input])
    d_1 = Dense(512, activation='relu')(merged)
    output0 = Dense(output_dimension, name='output0')(d_1)
    output = Softmax(name='output')(output0)

    optimizer = Adam(lr=learning_rate)
    model = Model(inputs=[current_input, previous_input], outputs=[output])
    model.summary()
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=["accuracy", f1])

    return model


def create_two_lstm_bert_model_with_previous_sentence(input_vector_size, learning_rate, output_dimension):
    current_input = Input(shape=(input_vector_size,), name='input')
    reshape_current = Reshape((1, input_vector_size))(current_input)
    lstm_current = LSTM(500, activation="relu", recurrent_dropout=0.2, return_sequences=False)(reshape_current)

    previous_input = Input(shape=(input_vector_size,), name='previous_input')
    reshape_current = Reshape((1, input_vector_size))(previous_input)
    lstm_previous = LSTM(500, activation="relu", recurrent_dropout=0.2, return_sequences=False)(reshape_current)

    merged = concatenate([lstm_current, lstm_previous])
    output = Dense(output_dimension, name='output', activation='softmax')(merged)

    optimizer = Adam(lr=learning_rate)
    model = Model(inputs=[current_input, previous_input], outputs=[output])
    model.summary()
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=["accuracy", f1])

    return model


def create_simple_bert_model_with_previous_sentence(input_vector_size, learning_rate, output_dimension):
    current_input = Input(shape=(input_vector_size,), name='input')
    reshape = Reshape((1, input_vector_size))(current_input)
    lstm = LSTM(500, activation="relu", recurrent_dropout=0.2, return_sequences=False)(reshape)

    previous_input = Input(shape=(input_vector_size,), name='previous_input')
    merged = concatenate([lstm, previous_input])
    output = Dense(output_dimension, name='output', activation='softmax')(merged)

    optimizer = Adam(lr=learning_rate)
    model = Model(inputs=[current_input, previous_input], outputs=[output])
    model.summary()
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=["accuracy", f1])

    return model

def create_kim_model(input_size, we_matrix, learning_rate, emb_training, output_dimension):
    kernels = [3, 4, 5]
    filters = 100
    convs, relus, pools, flats = [], [], [], []

    input_dimension = we_matrix.shape[0]
    embedding_dimension = we_matrix.shape[1]

    my_input = Input(shape=(input_size,), name='input')

    emb = Embedding(input_dim=input_dimension, weights=[we_matrix], output_dim=embedding_dimension,
                    input_length=input_size, trainable=emb_training)(my_input)

    reshape = Reshape((input_size, embedding_dimension, 1))(emb)
    for k in kernels:
        convs.append(Conv2D(filters, k, embedding_dimension)(reshape))
        relus.append(Activation('relu')(convs[-1]))
        pools.append(MaxPooling2D(pool_size=(input_size - k + 1, 1))(relus[-1]))
        flats.append(Flatten()(pools[-1]))
    merged = concatenate(flats)
    dense = Dense(256)(merged)
    relu = Activation('relu')(dense)
    drop = Dropout(0.5)(relu)
    output = Dense(output_dimension, activation='softmax', name='output')(drop)

    optimizer = Adam(lr=learning_rate)
    model = Model(input=my_input, output=output)
    model.layers[1].trainable = True
    model.summary()
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=["accuracy", f1])

    return model

### CNN model
def create_cnn_model_with_previous(input_size, we_matrix, learning_rate, emb_training, output_dimension, previous_sentence_input_vector_size):
    # print("WE_matrix shape: ", we_matrix.shape)
    input_dimension = 20003
    embedding_dimension = 300

    kernel = (4, 1)
    my_input = Input(shape=(input_size,), name='inputcnn')
    if we_matrix is None:
        emb = Embedding(input_dim=input_dimension, output_dim=embedding_dimension,
                        input_length=input_size, trainable=emb_training)(my_input)
    else:
        emb = Embedding(input_dim=input_dimension, weights=[we_matrix], output_dim=embedding_dimension,
                    input_length=input_size, trainable=emb_training)(my_input)

    reshape = Reshape((input_size, embedding_dimension, 1))(emb)
    conv = Conv2D(40, kernel, activation='relu', padding='valid')(reshape)
    pool = MaxPooling2D(pool_size=(input_size - kernel[0] + 1, 1))(conv)
    drop_1 = Dropout(0.2)(pool)
    flat = Flatten()(drop_1)
    d_1 = Dense(256, activation='relu', name="curuttd1")(flat)

    previous_input = Input(shape=(previous_sentence_input_vector_size,), name='previous_inputcnn')
    merged = concatenate([d_1, previous_input])
    drop_2 = Dropout(0.2)(merged)

    output = Dense(output_dimension, name='output', activation='softmax')(drop_2)

    optimizer = Adam(lr=learning_rate)
    model = Model(inputs=[my_input, previous_input], outputs=[output])
    model.summary()
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=["accuracy", f1])

    return model


def create_bilstm_model_with_previous(input_size, we_matrix, learning_rate, emb_training, output_dimension, previous_sentence_input_vector_size):
    #print("WE_matrix shape: ", we_matrix.shape)
    input_dimension = 20003
    embedding_dimension = 300

    my_input = Input(shape=(input_size,), name='input')
    if we_matrix is None:
        emb = Embedding(input_dim=input_dimension, output_dim=embedding_dimension,
                        input_length=input_size, trainable=emb_training)(my_input)
    else:
        emb = Embedding(input_dim=input_dimension, weights=[we_matrix], output_dim=embedding_dimension,
                    input_length=input_size, trainable=emb_training)(my_input)

    reshape = Reshape((input_size, embedding_dimension))(emb)
    lstm = Bidirectional(LSTM(100, activation="relu", recurrent_dropout=0.2, return_sequences=False))(reshape)

    previous_input = Input(shape=(previous_sentence_input_vector_size,), name='previous_input')
    merged = concatenate([lstm, previous_input])

    #d_1 = Dense(256, activation='relu')(lstm)

    drop_2 = Dropout(0.2)(merged)
    output = Dense(output_dimension, name='output', activation='softmax')(drop_2)

    optimizer = Adam(lr=learning_rate)
    model = Model(inputs=[my_input, previous_input], outputs=[output])
    # model.layers[1].trainable = False
    model.summary()
    # model.compile(loss='binary_crossentropy', optimizer='adam', metrics=["accuracy", f1])
    # learning rate 0.001
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=["accuracy", f1])

    return model


def create_bilstm_model(input_size, we_matrix, learning_rate, emb_training, output_dimension):
    input_dimension = we_matrix.shape[0]
    embedding_dimension = we_matrix.shape[1]

    my_input = Input(shape=(input_size,), name='input')

    emb = Embedding(input_dim=input_dimension, weights=[we_matrix], output_dim=embedding_dimension,
                    input_length=input_size, trainable=emb_training)(my_input)

    reshape = Reshape((input_size, embedding_dimension))(emb)
    lstm = Bidirectional(LSTM(100, activation="relu", recurrent_dropout=0.2, return_sequences=False))(reshape)
    d_1 = Dense(256, activation='relu')(lstm)

    drop_2 = Dropout(0.2)(d_1)
    output = Dense(output_dimension, name='output', activation='softmax')(drop_2)

    optimizer = Adam(lr=learning_rate)
    model = Model(inputs=[my_input], outputs=[output])
    # model.layers[1].trainable = False
    model.summary()
    # model.compile(loss='binary_crossentropy', optimizer='adam', metrics=["accuracy", f1])
    # learning rate 0.001
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=["accuracy", f1])

    return model

### CNN model
def create_cnn_model(input_size, we_matrix, learning_rate, emb_training, output_dimension):
    input_dimension = we_matrix.shape[0]
    embedding_dimension = we_matrix.shape[1]

    kernel = (4, 1)
    my_input = Input(shape=(input_size,), name='input')

    emb = Embedding(input_dim=input_dimension, weights=[we_matrix], output_dim=embedding_dimension,
                    input_length=input_size, trainable=emb_training)(my_input)

    reshape = Reshape((input_size, embedding_dimension, 1))(emb)
    conv = Conv2D(40, kernel, activation='relu', padding='valid')(reshape)
    pool = MaxPooling2D(pool_size=(input_size - kernel[0] + 1, 1))(conv)
    drop_1 = Dropout(0.2)(pool)
    flat = Flatten()(drop_1)
    d_1 = Dense(256, activation='relu')(flat)

    drop_2 = Dropout(0.2)(d_1)
    output = Dense(output_dimension, name='output', activation='softmax')(drop_2)

    optimizer = Adam(lr=learning_rate)
    model = Model(inputs=my_input, outputs=[output])
    model.summary()
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=["accuracy", f1])

    return model


def precision(y_true, y_pred):
    """Precision metric.
    Only computes a batch-wise average of precision.
    Computes the precision, a metric for multi-label classification of
    how many selected items are relevant.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    p = true_positives / (predicted_positives + K.epsilon())
    return p

def recall(y_true, y_pred):
    """Recall metric.
    Only computes a batch-wise average of recall.
    Computes the recall, a metric for multi-label classification of
    how many relevant items are selected.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    r = true_positives / (possible_positives + K.epsilon())
    return r

def f1(y_true, y_pred):
    prec = precision(y_true, y_pred)
    rec = recall(y_true, y_pred)
    return 2 * ((prec * rec) / (prec + rec))


def train_model(model, x_train, y_train, x_dev, y_dev, batch_size, epoch_count, plot, plot_filename, validation_res_filename):
    cb_list = []
    '''
    Apply early stopping on validation loss
    '''
    if epoch_count == 0:
        print("Training model with early stopping")
        es = EarlyStopping(monitor='val_accuracy', mode='auto', verbose=1, patience=50)
        cb_list.append(es)
        mc = ModelCheckpoint("best_model.h5", monitor="val_accuracy", save_best_only=True, mode="auto", verbose=1)
        cb_list.append(mc)
        nb_epoch = 500
    else:
        nb_epoch = epoch_count

    print("Training model...")
    history = model.fit(x_train, y_train, validation_data=[x_dev, y_dev], batch_size=batch_size, epochs=nb_epoch, callbacks=cb_list)
    print("Trained model successfully saved")
    '''
            Load the best saved model weights
            '''
    if epoch_count == 0:
        model.load_weights("best_model.h5")

    return model


def predict(model, x_test):
    print("Predicting values...")
    y_vectors = model.predict(x_test)
    return y_vectors

def evaluate_model_with_previous(model, x_test, x_test_previous, y_test):
    print("Evaluating model")
    print(x_test.shape, x_test_previous.shape, y_test.shape)
    scores = model.evaluate([x_test, x_test_previous], [y_test])
    print('Test loss:', scores[0])

    predicted = model.predict([x_test, x_test_previous])
    predicted = np.argmax(predicted, axis=1)
    Y_test = np.argmax(y_test, axis=1)

    acc = f1_score(Y_test, predicted, average="micro")
    f_macro_measure = f1_score(Y_test, predicted, average="macro")
    print('Test accuracy:', acc)
    print('Test fmeasure:', f_macro_measure)
    return acc, f_macro_measure

def evaluate_model_xlist(model, xlist, y_test):
    print("Evaluating model")
    scores = model.evaluate(xlist, [y_test])
    print('Test loss:', scores[0])

    predicted = model.predict(xlist)
    predicted = np.argmax(predicted, axis=1)
    Y_test = np.argmax(y_test, axis=1)

    acc = f1_score(Y_test, predicted, average="micro")
    f_macro_measure = f1_score(Y_test, predicted, average="macro")
    print('Test accuracy:', acc)
    print('Test fmeasure:', f_macro_measure)
    return acc, f_macro_measure

def evaluate_model(model, x_test, y_test):
    print("Evaluating model")
    print(x_test.shape, y_test.shape)
    scores = model.evaluate([x_test], [y_test])
    print('Test loss:', scores[0])

    predicted = model.predict(x_test)
    predicted = np.argmax(predicted, axis=1)
    Y_test = np.argmax(y_test, axis=1)

    acc = f1_score(Y_test, predicted, average="micro")
    f_macro_measure = f1_score(Y_test, predicted, average="macro")
    print('Test accuracy:', acc)
    print('Test fmeasure:', f_macro_measure)
    return acc, f_macro_measure

def save_keras_model(model, path):
    model.save(path)
    print("Model ", path, " has been saved...")

def load_keras_model(path):
    model = load_model(path, custom_objects={'f1': f1})
    print("Model ", path, " loaded")
    return model



def replace_intermediate_layer_in_keras(model, layer_id, new_layer):
    from keras.models import Model

    layers = [l for l in model.layers]

    x = layers[0].output
    for i in range(1, len(layers)):
        if i == layer_id:
            x = new_layer(x)
        else:
            x = layers[i](x)

    new_model = Model(input=layers[0].input, output=x)
    return new_model
