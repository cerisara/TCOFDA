"""
This code runs dialogue act recognition experiments on the French corpus
with the multi-head self attention model

Copyright Christophe Cerisara & Jiri Martinek

"""

import nn_model
import w2v_utils
import data_loader
import numpy as np
import config
import os
import pickle
from keras.layers import Dense
from keras.layers.merge import concatenate
from keras.engine import Model
from keras.optimizers import Adam, SGD
import keras
from keras_multi_head import MultiHeadAttention
from sklearn.metrics import f1_score
from keras.callbacks import EarlyStopping, ModelCheckpoint
from collections import Counter

# ========================================

# This method has been used to train
# the initial DA models on the (big) English Verbmobil corpus
# The resulting model is available in this repository: ./bert_en_models.h5
# So you don't need to train it on your own.
# This model will later be used for transfer learning to French.

def trainBigBERT_EN():
    # These vectors have been computed with BERT-Large on the (current,previous) sentence from the Verbmobil-EN corpus
    # We do neither distribute the Verbmobil corpus here because of license, nor do we distribute BERT-Large because
    # of space, but we do distribute the English model that is the result of this training
    train_en_vectors = w2v_utils.load_bin_data("EN_train_bert_vectors.bin")
    train_en_vectors_previous = w2v_utils.load_bin_data("EN_train_previous_sentences_bert_vectors.bin")
    test_en_vectors = w2v_utils.load_bin_data("EN_test_bert_vectors.bin")
    test_en_vectors_previous = w2v_utils.load_bin_data("EN_test_previous_sentences_bert_vectors.bin")

    keras_model = nn_model.create_no_lstm_bert_model_with_previous_sentence(1024, 0.002, len(config.categories))
    # train_X_verb_en = (9599, 15)
    print(train_en_vectors.shape)
    print(train_en_vectors_previous.shape)
    print(train_en_y.shape)
    print(test_en_vectors.shape)
    print(test_en_vectors_previous.shape)
    keras_model = nn_model.train_model(keras_model, x_train=[train_en_vectors, train_en_vectors_previous],
                                                y_train=train_en_y,
                                                x_dev=[test_en_vectors, test_en_vectors_previous],
                                                y_dev=test_en_y,
                                                batch_size=100, epoch_count=20, plot="yes",
                                                plot_filename="berta.pdf",
                                                validation_res_filename="bert_val_data.txt")
    nn_model.save_keras_model(keras_model, "bert_en_models.h5")

# ========================================

# Then, you need to translate from French to English, with any automatic translation system you want
# (we used Google translate) the french corpus in ../FrenchDialogActCorpus
# The result of this translation is available in the file ./FR2EN.txt

# Then you may pass FR2EN.txt into BERT-Large to get the following files
# (which we distribute in this corpus):

french_vectors = w2v_utils.load_bin_data("FR_bert_vectors.bin")
french_previous_vectors = w2v_utils.load_bin_data("FR_bert_previous_sentences_vectors.bin")

# ========================================

# load french data
train_all_sentences_french, train_all_classes_french = [], []
with open("FR2EN.txt", "r", encoding="utf8") as f:
    lines = f.readlines()
    for line in lines:
        line_elements = line.split()
        label = data_loader.create_labels_one_hot_repre_french(line_elements[0])
        sentence = ""
        for word in line_elements[1:]:
            sentence = sentence + " " + word
        train_all_sentences_french.append(sentence)
        train_all_classes_french.append(label)

_, all_y_french = data_loader.create_data_verb(train_all_sentences_french, train_all_classes_french, data_loader.create_dummy_vocabulary(), 15, "yes")

class Voc:
    def __init__(self,withUNK=True):
        self.co = Counter()
        self.unk = withUNK

    def buildVoc(self,mot):
        self.co.update([mot])

    def finalize(self,n=-1):
        if n>0: self.co = self.co.most_common(n)
        self.co = dict(self.co)
        self.voc={}
        if self.unk: self.voc['UNK']=0
        for w in self.co:
            self.voc[w]=len(self.voc)

    def widx(self,w):
        if w in self.voc: return self.voc[w]
        return 0

    def list(self):
        s=' '.join(self.voc.keys())
        return s

orfeodata = []
lexvoc = Voc()
posvoc = Voc()
davoc = Voc(False)

def loadORFEO():
    global lexvoc, davoc, posvoc
    fichs = ("Cadeaux_bon_08.trs.lif.w+p.orfeo", "pipelette.trs.lif.w+p.orfeo", "Reso_rich_06.trs.lif.w+p.orfeo", "voyage_USA.trs.lif.w+p.orfeo",) 
    for fich in fichs:
        curda=""
        curturn=[]
        with open("../FrenchDialogActCorpus/"+fich+".da","r") as f: dalines=f.readlines()
        with open("../FrenchDialogActCorpus/"+fich,"r") as f: lines=f.readlines()
        assert len(dalines)==len(lines)
        for i in range(len(lines)):
            st=lines[i].strip().split("\t")
            if len(st)<4:
                if len(curturn)>0: orfeodata.append(curturn)
                curturn=[]
                continue
            mot = st[1]
            pos = st[3]
            st = dalines[i].strip().split(" ")
            if len(st)==3:
                if len(curturn)>0: orfeodata.append(curturn)
                curturn=[]
                curda = st[0]
                curda = mergeDAORFEO(curda)
                curturn.append(curda)
            if len(curturn)>0: curturn.append((mot,pos))
        if len(curturn)>0: orfeodata.append(curturn)

    for turn in orfeodata:
        davoc.buildVoc(turn[0])
        for w,p in turn[1:]:
            lexvoc.buildVoc(w)
            posvoc.buildVoc(p)
    lexvoc.finalize(100)
    posvoc.finalize()
    davoc.finalize()

def mergeDAORFEO(da):
    merger = {
            'BB':'B',
            'E':'B',
            'RP':'B',
            'QX':'QO',
            'RC':'RO',
            'RZ':'RO',
            'O':'S',
            'PS':'S',
            'THX':'G',
            }
    if da in merger: return merger[da]
    return da

def selfattFR():
    # this is to enable/disable "stacking":
    addFRinputs = True
    uttlen=15
    nvoc = len(lexvoc.voc)

    # I've checked that all_y_french==orfeodata in the same order
    folds = 10
    total_accuracy = 0
    for fold in range(0, folds):
        print("Fold ", fold)

        # load EN model that has been trained on the big Verb-EN DA corpus: so this corresponds to "fine-tuning" transfer learning
        bert_mlp_model_original = nn_model.load_keras_model("bert_en_models.h5")
        bert_mlp_model_original.summary()
        for layer in bert_mlp_model_original.layers:
            # takes as input the BERT EN vectors corresponding to the translation of the FR input
            if layer.name == "input": xin = layer.output
            if layer.name == "previous_input": pxin = layer.output
            # This is the last-but-1 layer
            if layer.name == "dense_1": z = layer.output

        # now the self-att model on the french part:
        frinput = keras.Input(shape=(uttlen,),name='frinput')
        emb = keras.layers.Embedding(input_dim=nvoc, output_dim=100, mask_zero=False)(frinput)
        mhs = MultiHeadAttention(head_num=100, name='Multi-Head')(emb)
        gmp = keras.layers.GlobalMaxPooling1D()(mhs)
        bn1 = keras.layers.BatchNormalization()(gmp)
        de1 = keras.layers.Dense(units=100,activation='relu',name='dense2')(bn1)

        # now merge/stack
        if addFRinputs:
            lastlayinput = concatenate([de1,z],name="concatz")
            allinputs = [xin,pxin,frinput]
        else:
            lastlayinput = z
            allinputs = [xin,pxin]

        stackout = Dense(12,activation='softmax',name="densez")(lastlayinput)
        model = Model(allinputs,[stackout])
        # in French, we must use exactly the same hyperparms than in german:
        opt = Adam(lr=0.002)
        model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
        model.summary()

        offset = len(all_y_french) // folds # --> 47
        first_index = offset * fold

        print("Test samples: ", str(first_index), " -- ", str(first_index + offset))

        fold_test_data_X_list = []
        fold_test_data_X_previous_list= []
        fold_test_data_FR_list = []
        fold_test_data_y_list = []

        for i in range(first_index, first_index + offset):
            fold_test_data_X_list.append(french_vectors[i])
            fold_test_data_FR_list.append([lexvoc.widx(w[0]) for w in orfeodata[i][1:]])
            fold_test_data_X_previous_list.append(french_previous_vectors[i])
            fold_test_data_y_list.append(all_y_french[i])

        for i in range(len(fold_test_data_FR_list)):
            wrds = fold_test_data_FR_list[i]
            if len(wrds)>uttlen:
                wrds = wrds[0:uttlen-2]+wrds[-2:]
            else:
                wrds += [0]*(uttlen-len(wrds))
            fold_test_data_FR_list[i] = wrds

        fold_test_data_X = np.array(fold_test_data_X_list)
        fold_test_data_FR = np.array(fold_test_data_FR_list, dtype=np.float32)
        fold_test_data_X_previous = np.array(fold_test_data_X_previous_list)
        fold_test_data_y = np.array(fold_test_data_y_list)

        fold_train_data_X_list = []
        fold_train_data_FR_list = []
        fold_train_data_X_previous_list = []
        fold_train_data_y_list = []

        for i in range(0, first_index):
            fold_train_data_X_list.append(french_vectors[i])
            fold_train_data_FR_list.append([lexvoc.widx(w[0]) for w in orfeodata[i][1:]])
            fold_train_data_X_previous_list.append(french_previous_vectors[i])
            fold_train_data_y_list.append(all_y_french[i])

        for i in range(first_index+offset, len(all_y_french)):
            fold_train_data_X_list.append(french_vectors[i])
            fold_train_data_FR_list.append([lexvoc.widx(w[0]) for w in orfeodata[i][1:]])
            fold_train_data_X_previous_list.append(french_previous_vectors[i])
            fold_train_data_y_list.append(all_y_french[i])

        for i in range(len(fold_train_data_FR_list)):
            wrds = fold_train_data_FR_list[i]
            if len(wrds)>uttlen:
                wrds = wrds[0:uttlen-2]+wrds[-2:]
            else:
                wrds += [0]*(uttlen-len(wrds))
            fold_train_data_FR_list[i] = wrds

        fold_train_data_X = np.array(fold_train_data_X_list)
        fold_train_data_FR = np.array(fold_train_data_FR_list, dtype=np.float32)
        # fold_train_data_FR = np.expand_dims(fold_train_data_FR, axis=2)
        fold_train_data_X_previous = np.array(fold_train_data_X_previous_list)
        fold_train_data_y = np.array(fold_train_data_y_list)

        if addFRinputs:
            xtrin = [fold_train_data_X, fold_train_data_X_previous, fold_train_data_FR]
            xdevin = [fold_test_data_X, fold_test_data_X_previous, fold_test_data_FR]
            xtstin = [fold_test_data_X, fold_test_data_X_previous, fold_test_data_FR]
        else:
            xtrin = [fold_train_data_X, fold_train_data_X_previous]
            xdevin = [fold_test_data_X, fold_test_data_X_previous]
            xtstin = [fold_test_data_X, fold_test_data_X_previous]

        # model.fit(xtrin, fold_train_data_y)
        history = model.fit(xtrin, fold_train_data_y, validation_data=[xdevin,fold_test_data_y], batch_size=10, epochs=50, shuffle=True, callbacks=[], validation_freq=10)

        acc, f_measure = nn_model.evaluate_model_xlist(model, xtstin, fold_test_data_y)
        total_accuracy += acc
        model = None
    testacc = (total_accuracy / folds)
    return testacc, 0.

# ##########################"

# run 10 times the same XP because of variability in results
sacc = 0.
finacc = 0.
loadORFEO()
for i in range(10):
    tstacc, trloss = selfattFR()
    print("finrun %d trloss %f tstacc %f" % (i,trloss,tstacc))
    sacc += tstacc
sacc /= 10.

print("finacc %f %f" % (finacc,sacc))

