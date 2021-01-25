
#encoding=utf8
import sys
import numpy as np
import codecs as cs
import keras
import subprocess
from keras.models import Sequential,Model
from keras.layers import Dense,Bidirectional,Embedding,Dropout,Input,Activation,TimeDistributed,Conv1D,LSTM
from keras.utils import np_utils
from keras.optimizers import RMSprop
from ChainCRF import ChainCRF
from cwGenarateXY import GetXY,label2index
from wutils import loadtokens,GetModel,chinese_to_stroke,ELMO_MR_stroke,ypre2label,Output_BIOES,newcomputeFe
from constants import num_class,bils,ls,len_sentence,i2l_dic,l2i_dic18,num_class18
import pickle
from keras.layers.merge import Concatenate, Add
from highway import *
from conv1d_highway import *
from ELMoLayer import ELMo
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF 

if __name__ == '__main__':
    data_set={'18data':1,
              '19data':1}
    fea_dict = {'char': 1,
                'elmo': 1}
    mask = 0
    dicmodel = 25
    dicvec = 50
    model_type = 'LSTM'
    batchsize = 32
    
    bmpath = u'../output/LSTM19.h5'
    bmpath18 = u'../output/LSTM18.h5'
    outputfile18 = u'../output/FS-18.txt'
    outputfile19 = u'../output/FS-19.txt'

    charvecmodel=GetModel(u'../w2v/ccks200.vec',1)
    charfile = u'../w2v/ccks200.vec'

    devtokens, devlabel = loadtokens(u'../data/dev_BIOES.txt')
    traintokens_all, trainlabel = loadtokens(u'../data/train_BIOES.txt')
    xtrain_all, y_train_all = GetXY(u'../data/train_BIOES.txt',charfile, 1)

    x_dtest, y_dtest = GetXY(u'../data/dev_BIOES.txt',charfile, 1)
    ori_testlabel19 = copy.deepcopy(devlabel)

    numtrain = len(y_train_all)
    ytrain_all = np_utils.to_categorical(y_train_all, num_class)
    ytrain_all = np.reshape(ytrain_all, (numtrain, len_sentence, num_class))
    nytest = ypre2label(ytrain_all)
    xtrain = []
    ytrain = []
    xtrain_all = xtrain_all.tolist()
    ytrain_all = ytrain_all.tolist()
    vaildtokens = []
    traintokens = []
    for i in range(len(xtrain_all)):
        xtrain.append(xtrain_all[i])
        ytrain.append(ytrain_all[i])
    xtrain = np.array(xtrain)
    ytrain = np.array(ytrain)

    #18
    traintokens_all18, trainlabel18 = loadtokens(u'../data/train_BIOES_18.txt')
    devtokens18, devlabel18 = loadtokens(u'../data/dev_BIOES_18.txt')
    xtrain_all18, y_train_all18 = GetXY(u'../data/train_BIOES_18.txt',charfile, 2)
    x_dtest18, y_dtest18 = GetXY(u'../data/dev_BIOES_18.txt',charfile, 2)
    ori_testlabel18 = copy.deepcopy(devlabel18)

    numtrain18 = len(y_train_all18)
    ytrain_all18 = np_utils.to_categorical(y_train_all18, num_class18)
    ytrain_all18 = np.reshape(ytrain_all18, (numtrain18, len_sentence, num_class18))
    xtrain18 = []
    ytrain18 = []
    xtrain_all18 = xtrain_all18.tolist()
    ytrain_all18 = ytrain_all18.tolist()
    for i in range(len(xtrain_all18)):
        xtrain18.append(xtrain_all18[i])
        ytrain18.append(ytrain_all18[i])
    xtrain18 = np.array(xtrain18)
    ytrain18 = np.array(ytrain18)

    input_train18 = []
    input_dev18 = []
    input_train19 = []
    input_dev19 = []

    if fea_dict['char'] == 1:
        input_train19.append(xtrain)
        input_dev19.append(x_dtest)
        input_dev18.append(x_dtest18)
        input_train18.append(xtrain18)
    if fea_dict['elmo'] == 1:
        stroke_train18 = chinese_to_stroke(traintokens_all18)
        stroke_train19 = chinese_to_stroke(traintokens_all)
        stroke_dev19 = chinese_to_stroke(devtokens)
        stroke_dev18 = chinese_to_stroke(devtokens18)

        elmoembed_train18 = ELMO_MR_stroke(stroke_train18)
        tf.get_variable_scope().reuse_variables()
        elmoembed_train19 = ELMO_MR_stroke(stroke_train19)
        elmoembed_dev19 = ELMO_MR_stroke(stroke_dev19)
        elmoembed_dev18 = ELMO_MR_stroke(stroke_dev18)

        input_dev18.append(elmoembed_dev18)
        input_train18.append(elmoembed_train18)
        input_train19.append(elmoembed_train19)
        input_dev19.append(elmoembed_dev19)

    all_input = []
    all_input18 = []
    fea_list = []
    if fea_dict['char'] == 1:
        char_input = Input(shape=(len_sentence,), dtype='int32', name='char_input')
        all_input.append(char_input)
        char_fea = Embedding(charvecmodel.shape[0], charvecmodel.shape[1], weights=[charvecmodel], trainable=True,
                             mask_zero=mask, input_length=len_sentence, name='char_emd')(char_input)
        fea_list.append(char_fea)
    if fea_dict['elmo'] == 1:
        elmo_input = Input(shape=(len_sentence, 512, 3), dtype='float32', name='elmo_input')
        all_input.append(elmo_input)
        elmo_fea = ELMo(name='elo_fea')(elmo_input)
        fea_list.append(elmo_fea)
    if len(fea_list) == 1:
        concate_vec = fea_list[0]
    else:
        concate_vec = Concatenate()(fea_list)
    concate_vec = Dropout(0.5)(concate_vec)

    # model
    if model_type == 'LSTM':
        bilstm_shared = Bidirectional(LSTM(400, return_sequences=True, implementation=2, dropout=0.5, recurrent_dropout=0.5), name='bilstm_shared')
        bilstm = bilstm_shared(concate_vec)
        dense_shared = TimeDistributed(Dense(200, activation='tanh'), name='dense_shared')
        dense = dense_shared(bilstm)

    elif model_type == 'CNN':
        cnn_model1 = Conv1D(200, 3, padding='same', activation='relu')(concate_vec)
        dense_shared = TimeDistributed(Dense(200, activation='tanh'), name='dense_shared')
        dense = dense_shared(cnn_model1)

    dense = Dropout(0.5)(dense)
    if  data_set['18data']==1:
        crf_num18=num_class18
        dense18 = TimeDistributed(Dense(crf_num18, activation=None), name='dense2_c18')(dense)
        crf18 = ChainCRF(name='crf_out_c18')
        output18 = crf18(dense18)
    if data_set['19data']==1:
        crf_num19=num_class
        dense19 = TimeDistributed(Dense(crf_num19, activation=None), name='dense2_c19')(dense)
        crf19 = ChainCRF(name='crf_out_c19')
        output19 = crf19(dense19)

    model18 = Model(inputs=all_input, outputs=output18)

    opt18 = RMSprop(lr=0.0005, rho=0.9, epsilon=1e-06)
    model18.compile(loss=crf18.loss, optimizer=opt18, metrics=['accuracy'])
    model18.summary()
    
    model19 = Model(inputs=all_input, outputs=output19)
    opt19 = RMSprop(lr=0.0005, rho=0.9, epsilon=1e-06)
    model19.compile(loss=crf19.loss, optimizer=opt19, metrics=['accuracy'])
    model19.summary()

    # 为每组实验记录F值
    dev_best_epoch18 = 0
    dev_best_epoch19 = 0

    best_devf19 = 0.0
    best_devf18 = 0.0
    
    for i in range(70):
        labelresult = []
        numresult = []
        if data_set['18data']==1:
            model18.fit(input_train18,ytrain18,batch_size=batchsize, epochs=1)
            dev_predict18 = model18.predict(input_dev18, batch_size=32)
            devfpre_label18 = ypre2label(dev_predict18)

            Output_BIOES(outputfile18, devtokens18, ori_testlabel18, devfpre_label18)
            commond = './conlleval.pl < ' + outputfile18
            retval = subprocess.call(commond, shell=True)
            model18.save_weights(bmpath18)

        if data_set['19data']==1:
            model19.fit(input_train19,ytrain,batch_size=batchsize, epochs=1)
            y_predict = model19.predict(input_dev19, batch_size=32)
            devpre_label19 = ypre2label(y_predict)

            Output_BIOES(outputfile19, devtokens, ori_testlabel19, devpre_label19)
            commond = './conlleval.pl < ' + outputfile19
            retval = subprocess.call(commond, shell=True)
            model18.save_weights(bmpath)
