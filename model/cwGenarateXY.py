 #-*-coding:utf-8 -*-
import sys
import numpy as np
import codecs as cs
from wutils import GetToken2Index
from constants import l2i_dic,len_sentence,i2l_dic,l2i_dic18,i2l_dic18
import json

def label2index(labels,num,mask):
    for j in range(len(labels)):
        for k in range(len(labels[j])):
            if mask == 1:
                labels[j][k] = l2i_dic[labels[j][k]]
            elif mask == 2:
                labels[j][k] = l2i_dic18[labels[j][k]]
            if labels[j][k]>4:
                num+=1
    return labels,num


def GetXY(path,charvec, mask):
    f = cs.open(path, 'r', 'utf-8')
    text = f.read()
    f.close()
    Token2index = GetToken2Index(charvec, mask)
    labels = []
    tokens = []
    if '\r' in text:
        text = text.replace('\r\n\r\n\r\n', '\r\n\r\n')
        sentences = text.split(u'\r\n\r\n')[:-1]
    else:
        text = text.replace('\n\n\n', '\n\n')
        sentences = text.split(u'\n\n')[:-1]
    for i in range(len(sentences)):
        sentence_token = []
        sentence_label = []
        sentence = sentences[i]
        if '\r' in sentence:
            sentence = sentence.split(u'\r\n')[:-1]
        else:
            sentence = sentence.split(u'\n')[:-1]
        for k in range(len(sentence)):
            lable1 = sentence[k].split('\t')[1]
            sentence_label.append(lable1)
            word1 = sentence[k].split('\t')[0]
            try:
                sentence_token.append(Token2index[word1.lower()])
            except:
                sentence_token.append(0)
        labels.append(sentence_label)
        tokens.append(sentence_token)
    if mask == 1:
        labelindexs, num = label2index(labels, 0,1)
    elif mask == 2:
        labelindexs, num = label2index(labels, 0,2)
    for i in range(len(labelindexs)):
        if len(labelindexs[i]) >= len_sentence:
            labelindexs[i] = labelindexs[i][0:len_sentence]
            tokens[i] = tokens[i][0:len_sentence]
        else:
            k = len(labelindexs[i])
            for j in range(len_sentence - k):
                labelindexs[i].append(0)
                tokens[i].append(0)

    return np.array(tokens), np.array(labelindexs)

