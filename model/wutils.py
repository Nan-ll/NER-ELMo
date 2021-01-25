# -*- coding: utf-8 -*-

import codecs as cs
import h5py
import numpy as np
from constants import i2l_dic18, i2l_dic
from constants import len_sentence,wv,cv


import os
from bilm import Batcher,BidirectionalLanguageModel,TokenBatcher
from bilm.elmo import weight_layers
import tensorflow as tf
import copy

def Output_BIOES(file,testtokens,testlabel,pre_label):
    foutput = cs.open(file, 'w', 'utf-8')
    for line, ans, per in zip(testtokens, testlabel, pre_label):
        for sline, sans, sper in zip(line, ans, per[:len(line)]):
            if sline==' ':
                sline='*'
            foutput.write(str(sline) + ' '+str(sans)+' ' + str(i2l_dic18[sper]) + '\n')
        foutput.write('\n')
    foutput.close()

def chinese_to_stroke(sentence_list):
    replace_dcit={'一':'1', 'フ': '2', 'ノ': '3', '丨': '4', '丶': '5'}
    table = {ord(f):ord(t) for f,t in zip(u'，。！？、“”《》【】（）％＃＠＆１２３４５６７８９０', u',.!?,""<>[]()%#@&1234567890')}
    table[ord("‘")]=ord("'")
    table[ord("’")]=ord("'")
    infile='./chinese_all_feats_from_handian.txt'
    fin=open(infile,'r',encoding='utf-8')
    handi_dict={}
    line=next(fin)
    for line in fin:
        segs=line.strip().split('\t')
        if len(segs)>7:
            handi_dict[segs[0]]=segs[1]+'@'+segs[-4]+'@'+segs[-1]
    print(len(handi_dict))
    fin.close()
    num=0
    new_list=copy.deepcopy(sentence_list)
    for sen_i in range(len(new_list)):
        num+=1
        if num%10000==0:
            sys.stdout.flush()
        for tok_i in range(len(new_list[sen_i])):
            tokens=new_list[sen_i][tok_i].translate(table)
            if tokens.isdigit():
                tokens='0'
            elif tokens >= '\u4e00' and tokens <= '\u9fa5':
                if tokens in handi_dict.keys():
                    strout=''
                    segs=handi_dict[tokens].split('@')
                    for j in range(len(segs[0])):
                        strout=strout+replace_dcit[segs[0][j]]
                    tokens=strout
                else:
                    tokens='<UNK>'
            new_list[sen_i][tok_i]=tokens
    return(new_list)


def ELMO_MR_stroke(sentences):
    batch_size = 20
    use_top_only=False
    vocab_file = os.path.join('../chinese_elmo/stroke-elmo/vocab_ccks+chip+ch-md_word.stroke')
    options_file = os.path.join('../chinese_elmo/stroke-elmo/options.json')
    weight_file = os.path.join('../chinese_elmo/stroke-elmo/weights.hdf5')
    input_ids = tf.placeholder('int32', shape=(None, None, 50))
    model = BidirectionalLanguageModel(options_file, weight_file)
    bilm_ops = model(input_ids)
    session = tf.Session()
    session.run(tf.global_variables_initializer())
    batcher = Batcher(vocab_file, 50)
    sentence_ids = batcher.batch_sentences(sentences,len_sentence)
    print('***********')
    lm_embeddings= session.run(bilm_ops,feed_dict={input_ids: sentence_ids[0:batch_size]})
    lm_embeddings = lm_embeddings['lm_embeddings']
    lm_embeddings = np.transpose(lm_embeddings,(0, 2, 3, 1))
    print('$$$$$$$$$$$$$$$$$$')
    for batch in range(1,int(len(sentences)/batch_size)+1):
        print('batch:'+str(batch))
        if (batch+1)*batch_size<=len(sentences):
            lm_embedding= session.run(bilm_ops,feed_dict={input_ids: sentence_ids[batch*batch_size:(batch+1)*batch_size]})
        else:
            lm_embedding= session.run(bilm_ops,feed_dict={input_ids: sentence_ids[batch*batch_size:len(sentences)]})
        lm_embedding = lm_embedding['lm_embeddings']
        lm_embedding = np.transpose(lm_embedding, (0, 2, 3, 1))
        lm_embeddings=np.vstack((lm_embeddings,lm_embedding))
    print('elmo_done!')
    return lm_embeddings

def ypre2label(ypredict):
    predict_label = []
    y_predict = ypredict.tolist()
    for i in range(len(y_predict)):
        sentence_label = []
        for j in range(len(y_predict[i])):
            x=y_predict[i][j]
            maxindex = y_predict[i][j].index(max(y_predict[i][j]))
            sentence_label.append(maxindex)
        predict_label.append(sentence_label)
    return predict_label


def newcomputeFe(gold_entity,pre_entity):
    truenum = 0
    prenum = 0
    goldnum = 0
    if len(pre_entity)!=gold_entity:
        precise = 0
        recall = 0
        f = 0
    for i in range(len(gold_entity)):
        goldnum += len(gold_entity[i])
        prenum  += len(pre_entity[i]) 
        if len(gold_entity[i]) == 0 or len(pre_entity[i]) == 0:
            continue
        else:
            for pre in pre_entity[i]:
                for gold in gold_entity[i]:
                    if pre[0] == gold[0] and pre[1] == gold[1] and pre[2] == gold[2]:
                        truenum +=1
                        break
    try:
        precise = float(truenum) / float(prenum)
        recall = float(truenum) / float(goldnum)
        f = float(2 * precise * recall /( precise + recall))
    except:
        precise = recall = f = 0
    print('本轮实体的准确率是%f %f %f' %(precise,recall,f))
    return precise,recall,f

def GetModel(filepath,mask):
    model = []
    fp = cs.open(filepath,'r','utf-8')
    content = fp.readlines()[1:]
    fp.close()
    if mask == 1:
        if mask:
            word = [0 for i in range(cv)]
            model.append(word)
        for each in content:
            word = []
            each = each.split(' ')
            #each = each.split('\t')
            for i in range(1,cv+1):
                word.append(float(each[i]))
            model.append(word)
    elif mask == 2:
        if mask:
            word = [0 for i in range(wv)]
            model.append(word)
        for each in content:
            word = []
            each = each.split(' ')
            for i in range(1,wv+1):
                word.append(float(each[i]))
            model.append(word)
    return np.array(model)

def GetToken2Index(filepath,mask):
    token2index = {}
    fp = cs.open(filepath,'r','utf-8')
    content = fp.readlines()[1:]
    fp.close()
    for i in range(len(content)):
        each = content[i].split(' ')
        if mask:
            token2index[each[0]] = i + 1 
        else:
            token2index[each[0]] = i
    return token2index

def save_model(address,model):
    f = h5py.File(address,'w')
    weight = model.get_weights()
    for i in range(len(weight)):
        f.create_dataset('weight' + str(i),data = weight[i])
    f.close()

def load_model(address, model):
    f = h5py.File(address, 'r')
    weight = []
    for i in range(len(f.keys())):
        weight.append(f['weight' + str(i)][:])
    model.set_weights(weight)

def SaveGoldEntity(path,goldentity):
    fp = cs.open(path,'w','utf-8')
    for sen in goldentity:
        num = len(sen)
        if num == 0:
            fp.write('\n')
            continue
        for i in range(num):
            entity = sen[i]
            if i == (num -1):
                fp.write(str(entity[0]) + '\t' + str(entity[1]) + '\t' + str(entity[2])+'\n')
            else:
                fp.write(str(entity[0]) + '\t' + str(entity[1]) + '\t' + str(entity[2]) + '\t')
    fp.close()

def LoadGoldEntity(path):
    fp = cs.open(path,'r','utf-8')
    goldentity = []
    text = fp.read().split('\n')[0:-1]
    for sen in text:
        if len(sen)==0:
            goldentity.append([])
            continue
        locs = sen.split('\t')
        senentity = []
        for i in range(0,len(locs),2):
            senentity.append([int(locs[i]),int(locs[i+1])])
        goldentity.append(senentity)
    return goldentity

def loadtokens(path):
    f = cs.open(path,'r','utf-8')
    text = f.read()
    f.close()
    tokens = []
    lableresult = []
    if '\r' in text:
        sentences = text.split(u'\r\n\r\n')[:-1]
    else:
        sentences=text.split(u'\n\n')[:-1]
    for i in range(len(sentences)):
        x=len(sentences)
        sentence=[]
        sentence_token = []
        sentence_lable = []
        sentence = sentences[i]
        if '\r' in text:
           sentence = sentence.split('\r\n')
        else:
           sentence = sentence.split('\n')
        for k in range(len(sentence)):
            word1 = sentence[k].split('\t')[0]
            label = sentence[k].split('\t')[1]
            sentence_token.append(word1)
            sentence_lable.append(label)
        tokens.append(sentence_token)
        lableresult.append(sentence_lable)
    return tokens,lableresult

