import texar as tx
import tensorflow as tf
import numpy as np
import os
import random
import re
random.seed(5)
def clean(s, char=False):
    ns = s.lower()
    ns = ns.replace('<br />', ' ')
    ns = re.sub('[0-9]+', 'N', ns)
    ns = re.sub('[^a-zA-Z0-9 \-.,\'\"!?()]', ' ', ns) # Eliminate all but these chars
    ns = re.sub('([.,!?()\"\'])', r' \1 ', ns) # Space out punctuation
    #if char:
    #    ns = re.sub('(\S)', r' \1 ', ns) # Space out all chars
    ns = re.sub('\s{2,}', ' ', ns) # Trim ws
    str.strip(ns)
    return ns

def split_valid(textpath, labpath, tr_outtxt, tr_outlab,
                val_outtxt, val_outlab, split_count ):
    with open(textpath, 'r') as txtf, open(labpath, 'r') as labf:
        texts = txtf.readlines()
        labs = labf.readlines()
    shfl_idx = random.sample(range(len(texts)), len(texts))
    texts = [clean(texts[i]) for i in shfl_idx]
    labs = [labs[i] for i in shfl_idx]

    val_texts = texts[:split_count]
    val_labs = labs[:split_count]
    train_texts = texts[split_count:]
    train_labs = labs[split_count:]
    with open(tr_outtxt, 'w') as txtf, open(tr_outlab, 'w') as labf:
        for r, l in zip(train_texts, train_labs):
            txtf.write(r + '\n')
            labf.write(l)
    with open(val_outtxt, 'w') as txtf, open(val_outlab, 'w') as labf:
        for r, l in zip(val_texts, val_labs):
            txtf.write(r + '\n')
            labf.write(l)








if __name__=='__main__':
    split_valid('./opspam_reviews.txt', './opspam_labels.txt',
                './opspam_train_reviews.txt', './opspam_train_labels.txt',
                './opspam_val_reviews.txt', './opspam_val_labels.txt',
                320)
    split_valid('./opspam_val_reviews.txt', './opspam_val_labels.txt',
                './opspam_val_reviews.txt', './opspam_val_labels.txt',
                './opspam_test_reviews.txt', './opspam_test_labels.txt',
                160)

    
    vocab_words = tx.data.make_vocab(['./opspam_train_reviews.txt',
                                      './opspam_val_reviews.txt',
                                     './opspam_test_reviews.txt'], max_vocab_size=10000)
    with open('opspam_vocab.txt', 'w') as vf:
        for v in vocab_words:
            vf.write(v + '\n')





    




    

