import texar as tx
import tensorflow as tf
import numpy as np
import os
import random
import re
random.seed(5)

def rewrite_tripadvisor(tafilepath, trainf, valf, testf):
    with open(tafilepath, 'r') as tarawfile,\
    open(trainf, 'w') as trainf,\
    open(valf, 'w') as valf,\
    open(testf, 'w') as testf:
        for l in tarawfile:
            cl = clean(l)
            randint = random.randint(1, 10)
            if randint == 1:
                valf.write(cl + '\n') # With newlines
            elif randint == 2:
                testf.write(cl + '\n')
            else:
                trainf.write(cl + '\n')
        # 80% train/10% val/ 10% test
    


def clean(s):
    ns = s.lower()
    ns = ns.replace('<br />', ' ')
    ns = re.sub('[0-9]+', 'N', ns)
    ns = re.sub('[^a-zA-Z0-9 \-.,\'\"!?()]', ' ', ns) # Eliminate all but these chars
    ns = re.sub('([.,!?()\"\'])', r' \1 ', ns) # Space out punctuation
    ns = re.sub('\s{2,}', ' ', ns) # Trim ws
    str.strip(ns)
    return ns


if __name__=='__main__':
    #rewrite_tripadvisor('/home/gray/code/seqgan-opinion-spam/data/raw/tripadvisor.txt',
    #                   '/home/gray/code/stepGAN/tripadvisor_train.txt',
    #                   '/home/gray/code/stepGAN/tripadvisor_val.txt',
    #                   '/home/gray/code/stepGAN/tripadvisor_test.txt'
    #                   )
    vocab_words = tx.data.make_vocab([
       # './tripadvisor_train.txt', 
        './tripadvisor_val.txt',
        './tripadvisor_test.txt'
    ],
        max_vocab_size=10000)
    with open('./tripadvisor_vocab.txt', 'w') as vf:
        for v in vocab_words:
            vf.write(v + '\n')

