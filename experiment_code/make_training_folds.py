import random
import texar
import os
SEED = 5 # 5
random.seed(SEED)
TRAIN_REVS = '/home/gray/code/stepGAN/opspam_train_reviews.txt'
TRAIN_LABS = '/home/gray/code/stepGAN/opspam_train_labels.txt'
UNSUP_REVS = '/home/gray/code/stepGAN/chicago_unlab_reviews50.txt'
OUTDIR = '/home/gray/code/stepGAN/opspam_final'



with open(TRAIN_REVS, 'r') as f:
    revs = f.readlines()

with open(TRAIN_LABS, 'r') as f:
    labs = f.readlines()

shfl_idx = random.sample(list(range(len(revs))), len(revs))
revs = [str(revs[i]) for i in shfl_idx]
labs = [str(labs[i]) for i in shfl_idx]

if UNSUP_REVS is not None: 
    vocab = texar.data.make_vocab([TRAIN_REVS, UNSUP_REVS], 10000)
else:
    vocab = texar.data.make_vocab([TRAIN_REVS], 10000)
vocab_path = os.path.join(OUTDIR, 'opspam_vocab.txt')
with open(vocab_path, 'w') as vf:
    for v in vocab:
        vf.write(v + '\n')


for labeled_prop in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
    tr = revs[:round(labeled_prop *len(revs))]
    vr = revs[round(labeled_prop * len(revs)):]
    tl = labs[:round(labeled_prop * len(revs))]
    vl = labs[round(labeled_prop * len(revs)):]

    train_rev_name = os.path.join(OUTDIR, 'train_' + str(round(100 * labeled_prop)) + '_reviews.txt')
    train_lab_name = os.path.join(OUTDIR, 'train_' + str(round(100 * labeled_prop)) + '_labels.txt')
    val_rev_name = os.path.join(OUTDIR, 'val_' + str(round(100 * labeled_prop)) + '_reviews.txt')
    val_lab_name = os.path.join(OUTDIR, 'val_' + str(round(100 * labeled_prop)) + '_labels.txt')
    
    if UNSUP_REVS is not None:
        with open(UNSUP_REVS, 'r') as f:
            unsup_revs = f.readlines()
        unsup_labs = [-1] * len(unsup_revs)
        unsup_labs = [str(u) + '\n' for u in unsup_labs]
        tr = tr + unsup_revs
        tl = tl + unsup_labs

    with open(train_rev_name, 'w') as f:
        for x in tr:
            f.write(str(x))
    with open(train_lab_name, 'w') as f:
        for x in tl:
            f.write(str(x))
    with open(val_rev_name, 'w') as f:
        for x in vr:
            f.write(str(x))
    with open(val_lab_name, 'w') as f:
        for x in vl:
            f.write(str(x))






    

