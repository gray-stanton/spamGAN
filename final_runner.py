import os
import sys
import importlib
import stepGAN_train
import texar
import random
BASEDIR = './opspam_final/out/'

def get_config_file(trp, usp):
    if usp == -1:
        return 'stepGAN_base_config_nogan'
    if usp == 0.0:
        return 'stepGAN_base_config_nounsup'
    if usp == 0.5 or usp == 0.6:
        return 'stepGAN_base_config_smallunsup'
    if usp == 0.7 or usp == 0.8:
        return 'stepGAN_base_config_smallunsup'
    if usp == 0.9 or usp == 1.0:
        return 'stepGAN_base_config_smallunsup'

unsup_revs_path = './chicago_unlab_reviews.txt'

train_revs = './opspam_train_reviews.txt'
train_labs = './opspam_train_labels.txt'
test_revs = './opspam_test_reviews.txt'
test_labs = './opspam_test_labels.txt'

def make_data(trp, usp, run):
    nogan = False
    if usp == -1:
        usp = 0.0
        nogan = True
    with open(train_revs, 'r') as f:
        revs = f.readlines()
    with open(train_labs, 'r') as f:
        labs = f.readlines()
    
    shfl_idx = random.sample(list(range(len(revs))), len(revs))
    revs = [str(revs[i]) for i in shfl_idx]
    labs = [str(labs[i]) for i in shfl_idx]

    tr = revs[:round(trp *len(revs))]
    vr = revs[round(trp * len(revs)):]
    tl = labs[:round(trp * len(revs))]
    vl = labs[round(trp * len(revs)):]
 
    if len(vr) == 0 or trp == 1.0:
        # just add a fake as a workaround
        vr = revs[0:100]
        vl = labs[0:100]
    with open(unsup_revs_path, 'r') as f:
        unsup_revs_full = f.readlines()
    random.shuffle(unsup_revs_full)
    unsup_revs = unsup_revs_full[:round(usp * len(unsup_revs_full))]

    unsup_labs = ['-1\n'] * len(unsup_revs)


    dir_name = 'tr{}_usp{}_{}'.format(int(trp*100), int(usp * 100), run)
    if nogan:
        dir_name = dir_name + '_nogan/'
    os.mkdir(os.path.join(BASEDIR, dir_name))
    curdir = os.path.join(BASEDIR, dir_name)
    data_paths = {
        'train_data_reviews' : os.path.join(curdir, 'trevs.txt'),
        'train_data_labels'  : os.path.join(curdir, 'tlabs.txt'),
        'val_data_reviews' : os.path.join(curdir, 'vrevs.txt'),
        'val_data_labels' : os.path.join(curdir, 'vlabs.txt'),
        'unsup_train_data_reviews' : os.path.join(curdir, 'unsup_trevs.txt'),
        'unsup_train_data_labels' : os.path.join(curdir, 'unsup_tlabs.txt'),
        'vocab' : os.path.join(curdir, 'vocab.txt'),
        'clas_test_ckpt' : os.path.join(curdir, 'ckpt-bestclas'),
        'clas_pred_output' : os.path.join(curdir, 'testpreds.txt'),
        'dir' : curdir
    }


 
    with open(data_paths['train_data_reviews'], 'w') as f: 
        for x in tr: 
            f.write(x)

    with open(data_paths['train_data_labels'], 'w') as f:
        for x in tl:
            f.write(str(x))
 
    with open(data_paths['unsup_train_data_reviews'], 'w') as f: 
        for x in unsup_revs: 
            f.write(x)
  
    with open(data_paths['unsup_train_data_labels'], 'w') as f:
        for x in unsup_labs:
            f.write(str(x))

    with open(data_paths['val_data_reviews'], 'w') as f:
        for x in vr:
            f.write(x)

    with open(data_paths['val_data_labels'], 'w') as f:
        for x in vl:
            f.write(str(x))


    vocab = texar.data.make_vocab([train_revs, test_revs, data_paths['unsup_train_data_reviews']], 10000)

    with open(data_paths['vocab'], 'w') as f:
        for v in vocab:
            f.write(v + '\n')

    return data_paths

# 0.5, 0.8 x 0.5, 0.8
for train_pcent in [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]:
    for unsup_pcent in [-1, 0.0, 0.5, 0.7, 1.0]:
        for run in range(0, 10):
            base_config_file = get_config_file(train_pcent, unsup_pcent)
            data_paths = make_data(train_pcent, unsup_pcent, run)
            importlib.invalidate_caches()
            base_config = importlib.import_module(base_config_file)
            base_config = importlib.reload(base_config)
            # inject file paths
            base_config.train_data['datasets'][0]['files'] = [data_paths['train_data_reviews'],
                                                              data_paths['unsup_train_data_reviews']]
            base_config.train_data['datasets'][1]['files' ] = [data_paths['train_data_labels'],
                                                               data_paths['unsup_train_data_labels']]
            base_config.clas_train_data['datasets'][0]['files'] = data_paths['train_data_reviews']
            base_config.clas_train_data['datasets'][1]['files'] = data_paths['train_data_labels']
            base_config.val_data['datasets'][0]['files'] = data_paths['val_data_reviews']
            base_config.val_data['datasets'][1]['files'] = data_paths['val_data_labels']
            base_config.test_data['datasets'][0]['files'] = test_revs
            base_config.test_data['datasets'][1]['files'] = test_labs
            base_config.train_data['datasets'][0]['vocab_file'] = data_paths['vocab']
            base_config.clas_train_data['datasets'][0]['vocab_file'] = data_paths['vocab']
            base_config.val_data['datasets'][0]['vocab_file'] = data_paths['vocab']
            base_config.test_data['datasets'][0]['vocab_file'] = data_paths['vocab']
            base_config.clas_test_ckpt = data_paths['clas_test_ckpt']
            base_config.clas_pred_output = data_paths['clas_pred_output']
            base_config.log_dir = data_paths['dir']
            base_config.checkpoint_dir = data_paths['dir']
            print(base_config.train_data['datasets'][0]['files'])
            print('Train Pcent {} Unsup Pcent {} Run {}'.format(train_pcent, unsup_pcent, run))
            # Run
            stepGAN_train.main(base_config)
            # Run test
            base_config.clas_test = True
            stepGAN_train.main(base_config)




