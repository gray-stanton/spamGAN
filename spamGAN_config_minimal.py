import tensorflow

# Overarching


gen_test = False # Whether training or testing the generator perplexity

clas_test = False # Whether training or testing the classifier test performance
clas_test_ckpt = None # Which checkpoint to use for classifier testing
clas_pred_output = None # Where to save classifier test predictions

# Saving/logging Config
restore_model= False # Whether to reinitialize or restore weights
clear_run_logs = False # Whether to delete prior run logs
log_dir= '/tmp/' # Where to store logs
checkpoint_dir= '/tmp' # Where to store ckpt files
load_checkpoint_file = None # Which checkpoint to load

# Logging frequency/verbosity
log_verbose_mle = True
log_verbose_rl = True
batches_per_summary = 100
batches_per_text_summary = 100

# Number of epochs to run
g_unlab_every_n = 20 # Balance of exposure to labeled/unlabeled datasets
g_pretrain_epochs = 1# 60
d_pretrain_epochs = 1# 60
d_pretrain_critic_epochs = 0#20
c_pretrain_epochs =1 # 20
adversarial_epochs = 1 # How many total adversarial epochs, for all components

# During adversarial training, how many epochs to run for discriminator/classifier
disc_adv = 1
clas_adv = 10

gen_adv_epoch = 4 # Number of generator adversarial epochs
g_unlab_every_n_adv = -1 # Frequency of generator ML epochs with unlabeled data
gen_mle_adv_epoch = 2 # Number of generator ML epochs with labeled data

adv_train_max_gen_examples = 1000 # Maximum size of training epoch for gen in adv
adv_disc_max_ex = 5000 # Maximum size of training epoch for disc in adv
adv_gen_train_with_unsup = False # Whether or not to use unlabeled examples in adv

# Early stopping parameters
gen_patience=20
gen_es_tolerance = 0.005
clas_es_tolerance = 0.005
clas_patience = 10

# Controls ML/generation max sentence length (in words)
max_decoding_length = 128
max_decoding_length_infer = 128
annealing_length = 128 # Can use shorter sentences for initial training
adversarial_length = 128 # Can use shorter sentences for adversarial training

use_unsup=False # Whether unsupervised data is used at all for this model
sampling_temperature = 1.0 # Sampling temperature in generation


linear_decay_pg_weights = True # Place more importance on initial sentence rewards

# Context configs
prior_prob=0.5 # probability of class 1 in generated/unlabeled data.
noise_size=10 # dim of noise vector


# Training tweaks
disc_label_smoothing_epsilon = 0.05 # label smoothing for discriminator

# Set to experiement with clipping policy gradients at various points
adv_max_clip = 100
min_log_prob = 0.1
max_log_prob = 100
min_pg_loss = -200
max_pg_loss = 200


add_sentence_progress = True # Includes indicator of sentence length (decays 1 to 0)
                             # avoids mode collapse to extremely short sentences

clas_loss_on_fake_lambda = 1.0 # Balancing param on real/generated clas
disc_crit_train_on_fake_only = True # Only train disc crit on generated sentences
clas_crit_train_on_fake_only = True # Only train disc crit on generated sentences

reward_blending = 'f1' # Additive vs f1 clas-disc reward blending

clas_min_ent_lambda = 1.0 # Controls strength of entropy minimization

clas_has_own_embedder = True # Pool or share embedders
disc_has_own_embedder = True

# Different loss functions
mle_loss_in_adv = True # Whether or not to include ML optimization in adversarial 

# Relative weighting of discriminator and classifier in pg loss
discriminator_loss_lambda = 1.0 
classifier_loss_lambda = 1.0

norm_advantages = True # Normalize advantages
let_discriminator_train_embedder = True # whether discriminator can update embedder


train_data = {
    "num_epochs": 1,
    "batch_size": 128,
    "allow_smaller_final_batch": True,
    "shuffle": True,
    "shuffle_buffer_size": None,
    "shard_and_shuffle": False,

    "num_parallel_calls": 1,
    "prefetch_buffer_size": 0,
    "max_dataset_size": -1,
    "seed": None,
    "name": "train_data",
    'datasets' : [ 
        {
            "files" : ['./minrun_train_reviews.txt'],
            'vocab_file' : './minrun_opspam_vocab.txt',
            'max_seq_length' : 128,
            'length_filter_mode' : 'truncate',
            'bos_token' : '<BOS>',
            'delimiter' : ' ',
            'eos_token' : '<EOS>',
            'data_name' : 'x',
            'pad_to_max_seq_length' : True
        },
        {
            'files' : './minrun_train_labels.txt',
            'data_type' : 'int',
            'data_name' : 'label'
        }
    ]
}

clas_train_data = {
    "num_epochs": 1,
    "batch_size": 128,
    "allow_smaller_final_batch": True,
    "shuffle": True,
    "shuffle_buffer_size": None,
    "shard_and_shuffle": False,

    "num_parallel_calls": 1,
    "prefetch_buffer_size": 0,
    "max_dataset_size": -1,
    "seed": None,
    "name": "train_data",
    'datasets' : [ 
        {
            "files" : ['./minrun_train_reviews.txt'],
            'vocab_file' : './minrun_opspam_vocab.txt',
            'max_seq_length' : 128,
            'length_filter_mode' : 'truncate',
            'bos_token' : '<BOS>',
            'delimiter' : ' ',
            'eos_token' : '<EOS>',
            'data_name' : 'x',
            'pad_to_max_seq_length' : True
        },
        {
            'files' : ['./minrun_train_labels.txt'],
            'data_type' : 'int',
            'data_name' : 'label'
        }
    ]
}



val_data = {
    "num_epochs": 1,
    "batch_size": 50,
    "allow_smaller_final_batch": True,
    "shuffle": True,
    "shuffle_buffer_size": None,
    "shard_and_shuffle": False,

    "num_parallel_calls": 1,
    "prefetch_buffer_size": 0,
    "max_dataset_size": -1,
    "seed": None,
    "name": "val_data",

    'datasets' : [ 
        {
            "files" : ['./minrun_val_reviews.txt'],
            'vocab_file' : './minrun_opspam_vocab.txt',
            'max_seq_length' : 128,
            'length_filter_mode' : 'truncate',
            'bos_token' : '<BOS>',
            'delimiter' : ' ',
            'eos_token' : '<EOS>',
            'data_name' : 'x',
            'pad_to_max_seq_length' : True
        },
        {
            'files' : ['./minrun_val_labels.txt'],
            'data_type' : 'int',
            'data_name' : 'label'
        }
    ]
}

test_data = { 
    "num_epochs": 1,
    "batch_size": 64,
    "allow_smaller_final_batch": True,
    "shuffle": False,
    "shuffle_buffer_size": None,
    "shard_and_shuffle": False,
    "num_parallel_calls": 1,
    "prefetch_buffer_size": 0,
    "max_dataset_size": -1,
    "seed": None,
    "name": "test_data",
    'datasets' : [ 
        {
            "files" : 'minrun_test_reviews.txt',
            'vocab_file' : 'minrun_opspam_vocab.txt',
            'max_seq_length' : 128,
            'length_filter_mode' : 'truncate',
            'bos_token' : '<BOS>',
            'delimiter' : ' ',
            'eos_token' : '<EOS>',
            'data_name' : 'x',
            'pad_to_max_seq_length' : True
        },
        {
            'files' : 'minrun_test_labels.txt',
            'data_type' : 'int',
            'data_name' : 'label'
        }
    ]
}

# EMBEDDER HPARAMS

emb_hparams = {
    "dim": 50,
    "dropout_rate": 0.2,
    "dropout_strategy": 'element',
    "trainable": True,
    "initializer": {
        "type": "random_uniform_initializer",
        "kwargs": {
            "minval": -0.1,
            "maxval": 0.1,
            "seed": None
        }
    },
    "regularizer": {
        "type": "L1L2",
       "kwargs": {
            "l1": 0.,
            "l2": 0
        }
    },
    "name": "gen_embedder",
}

disc_emb_hparams = {
    "dim": 50,
    "dropout_rate": 0.4,
    "dropout_strategy": 'element',
    "trainable": True,
    "initializer": {
        "type": "random_uniform_initializer",
        "kwargs": {
            "minval": -0.1,
            "maxval": 0.1,
            "seed": None
        }
    },
    "regularizer": {
        "type": "L1L2",
        "kwargs": {
            "l1": 0.,
            "l2": 0
        }
    },
    "name": "disc_embedder",
}


clas_emb_hparams = {
    "dim": 50,
    "dropout_rate": 0.4,
    "dropout_strategy": 'element',
    "trainable": True,
    "initializer": {
        "type": "random_uniform_initializer",
        "kwargs": {
            "minval": -0.1,
            "maxval": 0.1,
            "seed": None
        }
    },
    "regularizer": {
        "type": "L1L2",
        "kwargs": {
            "l1": 0.,
            "l2": 0
        }
    },
    "name": "clas_embedder",
}


# GENERATOR HPARAMS
g_decoder_hparams = {
    "rnn_cell": {
            "type": tensorflow.contrib.cudnn_rnn.CudnnCompatibleGRUCell,
            "kwargs": {
                "num_units": 1024,
                
            },
            "num_layers": 2,
            "dropout": {
                "input_keep_prob": 1,
                "output_keep_prob": 0.5,
                "state_keep_prob": 1.0,
                "variational_recurrent": True,
                "input_size": [emb_hparams['dim'] + noise_size + 1,
                               1024]
            },
            "residual": False,
            "highway": False,
        },

    "max_decoding_length_train": None,
    "max_decoding_length_infer": None,
    "helper_train": {
        "type": "TrainingHelper",
        "kwargs": {}
    },
    "helper_infer": {
        "type": "SampleEmbeddingHelper",
        "kwargs": {}
    },
    "name": "g_decoder"
}


# DISCRIMINATOR HPARAMS
disc_hparams = {
    'encoder' : {

        "rnn_cell": {
               'type':tensorflow.contrib.cudnn_rnn.CudnnCompatibleGRUCell,
              'kwargs': {'num_units': 512},
              'num_layers': 2,
              'dropout': {'input_keep_prob': 1.0,
              'output_keep_prob': 0.5,
              'state_keep_prob': 1,
              'variational_recurrent': True,
              'input_size': [emb_hparams['dim'] + 1, 512],
              '@no_typecheck': ['input_keep_prob',
              'output_keep_prob',
              'state_keep_prob']},
              'residual': False,
              'highway': False,
              '@no_typecheck': ['type']},

        "output_layer": {
            "num_layers": 1,
            "layer_size": 1,
            "activation": "identity",
            "final_layer_activation": None,
            "other_dense_kwargs": None,
            "dropout_layer_ids": [],
            "dropout_rate": 0.5,
            "variational_dropout": True
        },
        'name' : 'discriminator',
        
        }
}


disc_crit_hparams = {
    'units' : 1,
    'activation' : 'linear'
}



# CLASSIFIER HPARAMS

clas_hparams = {
    'encoder' : {

        "rnn_cell": {
               'type':tensorflow.contrib.cudnn_rnn.CudnnCompatibleGRUCell,
              'kwargs': {'num_units': 128},
              'num_layers': 2,
              'dropout': {'input_keep_prob': 1.0,
              'output_keep_prob': 0.5,
              'state_keep_prob': 1,
              'variational_recurrent': True,
              'input_size': [emb_hparams['dim'], 128],
              '@no_typecheck': ['input_keep_prob',
              'output_keep_prob',
              'state_keep_prob']},
              'residual': False,
              'highway': False,
              '@no_typecheck': ['type']},

        "output_layer": {
            "num_layers": 1,
            "layer_size": 1,
            "activation": "identity",
            "final_layer_activation": None,
            "other_dense_kwargs": None,
            "dropout_layer_ids": [],
            "dropout_rate": 0.5,
            "variational_dropout": True
        },
        'name' : 'classifier',

    }
}

clas_crit_hparams = {
    'units':1,
    'activation':'linear'
}

# OPTIMIZER HPARAMS 

g_opt_mle_hparams = {
    "optimizer": {
        "type": tensorflow.contrib.opt.AdamWOptimizer,
        "kwargs": {
            'weight_decay' : 5e-3,
            "learning_rate": 0.001
        }
    },
    "learning_rate_decay": {
        "type": "",
        "kwargs": {},
        "min_learning_rate": 0.,
        "start_decay_step": 0,
        "end_decay_step": 1e10
    },
    "gradient_clip": {
        "type": tensorflow.clip_by_global_norm,
        "kwargs": {'clip_norm':5}
    },
    "gradient_noise_scale": None,
    "name": None
}

g_opt_pg_hparams = {
    "optimizer": {
        "type": tensorflow.contrib.opt.AdamWOptimizer,
        "kwargs": {
            'weight_decay' : 1e-7,
            "learning_rate": 0.00005
        }
    },
    "learning_rate_decay": {
        "type": "",
        "kwargs": {},
        "min_learning_rate": 0.,
        "start_decay_step": 0,
        "end_decay_step": 1e10
    },
    "gradient_clip": {
        "type": tensorflow.clip_by_global_norm,
        "kwargs": {'clip_norm':5}
    },
    "gradient_noise_scale": None,
    "name": None
}

c_opt_hparams = {
    "optimizer": {
        "type": tensorflow.contrib.opt.AdamWOptimizer,
        "kwargs": {
            'weight_decay' : 1e-4,
            "learning_rate": 0.0001
        }
    },
    "learning_rate_decay": {
        "type": "",
        "kwargs": {},
        "min_learning_rate": 0.,
        "start_decay_step": 0,
        "end_decay_step": 1e10
    },
    "gradient_clip": {
        "type": tensorflow.clip_by_global_norm,
        "kwargs": {'clip_norm':1}
    },
    "gradient_noise_scale": None,
    "name": None
}

d_opt_hparams = {
    "optimizer": {
        "type": tensorflow.contrib.opt.AdamWOptimizer,
        "kwargs": {
            'weight_decay' : 1e-4,
            "learning_rate": 0.0001,
        }
    },
    "learning_rate_decay": {
        "type": "",
        "kwargs": {},
        "min_learning_rate": 0.,
        "start_decay_step": 0,
        "end_decay_step": 1e10
    },
    "gradient_clip": {
        "type": tensorflow.clip_by_global_norm,
        "kwargs": {'clip_norm':1}
    },
    "gradient_noise_scale": None,
    "name": None
}

d_crit_opt_hparams = {
    "optimizer": {
        "type": tensorflow.contrib.opt.AdamWOptimizer,
        "kwargs": {
            'weight_decay' : 1e-3,
            "learning_rate": 0.001
        }
    },
    "learning_rate_decay": {
        "type": "",
        "kwargs": {},
        "min_learning_rate": 0.,
        "start_decay_step": 0,
        "end_decay_step": 1e10
    },
    "gradient_clip": {
        "type": tensorflow.clip_by_global_norm,
        "kwargs": {'clip_norm':1e6}
    },
    "gradient_noise_scale": None,
    "name": None
}
c_crit_opt_hparams = {
    "optimizer": {
        "type": tensorflow.contrib.opt.AdamWOptimizer,
        "kwargs": {
            'weight_decay' : 1e-3,
            "learning_rate": 0.001
        }
    },
    "learning_rate_decay": {
        "type": "",
        "kwargs": {},
        "min_learning_rate": 0.0,
        "start_decay_step": 0,
        "end_decay_step": 1e10
    },
    "gradient_clip": {
        "type": tensorflow.clip_by_global_norm,
        "kwargs": {'clip_norm':1e6}
    },
    "gradient_noise_scale": None,
    "name": None
}
