import tensorflow

# Overarching



clas_test = False
clas_test_ckpt = None
clas_pred_output = None
# Saving/logging Config
restore_model= False
clear_run_logs = False
log_dir= None
checkpoint_dir= None
load_checkpoint_file = None
log_verbose_mle = True
log_verbose_rl = True
batches_per_summary = 100
batches_per_text_summary = 100
use_char_sep=False
compute_grad_norms = False

# Epoch count
train_lm_only = False
g_unlab_every_n = 300
g_pretrain_epochs = 0# 60
d_pretrain_epochs = 0# 60
d_pretrain_critic_epochs = 0#20
div_pretrain_epochs = 0
c_pretrain_epochs = 50 # 20
preadversarial_epochs = 0
adversarial_epochs = 0

disc_adv = 0
clas_adv = 0

gen_adv_epoch = 4
g_unlab_every_n_adv = -1
gen_mle_adv_epoch = 2

adv_train_max_gen_examples = 1000 #
adv_gen_train_with_unsup = False

# Training configs
min_disc_pg_acc = 0.85 # Train disc in PG when acc less than
max_div_pg_loss = 5
min_clas_pg_fakeacc = 0.51

gen_patience=20
gen_es_tolerance = 0.005
clas_es_tolerance = -0.05
clas_patience = 10

max_extra_disc_adv_epochs = 1
max_extra_div_adv_epochs = 5
max_extra_clas_adv_epochs = 5

max_decoding_length = 128
max_decoding_length_infer = 128
use_unsup=False
sampling_temperature = 1.0

annealing_length = 128
adversarial_length = 128

linear_decay_pg_weights = True

# Context configs
prior_prob=0.5 # probability of class 1 in generated/unlabeled data.
noise_size=10

advantage_var_reduc = 1



# Training tweaks
disc_label_smoothing_epsilon = 0.05
adv_max_clip = 100
min_log_prob = 0.1
max_log_prob = 100
min_pg_loss = -200
max_pg_loss = 200
add_sentence_progress = True

clas_loss_on_fake_lambda = 1.0 # Balancing param on real/generated clas
disc_crit_train_on_fake_only = True
clas_crit_train_on_fake_only = True
use_alt_disc_loss = False
use_alt_disc_reward = False
use_sigmoided_rewards = False

reward_blending = 'f1'

clas_min_ent_lambda = 1.0

clas_has_own_embedder = True
disc_has_own_embedder = True

# Different loss functions
mle_loss_in_adv = True
pg_max_ent_lambda = 0

discriminator_loss_lambda = 1.0
diversifier_loss_lambda = 0
diversity_discount = 1
classifier_loss_lambda = 1.0
norm_advantages = True
discriminator_random_stopping = False
classifier_random_stopping = False
let_discriminator_train_embedder = True

bleu_test = False

use_beam_search = False
beam_width = 4
train_data = {
    "num_epochs": 1,
    "batch_size": 32,
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
            "files" : None,
            'vocab_file' : None,
            'max_seq_length' : 128,
            'length_filter_mode' : 'truncate',
            'bos_token' : '<BOS>',
            'delimiter' : ' ',
            'eos_token' : '<EOS>',
            'data_name' : 'x',
            'pad_to_max_seq_length' : True
        },
        {
            'files' : None,
            'data_type' : 'int',
            'data_name' : 'label'
        }
    ]
}

clas_train_data = {
    "num_epochs": 1,
    "batch_size": 32,
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
            "files" : None,
            'vocab_file' : None,
            'max_seq_length' : 128,
            'length_filter_mode' : 'truncate',
            'bos_token' : '<BOS>',
            'delimiter' : ' ',
            'eos_token' : '<EOS>',
            'data_name' : 'x',
            'pad_to_max_seq_length' : True
        },
        {
            'files' : None,
            'data_type' : 'int',
            'data_name' : 'label'
        }
    ]
}



val_data = {
    "num_epochs": 1,
    "batch_size": 64,
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
            "files" : None,
            'vocab_file' : None,
            'max_seq_length' : 128,
            'length_filter_mode' : 'truncate',
            'bos_token' : '<BOS>',
            'delimiter' : ' ',
            'eos_token' : '<EOS>',
            'data_name' : 'x',
            'pad_to_max_seq_length' : True
        },
        {
            'files' : None,
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
            "files" : None,
            'vocab_file' : None,
            'max_seq_length' : 128,
            'length_filter_mode' : 'truncate',
            'bos_token' : '<BOS>',
            'delimiter' : ' ',
            'eos_token' : '<EOS>',
            'data_name' : 'x',
            'pad_to_max_seq_length' : True
        },
        {
            'files' : None,
            'data_type' : 'int',
            'data_name' : 'label'
        }
    ]
}
unsup_data = { 
    "num_epochs": 1,
    "batch_size": 32,
    "allow_smaller_final_batch": True,
    "shuffle": True,
    "shuffle_buffer_size": None,
    "shard_and_shuffle": False,
    "num_parallel_calls": 1,
    "prefetch_buffer_size": 1,
    "max_dataset_size": -1,
    "seed": None,
    "name": "unsup_data",
    'shuffle' : True,
    'dataset' :  
        {
            "files" : "./unsup_reviews.txt",
            'vocab_file' : './opspam_vocab.txt',
            'max_seq_length' : 128,
            'length_filter_mode' : 'truncate',
            'bos_token' : '<BOS>',
            'delimiter' : ' ',
            'eos_token' : '<EOS>',
            'pad_to_max_seq_length' : True
        },
    
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


# DIVERSIFIER HPARAMS

div_hparams = {
    'encoder' : {

        "rnn_cell": g_decoder_hparams['rnn_cell'],

        "output_layer": {
            "num_layers": 1,
            "layer_size": None,
            "activation": "identity",
            "final_layer_activation": None,
            "other_dense_kwargs": None,
            "dropout_layer_ids": [],
            "dropout_rate": 0,
            "variational_dropout": False 
        },
        'name' : 'discriminator',
        }
}







# OPTIMIZER HPARAMS ----------------------------

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

div_opt_hparams = {
    "optimizer": {
        "type": "AdamOptimizer",
        "kwargs": {
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
        "kwargs": {'clip_norm':50}
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
