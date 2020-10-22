ARCHIVED VERSION, NOT SUPPORTED. For most recent SpamGAN work, see: https://github.com/YankunShen/spamGAN

# SpamGAN Code
GANs for Semi-Supervised Opinion Spam Detection
Code from: https://arxiv.org/abs/1903.08289

# Instructions
You will need to install Texar: https://texar.io/

The code was developed in *Tensorflow 1.12.0*, with Python *3.6.7*.

# Data
The cleaned original data is present in the `data` folder, with 
`opspam_reviews.txt` and `opspam_labels.txt` including the 1600 raw reviews and labels from the Ott dataset.
The `opspam_train_reviews.txt` and associated labels include the 1276 base training examples 
(post-removal of the 4 duplicate reviews) and the `opspam_test_reviews.txt` includes the common test set for all the experiments.
The various `chicago_unlab_reviews` files include the extracted unlabeled reviews from TripAdvisor

The labeled training examples and the unlabeled examples are further chopped up for the experiments to identify variation in performance for
the model depending on amount of labeled/unlabeled data used.

In the base folder, the `minrun` series of files are one of the best performing model configurations: spamGAN-50. This should serve as a minimal
working example (hopefully!). The full experiment code and such is in the `experiment_code` folder, and the data processing code is in the `data`
folder.

# Usage
Once you have texar installed and have cloned the project, try:

`python spamGAN_train.py spamGAN_config_minimal` 

Which will run one epoch of each of the components.

Running

`python spamGAN_train.py spamGAN_config_smallunsup`

will run a full training process for the data in the base folder.


Adjusting the model file or the model configuration file should provide a springboard for exploration. If you wanted to precisely replicate the
experiments in the paper, the code from the data and experiment_code folder can do that, albeit requiring some tweaking.
