import random
import clean_text
with open('./opspam_reviews.txt', 'r') as f:
    revs = f.readlines()

with open('./opspam_labels.txt', 'r') as f:
    labs = f.readlines()

labs = [int(l.strip()) for l in labs]

revs = [clean_text.clean(r) for r in revs]

decep = [r for r, l in zip(revs, labs) if l == 1]
nondecep = [r for r, l in zip(revs, labs) if l == 0]
nondecep = list(set(list(nondecep)))

decep_idx = random.sample(list(range(len(decep))), len(decep))
nondecep_idx = random.sample(list(range(len(nondecep))), len(nondecep))

decep = [decep[i] for i in decep_idx]
nondecep = [nondecep[i] for i in nondecep_idx]

train_decep = decep[0:640]
test_decep = decep[640:]
train_nondecep = nondecep[0:636]
test_nondecep = nondecep[636:]

train = train_decep + train_nondecep
train_labs = [1] * len(train_decep) + [0] * len(train_nondecep)
test = test_decep + test_nondecep
test_labs = [1] * len(test_decep) + [0] * len(test_nondecep)

test_idx = random.sample(list(range(len(test))), len(test))
train_idx = random.sample(list(range(len(train))), len(train))

train = [train[i] for i in train_idx]
train_labs = [train_labs[i] for i in train_idx]
test = [test[i] for i in test_idx]
test_labs = [test_labs[i] for i in test_idx]

with open('./opspam_train_reviews.txt', 'w') as f:
    for r in train:
        f.write(r + '\n')

with open('./opspam_train_labels.txt', 'w') as f:
    for l in train_labs:
        f.write(str(l) + '\n')

with open('./opspam_test_reviews.txt', 'w') as f:
    for r in test:
        f.write(r + '\n')

with open('./opspam_test_labels.txt', 'w') as f:
    for l in test_labs:
        f.write(str(l) + '\n')






