import theano
from Temporal_DBN.code import RBMLogistic
import numpy as np
import pandas as pd
import dnn_train
import matplotlib.pyplot as plt

set = 0  # if 2, mocap,resnet;if 1, mocap, CRBM;if 0, KX model CRBM.
num_cls = 2
num_samples = 5000
period = 51
freq = 1
rw = 0  # rw?
wide = 1  # wide youtube channel model?
batch = [32, 64, 128, 512]
pretrain_epoches = 2000
folds = 20
times_train = 9
lr = np.logspace(0, 6, times_train, base=0.4)
epoch = np.linspace(6000, 1000, times_train, dtype=int)
### prepare Data for training


for wide in [1]:
    if rw == 1:
        UVsa = pd.read_csv('~/PycharmProjects/network/RWseq').values
    elif wide == 1:
        UVsa = pd.read_csv('~/PycharmProjects/network/UVsa102_wide').values
    else:
        UVsa = pd.read_csv('~/PycharmProjects/network/UVsa102').values
    UVsa = UVsa.reshape([num_cls, num_samples, period, 2])

    num = 2200
    num_te =2000
    sample1 = UVsa[0, :num, :, :].reshape([num, period * 2])
    sample2 = UVsa[1, :num, :, :].reshape([num, period * 2])
    sample = np.concatenate([sample1, sample2], axis=0)
    labels = np.concatenate(
        [np.ones([num, 1]) * np.array([1, 0]), np.ones([num, 1]) * np.array([0, 1])], axis=0)

    order = np.random.permutation(num * num_cls)
    sample_shuffled = sample[order]
    labels_shuffled = labels[order]

    label_test = list(np.array(
        [np.zeros(num_te, dtype=np.int64), np.ones(num_te, dtype=np.int64)]).ravel())
    dataset_te = sample[200:]
    dataset_test = theano.shared(value=dataset_te, name='dataset_test')

for e, num_tr in enumerate([20]):
        for length in [14]:
            batch_size = 128  # batch[e]
            Acc = []
            for i in range(folds):
            # Model Data
                # np.random.randint(0, 1000)
                # assign training data according number of classes in experiment

                seqlen = [num, num]
                seqlen_tr = [num_tr, num_tr];
                num_te = 2000    # number of periods in testing sequence
                seqlen_te = [num_te, num_te];

                training_set = [];
                label_set = []
                idx = np.random.randint(0, 500)
                for k in range(num_tr):
                    for l in range(period - int(length / 2)):
                        training_set.append(sample_shuffled[idx + k, l * 2:l * 2 + length])
                        label_set.append(labels_shuffled[idx + k])
                sample_tr = np.array(training_set)
                label_tr = np.array(label_set)
                dataset_train = theano.shared(value=sample_tr, name='dataset_train')
                #dataset_tr = np.concatenate([sample1[0:num_tr], sample2[0:num_tr]], axis=0)
                # dataset_tr = np.concatenate([dataset_tr,dataset_tr,dataset_tr,dataset_tr], axis=1)
                #dataset_te = np.concatenate(
                            #[sample1[num_tr:num_tr + num_te], sample2[num_tr:num_tr + num_te]], axis=0)
                # dataset_te = np.concatenate([dataset_te, dataset_te, dataset_te, dataset_te], axis=1)

                log_rbm, train_fn, validate_model, test_model = RBMLogistic.create_train_LogisticCrbm(
                        finetune_lr=lr[0], pretraining_epochs=pretrain_epoches,
                        pretrain_lr=10e-2, k=1, training_epochs=epoch[0],
                        dataset_train=dataset_train, labelset_train=label_tr, seqlen_train=seqlen_tr,
                        dataset_validation=dataset_train, labelset_validation=label_tr, seqlen_validation=seqlen_tr,
                        dataset_test=dataset_train, labelset_test=label_tr, seqlen_test=seqlen_tr,
                        batch_size=batch_size, number_hidden_rbm=80, n_label=2,
                        retrain_rbm=True,
                        log_rbm=None
                    )
                for o in range(times_train - 1):
                        log_rbm, train_fn, validate_model, test_model = RBMLogistic.create_train_LogisticCrbm(
                            finetune_lr=lr[o + 1], pretraining_epochs=pretrain_epoches,
                            pretrain_lr=10e-2, k=1, training_epochs=epoch[o + 1],
                            dataset_train=dataset_train, labelset_train=label_tr, seqlen_train=seqlen_tr,
                            dataset_validation=dataset_train, labelset_validation=label_tr, seqlen_validation=seqlen_tr,
                            dataset_test=dataset_train, labelset_test=label_tr, seqlen_test=seqlen_tr,
                            batch_size=batch_size, number_hidden_rbm=80,  n_label=2,
                            retrain_rbm=False,
                            log_rbm=log_rbm
                        )
                # Test
                predictor = log_rbm
                batch_size_test = 100
                predict, prob = predictor.recognize_dataset(dataset_test, seqlen=[400, 400],
                                                                batch_size=batch_size_test)
                # prediction of first class: predict[delay*freq:end of the first sequence]
                # prediction of second class: predict[end of the first sequence+ delay*freq:end of the second sequence - batch size + delay]
                correct = 1 - abs(predict - label_test)
                correct = correct.reshape([num_te * 2])
                accuracy = np.mean(correct[1:num_cls * num_te - 2], axis=0)
                print(i, accuracy)
                Acc.append(accuracy)

            acc_ave = np.mean(Acc, 0)

            df = pd.DataFrame(acc_ave)
            dir = '~/PycharmProjects/network/results/'
            if rw == 1:
                df.to_csv(dir + 'RW%dseq%dinputs' % (num_tr, length + 1), index=False, header=False)
            elif wide == 1 and pretrain_epoches != 0:
                df.to_csv(dir + 'wide%dseq%dinputs' % (num_tr, length + 1), index=False, header=False)
            elif wide == 1 and pretrain_epoches == 0:
                df.to_csv(dir + 'wide%dseq%dinputs_nopretr' % (num_tr, length + 1), index=False, header=False)
            else:
                df.to_csv(dir + '%dseq%dinputs' % (num_tr, length + 1), index=False, header=False)

a = 1