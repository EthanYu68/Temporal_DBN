import theano
import motion
import CRBMLogistic
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

set = 0 # if 2, mocap,resnet;if 1, mocap, CRBM;if 0, KX model CRBM.
num_cls = 2 # number of classes
num_samples = 5000 # number of samples
dimension=2 # number of variables(units) in one phase
period = 51 # length of a period(initial phase plus 50 phases)
freq = 1 # freq is frequency of extracting past inputs

# choose what kind of data to load
# if ar is 1, choose AR model.
# if rw is 1, choose Random Walk data.
# if wide is 1, choose ES(Easy to Separate) model data.
ar = 0
rw = 0 # rw?
wide = 0 # wide youtube channel model?
diff = 0 # If 1, get differenced series. If 0, not.

batch = [32,64,128,512] # batch size for SGD
pretrain_epoches = 0
folds = 10 # number of trails
times_train = 7 # the number of training steps with different learning rates.
lr = np.logspace(0,9,times_train,base=0.4) # set decreasing learning rate
epoch = np.linspace(2000,900,times_train,dtype=int) # set decreasing training epochs

### prepare Data for training

for wide in [0,1]:

    # load data
    if ar == 1:
        dimension=5
        period = 25
        UVsa = pd.read_csv('~/PycharmProjects/network/ARseq').values
        UVsa = UVsa.reshape([num_cls, int(10000/dimension), dimension]) # AR time-series data
    elif rw == 1:

        UVsa = pd.read_csv('~/PycharmProjects/network/RW-B').values
        UVsa = UVsa.reshape([num_cls, num_samples, period, dimension])

    elif wide == 1:
        UVsa = pd.read_csv('~/PycharmProjects/network/UVsa102_wide').values
        UVsa = UVsa.reshape([num_cls, num_samples, 51, 2])

    else:
        UVsa = pd.read_csv('~/PycharmProjects/network/UVsa102').values
        UVsa = UVsa.reshape([num_cls, num_samples, 51, 2])


    # loop over different number of training sequences for each class
    # num_tr is number of sample sequences for training for each class
    # Concatenate num_tr of sample sequences to a long sequence by np.reshape().
    for e, num_tr in enumerate([10,20,50,100]):
        for delay in [6]: # delay is the number of units in past input layer.
            batch_size = batch[e] # batch size will influence training time. When small number of sequences is trained, small batch size is better.
            Acc = [] # Accuracy list
            for i in range(folds):
                    if set == 0:
                        # Model Data
                        num = 2200
                        begin = np.random.randint(0, 1000) # for each fold, we want training samples to be different
                        if ar==1:
                            sample1 = UVsa[0].reshape([np.alen(UVsa[0]),dimension])
                            sample2 = UVsa[1].reshape([np.alen(UVsa[0]),dimension])
                        else:
                            rawdata1 = UVsa[0, begin:begin + num, :, :].reshape([num * period, dimension])
                            rawdata2 = UVsa[1, begin:begin + num, :, :].reshape([num * period, dimension])
                            if diff ==1:# if diff is 1, get differenced data
                                sample1 = (rawdata1[1:] - rawdata1[:np.alen(rawdata1) - 1])
                                sample2 = (rawdata2[1:] - rawdata2[:np.alen(rawdata1) - 1])
                                sample1[sample1<0] = 0
                                sample2[sample2<0] = 0
                            else:
                                sample1 = rawdata1
                                sample2 = rawdata2

                        num_test = 2000  # number of periods in testing sequence for one class
                        len = num * period  # total length of sequences for one class
                        len_tr = num_tr * period  # total length of training sequence for one class
                        len_te = num_test * period # total length of testing sequence for one class

                        # assign training data
                        seqlen_tr = [len_tr, len_tr]
                        seqlen_te = [len_te, len_te]
                        dataset_tr = np.concatenate([sample1[0:len_tr], sample2[0:len_tr]], axis=0)
                        dataset_te = np.concatenate([sample1[len_tr:len_tr + len_te], sample2[len_tr:len_tr + len_te]], axis=0)
                        # assign labels for training and testing data
                        label_tr = list(
                            np.array([np.zeros(len_tr, dtype=np.int64), np.ones(len_tr, dtype=np.int64)]).ravel())
                        label_test = list(
                            np.array([np.zeros(num_test, dtype=np.int64), np.ones(num_test, dtype=np.int64)]).ravel())

                        dataset_test = theano.shared(value=dataset_te, name='dataset_test')
                        dataset_train = theano.shared(value=dataset_tr, name='dataset_train')

                    else: # if set is 0, use Motion Captured data.
                        ## mocap data for CRBM
                        files = ['data/geste3a.bvh', 'data/geste3b.bvh', 'data/geste3c.bvh', 'data/geste3d.bvh'
                            , 'data/geste7a.bvh', 'data/geste7b.bvh', 'data/geste7c.bvh', 'data/geste7d.bvh']
                        labels = [1, 1, 1, 1, 2, 2, 2, 2]
                        trunk, larm, rarm, lleg, rleg, label_tr, seqlen_tr = motion.generate_dataset(files, labels)
                        trunk = trunk.get_value()
                        larm = larm.get_value()
                        rarm = rarm.get_value()
                        lleg = lleg.get_value()
                        rleg = rleg.get_value()
                        dataset_tr = np.concatenate([trunk, larm, rarm, lleg, rleg], axis=1)
                        dataset_train = theano.shared(value=dataset_tr, name='dataset')

                    # CRBM
                    if set == 0 or set == 1:
                        # if pre-training, retrain_crbm = True, log_crbm = None.
                        # if using pre-trained model, retrain_crbm = False, log_crbm = log_crbm(returned from last call)
                        # delay is the number of units in past input history.
                        # freq is the frequency for extracting past inputs.
                        # For example, if freq = 1, indexes for past inputs might be [13,12,11,10]
                        # If freq = 2, indexes for past inputs might be [13,11,9,7]

                        log_crbm = CRBMLogistic.create_train_LogisticCrbm(
                            finetune_lr=lr[0], pretraining_epochs=pretrain_epoches,
                            pretrain_lr=10e-2, training_epochs=epoch[0],
                            dataset_train=dataset_train, labelset_train=label_tr, seqlen_train=seqlen_tr,
                            dataset_validation=dataset_train, labelset_validation=label_tr, seqlen_validation=seqlen_tr,
                            dataset_test=dataset_train, labelset_test=label_tr, seqlen_test=seqlen_tr,
                            batch_size=batch_size, number_hidden_crbm=80, n_delay=delay, freq=freq, n_label=3,forward=1,
                            retrain_crbm=True,
                            log_crbm=None
                        )

                        for o in range(times_train-1):
                            log_crbm = CRBMLogistic.create_train_LogisticCrbm(
                                finetune_lr=lr[o+1], pretraining_epochs=pretrain_epoches,
                                pretrain_lr=10e-2, training_epochs=epoch[o+1],
                                dataset_train=dataset_train, labelset_train=label_tr, seqlen_train=seqlen_tr,
                                dataset_validation=dataset_train, labelset_validation=label_tr, seqlen_validation=seqlen_tr,
                                dataset_test=dataset_train, labelset_test=label_tr, seqlen_test=seqlen_tr,
                                batch_size=batch_size, number_hidden_crbm=80, n_delay=delay, freq=freq, n_label=3,forward=1,
                                retrain_crbm=False, # not retrain
                                log_crbm=log_crbm
                            )

                    # Mocap data for testing
                    if set == 1 or set == 2:
                        files = ['data/geste3e.bvh',
                                 'data/geste7e.bvh']
                        labels = [1, 2]
                        trunk, larm, rarm, lleg, rleg, label_te, seqlen_te = motion.generate_dataset(files, labels)
                        trunk = trunk.get_value()
                        larm = larm.get_value()
                        rarm = rarm.get_value()
                        lleg = lleg.get_value()
                        rleg = rleg.get_value()
                        dataset_te = np.concatenate([trunk, larm, rarm, lleg, rleg], axis=1)
                        num_test = np.sum(seqlen_te) - delay * freq  # number of testing samples
                        dataset_test = theano.shared(dataset_te)

                    # Test
                    predictor = log_crbm
                    batch_size_test = 100
                    predict, prob = predictor.recognize_dataset(dataset_test, seqlen=seqlen_te,
                                                                batch_size=batch_size_test)
                    # prediction of first class: predict[delay*freq:end of the first sequence]
                    # prediction of second class: predict[end of the first sequence+ delay*freq:end of the second sequence - batch size + delay]

                    label_test = np.concatenate(
                        [np.zeros(len_te - delay * freq), np.ones(len_te - batch_size_test + delay * freq)])
                    correct = np.zeros(num_cls * len_te)
                    correct[delay * freq:(num_cls * len_te - batch_size_test + delay * freq)] = 1 - abs(predict - label_test)
                    correct = correct.reshape([num_test * 2, period])
                    accuracy = np.mean(correct[1:num_cls * num_test - 10], axis=0)
                    print(i, accuracy[:20])
                    Acc.append(accuracy)

            acc_ave = np.mean(Acc, 0)
            # save results
            df = pd.DataFrame(Acc)
            dir = '~/PycharmProjects/network/results/' # replace the directory
            if rw == 1:
                df.to_csv(dir + 'RWA%dseq%dinputs' % (num_tr, delay + 1), index=False, header=False)
            elif wide == 1 and pretrain_epoches != 0:
                df.to_csv(dir + 'wide%dseq%dinputs' % (num_tr, delay + 1), index=False, header=False)
            elif wide == 1 and pretrain_epoches == 0:
                df.to_csv(dir + 'wide%dseq%dinputs_nopretr' % (num_tr, delay + 1), index=False, header=False)
            else:
                df.to_csv(dir + '%dseq%dinputs' % (num_tr, delay + 1), index=False, header=False)


# plot standard deviation of accuracy
acc1=pd.read_csv('~/PycharmProjects/network/results/wide50seq8inputs_nopretr',header=None)
std1 = np.std(acc1,0)
acc2=pd.read_csv('~/PycharmProjects/network/results/wide_NN_whole_input7_seq50',header=None)
std2 = np.std(acc2,0)
x = np.arange(7,17)
plt.plot(x,std1[8:18],label='DCRBM');plt.plot(x,std2,label='NN');plt.legend();plt.xlabel('phase');plt.ylabel('stdv of accuracy');plt.show()
