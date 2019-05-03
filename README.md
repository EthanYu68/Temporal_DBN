# Temporal_DBN

This repository contains files that create data, build and optimize DCRBM classifiers and Neural Network classifiers and plot testing results. 

## Packages 
[Numpy](https://docs.scipy.org/doc/numpy/user/whatisnumpy.html), [pandas](https://pandas.pydata.org/), [matplotlib](https://matplotlib.org/users/installing.html), [theano](http://deeplearning.net/software/theano/install.html), [torch](https://pytorch.org/get-started/locally/)

## Modules
`CRBMLogistic.py`: return a classifer consisting of a CRBM and a logistic output layer on top

`RBMLogistic.py`: return a classifier consisting of a RBM and a logistic output layer on top

`crbm.py`: return a CRBM layer

`logistic_sgd.py`: consist of logistic output layer

`motion.py`: processes Mocap data 

`rbm.py`: return a RBM layer

`tdbn.py`: return a Temporal DBN classifier, consisting of sevaral RBM, a CRBM and a logistic output layer.
## Files
* In `create_data.py`:
  * Three models: ARModel, RWModel, YCModel.
  * Build model
  ```python
  rw_model = RWModel(50,parameters=params_rw)
  ```
  * Generate data
  ```python
  dataset_rw = rw_model.create_data(number_of_sequence=5000)
  ```
  * Normalize/standardize data
  ```python
  dataset_rw_normal = rw_model.normalize(dataset_rw)
  ```
  * Compute benchmark
  ```python
  ratio = rw_model.compute_benchmark(dataset_rw_normal,phase=i)
  ```
  
  

* In `CRBMclassify.py`:
  * Module `CRBMLogistic` will be imported.
  ```python
  import CRBMLogistic
  ```
  * Samples will be loaded (Mocap data, Youtube Channnel model, Random Walk model) 
  * Samples will be re-organized and be transformed to `Theano.tensor`.
  ```python
  dataset_tr = np.concatenate([sample1[0:len_tr], sample2[0:len_tr]], axis=0)
  dataset_train = theano.shared(value=dataset_tr, name='dataset_train')
  ```
  * Corresponding labels will be generated in a list
  ```python
  label_tr = list(np.array([np.zeros(len_tr, dtype=np.int64), np.ones(len_tr, dtype=np.int64)]).ravel())                      
  ```
  * Then, a DCRBM classifier `log_crbm` will be built and optimized using`CRBMLogistic`.
  ```python
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
  ```
  `seqlen_tr` is a list containing length of concatenated sequences. In case of two classes, it will contain two numbers.
  `freq` and `delay` are for defineing the past inputs. If index for current input is 20 and `freq=2`,`delay=3`, the indexes for past inputs will be `19 17 15`. If index for current input is 20 and `freq=1`,`delay=4`, the indexes for past inputs will be `19 18 17 16`. 
  `retrain_crbm` and `log_crbm` are for determining whether pre-training is needed or not. If `retrain_crbm` is `True`, `log_crbm = None`. Otherwise, `logcrbm` is supposed to be fed in a pre-trained model.
  * Performance will be tested by taking advantage of the `recognize_dataset` function
  ```python
  predictor = log_crbm
  predict, prob = predictor.recognize_dataset(dataset_test, seqlen=seqlen_te,                                                           batch_size=batch_size_test)
  ```
  `predict` and `prob` are recognition results arrays for whole periodic sequences. To get accuracy of classification at each phase inside a period, results arrays can be reshaped and averaged
  ```python
  label_test = np.concatenate(
                        [np.zeros(len_te - delay * freq), np.ones(len_te - batch_size_test + delay * freq)])
  correct = np.zeros(num_cls * len_te)
  correct[delay * freq:(num_cls * len_te - batch_size_test + delay * freq)] = 1 - abs(predict - label_test)
  correct = correct.reshape([num_test * 2, period])
  accuracy = np.mean(correct[1:num_cls * num_test - 10], axis=0)
  ```
* In `NNclassify.py`:
  * A class `Net` is defined.
  * Generate a `Net`
  ```python
  net = Net(n_feature=d, n_hidden=80, n_output=2)
  ```
  * Train the `Net`
  ```python
  net.train(training_samples=sample_tr, target=label_tr, iter=80000)
  ```
  `training_samples` and `target` are array type.
  
  * Predict using the `Net`
  ```python
  out2 = net.forward(test)
  ```
  `test` is a torch.tensor.
 
* In `plot.py`:
  * Accuracy results are plotted.
  
## Data
`data` : contains Motion Capture data.
  

