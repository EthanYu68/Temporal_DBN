from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import math
import matplotlib.mlab as mlab
class ARmodel():
    def __init__(self,length_of_sequence,parameters):
        '''
        :param length_of_sequence: length of the sequence
        :param parameters: list for coefficients. Each row has coefficients for one class.
        '''
        self.parameters = parameters
        self.num_cls = len(parameters)
        self.length = length_of_sequence
    def create_data(self):
        '''
        :return: an array contains two sequences.
        '''
        data = np.zeros([self.num_cls,self.length])
        for j in range(self.num_cls):
            order = len(self.parameters[j])
            s = list(np.random.normal(size=order))
            for i in range(self.length):
                w = np.random.normal(scale=1,size=1)
                for k in range(order):
                    w = w + self.parameters[j][k]*s[i+k]
                s.append(w)
            data[j] = np.array(s[order:]).ravel()
        return data
    def plot(self,data,length_to_show=100):
        length_to_show =100
        num_cls = np.alen(data)
        for cls in range(num_cls):
            plt.plot(data[cls][:length_to_show],'o-',label='class %d'%(cls+1))
        plt.title('AR model')
        plt.legend()
        plt.show()

class RWModel():
    def __init__(self,length_of_sequence,parameters):
        '''
        :param length_of_sequence: length of sequence

        :type parameters: list
        :param parameters: A list for coefficients. Each row contains coefficients for each variable (feature) of each class.
        '''
        self.parameters = parameters
        self.num_cls = len(parameters)
        self.num_features = 2
        self.length = length_of_sequence
    def create_data(self,number_of_sequence):
        x = np.zeros([self.num_cls,number_of_sequence,self.length+1,self.num_features])
        for j in range(self.num_cls):
            P = self.parameters[j]
            xset = []
            for i in range(number_of_sequence):
                xseq = [np.zeros(self.num_features)]
                xt = np.zeros([self.num_features])
                for k in range(self.length):
                    r = np.random.uniform(0,1)
                    if r< P[0]:
                        xt = xt + [1,0]
                    elif r>P[0] and r<P[1]:
                        xt = xt + [-1,0]
                    elif r>P[1] and r< P[2]:
                        xt = xt + [0,1]
                    elif r>P[2] and r<1:
                        xt = xt + [0,-1]
                    xseq.append(xt)
                xseq = np.array(xseq)
                xset.append(xseq)
            xset = np.array(xset)
            x[j,:,:,:] = xset
        return x

    def normalize(self, dataset, max_after_normalize=1):
        maximum = np.max(dataset)
        divide = maximum / max_after_normalize
        dataset_normalized = dataset / divide
        return dataset_normalized

    def plot_curve(self, dataset, index=None):
        if index != None:
            idx = index
        else:
            idx = np.random.randint(0, len(dataset[0]), 1)
        num_cls = len(dataset)
        #for cls in range(num_cls):
        plt.plot(dataset[0, idx, :, 0][0], dataset[0, idx, :, 1][0],marker="x",label='class %d'%(1))
        plt.plot(dataset[1, idx, :, 0][0], dataset[1, idx, :, 1][0], label='class %d' % (2))

        plt.xlabel("X");
        plt.ylabel("K");
        plt.legend()
        plt.show()
    def plot_scatter(self,dataset, index=None,number_points = 50):
        if index != None:
            idx = index
        else:
            idx = np.random.randint(0, len(dataset[0]))
        num_cls = len(dataset)
        color = ['r','g']
        for cls in range(num_cls):
            for i in range(number_points):
                plt.scatter(dataset[cls, i, idx, 0], dataset[cls,i, idx, 1],c=color[cls],
                         label='class %d:p=%.2f, q=%.2f' % (cls + 1, self.paramters[cls][0], self.paramters[cls][1]))
        plt.show()
    def compute_benchmark(self, dataset, phase):
            '''
            :type dataset: np.array
            :param dataset: a dataset returned from create_data
            :param phase: at which phase to test accuracy
            :return: an array containing accuracy of all phases
            '''
            num_cls = len(dataset)
            num_samples = len(dataset[0])
            data_i = dataset[:, :, phase, :].reshape([num_cls, num_samples, 2])
            sample_mean = np.mean(data_i, axis=1)
            sample_inv_cov = np.zeros([num_cls, 2, 2])
            for j in range(num_cls):
                sample_inv_cov[j, :, :] = np.linalg.inv(np.cov(data_i[j, :, :].T))

            correct = 0
            for label in range(num_cls):
                # Compute Mahalabonis Distance
                for k in range(num_samples):
                    sample = data_i[label, k, :]
                    distance = np.zeros(num_cls)
                    for c in range(num_cls):
                        miu = sample_mean[c, :]
                        icov = sample_inv_cov[c, :, :]
                        delta = sample - miu
                        distance[c] = np.dot(np.dot(delta, icov), delta.T)
                    pred = np.argmin(distance)
                    correct = correct + (1 - np.abs(label - pred))
            benchmark = correct / (num_cls * num_samples)
            return benchmark

class YCModel(): # Youtube Channel Model
    def __init__(self,length_of_sequence,parameters,number_of_potential_users):
        '''
        :param length_of_sequence:
        :type parameters: list
        :param parameters: A list containing parameters for each class. ex. two classes:[[Pnew1,Prew1,Dnew1,Drew1],[Pnew2,Prew2,Dnew2,Drew2]]
        :type number_of_potential_users: int
        :param number_of_potential_users: N, which is assumed to be the number of all users.
        '''
        self.N = number_of_potential_users
        self.num_cls = len(parameters)
        self.length = length_of_sequence
        self.parameters = parameters

    def create_data(self,number_of_sequence):
        '''
        :param number_of_sequence: number of sequences to be generated
        :return: an array containing two classes' sequences.
        '''
        UV_samples = np.zeros([self.num_cls,number_of_sequence,self.length+1,2])
        for j in range(self.num_cls):
            Pnew,Prew,Dnew,Drew = self.parameters[j]
            for l in range(number_of_sequence):
                usr = 0;views = 0
                U = np.zeros(self.length+1)
                V = np.zeros(self.length+1)
                for i in range(self.length):
                    r1 = np.random.rand(self.N - usr) # generate N-usrs r.v. from uniform distribution
                    newu = r1[r1<Pnew/(1+i*Dnew)]
                    num_newu = np.alen(newu) # number of new users at time t
                    usr = usr + num_newu  # number of total users
                    # calculate number of re-watching
                    r2 = np.random.rand(usr)
                    rewa = r2[r2<Prew / (1 + i * Drew)]
                    num_rewa = np.alen(rewa)
                    views = num_newu + views + num_rewa
                    U[i + 1] = usr
                    V[i + 1] = views
                UV_samples[j,l] = np.array([U,V]).T
        return UV_samples
    def normalize(self, dataset, max_after_normalize = 1 ):
        maximum = np.max(dataset)
        divide = maximum/max_after_normalize
        dataset_normalized = dataset/divide
        return dataset_normalized
    def plot_curve(self, dataset,index = None):
        '''
        :param dataset:
        :param index: indicate specific sequence to show
        :return: figures
        '''
        if index!=None:
            idx = index
        else:
            idx = np.random.randint(0,len(dataset[0]),1)
        num_cls = len(dataset)
        #for cls in range(num_cls):
        plt.plot(dataset[0,idx,:,0][0],dataset[0,idx,:,1][0],marker="x",label='class %d'%(1))
        plt.plot(dataset[1, idx, :, 0][0], dataset[1, idx, :, 1][0], label='class %d' % (2))
        plt.xlabel("Subscribers");plt.ylabel("Views");plt.legend()
        plt.show()

    def compute_benchmark(self,dataset,phase):
        num_cls = len(dataset)
        num_samples = len(dataset[0])
        data_i = dataset[:,:,phase,:].reshape([num_cls,num_samples,2])
        sample_mean = np.mean(data_i,axis=1)
        sample_inv_cov = np.zeros([num_cls,2,2])
        sample_cov = np.zeros([num_cls,2,2])
        for j in range(num_cls):
            sample_cov[j,:,:] = np.cov(data_i[j, :, :].T)
            sample_inv_cov[j,:,:] = np.linalg.inv(np.cov(data_i[j,:,:].T))

        correct = 0
        for label in range(num_cls):
#           Compute Mahalabonis Distance
            for k in range(num_samples):
                sample = data_i[label, k, :]
                prob = np.zeros(num_cls)
                for c in range(num_cls):
                    miu = sample_mean[c,:]
                    icov = sample_inv_cov[c,:,:]
                    cov = sample_cov[c,:,:]
                    delta = sample - miu
                    prob[c]= np.dot(np.dot(delta,icov),delta.T) - np.log(np.sqrt(np.linalg.det(cov)))
                pred = np.argmin(prob)
                correct = correct + (1-np.abs(label-pred))
        benchmark = correct/(num_cls*num_samples)
        return benchmark
ar=0
if ar ==1:
    ar = ARmodel(length_of_sequence=10000,parameters=[[0.05,-0.1,0.2,0.3,0.2,-0.1,0.05,0.2,0.1],[0.3,-0.2,0.15,-0.1,0.05,0.05]])
    ar_data = ar.create_data()
    ar.plot(ar_data)
    df = pd.DataFrame(ar_data)
    df.to_csv('ARseq', index=False)


rw = 1
if rw == 1:
    #params_rw = [[0.25,0.5,0.75],[0.4,0.6,0.8]]# RW-A
    P1 = 0.4; P2 = 1- P1 # P1 is probability of moving one-step towards y-axis. P2 is probability of moving one-step towards x-axis.
    params_rw = [P1,P2] # RW-B
    rw_model = RWModel(50,parameters=params_rw)
    dataset_rw = rw_model.create_data(number_of_sequence=5000)
    dataset_rw_normal = rw_model.normalize(dataset_rw)
    benchmark = []
    for i in range(2,50):
        ratio = rw_model.compute_benchmark(dataset_rw_normal,phase=i)
        benchmark.append(ratio)
    benchmark = np.array(benchmark)
    rw_model.plot_curve(dataset_rw)
    df = pd.DataFrame(dataset_rw_normal.ravel())
    df.to_csv('RW-A', index=False)

    df = pd.DataFrame(benchmark)
    df.to_csv('RW-A-benchmark', index=False,header=False)
else:
    #params = [[0.005,0.1,0.03,0.05],[0.006,0.1,0.05,0.03]] # close
    params = [[0.01, 0.12, 0.03, 0.05], [0.008, 0.16, 0.05, 0.03]]# wide
    mmodel= YCModel(50, params,1000)
    dataset = mmodel.create_data(50)
    dataset_normal = mmodel.normalize(dataset,max_after_normalize=1)
    benchmark = []
    for i in range(1,40):
        ratio = mmodel.compute_benchmark(dataset,phase=i)
        benchmark.append(ratio)
    benchmark = np.array(benchmark)
    mmodel.plot_curve(dataset)
    df = pd.DataFrame(dataset_normal.ravel())
    df.to_csv('UVsa102_wide', index=False,header=False)

    df = pd.DataFrame(benchmark)
    df.to_csv('benchmark_wide', index=False,header=False)
