#!/usr/bin/env python
# coding: utf-8
# get_ipython().system(' pip install imblearn')
import math
from math import e
import warnings
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.optimizers import SGD
from imblearn.over_sampling import SMOTE

warnings.filterwarnings('ignore')

#100 porcento de distintos
path="dataset100/"

#20 porcento distintos
#path="dataset80/"

X1 = np.genfromtxt(path+"X_train1.csv", delimiter=",")
Y1 = np.genfromtxt(path+"y_train1.csv", delimiter=",")


X2 = np.genfromtxt(path+"X_train2.csv", delimiter=",")
Y2 = np.genfromtxt(path+"y_train2.csv", delimiter=",")

# transform the dataset

oversample = SMOTE()

X1, Y1 = oversample.fit_resample(X1, Y1)
X2, Y2 = oversample.fit_resample(X2, Y2)

XALL = np.vstack((X1, X2))
yALL = np.append(Y1,Y2,axis = 0)


print("Number of samples in test set: %d" % yALL.shape)
print("Number of positive samples in test set: %d" % (yALL == 1).sum(axis=0))
print("Number of negative samples in test set: %d" % (yALL == 0).sum(axis=0))

print("Number of samples in test set: %d" % Y1.shape)
print("Number of positive samples in test set: %d" % (Y1 == 1).sum(axis=0))
print("Number of negative samples in test set: %d" % (Y1 == 0).sum(axis=0))

print("Number of samples in test set: %d" % Y2.shape)
print("Number of positive samples in test set: %d" % (Y2 == 1).sum(axis=0))
print("Number of negative samples in test set: %d" % (Y2 == 0).sum(axis=0))

X_test1 = np.genfromtxt(path+"X_test1.csv", delimiter=",")
Y_test1 = np.genfromtxt(path+"Y_test1.csv", delimiter=",")

X_test2 = np.genfromtxt(path+"X_test2.csv", delimiter=",")
Y_test2 = np.genfromtxt(path+"Y_test2.csv", delimiter=",")


X_test = np.vstack((X_test1, X_test2))
Y_test = np.append(Y_test1,Y_test2,axis = 0)

# transform the dataset
oversample = SMOTE()
X_test, Y_test = oversample.fit_resample(X_test, Y_test)


print("Number of samples in test set: %d" % Y_test.shape)
print("Number of positive samples in test set: %d" % (Y_test == 1).sum(axis=0))
print("Number of negative samples in test set: %d" % (Y_test == 0).sum(axis=0))

test_batched = tf.data.Dataset.from_tensor_slices((X_test, Y_test)).batch(len(Y_test))

def CreatedModelINN(WeightN1,
                    WeightN2,
                    porc1,
                    porc2, euler, conv):

   # (30, 8) (8,) (8, 8) (8,) (8, 1) (1,)

    WeightINN0 = WeightN1
    n=0
    WeightINN0[n]  =  INN_CalculateWeights(WeightN1[n],WeightN2[n],
                                           porc1,porc2,euler,conv)

    n=2
    WeightINN0[n]  =  INN_CalculateWeights(WeightN1[n],WeightN2[n],
                                               porc1,porc2,euler,conv)

    n=4
    WeightINN0[n]  =  INN_CalculateWeights(WeightN1[n],WeightN2[n],
                                           porc1,porc2,euler,conv)

    n=1
    WeightINN0[n]  =  INN_CalculateBIAS(WeightN1[n],WeightN2[n],
                                        porc1,porc2,euler,conv)


    n=3
    WeightINN0[n]  =  INN_CalculateBIAS(WeightN1[n],WeightN2[n],
                                            porc1,porc2,euler,conv)


    n=5
    WeightINN0[n]  =  INN_CalculateBIAS(WeightN1[n],WeightN2[n],
                                        porc1,porc2,euler,conv)

    return WeightINN0



# In[14]:


def INN_CalculateWeights(a, b, porc1, porc2,euler, conv):
    #calculate Euclides Distance
    N1 = a.reshape(-1,1)
    N2 = b.reshape(-1,1)

    #print('inicio',N1,N2)
    dim, soma = len(N1), 0
    for i in range(dim):
        soma += (math.pow(N1[i] - N2[i], 2))

    dist_euclidiana_total = math.sqrt(soma)
    #print(dist_euclidiana_total)

    threshold = (dist_euclidiana_total / dim)


    conta=0
    contb=0
    inv=a
    ii, jj =a.shape
    for i in range(ii):
        for j in range(jj):
            n1 = (a[i,j] * porc1)
            n2 = (b[i,j] * porc2)


            dist = math.pow(n1 - n2, 2)
            dist_euclidiana = math.sqrt(dist)
            n4_1 = dist_euclidiana

            if euler==0:
                n4_sum=((n1 * (1 + porc1) )  +
                        (n2 * (1 + porc2) )
                       )
            elif euler==1:
                n4_sum=((((n1 * (((1/2)*(1 + math.sqrt(5))) ** porc1))) +
                         ((n2 * (((1/2)*(1 + math.sqrt(5))) ** porc2)))
                        ))

            else:
                n4_sum=((    (n1 * (e ** (conv*porc1))) +
                             (n2 * (e ** (conv*porc2)))
                       ))


            if dist_euclidiana<=threshold:
                n4 =  n4_sum  + n4_1
                conta+= 1
            else:
                n4 = n4_sum
                contb+= 1

            inv[i,j] = n4

    return inv

def INN_CalculateBIAS(a, b, porc1, porc2, euler, conv):
    #calculate Euclides Distance
    N1 =a.reshape(-1,1)
    N2 =b.reshape(-1,1)


    dim, soma = len(N1), 0
    for i in range(dim):
        soma += (math.pow(N1[i] - N2[i], 2))

    dist_euclidiana_total = math.sqrt(soma)


    threshold = (dist_euclidiana_total / dim)

    conta=0;
    contb=0;

    ii = len(a)
    inv = a
    if (ii==0):
        ii = 1
    for i in range(ii):

        n1 = a[i] * porc1
        n2 = b[i] * porc2

        dist = math.pow(n1 - n2, 2)

        dist_euclidiana = math.sqrt(dist)
        n4_1 = dist_euclidiana

        n4_sum=(((n1 * (e ** (conv*porc1))) +
                 (n2 * (e ** (conv*porc2)))
                ))

        if dist_euclidiana<=threshold:
            n4 = n4_sum + n4_1
            conta+= 1
        else:
            n4 = n4_sum
            contb += 1

        inv[i] = n4

    return inv

def createArray(comms_round):
    arr = np.array([], dtype=float)
    arr = np.empty(shape=comms_round, dtype=float)
    return arr

def sum_scaled_weights_e(WeightN1, WeightN2, conv=1):
    '''Return the sum of the listed scaled weights. The is equivalent to scaled avg of the weights'''
    porc1, porc2 = 0.5 , 0.5
    avg_grad = CreatedModelINN(WeightN1, WeightN2,porc1,porc2,2,conv)

    return avg_grad

def test_model(X_test, Y_test,  model, comm_round):
    loss, acc = model.evaluate(X_test,Y_test, verbose = 0, batch_size = 128)
    print('comm_round: {} | global_acc: {:.3%} | global_loss: {}'.format(comm_round, acc, loss))
    return acc, loss

class SimpleMLP:
    @staticmethod
    def build():
        optimizer = SGD(lr=lr,
                        decay=lr / comms_round,
                        momentum=0.9)

        classifier_cv = Sequential()
        classifier_cv.add(Dense(units = 16, activation = 'relu',
                                kernel_initializer = 'glorot_uniform',
                                #kernel_regularizer=tf.keras.regularizers.L1(0.01),
                                #activity_regularizer=tf.keras.regularizers.L2(0.01),
                                input_dim = 30))
        classifier_cv.add(Dropout(0.25))
        classifier_cv.add(Dense(units = 16, activation = 'relu',
                                kernel_initializer = 'glorot_uniform',
                                #kernel_regularizer=tf.keras.regularizers.L1(0.01),
                                #activity_regularizer=tf.keras.regularizers.L2(0.01)
                               ))
        classifier_cv.add(Dropout(0.25))
        classifier_cv.add(Dense(units = 1, activation = 'sigmoid'))
        #TODO: Perguntar pq dessa linha otimizador = tf.keras.optimizers.Adam(lr = 0.001, decay = 0.0001, clipvalue = 0.5)
        classifier_cv.compile(optimizer = optimizer, loss = 'binary_crossentropy',
                              metrics = ['accuracy'])
        return classifier_cv

lr = 0.001
comms_round = 20
loss='binary_crossentropy'
metrics = ['accuracy']
optimizer = SGD(lr=lr,
                decay=lr / comms_round,
                momentum=0.9
               )
#initialize global model
smlp_global = SimpleMLP()
global_model_fl = smlp_global.build()
global_model_bl = smlp_global.build()
global_model_e = smlp_global.build()
global_model_h = smlp_global.build()

global_loss_bl = 10
global_loss_e  = 10
global_loss_h  = 10


epoch= 100
conv= lr

global_model_fl.summary()

#Modelo Contralizado
cent_model_bl = smlp_global.build()

cent_model_bl.fit(XALL,yALL, epochs=200, verbose=0)

global_acc_central, global_loss = test_model(X_test, Y_test, cent_model_bl, 1)
global_acc, global_loss_e = test_model(X_test1, Y_test1, cent_model_bl, 1)
global_acc, global_loss_e = test_model(X_test2, Y_test2, cent_model_bl, 1)

local_model1 = smlp_global.build()
local_model2 = smlp_global.build()

local_model = smlp_global.build()

scaled_weights = local_model.get_weights()


#Equal initialization
local_model1.set_weights(scaled_weights)
local_model2.set_weights(scaled_weights)


def retornarPesos(modelo):

    model1 = modelo
    W1 =model1.get_weights()
    N1 = np.append(W1[0].reshape(-1,1),W1[1].reshape(-1,1),axis = 0)
    N1 = np.append(N1,W1[2].reshape(-1,1),axis = 0)
    N1 = np.append(N1,W1[3].reshape(-1,1),axis = 0)
    N1 = np.append(N1,W1[4].reshape(-1,1),axis = 0)
    N1 = np.append(N1,W1[5].reshape(-1,1),axis = 0)

    return N1

#commence global training loop

for comm_round in range(comms_round -1):
    comm_round =  comm_round  + 1

   #model 1
    local_model1.fit(X1,Y1, epochs=epoch, verbose=0)


    #model 2
    local_model2.fit(X2,Y2, epochs=epoch, verbose=0)



    #model cknn
    average_weights_e =  sum_scaled_weights_e(local_model1.get_weights(),
                                              local_model2.get_weights(),
                                              conv)

    #update global model
    global_model_e.set_weights(average_weights_e)



    print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")

    global_acc, global_loss_e = test_model(X_test, Y_test, local_model1, comm_round)
    global_acc, global_loss_e = test_model(X_test, Y_test, local_model2, comm_round)
    global_acc, global_loss_e = test_model(X_test, Y_test, global_model_e, comm_round)



    #update local model
    local_model2.set_weights(average_weights_e)
    local_model1.set_weights(average_weights_e)
