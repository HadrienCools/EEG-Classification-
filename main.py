
from __future__ import division, print_function

# Import data
    # 52 channels
    # +1 label channel
#2  pre-processing
# Before segmentation
    # apply low pass filter (<40)
    # reject  electrodes after (visual inspection)
    # perform  ICA  to filer noise
# segmentation
    # split  in 2 -> eyes open/close
# after segmentation
    # check lag before change of state
    # cut composante DC 
    # cut unrelevant and too noisy channels
# 3 classification
    # feature extraction
    # feature selection (3dth seance)
        # PCA
        # wavelet
        # PSD
        # Mean
        # Power of amplitudes
    # classification
    # Separe en K folds
    # decision tree
    # NN
    # svm
    # knn
    # report  performances
    # 
#############################################################
import os
import numpy as np
import scipy as scp
import scipy.io as sio
import sys
import pandas as pd 
from numpy.random import randn
from numpy.fft import rfft
from scipy import signal
import matplotlib.pyplot as plt
# pca
from sklearn.decomposition import PCA
# knn
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
# svm
from sklearn.svm import SVC
#filter
import pywt
from scipy.signal import butter, lfilter
from scipy.signal import freqz
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
# metric
import timeit
import resource
import time
#shuffle
from sklearn.utils import shuffle

# Variables environnement
path_target = os.getcwd()+"/Data/target.csv"
path_datas = os.getcwd()+"/Data/EEG_data.csv"

data = pd.read_csv(path_datas, header=None)
target = pd.read_csv(path_target, header=None)

def print_target():
    print(path_target)
    print(path_datas)

def plot1(features):
    print(features.head())
    features.plot.line(subplots=True)
    plt.show()

# clean datas #
def delete_na(data,labels,value="NaN"):
    #https://thispointer.com/python-pandas-how-to-drop-rows-in-dataframe-by-conditions-on-column-values/
    # Get names of indexes for which column value == "NAN"
    if(data.shape[0]!=labels.shape[0]):
        print("erreur in dim")
        return -1
    ensembl = pd.concat([data, labels], axis=1, ignore_index=True)
    ensembl = ensembl.apply (pd.to_numeric, errors='coerce')
    ensembl = ensembl.dropna()
    ensembl.reset_index(drop=True)
    return ensembl
    #print(ensembl.head())
    #ensembl.iloc[:,-1]
    #https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.to_csv.html
    #df.to_csv(index=False,header)

def fft_signal(data):
    #https://courspython.com/fft-introduction.html
    plt.plot( data[0] )
    A = np.fft.fft(data[0])
    B = np.append(A, A[0])
    plt.subplot(312)
    plt.plot(np.real(B))
    plt.ylabel("partie reelle")

    plt.subplot(313)
    #plt.plot(np.imag(B))
    plt.ylabel("partie imaginaire")
    #
    # calcul de la transformee de Fourier et des frequences
    dt = 0.001
    signal = data[0]
    fourier = np.fft.fft(signal)
    n = signal.size
    freq = np.fft.fftfreq(n, d=dt)

    # affichage de la transformee de Fourier
    plt.subplot(212)
    plt.plot(freq, fourier.real, label="real")
    #plt.plot(freq, fourier.imag, label="imag")
    plt.legend()
    #plt.show()
    plt.show()

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = 0.001 / nyq
    high = 100 / nyq
    b, a = signal.butter(order, 100, btype='low')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def filter3(data,noplot):
    print("perform filtering")
    signal_d = data[1]
    
    #b, a = signal.butter(order, 100, btype='low')# command doesnt work
    t = np.linspace(0, 600, 626637, False)
    sos = signal.butter(10, 40, 'low', fs=2000, output='sos')
    filtered = signal.sosfilt(sos, signal_d)
    if (noplot != True):
        plt.plot(signal_d, label='Noisy signal')
        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
        ax1.plot(t, signal_d)
        ax1.set_title('Raw signal')
        ax2.plot(t, filtered)
        ax2.set_title('After 40 Hz high-pass filter')
        plt.show()
    return filtered

def segmentation(data):
    pos_values = []
    neg_values = []
    state = 0
    buffer = []
    for i in data.values:
        
        if(i[-1] == 0):
            if(state == 1):
                pos_values.append(buffer)
                buffer = []
                state = 0
            
            buffer.append(i)        

        if(i[-1] == 1):
            if(state == 0):
                neg_values.append(buffer)
                buffer = []
                state = 1
            
            buffer.append(i)
    data_set =[pos_values,neg_values]
    return data_set

def print_segment(data):
    df = pd.DataFrame(data=data)
    plt.plot(df, label='segment signal')
    #plt.plot(signal_d, label='Noisy signal')
    plt.show() 

def cut_subset(data, nb_cut = 700):
    """ Cut 700 first and last samples in our segments """
    res = []
    for label in data:
        #print('r')
        buffer  =[]
        for subset in label:
            sbs = subset[700:-700] 
            buffer.append(sbs)
        res.append(buffer)
    return res

# feature extraction & selection #
def loop_on_sbs(dataloop):
    """ Wrapper to the model of our data ( 2 dataframe the [0] -> features label 0 [1] -> features label 0 ) """
    res = []
    for label in dataloop: # 0 et 1
        print("label")
        buffer  =[]
        i=0
        for subset in label:
            i=i+1
            #print(i)
            df = pd.DataFrame(data=subset)
            #print(df.shape)#33
            elageddata = df.iloc[:,:-1]#df.drop(df.columns[len(df.columns)-1], axis=1, inplace=True)
            #print(elageddata.shape)#32
            # faire les opérations ici
            
            #mean = calcul_means(df)
            #cwt_elem = calcul_cwt(df)
            #spower_elem = spectral_power(df)
            elem_power = power(elageddata)

            buffer.append(elem_power)
        res.append(buffer)
    return res

# feature extraction & selection #
def loop2(dataloop):
    """ Wrapper to the model of our data ( 2 dataframe the [0] -> features label 0 [1] -> features label 0 ) """
    res = []
    for label in dataloop: # 0 et 1
        print("label")
        buffer  =[]
        i=0
        df_buff = pd.DataFrame(data=label[0])
        for subset in label:
            i=i+1
            df = pd.DataFrame(data=subset)
            elageddata = df.iloc[:,:-1]
            #print(elageddata)
            #buffer.append(elageddata.values)
            #df_buff
            df_buff.append(elageddata) 
        res.append(df_buff.values)
        
    return res

def calcul_means(df):
    moy = df.mean(axis = 0)
    #print()
    return moy

def calcul_cwt(df):
    """  Retrieve the max coef after aplying Continuous wavelet transformation on our fréquencies """
    return_vector = []
    for  columnData in df.iteritems():#[:-1]:
        #print(len(columnData))
        #for (columnName, columnData) in calcul_cwt.iteritems():
        sign_filtr = pywt.cwt(columnData[1].values, 5, 'gaus1')[0][0]
        return_vector.append(max(sign_filtr))
        # a check
        #print("""sssssssssssssssssssssssss""")
        #print(return_vector)
    return return_vector

def spectral_power(df,no_plot=True):
    """ Calcul the PSD of the signal on freq 7:13 corresponding to alpha waves and the most relevant ones """
    fs = 2000
    return_vector = []
    for columnData in df.iteritems():
        f, Pxx_den = signal.periodogram(columnData[1], fs)
        #print(len(Pxx_den))
        return_vector.append(sum(Pxx_den[7:13]))
        #return_vector.append(sum(Pxx_den[4:7]))
        #return_vector.append(sum(Pxx_den[8:11]))
        #return_vector.append(sum(Pxx_den[12:15]))
        #ou
        #X = fftpack.fft(x)
        #return_vector.append(sum(X[0:3]))
        #return_vector.append(sum(X[4:7]))
        #return_vector.append(sum(X[8:11]))
        #return_vector.append(sum(X[12:15]))
    if ( no_plot != True):
        plot_psd(fs, Pxx_den)
    return return_vector
    #plot_psd(fs, Pxx_den)

def plot_psd(f,Pxx_den):
    """ plot the PSD of the signal """
    plt.semilogy(f, Pxx_den)
    plt.ylim([1e-7, 1e2])
    plt.xlabel('frequency [Hz]')
    plt.ylabel('PSD [V**2/Hz]')
    plt.show()

def power(df):
    """ Calcul the mean of the square of the amplitudes of the signal """
    return_vector = []
    for  columnData in df.iteritems():
        #print(len(sum(np.square(columnData[1]))/len(columnData[1])))
        return_vector.append(sum(np.square(columnData[1]))/len(columnData[1]))
        #print(return_vector)
    return return_vector

def discard_column(data, selected_datas):
    """ Slice column in a dataframe following entry list (type: int < data.len()) """
    sliced_dataframe= data.iloc[:, selected_datas]
    return sliced_dataframe

def fusion_data(data):
    """ This function merge the two dataframe with features label 0 and features labeled 1 """
    feat0 = pd.DataFrame(data=data[0])
    #print("shape",feat0.shape)
    feat1 = pd.DataFrame(data=data[1])
    feat0[len(feat0.columns)] = 0
    feat1[len(feat1.columns)] = 1
    fusioned = feat0.append(feat1)
    #print(fusioned.head)
    return fusioned

def standardize(data_fusionned):
    """ Apply a standard scaler to limit the standard deviations,
     apply this function on data ( ! not label) before apply a PCA or a model """
    labels = data_fusionned.iloc[:,-1]
    #print(data_fusionned.shape)
    #print(data_fusionned.head)
    data_fusionned = data_fusionned.iloc[:,:-1]
    #print(data_fusionned.shape)
    #print(data_fusionned.head)
    scaler = StandardScaler() # Fit on training set only.
    scaler.fit(data_fusionned) # Apply transform to both the training set and the test set.
    standardized_dataset = scaler.transform(data_fusionned)
    standardized_dataset = pd.DataFrame(data = standardized_dataset)
    lab2 = pd.DataFrame(data = labels)

    #print(standardized_dataset.shape)
    #print(standardized_dataset.head)
    #print(lab2)
    #print(lab2.head)
    #bigdata = pd.concat([standardized_dataset, lab2], ignore_index=True, sort =False,axis=1)
    #print(bigdata)
    #ds2 = standardized_dataset.append(lab2)
    #ds3 = pd.concat([standardized_dataset, lab2],  axis=1)
    #standardized_dataset.insert(lab2)
    #print(standardized_dataset.head,"std values")
    #ds4 = pd.concat([standardized_dataset, lab2.reindex(standardized_dataset.index)], axis=1)
    #return ds4
    return_v = [standardized_dataset,lab2]
    return return_v

def pca2(data):
    """ Do a principal component analysis and extract most valuable features """
    pca = PCA(.95)
    pca.fit(data)
    #n_comp = pca.n_components_
    dataset = pca.transform(data)
    #print(dataset,"valeurs")
    return dataset
    #target = breastCancer_target_cell

def draw_pca(data):
    """ Draw values of cumulated variance between the features """
    pca = PCA()
    pca.fit(data)
    n_comp = pca.explained_variance_ratio_
    add_v = []
    buff = 0
    #print(n_comp[0])
    for item in n_comp:
        print(item,"item")
        buff = buff + item
        add_v.append(buff)
    #print(n_comp)
    #plt.plot(df, label='segment signal')
    plt.plot(add_v, label='PCA variance ratio')#.show()
    plt.show()

# Classification #
def kfolds_making(dataset,label,K=10):
    """Doing K folds on 10 folds, and launching all algorithms of 
    classification to have the same dataset and so we can compare accuracy between models"""
    #data = dataset-1
    #label = dataset-1
    dataset, label = shuffle(dataset, label, random_state=0)
    print(label)
    kfold_DT = KFold(K)

    performances_DT = 0
    performances_NN = 0
    performances_SVM = 0
    performances_KNN = 0
    #print()

    i=0
    for trn_idx, tst_idx in kfold_DT.split(dataset):
        print(i)
        i = i+1
        performances_DT = performances_DT + decision_tree2(dataset,label.ravel(),trn_idx,tst_idx)
        performances_NN = performances_NN + nn2(dataset,label.ravel(),trn_idx,tst_idx)
        performances_SVM = performances_SVM + svm2(dataset,label.ravel(),trn_idx,tst_idx)
        performances_KNN = performances_KNN + knn(dataset,label.ravel(),trn_idx,tst_idx)

    
    performances_DT = performances_DT/K
    performances_NN = performances_NN/K
    performances_SVM = performances_SVM/K
    performances_KNN = performances_KNN/K

    print(round(performances_DT, 3), "accuracy for decision tree")
    print(round(performances_NN, 3), "accuracy for multi layer perceptron")
    print(round(performances_SVM, 3), "accuracy for support vector  classifier ")
    print(round(performances_KNN, 3), "accuracy for K nearest neighbors ")
    
def nn2(data,label,trn_idx,tst_idx):
    time_start = time.perf_counter()
    training_data = data[trn_idx]
    training_target = label[trn_idx]
    validation_data = data[tst_idx]
    validation_target = label[tst_idx]

    model = MLPClassifier(hidden_layer_sizes=(70, ))#hidden_layer_sizes=(5, 2))
    model.fit(training_data, training_target)
    score = model.score(validation_data, validation_target)
    print("time for neural network",(time.perf_counter() - time_start))
    return score
    #performances_NN = performances_NN + score
    #count_NN = count_NN + 1

def svm2(data,label,trn_idx,tst_idx):
    time_start = time.perf_counter()
    training_data = data[trn_idx]
    training_target = label[trn_idx]
    validation_data = data[tst_idx]
    validation_target = label[tst_idx]

    model = SVC()
    model.fit(training_data, training_target)
    score = model.score(validation_data, validation_target)
    print("time for svm",(time.perf_counter() - time_start))
    return score
    #performances_SVM = performances_SVM + score
    #count_SVM = count_SVM + 1

def decision_tree2(data,label,trn_idx,tst_idx):
    time_start = time.perf_counter()
    
    training_data = data[trn_idx]
    training_target = label[trn_idx]
    validation_data = data[tst_idx]
    validation_target = label[tst_idx]
    model = DecisionTreeClassifier()
    model.fit(training_data, training_target)
    score = model.score(validation_data, validation_target)
    #performances_DT = performances_DT + score
    #count_DT = count_DT + 1 
    #performances_DT = performances_DT/count_DT
    print("time for decision tree",(time.perf_counter() - time_start))
    return score

def knn(data,label,trn_idx,tst_idx):
    #X_train, X_test, y_train, y_test = train_test_split(features.values, labels_knn.values, test_size=0.20)
    #scaler = StandardScaler()
    #scaler.fit(X_train)

    #X_train = scaler.transform(X_train)
    #X_test = scaler.transform(X_test)
    training_data = data[trn_idx]
    training_target = label[trn_idx]
    validation_data = data[tst_idx]
    validation_target = label[tst_idx]

    classifier = KNeighborsClassifier(n_neighbors=5)
    classifier.fit(training_data, training_target)

    #predictions
    score = classifier.score(validation_data, validation_target)
    y_pred = classifier.predict(validation_data)
    print(confusion_matrix(validation_target, y_pred),'conf matrix')
    accuracy = classification_report(validation_target, y_pred)
    print(accuracy, "c")
    return score

if __name__ == '__main__':
    """ Main: change no_plot to False to have de graphs
        to use the model with feature extraction use"feature1" argument
        or to use the model with all amplitude of the signal, user "feature2" as argument """
    no_plot = True
    feature = "feature1" #"feature2"
    
    if ( no_plot != True):
        plot1(data)
    data = delete_na(data,target) #[32x67xxxx]
    #fft_signal(data)
    filtered_datas = filter3(data, no_plot)
    segmented_set = segmentation(data) # [[[54x32[4400]]],[]]
    
    if ( no_plot != True):
        print_segment(segmented_set[1][47])
    cutted_subset = cut_subset(segmented_set)# [[[54x32[3000]]],[]]
    
    if ( feature == "feature1"):
        data_set  = loop_on_sbs(cutted_subset)

    if ( feature == "feature2"):
        data_set  = loop2(cutted_subset)

    try:
        data_fusionned = fusion_data(data_set)
    except :
        print('error probably in feature set declaration')
    #print(data_fusionned.head,"data_fusionned")
    
    data_standzed = standardize(data_fusionned) 
    
    if ( no_plot != True):
        draw_pca(data_standzed[0])

    data_pca = pca2(data_standzed[0])
    # performances
    tttest = pd.DataFrame(data = cutted_subset)
    kfolds_making(data_pca,data_standzed[1].values, K = 10)


    #print(len(data_set[0][0]),'loo') #32
    #print(data_set[0][0])
    #print (type(cutted_subset))
    #print (type(data_set))
    #print (len(cutted_subset[0][0]))
    #print (len(data_set[0][0]))
    #tttest = pd.DataFrame(data = cutted_subset)
    #print(type(data_standzed[1]))
    #print(type(data_standzed[1].values))
    #time_start = time.perf_counter()
    #print("time",(time.perf_counter() - time_start))
    #print(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    #print(target.head())
    #print_target()


###################
# DEPRECIED/legacy#
###################

#deprecied
def perform_filtering(data):
    #https://dsp.stackexchange.com/questions/19084/applying-filter-in-scipy-signal-use-lfilter-or-filtfilt
    # perform on each channel the pass band filter
    b, a = signal.butter(4, 0.03, analog=False)
    sig = [0,0]# a changer
    sig_ff = signal.filtfilt(b, a, sig)
    # cwt
    sign_filtr = pywt.cwt(datas, 5, 'gaus1')[0]

#deprecied
def perform_pca(entry_matrix):
    #pca = PCA(n_components=2)
    pca = PCA(0.95)
    # mutualinformation
    #pca = PCA(n_components=2, svd_solver='full') 
    pca.fit(entry_matrix)
    pca.transform(entry_matrix)
    print(pca.explained_variance_ratio_)
    print(pca.singular_values_)

#deprecied
def apply_svc(features, labels_svc):
    #https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
    clf = SVC(gamma='auto')
    clf.fit(X, y)
    
    return 0

#deprecied
def perform_filt(data):
    signal=data[0]
    fs = 1000.0
    lowcut = 0.5
    highcut = 40.0
    f0 = 1000.0
    # calcul des coefficiants
    #b, a = butter_bandpass(lowcut, highcut, fs, order=5)
    # dessin du signal bruité
    plt.figure(2)
    #plt.clf()
    #plt.plot(signal, label='Noisy signal')
    # application du filtre passe bande
    y = butter_bandpass_filter(signal, lowcut, highcut, fs, order=6)
    plt.plot( y)
    #plt.xlabel('time (seconds)')
    ##plt.hlines([-a, a], 0, T, linestyles='--')
    #plt.grid(True)
    #plt.axis('tight')
    #plt.legend(loc='upper left')

    plt.show()

    

