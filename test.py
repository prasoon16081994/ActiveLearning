import matplotlib
#matplotlib.use('Agg')
import os
from time import time
import numpy as np
import pylab as pl
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import RidgeClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.extmath import density
from sklearn import metrics
from sklearn import cross_validation
import itertools
import shutil
import bench
import active

from  sklearn.datasets import load_files
from sklearn.feature_extraction.text import TfidfVectorizer

rep=raw_input("Enter the number of times youwant to label data:")
rep=int(rep)
que=2		#number of questions asked
PLOT_RESULTS = False
data_folder = "al"
train_folder="al/train"
test_folder="al/test"
unlabeled_folder="al/unlabeled"

p=0
while 1:
	testdata = load_files(test_folder, encoding = 'latin1')
	traindata = load_files(train_folder, encoding ='latin1')
    	unlabeleddata = load_files(unlabeled_folder, encoding = 'latin1')
	

	print " classes are "
	classes = traindata.target_names
    	print classes
	print()

	y_train = traindata.target 		
    	y_test =  testdata.target

	u_vector = TfidfVectorizer(encoding= 'latin1', use_idf=True, norm='l2', binary=False, sublinear_tf=True,min_df=0.001, max_df=1.0, ngram_range=(1, 2), analyzer='word', stop_words=None)

	X_train = u_vector.fit_transform(traindata.data)
	X_test = u_vector.transform(testdata.data)
	X_unlabeled = u_vector.transform(unlabeleddata.data)

	results = []
	args=[]
	args.append(LinearSVC(loss='l2', penalty='l2',dual=False, tol=1e-3, class_weight='auto'))
	args.append(X_train)
	args.append(y_train)
	args.append(X_test)
	args.append(y_test)
	args.append(X_unlabeled)
	args.append(classes)
	args.append(que)
    	results.append(bench.mtrain(args))
	
	if p>=rep:
		break
	
	indices = np.arange(len(results))
    	results = [[x[i] for x in results] for i in range(3)]


	clf_names, score, question_samples = results
	
	active.activate(question_samples,unlabeleddata,classes,train_folder)
	p=p+1
	
	

