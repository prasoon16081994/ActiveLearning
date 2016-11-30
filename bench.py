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

from  sklearn.datasets import load_files
from sklearn.feature_extraction.text import TfidfVectorizer

def mtrain(args):
	print ""
	print ""
	print("Training our machine :  ")
	print ""
	print ""
	
	args[0].fit(args[1],args[2])
	pred = args[0].predict(args[3])
	score = metrics.f1_score(args[4], pred)
        accscore = metrics.accuracy_score(args[4], pred)
	
	print ("pred count is %d" %len(pred))
        print ('accuracy score:     %0.3f' % accscore)
        print("f1-score:   %0.3f" % score)
	
	conf_matrix = metrics.confusion_matrix(args[4], pred)
	confide = np.abs(args[0].decision_function(args[5]))
	if(len(args[6]) > 2):
            confide = np.average(confide, axis=1)
	sorted_confidences = np.argsort(confide)	

	question_samples = []
        low_confidence_samples = sorted_confidences[0:args[7]]
        high_confidence_samples = sorted_confidences[(-1)*args[7]:]
        question_samples.extend(low_confidence_samples.tolist())
        question_samples.extend(high_confidence_samples.tolist())
	

	clf_descr = str(args[0]).split('(')[0]
	print ""
	print ""
        return clf_descr,score,question_samples


