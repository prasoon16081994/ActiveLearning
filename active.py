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


def activate(question_samples,unlabeleddata,classes,train_folder):
          for i in question_samples[0]:
            n_file = unlabeleddata.filenames[i]
            print n_file
            print '/////////////////////////content///////////////////////'
            print unlabeleddata.data[i]
            print '////////////////////content ends here///////////////////////'
            print "(selecting a label):"
            for i in range(0, len(classes)):
                print ("%d = %s" %(i+1, classes[i]))
            n_label = raw_input("Enter the correct label number:")
            while n_label.isdigit()== False:
                n_label = raw_input("Enter the correct label number (a number please):")
            n_label = int(n_label)
            category = classes[n_label - 1] 
            rdstDi = os.path.join(train_folder, category) 
            shutil.move(n_file, rdstDi)
