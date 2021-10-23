#!/usr/bin/env python
'''
    File name: build_models.py
    Author: Nicolas Bockstael && Alexandre Jadin
    Date created: 10/2/2018
    Date last modified: 04/06/2018
    Python Version: 3.6.3

	Description : Helper file Building the different machine learning models and their parameter range
					This makes an extensive use of the sci-kit library
'''

from sklearn.gaussian_process.kernels import ExpSineSquared, WhiteKernel, RBF, Matern
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR

# IMPORT LINEAR MODELS
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.linear_model import Lasso, LassoCV, LassoLarsCV, LassoLars, LassoLarsIC, Lars
from sklearn.linear_model import ElasticNet, ElasticNetCV
from sklearn.linear_model import OrthogonalMatchingPursuit, OrthogonalMatchingPursuitCV
from sklearn.linear_model import BayesianRidge, ARDRegression
from sklearn.linear_model import SGDRegressor, Perceptron
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model.passive_aggressive import PassiveAggressiveRegressor

# IMPORT DISCRIMINANT ANALYSIS CLASSIFIERS (not regression !!!)
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis

# IMPORT SVM
from sklearn.svm import SVR, LinearSVR, NuSVR, SVC

from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier

# Naive Bayes
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB

# KNN
from sklearn.neighbors import KNeighborsRegressor

# GAUSSIAN PROCESS

from sklearn.gaussian_process import GaussianProcessRegressor

# IMPORT DECISION TREES AND RANDOM FORESTS
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

# IMPORT NEURAL NETWORKS
from sklearn.neural_network import MLPRegressor, MLPClassifier

#  IMPORT ENSEMBLE METHODS
from sklearn.ensemble import VotingClassifier

import numpy as np
import pydevd


"""CLASSIC INTERVALS DECLARATION"""

alpha_interval = [1, 10, 100, 1000]
alpha_1_interval = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3]
alpha_SGD_interval = [1e-5, 1e-4, 1e-3]
alpha_1_short_interval = [1e-7, 1e-6, 1e-5]
alpha_MLP_interval = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100]
C_interval = [1, 10, 20, 50, 100]
C_short_interval = [1, 3, 5]
coef0_interval = np.arange(0.0, 1.0, 0.2)
decision_criterion_interval = ["gini", "entropy"]
degree_interval = range(1, 6)  # 1..5 included only used for poly kernel
eps_interval = [1e-3, 1e-2, 1e-1, 1]
epsilon_interval = np.arange(0.0, 1.0, 0.2)
eta0_interval = [1, 0.5, 0.25, 1e-1, 1e-2, 1e-3, 1e-4]
eta0_short_interval = [0.5, 0.25, 1e-1, 1e-2]
gamma_interval = [1e-3, 1e-4]
kernel_interval = ['linear', 'rbf', 'poly', 'sigmoid', 'precomputed']
kernel_short_interval = ['linear', 'rbf', 'poly']
hidden_layer_sizes_interval = [(100,), (200,), (50,), (25,), (10, 10), (25, 25), (50, 50), (10, 15, 10)]
KNNAlgo = ['auto', 'ball_tree', 'kd_tree', 'brute']
KNNweigths = ['uniform', 'distance']

l1_ratio_interval = np.arange(0.1, 1.0, 0.2)  # The Elastic Net mixing parameter, with 0 <= l1_ratio <= 1. l1
l1_ratio_short_interval = np.arange(0.1, 0.5, 0.2)  # The Elastic Net mixing parameter, with 0 <= l1_ratio <= 1. l1
learning_rate_interval = ['constant', 'optimal', 'invscaling']
loss_interval = ['squared_loss', 'huber', 'epsilon_insensitive', 'squared_epsilon_insensitive']

max_iter_interval = [100, 250, 500, 1000, 2000]
max_iter_short_interval = [100, 250, 500]
max_iter_elastic_interval = [1000, 2000, 3000]
max_depth_interval = [None, 10, 30, 50, 100]

MLP_activation = ['identity', 'logistic', 'tanh', 'relu']
MLP_learning_rate = ['constant', 'invscaling', 'adaptive']
MLP_solver = ['lbfgs', 'sgd', 'adam']
min_samples_split_interval = [2, 4, 10, 30, 50]

n_alphas_interval = [200, 300, 500]

n_estimators_interval = [10, 30, 50, 100, 200]

n_nonzero_coefs = [100, 200, 500, 750, 1000]
nu_interval = np.arange(0.01, 1.0, 0.2)  # 0 value not accepted
penalty_interval = ['str', 'none', 'l2', 'l1', 'elasticnet']
power_t_interval = [1e-2, 0.1, 0.25, 0.5, 0.75, 1]
splitter_interval = ['best', 'random']
tol_interval = [1e-4, 1e-3, 1e-2, 1e-1, 1]
tol_short_interval = [1e-3]
true_false_interval = [True, False]

gp_kernels = []
gp_kernels.append(ExpSineSquared(1.0, 5.0, periodicity_bounds=(1e-2, 1e1)) + WhiteKernel(1e-1))
gp_kernels.append(RBF(length_scale=1.0, length_scale_bounds=(1e-5, 1e5)))
gp_kernels.append(Matern(length_scale=1.0, length_scale_bounds=(1e-5, 1e5), nu=1.5))

"""EXTENDED INTERVALS"""

max_depth_interval_h = [20, 30, 40]
min_samples_split_interval_h = [10, 20, 30, 50]
n_estimators_interval_h = [50, 75, 150, 200]

alpha_1_interval_h = [1e-5, 1e-4, 1e-3]
hidden_layer_sizes_interval_h = [(100,), (200,), (50,), (25,), (10, 10), (25, 25), (50, 50), (10, 15, 10)]

"""FUNCTIONS BUILDING MODELS"""

def build_custom_voting_clf():
	KNeighbors_R_grid = [{'n_neighbors': [3, 6, 9], 'weights': ['distance'], 'algorithm': ['auto']}]
	KNeighbors_C_grid = [{'n_neighbors': [3, 6, 9], 'weights': ['distance'], 'algorithm': ['auto']}]

	Random_Forest_classifier_grid = [{'max_depth': [40, 50], 'min_samples_split': [10], 'n_estimators': [200]}]
	Random_Forest_regressor_grid = [{'max_depth': [30, 40], 'min_samples_split': [10], 'n_estimators': [200]}]
	MLPClassifier_grid = [{'hidden_layer_sizes': [(50, 50)]}]
	MLPClassifier_grid = [{'hidden_layer_sizes': [(50, 50)]}]

	eclf_grid = {'rfc__max_depth': [40, 50], 'rfc__min_samples_split': [10], 'rfc__n_estimators': [200],
				 'rfr__max_depth': [40, 50], 'rfr__min_samples_split': [10], 'rfr__n_estimators': [200],
				 'knnr__n_neighbors': [3, 6, 9], 'knnr__weights': ['distance'], 'knnr__algorithm': ['auto'],
				 'knnc_n_neighbors': [3, 6, 9], 'knnc__weights': ['distance'], 'knnc__algorithm': ['auto'],
				 'mlpc__hidden_layer_sizes': [(50, 50)], 'mlpr__hidden_layer_sizes': [(50, 50)], }

	models = []
	# models.append(('rfc', RandomForestClassifier()))
	# models.append(('rfr', RandomForestRegressor()))
	# models.append(('knnr', KNeighborsRegressor()))
	# models.append(('knnc', KNeighborsClassifier()))
	# models.append(('mlpc', MLPClassifier()))
	# models.append(('mlpr', MLPRegressor()))

	rfc = RandomForestClassifier(max_depth=40, min_samples_split=10, n_estimators=200)
	rfr = RandomForestRegressor(max_depth=40, min_samples_split=10, n_estimators=200)
	knnr = KNeighborsRegressor(n_neighbors=3, weights='distance', algorithm='auto')
	knnc = KNeighborsClassifier(n_neighbors=7, weights='distance', algorithm='auto')
	mlpc = MLPClassifier(hidden_layer_sizes=(50, 50))
	mlpr = MLPRegressor(hidden_layer_sizes=(50, 50))

	# models.append(('rfc', rfc))
	# models.append(('rfr', rfr))
	# models.append(('knnr', knnr))
	# models.append(('knnc', knnc))
	# models.append(('mlpc', mlpc))
	# models.append(('mlpr', mlpr))

	eclf = VotingClassifier(estimators=[('rfc', rfc), ('rfr', rfr), ('knnr', knnr), ('knnc', knnc), ('mlpc', mlpc),
										('mlpr', mlpr)], voting='soft', weights=[2, 1, 1, 2, 2, 1])
	eclf = VotingClassifier(estimators=[('rfc', rfc), ('knnc', knnc), ('mlpc', mlpc)],
										voting='hard')  #
	# eclf = VotingClassifier(estimators=[('rfc', rfc), ('rfr', rfr), ('knnr', knnr), ('knnc', knnc), ('mlpc', mlpc),
	# 									('mlpr', mlpr)], voting='hard')
	clf1 = DecisionTreeClassifier(max_depth=4)
	clf2 = KNeighborsClassifier(n_neighbors=7)
	clf3 = SVC(kernel='rbf', probability=True)
	# eclf = VotingClassifier(estimators=[('dt', clf1), ('knn', clf2), ('svc', clf3)], voting='soft',
	# 							 weights=[2, 1, 2])
	models.append(rfc)
	models.append(rfr)
	models.append(knnr)
	models.append(knnc)
	models.append(mlpc)
	models.append(mlpr)

	# return [(eclf, eclf_grid)]
	return [(models, [])]

def build_voting_clf():
	KNeighbors_R_grid = [{'n_neighbors': [3, 6, 9], 'weights': ['distance'], 'algorithm': ['auto']}]
	KNeighbors_C_grid = [{'n_neighbors': [3, 6, 9], 'weights': ['distance'], 'algorithm': ['auto']}]

	Random_Forest_classifier_grid = [{'max_depth': [40, 50], 'min_samples_split': [10], 'n_estimators': [200]}]
	Random_Forest_regressor_grid = [{'max_depth': [30, 40], 'min_samples_split': [10], 'n_estimators': [200]}]
	MLPClassifier_grid = [{'hidden_layer_sizes': [(50, 50)]}]
	MLPClassifier_grid = [{'hidden_layer_sizes': [(50, 50)]}]

	eclf_grid = {'rfc__max_depth': [40, 50], 'rfc__min_samples_split': [10], 'rfc__n_estimators': [200],
				 'rfr__max_depth': [40, 50], 'rfr__min_samples_split': [10], 'rfr__n_estimators': [200],
				 'knnr__n_neighbors': [3, 6, 9], 'knnr__weights': ['distance'], 'knnr__algorithm': ['auto'],
				 'knnc_n_neighbors': [3, 6, 9], 'knnc__weights': ['distance'], 'knnc__algorithm': ['auto'],
				 'mlpc__hidden_layer_sizes': [(50, 50)], 'mlpr__hidden_layer_sizes': [(50, 50)], }

	models = []
	rfc = RandomForestClassifier(max_depth=40, min_samples_split=10, n_estimators=200)
	knnc = KNeighborsClassifier(n_neighbors=7, weights='distance', algorithm='auto')
	mlpc = MLPClassifier(hidden_layer_sizes=(50, 50))

	# rfc = RandomForestClassifier()
	# knnc = KNeighborsClassifier()
	# mlpc = MLPClassifier()

	eclf = VotingClassifier(estimators=[('rfc', rfc), ('knnc', knnc), ('mlpc', mlpc)],
										voting='hard')

	return [(eclf, [])]

def build_extended():
	"""
	Builds a list of models along with their range of hyper-parameters. The range of params is increased
	:return: The list of (models, param_grid)
	"""
	KNeighborsRegressor_grid = [{'n_neighbors': range(3, 25, 3), 'weights': KNNweigths, 'algorithm': KNNAlgo}]
	Random_Forest_grid = [{'max_depth': max_depth_interval_h, 'min_samples_split': min_samples_split_interval_h,
						   'n_estimators': n_estimators_interval_h}]
	MLPClassifier_grid = [{'alpha': alpha_1_short_interval, 'hidden_layer_sizes': hidden_layer_sizes_interval,
						   'activation': MLP_activation, 'solver': MLP_solver, 'shuffle': true_false_interval}]
	MLPClassifier_short_grid = [{'alpha': alpha_1_interval_h, 'hidden_layer_sizes': hidden_layer_sizes_interval_h,
								 'activation': MLP_activation}]

	MLPClassifier_tiny_grid = [{'alpha': alpha_1_interval_h, 'hidden_layer_sizes': hidden_layer_sizes_interval_h}]
	MLPClassifier_ultra_tiny_grid = [{'hidden_layer_sizes': hidden_layer_sizes_interval_h}]

	models = []

	models.append((RandomForestClassifier(), Random_Forest_grid))
	models.append((RandomForestRegressor(), Random_Forest_grid))
	models.append((MLPClassifier(), MLPClassifier_ultra_tiny_grid))
	models.append((MLPRegressor(), MLPClassifier_ultra_tiny_grid))
	models.append((KNeighborsRegressor(), KNeighborsRegressor_grid))
	models.append((KNeighborsClassifier(), KNeighborsRegressor_grid))

	return models



def build_regressors():
	"""
	build the list of regressors along with their range of hyper-parameters
	parameter list, copy paste this to avoid re-typing :
	[{'loss': , 'alpha': , 'l1_loss': , 'penalty': ,
	'max_iter': , 'l1_ratio': , 'tol': , 'epsilon': ,
	'learning_rate': , 'eta0': , 'power_t': , 'degree': , 'kernel': , 'coef0': , 'nu': '}]
	:return: the list of (regressors, param_grid)
	"""

	regressors = []
	BayesianRidge_grid = [
		{'n_iter': max_iter_interval, 'tol': tol_interval, 'alpha_1': alpha_1_interval,
		 'alpha_2': alpha_1_interval, 'lambda_1': alpha_1_interval, 'lambda_2': alpha_1_interval}]
	BayesianRidge_short_grid = [
		{'n_iter': max_iter_short_interval, 'tol': tol_short_interval, 'alpha_1': alpha_1_short_interval,
		 'alpha_2': alpha_1_short_interval, 'lambda_1': alpha_1_short_interval, 'lambda_2': alpha_1_short_interval}]

	DecisionTreeRegressor_grid = [{'splitter': splitter_interval,
								   'max_depth': max_depth_interval, 'min_samples_split': min_samples_split_interval}]

	Elastic_grid = [{'tol': [1e-3, 1e-2, 1e-1, 1], 'alpha': [1, 10, 100, 1000],
					 'max_iter': max_iter_elastic_interval, 'l1_ratio': l1_ratio_interval}]
	Elastic_short_grid = [
		{'tol': tol_short_interval, 'alpha': [1, 10, 100], 'l1_ratio': l1_ratio_short_interval}]  # Otlet grid

	ElasticCV_grid = [{'eps': [1e-3, 1e-2], 'tol': [1e-4, 1e-3], 'n_alphas': n_alphas_interval,
					   'max_iter': max_iter_elastic_interval, 'l1_ratio': l1_ratio_short_interval}]

	ElasticCV_short_grid = [{'tol': tol_short_interval, 'n_alphas': n_alphas_interval,
							 'max_iter': max_iter_elastic_interval, 'l1_ratio': l1_ratio_short_interval}]

	GPR_grid = [
		{'kernel': gp_kernels, 'alpha': [1, 10, 100, 1000]}
	]

	KNeighborsRegressor_grid = [{'n_neighbors': range(1, 25, 3), 'weights': KNNweigths, 'algorithm': KNNAlgo}]
	KNeighborsRegressor_short_grid = [{'n_neighbors': range(1, 25, 6), 'weights': KNNweigths}]
	Lars_grid = [{'n_nonzero_coefs': [100, 200, 500, 750, 1000]}]
	Lasso_grid = [{'tol': [1e-3, 1e-2, 1e-1, 1], 'alpha': [1e-4, 1e-3, 0.01, 0.1, 0.5, 1, 10],
				   'max_iter': [100, 250, 500, 1000, 2000]}]
	LassoLars_grid = [
		{'alpha': [1e-4, 1e-3, 0.01, 0.1, 0.5, 1, 10], 'max_iter': [100, 250, 500, 1000, 2000]}]
	LassoCV_grid = [{'eps': [1e-3, 1e-2, 1e-1, 1], 'n_alphas': [1, 10, 100, 1000], 'tol': [1e-3, 1e-2, 1e-1, 1],
					 'max_iter': [3000]}]  # Otlet data set : convergence warning
	LinearSVR_grid = [
		{'C': C_short_interval, 'epsilon': epsilon_interval, 'tol': tol_short_interval,
		 'max_iter': max_iter_short_interval}]

	MLPRegressor_grid = [{'hidden_layer_sizes': hidden_layer_sizes_interval, 'alpha': alpha_1_short_interval,
						  'activation': MLP_activation, 'solver': MLP_solver, 'shuffle': true_false_interval}]

	MLPRegressor_short_grid = [{'alpha': alpha_1_short_interval, 'hidden_layer_sizes': hidden_layer_sizes_interval}]

	NUSVR_grid = [{'nu': nu_interval, 'kernel': kernel_interval, 'degree': degree_interval, 'gamma': gamma_interval,
				   'coef0': coef0_interval, 'C': C_short_interval, 'tol': tol_short_interval}]
	OMP_grid = [{'n_nonzero_coefs': [100, 200, 500, 750, 1000], 'tol': [0, 1e-3, 1e-2, 1e-1, 1]}]

	PassiveAggressive_grid = [{'C': C_interval, 'epsilon': epsilon_interval, 'max_iter': max_iter_interval,
							   'tol': tol_interval}]
	PassiveAggressive_short_grid = [{'C': C_interval, 'epsilon': epsilon_interval,
									 'tol': tol_interval}]

	Perceptron_grid = [{'penalty': ["l2"], 'alpha': alpha_interval,
						'max_iter': [3000], 'tol': tol_interval, 'eta0': eta0_interval}]

	RandomForestRegressor_grid = [{'max_depth': max_depth_interval, 'min_samples_split': min_samples_split_interval,
								   'n_estimators': n_estimators_interval}]

	RandomForestRegressor_short_grid = [{'n_estimators': n_estimators_interval}]
	Ridge_grid = [
		{'tol': [1e-3, 1e-2, 1e-1, 1], 'alpha': [1, 10, 100, 1000]}]

	SGD_grid = [{'loss': ['squared_loss'], 'alpha': alpha_SGD_interval, 'penalty': ["l2"],
				 'max_iter': max_iter_elastic_interval, 'l1_ratio': l1_ratio_interval, 'tol': tol_interval,
				 'epsilon': epsilon_interval,
				 'learning_rate': ["invscaling"], 'eta0': eta0_interval, 'power_t': power_t_interval,
				 'shuffle': [True]}]

	SGD_small_grid = [{'loss': ['squared_loss'], 'alpha': alpha_SGD_interval,
					   'epsilon': epsilon_interval, 'eta0': eta0_interval, 'power_t': power_t_interval}]

	SVR_grid = [{'kernel': kernel_interval, 'degree': degree_interval, 'gamma': gamma_interval, 'coef0': coef0_interval,
				 'C': C_short_interval, 'epsilon': epsilon_interval, 'tol': tol_short_interval}]
	SVR_grid = [
		{'kernel': kernel_short_interval, 'degree': degree_interval, 'gamma': gamma_interval, 'coef0': coef0_interval,
		 'C': [1]}]

	"""APPEND THE WANTED CLASSIFIERS : comment the regressors you do not want to use"""

	# regressors.append((ARDRegression(), []))  # Memory Error
	regressors.append((BayesianRidge(), BayesianRidge_short_grid))  # time > 300 secs + bug with Otlet data set Inf Nan

	regressors.append((ElasticNet(), Elastic_grid))  # did not converged because of max_iter or alpha
	regressors.append((ElasticNetCV(), ElasticCV_grid))  # alpha must be provided for l1+ratio !
	# Otlet data set : convergence warning

	regressors.append((KNeighborsRegressor(), KNeighborsRegressor_short_grid))  # time ~~ 2min30 (scoring)

	regressors.append((Lars(), Lars_grid))  # time < 10 secs
	regressors.append((Lasso(), Lasso_grid))  # time > 10 secs
	regressors.append((LassoCV(), LassoCV_grid))  # wrong params
	regressors.append((LassoLars(), LassoLars_grid))  # time ~~ 10 secs some scorer took too much time
	regressors.append((LassoLarsCV(), []))  # time < 10 secs + bug with Otlet data set ValueError: x and y
	# arrays must have at least 2 entries
	regressors.append((LassoLarsIC(), []))  # time < 10 secs

	regressors.append((LinearRegression(), []))  # time < 10 secs
	regressors.append((LinearSVR(), LinearSVR_grid))  # time > 300 secs

	regressors.append((NuSVR(), []))  # time > 10 secs + error about not converged
	regressors.append((OrthogonalMatchingPursuit(), OMP_grid))  # time > 10 secs
	# Orthogonal matching pursuit ended prematurely due to linear dependence in the dictionary.
	# The requested precision might not have been met.

	regressors.append((OrthogonalMatchingPursuitCV(), []))  # time < 10 secs
	# Otlet : ValueError: attempt to get argmin of an empty sequence even after pre_processing

	regressors.append((PassiveAggressiveRegressor(), PassiveAggressive_short_grid))  # time ~~ 180 secs
	regressors.append((Perceptron(), Perceptron_grid))  # time ~~ 180 secs
	# ValueError: The number of class labels must be greater than one

	regressors.append((Ridge(), Ridge_grid))  # time < 10 secs
	regressors.append((SGDRegressor(), SGD_small_grid))  # time > 300 secs + ConvergenceWarning
	regressors.append((SVR(), []))  # time > 300 secs
	# regressors.append((GaussianProcessRegressor(), []))  # Return memory error
	# numpy.linalg.linalg.LinAlgError: ("The kernel, ExpSineSquared(length_scale=1, periodicity=5) + WhiteKernel
	# (noise_level=0.1), is not returning a positive definite matrix. Try gradually increasing the 'alpha' parameter
	#  of your GaussianProcessRegressor estimator.", '64-th leading minor of the array is not positive definite')

	regressors.append((DecisionTreeRegressor(), DecisionTreeRegressor_grid))
	regressors.append((RandomForestRegressor(), RandomForestRegressor_short_grid))
	regressors.append((MLPRegressor(), []))  # convergence warning  Otlet data : no grid search

	return regressors


def build_classifiers():
	"""
	build the list of classifiers along with their range of hyper-parameters
	parameter list, copy paste this to avoid re-typing :
	[{'loss': , 'alpha': , 'l1_loss': , 'penalty': ,
	'max_iter': , 'l1_ratio': , 'tol': , 'epsilon': ,
	'learning_rate': , 'eta0': , 'power_t': , 'degree': , 'kernel': , 'coef0': , 'nu': '}]
	:return: the list of (classifiers, param_grid)
	"""
	classifiers = []
	SVC_grid = [{'kernel': kernel_short_interval, 'gamma': [1e-3, 1e-4], 'C': [1]}]
	SVC_grid = [{'kernel': ['rbf']}]
	LDA_grid = [{'solver': ['svd'], 'tol': [1e-4]}]
	QDA_grid = [{'reg_param': [0, 1, 2, 3, 5, 10], 'tol': [1e-4]}]
	NB_grid = [{'alpha': alpha_interval}]
	Decision_Tree_grid = [{'splitter': splitter_interval, 'max_depth': max_depth_interval,
						   'min_samples_split': min_samples_split_interval}]
	KNeighborsClassifier_grid = [{'n_neighbors': range(1, 25, 3), 'weights': KNNweigths, 'algorithm': KNNAlgo}]
	KNeighborsClassifier_short_grid = [{'n_neighbors': range(1, 25, 6), 'weights': KNNweigths}]
	Random_Forest_grid = [{'max_depth': max_depth_interval, 'min_samples_split': min_samples_split_interval,
						   'n_estimators': n_estimators_interval}]
	RandomForest_short_grid = [{'n_estimators': n_estimators_interval}]

	MLPClassifier_grid = [{'alpha': alpha_1_short_interval, 'hidden_layer_sizes': hidden_layer_sizes_interval,
						   'activation': MLP_activation, 'solver': MLP_solver, 'shuffle': true_false_interval}]
	MLPClassifier_short_grid = [{'alpha': alpha_1_short_interval, 'hidden_layer_sizes': hidden_layer_sizes_interval,
								 'shuffle': true_false_interval}]

	"""APPEND THE WANTED CLASSIFIERS : comment the clf you do not want to use"""

	classifiers.append((LinearDiscriminantAnalysis(), LDA_grid))
	# The priors do not sum to 1 + ConvergenceWarning
	classifiers.append((QuadraticDiscriminantAnalysis(), QDA_grid))
	# invalid value encountered in log u = np.asarray([np.sum(np.log(s)) for s in self.scalings_])
	# 'is ill defined.' % str(self.classes_[ind]))
	# ValueError: y has only 1 sample in class 10, covariance is ill defined. + ConvergenceWarning
	# classifiers.append((MultinomialNB(), NB_grid))  # X must be positive ... error

	classifiers.append((SVC(), SVC_grid))  # ???  Otlet : ValueError: The number of classes has to be greater than one; got 1
	classifiers.append((BernoulliNB(), NB_grid))
	classifiers.append((GaussianNB(), []))
	classifiers.append(
		(DecisionTreeClassifier(), Decision_Tree_grid))  # Otlet : The least populated class in y has only 1 members
	classifiers.append((RandomForestClassifier(), RandomForest_short_grid))
	classifiers.append((MLPClassifier(), []))  # Otlet data : no grid search
	classifiers.append((KNeighborsClassifier(), KNeighborsClassifier_short_grid))

	return classifiers


# @profile
def build_grid(model, X, Y, queue):
	"""
	Function performing a grid search and putting the best model found inside the queue
	Useful for the subprocess computation of the grid-search
	:param model: the model to be fitted and tested
	:param X: Sensor data
	:param Y: person count data
	:param queue: The queue (from processes library) to put the best model
	:return: the model itself (if the queue is unused)
	"""
	model.fit(X, Y)
	queue.put(model)
	return model