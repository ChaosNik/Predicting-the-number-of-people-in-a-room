#!/usr/bin/env python
'''
    File name: ML_helper.py
    Author: Nicolas Bockstael && Alexandre Jadin
    Date created: 10/2/2018
    Date last modified: 04/06/2018
    Python Version: 3.6.3

	Description : Helper file with numerous functions for various pruposes
'''

from collections import Counter
from queue import Empty

import matplotlib

matplotlib.use('Agg')  # avoid showing the figures while computing
import matplotlib.pyplot as plt
import multiprocessing

from sklearn import metrics, preprocessing

from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_validate, GridSearchCV, learning_curve
import random, math
import pickle
import logging
import logging.handlers
import os, sys, time
import itertools
import pydevd
import numpy as np

from MachineLearning.add_aggregate import add_aggregate
import MachineLearning.pre_processing as pre_processing
from MachineLearning.build_models import build_classifiers, build_regressors, build_grid, build_extended, \
	build_custom_voting_clf, build_voting_clf

import MachineLearning.settings as settings

def log_setup():
	"""
	Creates a log dirctory and initiate a log file with the current date
	:return: the created logger
	"""
	if not os.path.isdir("./logs/"):
		os.mkdir("./logs/")
	name = time.strftime("%Y-%m-%d")
	log_handler = logging.handlers.WatchedFileHandler("./logs/" + name + '.log')
	formatter = logging.Formatter(
		'%(levelname)s - %(asctime)s - %(filename)s - [%(funcName)s]  line %(lineno)d: %(message)s',
		'%b %d %H:%M:%S')
	formatter.converter = time.localtime
	log_handler.setFormatter(formatter)
	logger = logging.getLogger()
	#  Close the previous logger
	for handler in logger.handlers:
		handler.close()
		logger.removeHandler(handler)
	logger.addHandler(log_handler)
	logger.setLevel(logging.DEBUG)
	logging.captureWarnings(True)  # Capture warnings into logging
	return logger

def subproc_grid_search(model_grid, X_training, Y_training, method_name="", scorer_name=""):
	"""
	returning the cross-validated model computed in a subprocess with timeout
	returns an error while debugging
	:param model_grid: the grid of parameters to be considered according to sci-kit grid search method
	:param X_training: Input features values
	:param Y_training: Labels (person count) for the samples in features
	:return: the best classifier if found, None if process was killed
	"""
	clf_list_grid = multiprocessing.Queue()
	# Start bar as a process
	p = multiprocessing.Process(target=build_grid, args=(model_grid, X_training, Y_training, clf_list_grid))
	# p = Process(target=fun, args=(clf_list,))
	p.start()
	# Wait for CROSS_VALIDATION_TIMEOUT seconds or until process finishes
	toret = None
	try:
		toret = clf_list_grid.get(block=True, timeout=settings.CROSS_VALIDATION_TIMEOUT)
	except Empty:
		print("List of clf is empty, process timed-out > " + str(settings.CROSS_VALIDATION_TIMEOUT))
		logging.warning("List of clf is empty, process timed-out > " + str(settings.CROSS_VALIDATION_TIMEOUT))
		print("Killing process : " + method_name + " " + scorer_name)
		logging.warning("Killing process : " + method_name + " " + scorer_name)
		# Terminate
		p.terminate()
		p.join()
		return None

	p.join(settings.CROSS_VALIDATION_TIMEOUT)

	if p.is_alive():
		# If process is still active : Should never happen
		print("Killing process : " + method_name + " " + scorer_name)
		logging.warning("Killing process : " + method_name + " " + scorer_name)
		# Terminate
		p.terminate()
		p.join()
		return None
	else:
		print("grid_search succedeed")
		logging.info("grid_search succedeed")
		return toret
    
def get_pre_processed_data(verbose=False, filename=settings.DATA_FILENAME, form_filename=settings.COUNT_FILENAME,
						   consolidate_filename=settings.consolidate_FILENAME,
						   pre_processed_filename=settings.pre_processed_FILENAME,
						   remove_bad_features=False, remove_aggregates=False, other_input=None, other_form=None):
	"""
	Pre-process + feature engineered data
	:param verbose: whether to print additionnal information or not
	:param filename: File containing the sensor data
	:param form_filename: File containing the person count data, direct output of a Google Form
	:param consolidate_filename: filename to export the formatted, human readable csv
	:param pre_processed_filename: filename to export the formatted, human readable csv
	:param remove_bad_features: Whether to remove featured considered as "bad" aka "motion"
	:param remove_aggregates: Whether to remove the extracted features aka "co2_smoothed" and "co2_derived"
	:param other_input: Optionnal addiftionnal file containing the sensor data
	:param other_form: Optionnal addiftionnal file containing the person count data
	:return: keys, training data and person count unscaled
	"""
	# global settings.FILENAME, settings.FORM_FILENAME, settings.pre_processed_FILENAME
	features_to_remove = ["motion"]
	if not settings.PICKLE_DATA:

		data = pre_processing.pre_process(filename, form_filename, pre_processed_filename,
										  export_pre_processed=True, other_input=other_input, other_form=other_form)
		if verbose:
			print(filename + " has been pre-processed")
			logging.info(filename + " has been pre-processed")
		timed_data = []
		person_count = []
		for line in data:
			person_count.append(int(line["person_count"]))
			for k in settings.NAMES_TO_REMOVE:
				del line[k]
			if remove_bad_features:
				for k in features_to_remove:  # Remove the non important features
					del line[k]
			for k in line:
				line[k] = int(line[k])  # Convert to int for further computations
			timed_data.append(line)

		(keys, mydata) = add_aggregate(data, add_aggregate=not remove_aggregates)  # add smoothed, derived CO2

		with open(settings.PREPROCESSED_PICKLE_NAME, "wb") as myfile:  # dump for quicker retrieval if necessary
			pickle.dump((keys, mydata, person_count), myfile)

		"""CREATING CONSOLIDATED DATA SET"""
		consolidated = []
		for line, count in zip(mydata, person_count):
			consolidated.append(line + [count])
		pre_processing.export_to_csv(consolidated, consolidate_filename, keys + ["person_count"], True)

	else:
		if verbose:
			print("using pickle file")
			logging.warning("using pickle file")
		with open(settings.PREPROCESSED_PICKLE_NAME, "rb") as myfile:  # dump for quicker retrieval if necessary
			(keys, mydata, person_count) = pickle.load(myfile)
	return keys, mydata, person_count

def compute_learning_curve(estimator, X, Y, cv, train_sizes, score_fun):
	"""
	Computation of the learning curve of a model on a given set X:Y, performs the samples selection randomly
	:param estimator: The estimator to be tested
	:param X: the sensor data
	:param Y: the person count data
	:param cv: number of passes for the cross validation
	:param train_sizes: array with the different proportions of the set to train the estimator
	:param score_fun: scoring function to evaluate the performance
	:return: sizes, train_results, test_results
	"""
	test_results = []
	train_results = []
	sizes = []
	for i_ratio, ratio in enumerate(train_sizes):
		test_results.append([])
		train_results.append([])

		n_samples = len(X)
		n_train_samples = int(n_samples * ratio)
		sizes.append(n_train_samples)
		n_test_samples = n_samples - n_train_samples

		for iteration in range(cv):
			train_indices = random.sample(range(0, n_samples), n_train_samples)
			train_set = [X[index] for index in train_indices]
			train_person_count = [Y[index] for index in train_indices]
			test_set = [i for j, i in enumerate(X) if j not in train_indices]
			test_person_count = [i for j, i in enumerate(Y) if j not in train_indices]
			X_train = train_set
			Y_train = train_person_count
			X_test = test_set
			Y_test = test_person_count

			# scale data
			scaler = preprocessing.StandardScaler().fit(X_train)
			X_train = list(scaler.transform(X_train))
			X_test = list(scaler.transform(X_test))

			if isinstance(estimator, list):
				for m in estimator:
					m.fit(X_train, Y_train)
				Y_pred = predict_voting(X_test, estimator)
				test_score = score_fun(Y_test, Y_pred)
				test_results[i_ratio].append(test_score)
				Y_pred = predict_voting(X_train, estimator)
				train_score = score_fun(Y_train, Y_pred)
				train_results[i_ratio].append(train_score)
			else:
				estimator.fit(X_train, Y_train)
				Y_pred = estimator.predict(X_test)
				test_score = score_fun(Y_test, Y_pred)
				test_results[i_ratio].append(test_score)
				Y_pred = estimator.predict(X_train)
				train_score = score_fun(Y_train, Y_pred)
				train_results[i_ratio].append(train_score)

	return sizes, train_results, test_results

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
						n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5), scorer=None):
	"""
	Source : http://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html#sphx-glr-auto-
	examples-model-selection-plot-learning-curve-py
	With some modifications in order to store the result
	Generate a simple plot of the test and training learning curve.

	Parameters
	----------
	estimator : object type that implements the "fit" and "predict" methods
		An object of that type which is cloned for each validation.

	title : string
		Title for the chart.

	X : array-like, shape (n_samples, n_features)
		Training vector, where n_samples is the number of samples and
		n_features is the number of features.

	y : array-like, shape (n_samples) or (n_samples, n_features), optional
		Target relative to X for classification or regression;
		None for unsupervised learning.

	ylim : tuple, shape (ymin, ymax), optional
		Defines minimum and maximum yvalues plotted.

	cv : int, cross-validation generator or an iterable, optional
		Determines the cross-validation splitting strategy.
		Possible inputs for cv are:
		  - None, to use the default 3-fold cross-validation,
		  - integer, to specify the number of folds.
		  - An object to be used as a cross-validation generator.
		  - An iterable yielding train/test splits.

		For integer/None inputs, if ``y`` is binary or multiclass,
		:class:`StratifiedKFold` used. If the estimator is not a classifier
		or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

		Refer :ref:`User Guide <cross_validation>` for the various
		cross-validators that can be used here.

	n_jobs : integer, optional
		Number of jobs to run in parallel (default 1).
	"""
	plt.figure()
	plt.title(title)
	if ylim is not None:
		plt.ylim(*ylim)
	plt.xlabel("Training examples")
	plt.ylabel("Score")
	# train_sizes, train_scores, test_scores = learning_curve(
	# 	estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes, scoring=scorer)
	train_sizes, train_scores, test_scores = compute_learning_curve(
		estimator, X, y, cv=cv, train_sizes=train_sizes, score_fun=scorer)
	train_scores_mean = np.mean(train_scores, axis=1)
	train_scores_min = np.min(train_scores, axis=1)
	train_scores_max = np.max(train_scores, axis=1)
	test_scores_mean = np.mean(test_scores, axis=1)
	test_scores_min = np.min(test_scores, axis=1)
	test_scores_max = np.max(test_scores, axis=1)

	plt.grid()

	plt.fill_between(train_sizes, train_scores_min,
					 train_scores_max, alpha=0.1,
					 color="r")
	plt.fill_between(train_sizes, test_scores_min,
					 test_scores_max, alpha=0.1, color="g")
	plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
			 label="Training score")
	plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
			 label="Testing score")

	plt.legend(loc="best")
	plt.savefig(settings.FIGURES_FOLDER + title)
	plt.close()


def plot_feature_importance(forest, features, title, isForest=True):
	"""
	Source : http://scikit-learn.org/stable/auto_examples/ensemble/plot_forest_importances.html
	:param forest: forest to be analyzed
	:param features: list of the name of the features
	:param title: title of the image to export
	:return:
	"""
	importances = forest.feature_importances_
	if isForest:
		std = np.std([tree.feature_importances_ for tree in forest.estimators_],
					 axis=0)
		indices = np.argsort(importances)[::-1]  # minimizing standard deviation
	else:
		std = importances
		indices = np.argsort(-importances)[::-1]  # maximizing feature importance

	# Print the feature ranking
	logging.info("Feature ranking:")
	print("Feature ranking:")

	for f in range(len(features)):
		logging.info("%d. feature %s (%f)" % (f + 1, features[indices[f]], importances[indices[f]]))
		print("%d. feature %s (%f)" % (f + 1, features[indices[f]], importances[indices[f]]))

	# Plot the feature importances of the forest
	matplotlib.rcParams.update({'font.size': 8})
	plt.figure()
	plt.title("Feature importances")
	plt.bar(range(len(features)), importances[indices],
			color="r", yerr=std[indices], align="center")
	plt.xticks(range(len(features)), (features[index] for index in indices))
	plt.xlim([-1, len(features)])
	plt.savefig(settings.FIGURES_FOLDER + title)
	plt.close()


def plot_zero_accuracy(zero_scores, non_zero_scores, filename):
	"""
	Plot the zero vs non zero scores
	:param zero_scores: array containing the values of the strict accuracy on zero samples
	:param non_zero_scores: array containing the values of the strict accuracy on non-zero samples
	:param filename: file to export the final plot
	:return: /
	"""
	fig, f_axes = plt.subplots(ncols=2, nrows=1)

	f_axes[0].boxplot(zero_scores)
	f_axes[0].set_ylim([0, 1])
	f_axes[0].set_title("Zero score", fontsize=10)
	f_axes[1].boxplot(non_zero_scores)
	f_axes[1].set_ylim([0, 1])
	f_axes[1].set_title("non_zero score", fontsize=10)
	plt.savefig(settings.FIGURES_FOLDER + filename)
	plt.close(fig)


def featureImportance(model, score_fun, model_name="", title=""):
	"""
	Compute the feature importance by comparing the accuracy of the model with all features then with a single feature
	removed
	:param model: the estimator to be analyzed
	:param score_fun: scoring function, evaluation metric
	:param model_name: name of the evaluated model (useful for logs)
	:param title: name of the final plot Note the title is appended to the figure folder from settings.py
	:return: /
	"""
	print("compute feature importance")

	features = ["all", "co2", "co2_smoothed", "light", "motion", "temperature", "co2_derived", "humidity"]
	results = {}
	averages = {}
	bars = {}

	for feature in features:
		results[feature] = []

		(keys, X, Y) = get_pre_processed_data()

	for iteration in range(5):
		n_samples = len(X)
		n_train_samples = int(n_samples * settings.TEST_SET_RATIO)
		n_test_samples = n_samples - n_train_samples

		train_indices = random.sample(range(0, n_samples), n_train_samples)
		train_set = [X[index] for index in train_indices]
		train_person_count = [Y[index] for index in train_indices]
		test_set = [i for j, i in enumerate(X) if j not in train_indices]
		test_person_count = [i for j, i in enumerate(Y) if j not in train_indices]
		X_train = train_set
		Y_train = train_person_count
		X_test = test_set
		Y_test = test_person_count

		# scale data
		scaler = preprocessing.StandardScaler().fit(X_train)
		X_train = list(scaler.transform(X_train))
		X_test = list(scaler.transform(X_test))

		for feature in features:
			new_X_train = []
			new_X_test = []
			toDrop = []
			if feature != "all":
				if feature == "co2":
					toDrop.append(keys.index("co2_smoothed"))
				elif feature == "co2_smoothed":
					toDrop.append(keys.index("co2"))
				toDrop.append(keys.index(feature))
				for row in X_train:
					new_X_train.append([elem for num, elem in enumerate(row) if num not in toDrop])
				for row in X_test:
					new_X_test.append([elem for num, elem in enumerate(row) if num not in toDrop])
			else:
				new_X_train = X_train
				new_X_test = X_test

			if isinstance(model, list):
				for m in model:
					m.fit(new_X_train, Y_train)
			else:
				model.fit(new_X_train, Y_train)

			if isinstance(model, list):
				Y_pred = predict_voting(new_X_test, model)
			else:
				Y_pred = model.predict(new_X_test)

			test_score = score_fun(Y_test, Y_pred)

			results[feature].append(test_score)

	for feature in features:
		averages[feature] = sum(results[feature]) / len(results[feature])
		if feature != 'all':
			bars[feature] = (averages['all'] - averages[feature]) / averages['all']

	matplotlib.rcParams.update({'font.size': 7})
	plt.figure()
	plt.title("Feature importances %s" % (model_name))
	plt.bar(range(len(bars)), bars.values(), color="r", align="center")
	# plt.bar(range(len(features)), bars.values(),color="r", yerr=std[indices], align="center")
	plt.xticks(range(len(bars)), bars.keys())
	plt.xlim([-1, len(bars)])
	plt.savefig(settings.FIGURES_FOLDER + title)
	plt.close()
	print("end feature importance")


def singleFeatureImportance(model, score_fun, model_name="", title=""):
	"""
		Compute the feature importance by comparing the accuracy of the model with a single feature
		:param model: the estimator to be analyzed
		:param score_fun: scoring function, evaluation metric
		:param model_name: name of the evaluated model (useful for logs)
		:param title: name of the final plot Note the title is appended to the figure folder from settings.py
		:return: /
		"""
	print("compute single feature importance")

	features = ["all", "co2", "co2_smoothed", "light", "motion", "temperature", "co2_derived", "humidity"]
	results = {}
	averages = {}
	bars = {}

	for feature in features:
		results[feature] = []

		(keys, X, Y) = get_pre_processed_data()

	for iteration in range(5):
		n_samples = len(X)
		n_train_samples = int(n_samples * settings.TEST_SET_RATIO)
		n_test_samples = n_samples - n_train_samples

		train_indices = random.sample(range(0, n_samples), n_train_samples)
		train_set = [X[index] for index in train_indices]
		train_person_count = [Y[index] for index in train_indices]
		test_set = [i for j, i in enumerate(X) if j not in train_indices]
		test_person_count = [i for j, i in enumerate(Y) if j not in train_indices]
		X_train = train_set
		Y_train = train_person_count
		X_test = test_set
		Y_test = test_person_count

		# scale data
		scaler = preprocessing.StandardScaler().fit(X_train)
		X_train = list(scaler.transform(X_train))
		X_test = list(scaler.transform(X_test))

		for feature in features:
			new_X_train = []
			new_X_test = []
			toDrop = []
			if feature != "all":
				toDrop = [num for num, elem in enumerate(keys) if elem != feature]
				for row in X_train:
					new_X_train.append([elem for num, elem in enumerate(row) if num not in toDrop])
				for row in X_test:
					new_X_test.append([elem for num, elem in enumerate(row) if num not in toDrop])
			else:
				new_X_train = X_train
				new_X_test = X_test

			model.fit(new_X_train, Y_train)
			Y_pred = model.predict(new_X_test)
			test_score = score_fun(Y_test, Y_pred)

			results[feature].append(test_score)

	for feature in features:
		averages[feature] = sum(results[feature]) / len(results[feature])
		if feature != 'all':
			bars[feature] = averages[feature] / averages['all']

	matplotlib.rcParams.update({'font.size': 7})
	plt.figure()
	plt.title("Feature importances %s" % (model_name))
	plt.bar(range(len(bars)), bars.values(), color="r", align="center")
	# plt.bar(range(len(features)), bars.values(),color="r", yerr=std[indices], align="center")
	plt.xticks(range(len(bars)), bars.keys())
	plt.xlim([-1, len(bars)])
	plt.savefig(settings.FIGURES_FOLDER + title)
	plt.close()
	print("end single feature importance")

def correlation(X, keys):
	"""
	Compute the correlation between variables in a dataset using pandas and seaborn and yield a plot of the result
	:param X: the input data set
	:param keys: the name of the features. Note that the order must match the data set for the plot to be meaningful
	:return:
	"""
	from pandas import DataFrame
	import seaborn as sns

	print("correlation")
	matplotlib.rcParams.update({'font.size': 8})
	# plt.figure()
	f, ax = plt.subplots(figsize=(10, 8))
	corr = DataFrame(X).corr()
	sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),
				square=True, ax=ax)
	# corr = X.corr()
	plt.title("Feature correlation", fontsize=12)
	plt.xticks(range(len(X[0])), keys)
	plt.yticks(range(len(X[0])), keys)
	plt.savefig(settings.FIGURES_FOLDER + "correlation")
	plt.close()

	print("correlation done")


def predict_voting(data_to_label, models, soft_voting=None):
	"""
	Custom function used to compute the prediction of the voting algorithm (using classifiers and regressors)
	:param data_to_label: the samples to classify/assign a value
	:param models: the Voting algorithm is a list of models.
	:param soft_voting: None if hard voting, otherwise an array with the weigths given to each model
	:return: an array with the prediction for each input sample
	"""
	norm = 1
	if soft_voting is None:
		soft_voting = [1] * len(models)
		norm = len(models)
	else:
		assert len(models) == len(soft_voting)
		norm = sum(soft_voting)  # define the norm of the soft_voting vector to make the results coherent


	predictions = []
	for model in models:
		predictions.append(model.predict(data_to_label))

	to_ret = []
	for index in range(len(predictions[0])):  # Iterate over each example
		lst = [item[index] for item in predictions]
		# res = sum([a * b for a, b in zip(lst, soft_voting)])/norm  # multiply each clf output by weight, sum and div by norm

		# Rounding to nearest integer and making a vote for a number
		lst2 = [int(round(i)) for i in lst]
		res = most_common(lst2)

		to_ret.append(res)
	return to_ret


def most_common(lst):
	"""
	Helper function computing which element is the most common inside a list
	:param lst: the list where the element has to be found
	:return: the most common value
	"""
	data = Counter(lst)
	return data.most_common(1)[0][0]