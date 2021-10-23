#!/usr/bin/env python
'''
    File name: ML.py
    Author: Nicolas Bockstael && Alexandre Jadin
    Date created: 10/2/2018
    Date last modified: 04/06/2018
    Python Version: 3.6.3

	Description : Main file for launching the Machine Learning computations, also contains the main parameters
'''

import matplotlib

matplotlib.use('Agg')  # avoid showing the figures while computing

from MachineLearning.ML_helpers import *
from MachineLearning.scorings import *

random.seed(42)  # assure reproducibility


if __name__ == '__main__':

	# Creating a log also redirecting stdout and stderr towards log
	# Note : debug level to show the usual print
	log = log_setup()

	# Creating result folders if non-existing
	if not os.path.isdir(settings.FIGURES_FOLDER):
		os.makedirs(settings.FIGURES_FOLDER, exist_ok=True)
	if not os.path.isdir(settings.DATA_FOLDER):
		os.makedirs(settings.DATA_FOLDER, exist_ok=True)

	"""RETRIEVING PROCESSED DATA FROM INPUT FILES"""
	if settings.TO_COMPUTE_DATASET == settings.ONE_DATASET:
		(keys, mydata, person_count) = get_pre_processed_data()
		X = mydata
		Y = person_count

	if settings.TO_COMPUTE_DATASET == settings.TWO_DATASET_BALANCED:
		(First_keys, First_mydata, First_person_count) = get_pre_processed_data(
															filename=settings.DATA_FILENAME,
															form_filename=settings.COUNT_FILENAME)
		(Second_keys, Second_mydata, Second_person_count) = get_pre_processed_data(
																filename=settings.SECOND_DATA_FILENAME,
																form_filename=settings.SECOND_COUNT_FILENAME)
		# need first dataset to be bigger than second one
		assert len(First_mydata) >= len(Second_mydata)
		Parnas_indices = random.sample(range(0, len(First_mydata)), len(Second_mydata))
		Parnas_resized_data = [First_mydata[index] for index in Parnas_indices]
		Parnas_resized_person_count = [First_person_count[index] for index in Parnas_indices]

		X = Parnas_resized_data + Second_mydata
		Y = Parnas_resized_person_count + Second_person_count
		keys = First_keys

	if settings.TO_COMPUTE_DATASET == settings.COMPUTE_ON_ONE_TEST_ON_TWO:
		(First_keys, First_mydata, First_person_count) = get_pre_processed_data(
															filename=settings.DATA_FILENAME,
															form_filename=settings.COUNT_FILENAME)
		(Second_keys, Second_mydata, Second_person_count) = get_pre_processed_data(
															filename=settings.SECOND_DATA_FILENAME,
													   		form_filename=settings.SECOND_COUNT_FILENAME)
		X = First_mydata
		Y = First_person_count
		keys = First_keys

	if settings.TO_COMPUTE_DATASET == settings.TWO_DATASET:
		(both_keys, both_mydata, both_person_count) = get_pre_processed_data(
															filename=settings.DATA_FILENAME,
															form_filename=settings.COUNT_FILENAME,
															other_input=settings.SECOND_DATA_FILENAME,
															other_form=settings.SECOND_COUNT_FILENAME,
															remove_bad_features=False,
															remove_aggregates=False)
		X = both_mydata
		Y = both_person_count
		keys = both_keys


	log.info("Source data : " + settings.DATA_FILENAME)
	log.info("Source form data : " + settings.COUNT_FILENAME)
	log.info("figure folder : " + settings.FIGURES_FOLDER)
	print("figure folder : " + settings.FIGURES_FOLDER)


	"""SCALING X FOR THE LEARNING CURVE COMPUTATION"""
	scaler = preprocessing.StandardScaler().fit(X)
	X_all = list(scaler.transform(X))
	correlation(X_all, keys)
	# For polynomial regression
	# poly = PolynomialFeatures(degree=2)
	# X = poly.fit_transform(X)

	"""PRINTING THE PROPORTION OF 0 SAMPLES"""
	zeroes_count = 0
	for person in Y:
		if person == 0:
			zeroes_count += 1
	print("zeroes proportion : " + str(zeroes_count / len(Y)))
	log.info("zeroes proportion : " + str(zeroes_count / len(Y)))

	"""PRINTING THE DIMENSIONS OF THE DATA"""
	print("sensor data dimensions : " + str(len(X)) + " " + str(len(X[0])))
	log.info("sensor data dimensions : " + str(len(X)) + " " + str(len(X[0])))
	print("person count dimension : " + str(len(Y)))
	log.info("person count dimension : " + str(len(Y)))

	"""BUILDING REGRESSORS and CLASSIFIERS"""
	if settings.TO_COMPUTE_MODELS == settings.BASE:
		models_list = build_regressors()
		models_list += build_classifiers()

	"""BUILDING EXTENDED MODELS"""
	if settings.TO_COMPUTE_MODELS == settings.EXTENDED:
		models_list = build_extended()

	"""BUILDING VOTING CLASSIFIER"""
	if settings.TO_COMPUTE_MODELS == settings.VOTING:
		models_list = build_voting_clf()

	"""BUILDING GENETICS MODELS"""
	if settings.TO_COMPUTE_MODELS == settings.GENETICS:
		models_list = build_custom_voting_clf()

	"""BUILDING SCORERS"""

	scorers = [(strict_accuracy, make_scorer(strict_accuracy, greater_is_better=True)),
			   (threshold_accuracy, make_scorer(threshold_accuracy, greater_is_better=True)),
			   (proportional_threshold_accuracy, make_scorer(proportional_threshold_accuracy, greater_is_better=True)),
			   (metrics.mean_squared_error, make_scorer(metrics.mean_squared_error)),
			   (metrics.median_absolute_error, make_scorer(metrics.median_absolute_error)),
			   (metrics.r2_score, make_scorer(metrics.r2_score))]

	zero_scorer = make_scorer(on_zeroes_accuracy, greater_is_better=True)
	non_zero_scorer = make_scorer(on_non_zeroes_accuracy, greater_is_better=True)

	'''BUILDING TUPLES FOR BOXPLOTS FIGURES'''
	N_COL = 3
	N_ROW = 2
	tuple_list = []
	for (i, j) in itertools.product(range(N_ROW), range(N_COL)):
		tuple_list.append((i, j))

	all_scores_detailed = {}  # Dict keeping track of all results for all methods for further processing and stats

	"""CROSS-VALIDATION FOR EACH METHOD"""
	for model, param_grid in models_list:
		if isinstance(model, list):
			method_name = ("Genetic_Algorithm")
		else:
			method_name = str(model.__class__).split('.')[-1][:-2]
		print("Cross-validation beginning with " + method_name)
		logging.info("Cross-validation beginning with " + method_name)

		fig1, f1_axes = plt.subplots(ncols=N_COL, nrows=N_ROW)

		index = 0
		model_scores = []
		"""CROSS-VALIDATION FOR EACH SCORER"""
		for (score_fun, scorer) in scorers:
			scorer_name = str(scorer)
			scorer_name = scorer_name[scorer_name.find("(") + 1:scorer_name.find(")")]

			print("Beginning with " + method_name + " using " + scorer_name)
			logging.info("Beginning with " + method_name + " using " + scorer_name)

			train_scores = []
			test_scores = []
			zero_test_scores = []
			non_zero_test_scores = []
			best_clf = (None, -1)  # tuple (bestClassifier, accuracy)

			for i in range(settings.N_FOLDS_GRID_SEARCH):  # Using a reduced number of fold to increase speed
				"""SPLIT DATA FOR VALIDATION SET"""
				n_samples, n_features, = len(X), len(keys)
				smoothing, derivation = 5, 5

				assert len(X[0]) == n_features and len(X) == len(Y)
				n_train_samples = int(n_samples * settings.TEST_SET_RATIO)
				n_test_samples = n_samples - n_train_samples
				train_indices = random.sample(range(0, n_samples), n_train_samples)
				train_set = [X[index] for index in train_indices]
				train_person_count = [Y[index] for index in train_indices]
				test_set = [i for j, i in enumerate(X) if j not in train_indices]
				test_person_count = [i for j, i in enumerate(Y) if j not in train_indices]

				"""CHANGE THE TRAIN AND TEST SETS e.g. force Otlet as test set"""
				if settings.TO_COMPUTE_DATASET == settings.COMPUTE_ON_ONE_TEST_ON_TWO:
					X_train = First_mydata
					Y_train = First_person_count
					X_test = Second_mydata
					Y_test = Second_person_count
				else:
					X_train = train_set
					Y_train = train_person_count
					X_test = test_set
					Y_test = test_person_count


				"""SCALING DATA"""
				if not settings.PICKLE_DATA:
					scaler = preprocessing.StandardScaler().fit(X_train)
					X_train = list(scaler.transform(X_train))
					X_test = list(scaler.transform(X_test))

				if not param_grid == []:
					ret = None
					"""BUILDING GRID SEARCH"""
					clf = GridSearchCV(model, param_grid, cv=settings.N_FOLDS_GRID_SEARCH, scoring=scorer)
					"""APPLYING GRID SEARCH IN A SUBPROCESS"""
					if settings.USE_SUBPROCESS:
						# logging.warning("Use process is WIP and not operational at the moment while using debugger")
						# clf.fit(X,Y)
						ret = subproc_grid_search(clf, X_train, Y_train, method_name=method_name, scorer_name=scorer_name)
						clf = ret
					else:
						clf.fit(X_train, Y_train)
						print("grid_search succedeed")
						logging.info("grid_search succedeed")
					if clf is None:
						continue
					# print(clf.best_params_)
					"""SHOWING THE BEST ESTIMATOR"""
					logging.info("best parameters : " + str(clf.best_params_))
					clf = clf.best_estimator_
				else:  # No grid search is done since no hyper-parameters
					if isinstance(model, list):  # for voting algo with regressors
						print("No grid-search required")
						logging.info("No grid-search required")
						for m in model:
							m.fit(X_train, Y_train)
							clf = model
					else:
						print("No grid-search required")
						logging.info("No grid-search required")
						clf = model
						clf.fit(X_train, Y_train)
						print("model fitted")
						logging.info("model fitted")

				"""PREDICTING THE TEST SET AND ASSESSING ACCURACY"""
				if isinstance(clf, list):
					Y_pred = predict_voting(X_test, clf)
				else:
					Y_pred = clf.predict(X_test)
				test_score = score_fun(Y_test, Y_pred)
				test_scores.append(test_score)
				if test_score > best_clf[1]:
					best_clf = (clf, test_score)
				zero_score = on_zeroes_accuracy(Y_test, Y_pred)
				zero_test_scores.append(zero_score)

				non_zero_score = on_non_zeroes_accuracy(Y_test, Y_pred)
				non_zero_test_scores.append(non_zero_score)

				if isinstance(clf, list):
					Y_pred = predict_voting(X_train, clf)
				else:
					Y_pred = clf.predict(X_train)
				train_score = score_fun(Y_train, Y_pred)
				train_scores.append(train_score)

			print("Done with " + method_name + " using " + scorer_name)
			logging.info(scorer_name + " accuracy")
			logging.info("train_scores : " + str(train_scores))
			logging.info("test_scores : " + str(test_scores))

			if "accuracy" in scorer_name:  # Only plot the zero vs non zero scores and LC for the accuracy metrics
				logging.info("On zeroes accuracy : " + str(zero_test_scores))
				print("On zeroes accuracy : " + str(zero_test_scores))
				logging.info("On non-zeroes accuracy : " + str(non_zero_test_scores))
				print("On non-zeroes accuracy : " + str(non_zero_test_scores))
				plot_zero_accuracy(zero_test_scores, non_zero_test_scores, method_name + "-zero-vs-non-zero-scores")
				print("Computing LC")
				logging.info("Computing LC")
				plot_learning_curve(best_clf[0], method_name + "-Learning-curve-" + scorer_name, X_all, Y,
                                    cv=settings.N_FOLDS_VALIDATION,
                                    train_sizes=settings.LEARNING_CURVE_SIZES, scorer=score_fun)
				print("LC Done")
				logging.info("LC Done")
				if "Forest" in method_name:  # Computing the feature importance using Random Forest attribute
					logging.info("Feature importances")
					print("Feature importances")
					logging.info(str(keys))
					logging.info(str(best_clf[0].feature_importances_))
					plot_feature_importance(best_clf[0], keys, title=method_name + "-feature_importance-scikit-" +
																	 scorer_name)
				if not isinstance(best_clf[0], list):
					# feature importance
					featureImportance(best_clf[0], score_fun, model_name=method_name,
								  title=method_name + "-feature_importance-" + scorer_name)
					# single feature importance
					singleFeatureImportance(best_clf[0], score_fun, model_name=method_name,
									  title=method_name + "-single_feature_importance-" + scorer_name)

			"""Boxplot of the scores for the cross validation"""
			print("Boxplot " + scorer_name)
			(row, col) = tuple_list[index]
			f1_axes[row, col].boxplot(test_scores)
			f1_axes[row, col].get_xaxis().set_visible(False)
			if settings.FIXED_BOXPLOT_LIMITS:
				if "accuracy" in scorer_name:
					f1_axes[row, col].set_ylim(settings.ACCURACY_BOXPLOT_LIMITS)
				elif "score" in scorer_name:
					f1_axes[row, col].set_ylim(settings.ERROR_BOXPLOT_LIMITS)
				else:
					f1_axes[row, col].set_ylim(settings.ERROR_BOXPLOT_LIMITS)
			f1_axes[row, col].set_title(scorer_name, fontsize=8)
			index += 1

		filename = method_name + '-' + 'accuracy' + '.png'
		"""SAVING BOXPLOT OF VALIDATION"""
		# plt.show()
		plt.savefig(settings.FIGURES_FOLDER + filename)
		plt.close(fig1)
		print("Cross-validation done")
		logging.info("Cross-validation done")

		"""EXPORT DATAILED DATA TO PICKLE FILE"""
		print("Beggining data export")
		logging.info("Beggining data export")
		with open(settings.DATA_FOLDER + method_name + settings.ALL_SCORES_PICKLE, "wb") as f:
			pickle.dump((train_scores, test_scores, zero_test_scores, non_zero_test_scores), f)
		print("DONE")
		logging.info("DONE")