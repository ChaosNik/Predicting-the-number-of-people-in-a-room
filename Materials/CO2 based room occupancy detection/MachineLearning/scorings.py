#!/usr/bin/env python
'''
    File name: scorings.py
    Author: Nicolas Bockstael && Alexandre Jadin
    Date created: 10/2/2018
    Date last modified: 04/06/2018
    Python Version: 3.6.3

	Description : File containing the scoring functions
				The scoring functions follow the same signature as the methods from sci-kit for easy integration
'''

def strict_accuracy(y, y_pred, **kwargs):
	assert len(y) == len(y_pred)
	well_classified = 0
	for i in range(len(y)):
		if abs(round(y_pred[i]) - y[i]) == 0:  # predicted = actual
			well_classified += 1
	return well_classified / len(y)


def threshold_accuracy(y, y_pred, threshold=3):
	assert len(y) == len(y_pred)
	well_classified = 0
	for i in range(len(y)):
		if abs(round(y_pred[i]) - y[i]) <= threshold:  # diff predicted/actual <= threshold equal is important
			well_classified += 1
	return well_classified / len(y)


def proportional_threshold_accuracy(y, y_pred, percentage=0.35):
	assert len(y) == len(y_pred)
	well_classified = 0
	for i in range(len(y)):
		if abs(round(y_pred[i]) - y[i]) <= (y[i] * percentage):  # threshold depend on actual value equal is important
			well_classified += 1
	return well_classified / len(y)


def on_zeroes_accuracy(y, y_pred):
	assert len(y) == len(y_pred)
	well_classified = 0
	n_zeroes = 1  # avoid  division by zero
	for i in range(len(y)):
		if y[i] == 0:
			n_zeroes += 1
			if round(y_pred[i]) == 0:
				well_classified += 1
	return well_classified / n_zeroes


def on_non_zeroes_accuracy(y, y_pred):
	assert len(y) == len(y_pred)
	well_classified = 0
	n_non_zeroes = 1  # avoid division by zero
	for i in range(len(y)):
		if not y[i] == 0:
			n_non_zeroes += 1  # threshold depend on actual value equal is important
			if round(y_pred[i]) == y[i]:
				well_classified += 1
	return well_classified / n_non_zeroes