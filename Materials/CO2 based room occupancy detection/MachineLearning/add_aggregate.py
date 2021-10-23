import numpy as np
from sklearn import preprocessing
import csv
import pickle
import sys
import copy


def add_aggregate(data, smoothing_interval=5, derivation_interval=5, add_aggregate=True):
	"""
	smooth co2 values, add a new feature : derived co2 evolution
	:param the data, the smooting value, the derivation value
	:return the extended data without the time feature
	"""

	if add_aggregate:
		smoothed_data = add_smoothed_co2(data, smoothing_interval)
		derived_data = add_derived_co2(smoothed_data, derivation_interval)
	else:
		derived_data = data

	keys = list(derived_data[0].keys())
	keys.remove('time')
	keys.sort()

	# row0 = copy.copy(derived_data[0])
	# for k in list(row0.keys()):
	# 	row0[k] = k
	# keys = list(row0.values())
	# keys.remove('time')

	my_data = []
	for row in derived_data:
		del row['time']
		entry = list(row.items())
		entry.sort(key=lambda elem: keys.index(elem[0]))
		entry = [elem[1] for elem in entry]
		my_data.append(entry)
	return [keys, my_data]


def add_smoothed_co2(data, interval=5):
	"""
	create a new data set with the smoothed co2
	:param data: the original data set
	:return: the new dataset
	"""
	last_four_sum = 0
	ret = []
	for index in range(len(data)):
		if index < interval - 1:
			last_four_sum += data[index]['co2']
			ret.append(data[index])
			ret[index]['co2_smoothed'] = data[index]['co2']
		else:
			ret.append(data[index])
			last_four_sum = (data[index]['co2'] + last_four_sum)  # compute the sum of the 5 last elements
			ret[index]['co2_smoothed'] = last_four_sum / interval  # saves the mean of those 5 elements
			last_four_sum -= data[index - interval + 1]['co2']  # removes the oldest element of the sum
	return ret


def add_derived_co2(data, interval=5):
	"""create a new data set with the derived co2
	:param data: the data set REQUIRE co2_smoothed
	:return: the new dataset
	"""
	if interval < 2:
		return data

	ret = []
	for index in range(len(data)):
		ret.append(data[index])
		if index < interval - 1:
			ret[index]['co2_derived'] = 0
		co2x = data[index - interval + 1]['co2_smoothed']
		co2y = data[index]['co2_smoothed']
		timex = data[index - interval + 1]['time'] / (10 ** 9)
		timey = data[index]['time'] / (10 ** 9)
		ret[index]['co2_derived'] = (co2y - co2x) / (timey - timex)  # derivative at a point is the angular coefficient
	return data
