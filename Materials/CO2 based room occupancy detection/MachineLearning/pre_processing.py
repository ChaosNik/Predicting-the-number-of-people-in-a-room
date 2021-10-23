#!/usr/bin/env python
'''
    File name: pre_processing.py
    Author: Nicolas Bockstael && Alexandre Jadin
    Date created: 10/2/2018
    Date last modified: 04/06/2018
    Python Version: 3.6.3

	Description : Helper file dedicated to pre_processing and formatting the data
'''

import sys
import csv
import datetime
import time
import operator

"""CONSTANTS DECLARATION"""
DATE_FORMAT = "%d/%m/%Y %H:%M:%S"
NAMES = ['name', 'time', 'app_id', 'battery', 'co2', 'dev_id', 'hardware_serial', 'humidity', 'light', 'motion',
		 'person_count',
		 'temperature', 'time_device']
NAMES_TO_REMOVE = ['name', 'app_id', 'battery', 'dev_id', 'hardware_serial', 'time_device']

PERSON_COUNT_VALIDITY_NANO = 3600 * (10 ** 9)
PERSON_COUNT_VALIDITY_SEC = 3600


"""FUNCTIONS DECLARATIONS"""

def discard_outside_dates(data, from_date, to_date):
	"""

	:param data: the input data to remove some dates
	:param from_date: the beginning of the interval in nanoseconds from EPOCH
	:param to_date: the end of the interval in nanoseconds from EPOCH
	:return: the data for the concerned interval
	"""
	found_from = False
	found_to = False
	line_index = 0
	to_return = []
	length = len(data)
	# Part 1 find beginning of interval
	# TODO (can be improved by finding beginning with binary search)
	while not found_from and line_index < length:
		line = data[line_index]
		time = int(line['time'])
		if time > from_date:
			found_from = True
			to_return.append(line)
		line_index += 1

	# Part 2 adding the lines until to_date found
	while not found_to and line_index < length:
		line = data[line_index]
		time = int(line['time'])
		if time > to_date:
			found_to = True
		else:
			to_return.append(line)
		line_index += 1
	return to_return


# incorporate number of people in IoT records (reset to 0 at 3 am)
def add_people_figures(data):
	"""
	fills in the person_count attribute based on the closest previous data + sets to 0 every day at 3 am
	Also discard every line where the person count is older than PERSON_COUNT_VALIDITY_NANO
	Assumes lines without hardware serial are not IoT sensor data
	:param data: the data where the person count should be deduces
	:return: /
	"""
	to_return = []
	current_people_count = 0
	current_date = int(data[0]['time'])
	current_date = datetime.datetime.fromtimestamp(current_date / 10 ** 9)
	current_date = current_date.replace(hour=3, minute=0, second=0)  # 3 am
	current_date = int(current_date.replace(tzinfo=datetime.timezone.utc).timestamp() * 10 ** 9)  # to nanosec timestamp

	last_person_count_time = 0
	for row in data:
		myrow = row.copy()
		# special condition to handle person_count validity
		if int(myrow['time']) - current_date > (3600 * 24 * 10 ** 9):  # delay is higher than 24 hours
				myrow['person_count'] = '0'
				current_people_count = 0
				current_date = datetime.datetime.fromtimestamp(int(myrow['time']) / 10 ** 9)
				current_date = current_date.replace(hour=3, minute=0, second=0)  # 3 am
				current_date = int(current_date.replace(tzinfo=datetime.timezone.utc).timestamp() * 10 ** 9)

		if ('hardware_serial' not in myrow) or myrow['hardware_serial'] == '':
			# the current row is giving info only on people_count
			current_people_count = myrow['person_count']
			last_person_count_time = int(myrow['time'])
			continue  # do not append this empty line
		else:
			#  regular row
			if int(myrow['time']) - last_person_count_time < PERSON_COUNT_VALIDITY_NANO:
				# if the person count is recent enough
				myrow['person_count'] = current_people_count
			# else we skip the data
			else:
				continue

		if not last_person_count_time == 0 and int(
				myrow['time']) - last_person_count_time < PERSON_COUNT_VALIDITY_NANO:  # person count has a valid value
			to_return.append(myrow)

	return to_return


def export_to_csv(data, filename, fieldnames=NAMES, isNPArray=False):
	"""
	Export tthe data towards a well formatted csv
	:param data: the data to export
	:param filename: output filename
	:param fieldnames: the fields that should make the header of the csv file
	:param isNPArray: whether data is a list of disctionnary or a numpy array
	:return: /
	"""
	with open(filename, 'w', newline='') as csvfile:
		writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
		writer.writeheader()
		for row in data:
			if isNPArray:
				row = dict(zip(fieldnames, row))
			writer.writerow(row)


def pre_process_Otlet(source_form_filename, dest_form_filename):
	"""
	Take the form filename of room Paul Otlet, remove extensive 0's and export the modified file
	The 0's are removed if the value remains to be 0 during a period longer than PERSON_COUNT_VALIDITY_SEC
	:param source_form_filename: input csv file
	:param dest_form_filename: output csv file
	:return: /
	"""

	with open(source_form_filename, 'r', newline='') as csv_source:
		with open(dest_form_filename, 'w', newline='') as csv_dest:
			input_file = csv.DictReader(csv_source)
			first_line = True
			last_non_zero_date = datetime.datetime.now()
			last_non_zero_date = last_non_zero_date.replace(year=1970)
			for row_input in input_file:
				if first_line:
					fields = row_input
					writer = csv.DictWriter(csv_dest, fieldnames=fields)
					writer.writeheader()
					last_non_zero_date = row_input['time']
					first_line = False

				if not int(row_input["person_count"]) == 0:
					last_non_zero_date = row_input["time"]
				d1 = datetime.datetime.strptime(row_input["time"], DATE_FORMAT)
				d2 = datetime.datetime.strptime(last_non_zero_date, DATE_FORMAT)
				diff = d1 - d2
				diff_secs = diff.total_seconds()
				if diff_secs < PERSON_COUNT_VALIDITY_SEC:
					# row_input = dict(zip(fields, row_input))
					writer.writerow(row_input)


def pre_process(input_filename, form_filename, pre_processed_filename="", raw_plus_count_filename="",
				export_raw_plus_count=False, export_pre_processed=False, from_date_nano=0, to_date_nano=float("inf"),
				other_input = None, other_form = None):
	"""
	Pre process the data by parsing a csv from sensor, a csv with person count, discarding outside dates, merging the
	data into a single list of dictionnaries
	:param input_filename: input file containing the sensor data
	:param form_filename: input file containing the person count data
	:param pre_processed_filename: the name of the file to export the final pre_processed data as csv, ignored if flag
	 is false
	:param raw_plus_count_filename: the name of the file to export the sensor data merged with person_count as csv,
	 ignored if flag is false
	:param export_raw_plus_count: Whether the sensor data merged with person_count should be exported
	:param export_pre_processed: Whether the final pre processed file should be exported
	:param from_date_nano: the date from which we should consider the data, expressed in nanoseconds from EPOCH
	:param to_date_nano: the date up to which we should consider the data, expressed in nanoseconds from EPOCH
	:return: the pre processed data as a list of dictionnaries
	"""
	with open(input_filename, 'r', newline='') as csv_source:
		input_file = csv.DictReader(csv_source)
		raw_data = []
		for row_input_file in input_file:
			raw_data.append(row_input_file)
	if other_input is not None:
		with open(other_input, 'r', newline='') as csv_source_other:
			input_file = csv.DictReader(csv_source_other)
			raw_data = []
			for row_input_file in input_file:
				raw_data.append(row_input_file)

	# parse csv from google form and add rows to raw_data
	with open(form_filename, 'r', newline='') as csv_form:
		form_file = csv.DictReader(csv_form)
		for row_form_file in form_file:
			s = row_form_file["time"]
			epoch = str(int(time.mktime(datetime.datetime.strptime(s, DATE_FORMAT).timetuple()) * 10 ** 9))
			entry = {"time": epoch, "person_count": row_form_file["person_count"]}
			raw_data.append(entry)

	if other_form is not None:
		with open(other_form, 'r', newline='') as csv_form_other:
			form_file = csv.DictReader(csv_form_other)
			for row_form_file in form_file:
				s = row_form_file["time"]
				epoch = str(int(time.mktime(datetime.datetime.strptime(s, DATE_FORMAT).timetuple()) * 10 ** 9))
				entry = {"time": epoch, "person_count": row_form_file["person_count"]}
				raw_data.append(entry)

	# sort by timestamp
	raw_data = sorted(raw_data, key=operator.itemgetter("time"))

	# pre_process (formatting) the data and export

	if export_raw_plus_count:
		export_to_csv(raw_data, raw_plus_count_filename)

	raw_data = discard_outside_dates(raw_data, from_date_nano, to_date_nano)
	complete_data = add_people_figures(raw_data)
	if export_pre_processed:
		export_to_csv(complete_data, pre_processed_filename)
	# print("Pre process Done")
	consolidate_dataset(complete_data, pre_processed_filename + "_consolidated.csv")
	return complete_data


def consolidate_dataset(complete_data, filename):
	"""
	Exports a human readable, unscaled, data set grouping the sensor and person count data we considered in
	machine learning tests
	:param complete_data: thee entire data set (sensor + person count)
	:param filename: filename to output the result (csv format)
	:return: /
	"""
	to_ret = []
	for line in complete_data:
		myline = line.copy()
		for k in NAMES_TO_REMOVE:  # Remove irrelevant keys
			del myline[k]
		to_ret.append(myline)
	export_to_csv(to_ret, filename, list(set(NAMES) - set(NAMES_TO_REMOVE)))

"""MAIN"""
# Testing the pre processing of a file
if __name__ == "__main__":

	# This is used for the removal of extensive zeroes from the room Otlet data set (otherwise, more than 90% of the
	# data are samples while the room is empty
	pre_process_Otlet("person_count_otlet_11_03.csv", "person_count_otlet_11_03_pre_processed.csv")

	if len(sys.argv) < 5:
		print("wrong number of arguments\n"
			  "\tpython3 pre_processing.py [raw co2 data] [raw_person count data] "
			  "[from_date dd/mm/yyyy] [to_date dd/mm/yyyy] "
			  "[output filename raw_plus_count] [output filename pre_processed]")
		sys.exit(0)

	from_date_list = list(map(int, sys.argv[3].split("/")))
	to_date_list = list(map(int, sys.argv[4].split("/")))

	# FILENAME = "./Parnas_data.csv"
	aremplacerhgtrfde_FILENAME = sys.argv[1]
	# FORM_FILENAME = "./Parnas_form.csv"
	aremplacerhgtrfde_FORM_FILENAME = sys.argv[2]
	raw_plus_count_FILENAME = sys.argv[5]
	aremplacerhgtrfde_pre_processed_FILENAME = sys.argv[6]

	# FROM_DATE = datetime.datetime(2017, 12, 12)
	FROM_DATE = datetime.datetime(from_date_list[2], from_date_list[1], from_date_list[0])
	FROM_DATE_NANO = int(FROM_DATE.replace(tzinfo=datetime.timezone.utc).timestamp() * 10 ** 9)
	# TO_DATE = datetime.datetime(2018, 1, 31)
	TO_DATE = datetime.datetime(to_date_list[2], to_date_list[1], to_date_list[0])
	TO_DATE_NANO = int(TO_DATE.replace(tzinfo=datetime.timezone.utc).timestamp() * 10 ** 9)

	# This yields a list of dictionary data[row number][attribute]
	pre_process(aremplacerhgtrfde_FILENAME, aremplacerhgtrfde_FORM_FILENAME, aremplacerhgtrfde_pre_processed_FILENAME,
				raw_plus_count_FILENAME, True)
