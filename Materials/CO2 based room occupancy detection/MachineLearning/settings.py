#!/usr/bin/env python
'''
    File name: settings.py
    Author: Nicolas Bockstael && Alexandre Jadin
    Date created: 10/2/2018
    Date last modified: 04/06/2018
    Python Version: 3.6.3

	Description : File containing the settings used by the script
'''

import numpy as np

"""DEFINING WHAT TYPE OF MODELS SHOULD BE USED"""
BASE = 0
EXTENDED = 1
VOTING = 2
GENETICS = 3

TO_COMPUTE_MODELS = BASE

"""DEFINING WHAT DATASET TO USE"""
ONE_DATASET = 0
TWO_DATASET = 1
TWO_DATASET_BALANCED = 2
COMPUTE_ON_ONE_TEST_ON_TWO = 3
TO_COMPUTE_DATASET = TWO_DATASET

DATA_FILENAME = "input_data/Parnas_data.csv"
COUNT_FILENAME = "input_data/Parnas_form.csv"

SECOND_DATA_FILENAME = "input_data/Otlet_9_3.csv"
SECOND_COUNT_FILENAME = "input_data/person_count_otlet_11_03_pre_processed.csv"

"""DEFINING CONSTANTS AND GLOBAL PARAMETERS"""

PICKLE_DATA = False
PREPROCESSED_PICKLE_NAME = "input_data/Parnas_scaled_data.p"
# PREPROCESSED_PICKLE_NAME = "input_data/Otlet_scaled_data.p"

ALL_SCORES_PICKLE = "_all_scores_detailed_Parnas.p"
# ALL_SCORES_PICKLE = "_all_scores_detailed_Otlet.p"

FIGURES_FOLDER = "./export_figures/figures_Parnas_clean_2018-06-04/"
# FIGURES_FOLDER = "./figures_Otlet_genetics_2018-05-24/"
DATA_FOLDER = "./export_data/all_data_Parnas_clean_2018-06-04/"
# DATA_FOLDER = "./all_data_Otlet_genetics_2018-05-24/"
TEST_SET_RATIO = 9 / 10
N_FOLDS_GRID_SEARCH = 5
N_FOLDS_VALIDATION = 10

LEARNING_CURVE_SIZES = np.linspace(0.1, 0.9, 5)
CROSS_VALIDATION_TIMEOUT = 600
JOIN_TIMEOUT = 1
USE_SUBPROCESS = True
# USE_SUBPROCESS = False

PERSON_COUNT_VALIDITY = 3600 * (10 ** 9)

raw_plus_count_FILENAME = "input_data/Parnas_row_plus_count_genetics.csv"
# raw_plus_count_FILENAME = "input_data/Otlet_row_plus_count.csv"

pre_processed_FILENAME = "input_data/Parnas_pre_processed_genetics.csv"
# pre_processed_FILENAME = "input_data/Otlet_pre_processed.csv"

# consolidate_FILENAME = "input_data/Otlet_Consolidated.csv"
consolidate_FILENAME = "input_data/Parnas_Consolidated_genetics.csv"

"""CONSTANT FOR FORMATS"""

DATE_FORMAT = "%d/%m/%Y %H:%M:%S"
NAMES = ['name', 'time', 'app_id', 'battery', 'co2', 'dev_id', 'hardware_serial', 'humidity', 'light', 'motion',
            'person_count', 'temperature', 'time_device']
NAMES_TO_REMOVE = ['name', 'app_id', 'battery', 'dev_id', 'hardware_serial', 'time_device', 'person_count']

"""CONSTANTS FOR GRAPHS"""

FIXED_BOXPLOT_LIMITS = False
ACCURACY_BOXPLOT_LIMITS = [0.8, 1]
ERROR_BOXPLOT_LIMITS = [-2, 2]