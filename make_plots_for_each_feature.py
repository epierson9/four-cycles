import matplotlib
matplotlib.use('Agg')

import pandas as pd
from collections import Counter
import random
from dateutil import tz
import numpy as np

import copy
import matplotlib.pyplot as plt
import datetime
from constants_and_util import *
import dataprocessor
from collections import Counter
from copy import deepcopy
import statsmodels.api as sm
import compare_to_seasonal_cycles
import sys
import cPickle

def get_full_combined_symptom_list(min_count_in_chunk):
    d = pd.read_csv(big_annotated_symptom_filename(0))
    d['combined_type'] = d['category'] + '*' + d['type']
    all_combined_type_counts = Counter(d['combined_type'])
    all_combined_symptoms = sorted(all_combined_type_counts.keys(), key = lambda x:all_combined_type_counts[x])[::-1]
    all_combined_symptoms = [a for a in all_combined_symptoms if all_combined_type_counts[a] >= min_count_in_chunk]
    all_mood_symptoms = [a for a in all_combined_symptoms if any([x in a for x in MOOD_SYMPTOM_CATEGORIES])]
    print 'All symptoms with >= %i observations in a chunk are (mood symptoms have stars; symptoms are followed by counts)' % min_count_in_chunk
    for i, symptom in enumerate(sorted(all_combined_symptoms)):
        star = '*' if symptom in all_mood_symptoms else ''
        print '%i. %-50s %s %i' % (i+1, symptom, star, all_combined_type_counts[symptom])
    return all_combined_symptoms, all_mood_symptoms

def make_four_cycle_plots_for_symptom_list(process_idx, total_processes):
    all_combined_symptoms, all_mood_symptoms = get_full_combined_symptom_list(min_count_in_chunk=100)
    chunks_to_use = range(64)
    combined_symptom_list = all_combined_symptoms[process_idx::total_processes]
    for combined_symptom_idx, combined_symptom in enumerate(combined_symptom_list):
        enough_users_logging_symptom = True
        opposite_pair = 'logged_symptom_%s_versus_didnt*didnt_log_symptom' % combined_symptom
        print 'Analyzing symptom %s, %i/%i' % (opposite_pair, combined_symptom_idx + 1, len(combined_symptom_list))
        if combined_symptom in all_mood_symptoms:
            symptom_category = combined_symptom.split('*')[0] + '*'
            symptoms_on_screen = [a for a in all_combined_symptoms if symptom_category in a]
            symptom_and_opposite = None
            for canonical_pretty_symptom_name in CANONICAL_PRETTY_SYMPTOM_NAMES:
                good_symptom, bad_symptom = canonical_pretty_symptom_name.split('_versus_')
                if combined_symptom == good_symptom or combined_symptom == bad_symptom:
                    symptom_and_opposite = [good_symptom, bad_symptom]
                    print 'found symptom and its opposite'
                    print symptom_and_opposite
            if 'high_energy' in opposite_pair or 'low_energy' in opposite_pair:
                # these are not included in our original analysis. 
                symptom_and_opposite = ['energy*high_energy', 'energy*low_energy']


            print("Normalizing mood symptom %s using" % combined_symptom)
            symptom_lists_to_use_as_normalizer = {'symptoms_on_screen':symptoms_on_screen, 'all_mood_symptoms':all_mood_symptoms}
            if symptom_and_opposite is not None:
                symptom_lists_to_use_as_normalizer['symptom_and_opposite'] = symptom_and_opposite
            print symptom_lists_to_use_as_normalizer
        else:
            symptom_lists_to_use_as_normalizer = {'none':None}
        # compute the emission probability both grouping by hour and not grouping by hour. 
        all_ds = {'by_hour':{}, 'not_by_hour':{}}
        for normalizer_name in symptom_lists_to_use_as_normalizer:
            all_ds['by_hour']['normalizer: ' + normalizer_name] = []
            all_ds['not_by_hour']['normalizer: ' + normalizer_name] = []

        for chunk in chunks_to_use:
            print("Chunk %i/%i" % (chunk + 1, len(chunks_to_use)))
            user_feature_filename = dataprocessor.get_user_feature_filename(chunk)
            user_features = cPickle.load(open(user_feature_filename, 'rb'))
            d = pd.read_csv(big_annotated_symptom_filename(chunk))
            print("Prior to filtering for very active loggers, %i rows" % len(d))
            very_active_northern_logger_idxs = (d['user_id_hash'].map(lambda x:user_features[x]['very_active_northern_hemisphere_loggers']) == True)
            d = d.loc[very_active_northern_logger_idxs]
            d.index = range(len(d))
            print("After filtering for very active northern hemisphere loggers, %i rows" % len(d))
            d['combined_type'] = d['category'] + '*' + d['type']
            
            

            for by_hour in [True, False]:
                for normalizer_name in symptom_lists_to_use_as_normalizer:
                    symptom_list = symptom_lists_to_use_as_normalizer[normalizer_name]
                    d_conditional_on_logging = dataprocessor.compute_emission_probability_conditional_on_logging(d, 
                        opposite_pair,
                        filter_for_users_who_log_at_least_once=True, 
                        group_by_hour_as_well=by_hour, 
                        symptom_list_to_use_as_normalizer=symptom_list)
                    if d_conditional_on_logging is None:
                        enough_users_logging_symptom = False
                        continue
                    if by_hour:
                        all_ds['by_hour']['normalizer: ' + normalizer_name].append(d_conditional_on_logging)
                    else:
                        all_ds['not_by_hour']['normalizer: ' + normalizer_name].append(d_conditional_on_logging)
            if not enough_users_logging_symptom:
                break
        if not enough_users_logging_symptom:
            print("Cannot compute results for %s" % combined_symptom)
            continue
        regression_results = {}
        for k in ['by_hour', 'not_by_hour']:
            regression_results[k] = {}
            for normalizer_name in symptom_lists_to_use_as_normalizer:
                print("by hour: %s; normalizer name %s" % (k, normalizer_name))
                if k == 'by_hour':
                    covariates = ['C(year)', 'C(month)',  'C(weekday)', 'C(local_hour)', 'C(date_relative_to_period, Treatment(reference=0))']
                else:
                    covariates = ['C(year)', 'C(month)',  'C(weekday)', 'C(date_relative_to_period, Treatment(reference=0))']
                d_to_use_for_regression = pd.concat(all_ds[k]['normalizer: ' + normalizer_name])
                d_to_use_for_regression.index = range(len(d_to_use_for_regression))

                regression_results[k]['normalizer: ' + normalizer_name] = compare_to_seasonal_cycles.fit_linear_regression(d_to_use_for_regression, covariates)
                print(regression_results[k]['normalizer: ' + normalizer_name])
        file_handle = open(os.path.join(base_results_dir, 'fluctuations_in_symptom_%s.pkl' % opposite_pair), 'wb')
        #file_handle = open(os.path.join(base_results_dir, 'fluctuations_in_all_features.pkl' % ), 'wb')
        cPickle.dump(regression_results, file_handle)
        file_handle.close()

    print("Successfully completed all processes.")

if __name__ == '__main__':
    os.environ["OMP_NUM_THREADS"]="5"
    total_processes = 50
    if len(sys.argv) == 1:
        for i in range(total_processes):
            print("Starting process %i/%i" % (i + 1, total_processes))
            os.system('nohup python -u make_plots_for_each_feature.py %i > %s/make_plots_for_each_feature_%i.out &' % (i, base_processing_outfile_dir, i))
    else:
        assert len(sys.argv) == 2
        process_idx = int(sys.argv[1])
        assert process_idx < total_processes
        make_four_cycle_plots_for_symptom_list(process_idx, total_processes)

