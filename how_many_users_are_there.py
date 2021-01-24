from constants_and_util import *
import pandas as pd
import os
import cPickle
from collections import Counter
from dataprocessor import get_timezone_to_country_mapping

def count_rows_and_users_in_processed_data():
    print("\n\n\n*******COUNTING ROWS AND USERS IN PROCESSED DATA")
    total_users = 0
    total_rows = 0
    all_symptom_counts = {}
    for chunk_number in range(n_chunks):
        d = pd.read_csv(big_annotated_symptom_filename(chunk_number))
        total_users += len(set(d['user_id_hash']))
        total_rows += len(d)
        print("After reading chunk %i, total users: %i; total rows: %i" % (chunk_number, total_users, total_rows))
        d['combined_type'] = d['category'] + '*' + d['type']
        symptom_counts = Counter(d['combined_type'])
        for k in symptom_counts:
            if k not in all_symptom_counts:
                all_symptom_counts[k] = 0
            all_symptom_counts[k] += symptom_counts[k]
    print all_symptom_counts
    cPickle.dump(all_symptom_counts, open(os.path.join(base_results_dir, 'symptom_counts_in_processed_data.pkl'), 'wb'))

def count_countries_in_processed_data():
    print("\n\n\n*******COUNTING COUNTRIES IN PROCESSED DATA")
    tzs_to_countries = get_timezone_to_country_mapping()
    all_country_counts = {}
    for chunk_number in range(n_chunks):
        d = pd.read_csv(big_annotated_symptom_filename(chunk_number))
        d['country'] = d['timezone'].map(lambda x:tzs_to_countries[str(x)])
        grouped_d = d.groupby('country')
        for country, country_d in grouped_d:
            if country not in all_country_counts:
                all_country_counts[country] = {'n_users':0, 'n_obs':0}
            all_country_counts[country]['n_obs'] += len(country_d)
            all_country_counts[country]['n_users'] += len(set(country_d['user_id_hash']))

    for i, c in enumerate(sorted(all_country_counts, key=lambda country:all_country_counts[country]['n_users'])[::-1]):
        print("%3i. %-50s %9i %9i" % (i + 1, c, all_country_counts[c]['n_obs'], all_country_counts[c]['n_users']))
    cPickle.dump(all_country_counts, open(os.path.join(base_results_dir, 'country_counts_in_processed_data.pkl'), 'wb'))


def count_rows_and_users_in_original_data():
    print("\n\n\n*******COUNTING ROWS AND USERS IN ORIGINAL DOWNLOADED DATA")
    total_rows = 0
    all_user_id_hashes = set()
    filenames = ['tracking.csv'] + [os.path.join('downloaded_files', f) for f in CONTINUOUS_VARIABLE_FILENAMES]
    for f in filenames:
        full_path = os.path.join(base_data_dir, f)
        if 'gzip' in full_path:
            d = pd.read_csv(full_path, compression='gzip')
        else:
            d = pd.read_csv(full_path)
        all_user_id_hashes = all_user_id_hashes.union(set(d['user_id_hash']))
        total_rows += len(d)
        print("After reading %s, total users: %i; total rows: %i" % (f, len(all_user_id_hashes), total_rows))

if __name__ == '__main__':
    count_countries_in_processed_data()
    count_rows_and_users_in_processed_data()
    count_rows_and_users_in_original_data()
