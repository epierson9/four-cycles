from constants_and_util import *
from collections import Counter
import pandas as pd
import datetime
import copy
import pytz
import numpy as np
import os
import random
import sys
from traceback import print_exc
import cPickle
from scipy.stats import scoreatpercentile
max_tracking_rows_to_read_in = None # set this to a small number to test, None otherwise

"""
Processes the original Clue data.

The code sometimes refers to "symptoms" logged in the Clue data - this is wording used for consistency with
 earlier analyses where we looked specifically at things like headaches, pain, and so on. It is synonymous with "features" 
 described in the Nature Human Behaviour paper, which is the better term for the mood, behaviour, and vital signs we consider. 
"""

def compute_max_min_difference_by_user(d, max_min_diff_threshold):
    """
    filter out people with huge fluctuations in continuous features. 
    (Since these possibly indicate logging errors.)
    """
    min_max_by_user = d[['user_id_hash', 'value']].groupby('user_id_hash').agg(['min', 'max'])
    min_max_by_user = min_max_by_user.reset_index()
    min_max_by_user.columns = ['user_id_hash', 'min', 'max']
    min_max_by_user['diff'] = min_max_by_user['max'] - min_max_by_user['min']
    for percentile in range(100):
        print 'Maximum difference by user, percentile %i: %2.3f' % (percentile, 
                                                         scoreatpercentile(min_max_by_user['diff'].values, percentile))
    no_huge_fluctuations = set(min_max_by_user['user_id_hash'].loc[min_max_by_user['diff'] <= max_min_diff_threshold])
    print 'Prior to filtering for people with huge fluctuations, %i readings' % len(d)
    d = copy.deepcopy(d.loc[d['user_id_hash'].map(lambda x:x in no_huge_fluctuations)])
    print("After filtering out people with max - min >= %i, %i readings" % (max_min_diff_threshold, len(d)))
    d.index = range(len(d))
    return d

def average_continuous_readings_on_same_day(d):
    """
    Average together readings if there are multiple ones on the same day. 
    We use this for BBT and weight, although it does not hugely affect the data. 
    """
    print("Prior to averaging together readings on the same day, %i rows" % len(d))
    assert list(d.columns) == list(['user_id_hash', 'date', 'value'])
    grouped_d = d.groupby(['user_id_hash', 'date']).mean()
    grouped_d = grouped_d.reset_index()
    grouped_d.index = range(len(grouped_d))
    print("After averaging together readings on the same day, %i rows" % len(grouped_d))
    return grouped_d

def filter_for_frequent_continuous_loggers(d, min_readings):
    # filter for users who log a lot of continuous features, since other data may be unreliable. 
    user_counts = Counter(d['user_id_hash'])
    good_idxs = d['user_id_hash'].map(lambda x:user_counts[x] >= min_readings)
    d = copy.deepcopy(d.loc[good_idxs])
    d.index = range(len(d))
    
    print("After filtering for frequent loggers, %i/%i users remain, %2.1f%% of rows" % (len(set(d['user_id_hash'])), 
        len(user_counts),
        100*good_idxs.mean()))
    return d

def process_weight_data(d, lower_limit=50, upper_limit=500):
    """
    Puts weight data into our standard format. 
    """
    print("Processing weight data")
    d = copy.deepcopy(d)
    
    # convert to pounds. 
    d['weight_lb'] = d['weight_kg'] * 2.20462262185
    d = d[['user_id_hash', 'date', 'weight_lb']]
    d.columns = ['user_id_hash', 'date', 'value']
    
    # drop missing values
    print("Prior to dropping any missing values, %i readings" % len(d))
    d = d.dropna()
    print("After dropping any missing values, %i readings" % len(d))
    
    
    # Compute weight percentiles and filter out people with outlier weights which possibly indicate erroneous entries.
    print("Computing weight percentiles")
    for percentile in range(0, 100):
        score = scoreatpercentile(d['value'].values, percentile)
        print("Score at percentile %2i: %2.3f" % (percentile, score))
    plausible_weight_idxs = (d['value'] >= lower_limit) & (d['value'] <= upper_limit)
    print("Filtering for plausible weights in range %i-%i: %2.1f%% of values in this range" % 
        (lower_limit,
        upper_limit, 
        100*plausible_weight_idxs.mean()))
    d = copy.deepcopy(d.loc[plausible_weight_idxs])

    # combine multiple readings on same day. 
    d = average_continuous_readings_on_same_day(d)
    
    # filter out people with huge fluctuations
    d = compute_max_min_difference_by_user(d, max_min_diff_threshold=50)
    
    # filter out people with very few readings. 
    d = filter_for_frequent_continuous_loggers(d, min_readings=5)
    
    d['category'] = 'continuous_features'
    d['type'] = WEIGHT_SUBSTRING
    for c in ['backend_created_at', 'session_start', 'session_end']:
        d[c] = d['date'] + ' 00:00:00'
    assert (d['date'].map(len) == 10).all()
    return d



def process_heartrate_data(d):
    """
    Checked. puts the fitbit RHR data into a more standard format. 
    """
    d = copy.deepcopy(d[['user_id_hash', 'measured_at', 'resting_heart_rate']])
    d['measured_at'] = d['measured_at'].map(lambda x:x.split()[0])
    d.columns = ['user_id_hash', 'date', 'value']
    d['category'] = 'continuous_features'
    d['type'] = HEART_SUBSTRING

    early_date = '2016-04-01'
    print("Prior to filtering Fitbit HR data for dates after %s, %i rows" % (early_date, len(d)))
    d = d.loc[d['date'] >= early_date]
    print("After filtering, %i rows" % len(d))

    d = filter_for_frequent_continuous_loggers(d, min_readings=50)

    # filter out people with huge fluctuations
    d = compute_max_min_difference_by_user(d, max_min_diff_threshold=50)

    for c in ['backend_created_at', 'session_start', 'session_end']:
        d[c] = d['date'] + ' 00:00:00'
    assert (d['date'].map(len) == 10).all()
    return d
    
def process_bbt_data(d, lower_limit=90, upper_limit=110):
    """
    Checked. Puts the BBT data into a more standard format. 
    Filters for temperatures which are out of range. 
    """
    d.columns = ['user_id_hash', 'date', 'value']
    d['value'] = d['value'] * 1.8 + 32
    # filter out implausible values. 
    print("Computing BBT percentiles")
    for percentile in range(0, 100):
        score = scoreatpercentile(d['value'].values, percentile)
        print("Score at percentile %2i: %2.3f" % (percentile, score))
    plausible_temp_idxs = (d['value'] >= lower_limit) & (d['value'] <= upper_limit)
    print("Filtering for plausible temperatures in range %i-%i: %2.1f%% of values in this range" % 
        (lower_limit,
        upper_limit, 
        100*plausible_temp_idxs.mean()))
    d = copy.deepcopy(d.loc[plausible_temp_idxs])

    d = average_continuous_readings_on_same_day(d) # if there are multiple values on same day, average. This only removes a few rows.
    d['category'] = 'continuous_features'
    d['type'] = BBT_SUBSTRING

    d = filter_for_frequent_continuous_loggers(d, min_readings=50)

    # Filter for people with at least 5 unique measurements; body temperature should not remain perfectly constant. 
    print("Filtering for users with at least 5 unique BBT values")
    print("Prior to filtering, %i values" % len(d))
    grouped_d = d.groupby('user_id_hash')
    reliable_loggers = set()
    for user, user_d in grouped_d:
        if len(set(user_d['value'])) >= 5:
            reliable_loggers.add(user)
    good_idxs = d['user_id_hash'].map(lambda x:x in reliable_loggers)
    d = copy.deepcopy(d.loc[good_idxs])
    d.index = range(len(d))
    print("After filtering, %i values" % len(d))

    for c in ['backend_created_at', 'session_start', 'session_end']:
        d[c] = d['date'] + ' 00:00:00'
    assert (d['date'].map(len) == 10).all()
    assert len(d[['user_id_hash', 'date']].drop_duplicates()) == len(d)
    return d 

def remove_double_logged_symptoms(d):
    """
    some symptoms can only be logged once per date. Per Amanda/Daniel: 
    bleeding, sleep, fluid, ring, pill, patch, injection, IUD, weight, temperature (last two are continuous so we don't filter here)
    if a binary symptom is logged twice in one day, we take the later log (using backend_created_at as a sorting key, as Amanda suggests)
    """
    print("Removing symptoms logged twice on one day that should not be")
    print("Category counts are")
    print(Counter(d['category'].dropna()))
    once_per_day_cats = ['period', 'sleep', 'fluid', 'ring_hbc', 'pill_hbc', 'patch_hbc', 'injection_hbc', 'iud']
    for cat in once_per_day_cats:
        print("the number of symptoms in category %s is %i" % (cat, (d['category'] == cat).sum()))
        assert (d['category'] == cat).sum() > 0
    d = d.sort_values(by=['user_id_hash', 'backend_created_at'])
    should_not_be_logged_twice_on_one_day = d['category'].map(lambda x:x in once_per_day_cats)
    are_duplicated = d[['user_id_hash', 'category', 'date']].duplicated(keep='last') # keep the entry with the largest backend created at.
    should_be_removed = are_duplicated & should_not_be_logged_twice_on_one_day
    print("Prior to removing symptoms logged twice on one day, %i rows" % len(d))
    d = d.loc[~should_be_removed]
    print("After removing symptoms logged twice on one day, %i rows" % len(d))
    d.index = range(len(d))
    return d

def split_original_tracking_data_into_manageable_pieces(n_chunks = n_chunks):
    """
    Takes the (gigantic) tracking dataframe and splits it into evenly sized chunks of random users
    so that we can actually process it. 
    Checked. 
    """
    # read in the continuous variables
    all_continuous_data = {}
    for continuous_filename in CONTINUOUS_VARIABLE_FILENAMES:
        continuous_d = pd.read_csv(os.path.join(base_data_dir, 
                                                'downloaded_files', 
                                                continuous_filename),
                                   nrows = max_tracking_rows_to_read_in,
                                   compression='gzip')
        
        print('Read in %i measurements from %s' % (len(continuous_d), 
                                           continuous_filename))
        continuous_d = continuous_d.drop_duplicates()
        print("After dropping any duplicate rows, %i remaining." % len(continuous_d))
        nice_name = continuous_filename.replace('.csv.gzip', '')
        if continuous_filename == 'fitbit_resting_heartrates.csv.gzip':
            continuous_d = process_heartrate_data(continuous_d)
        elif continuous_filename == 'bbt.csv.gzip':
            continuous_d = process_bbt_data(continuous_d)
        elif continuous_filename == 'weight.csv.gzip':
            continuous_d = process_weight_data(continuous_d)
        else:
            raise Exception("Invalid continuous filename")
        print("Minimum date: %s; maximum date: %s" % (continuous_d['date'].min(), 
                                                      continuous_d['date'].max()))
        print("After preprocessing data looks like")
        print(continuous_d.head())
        assert len(continuous_d[['user_id_hash', 'date']].drop_duplicates()) == len(continuous_d) # make sure only one reading per day. 
        all_continuous_data[nice_name] = continuous_d
        
    # now read in all the binary variables. 
    d = pd.read_csv(original_tracking_csv, index_col=0, nrows = max_tracking_rows_to_read_in, dtype = {'us_state':str})
    d['value'] = 1.0
    print("%i rows read in from original tracking data %s" % (len(d), original_tracking_csv))
    print("Minimum date: %s; maximum date: %s" % (d['date'].min(), d['date'].max()))
    print(d.head())

    unique_ids = list(set(d['user_id_hash'].dropna()))
    random.shuffle(unique_ids)
    print("Number of unique user IDs: %i" % len(unique_ids))
    
    # set up variables to make sure the ids and row counts for each binary data chunk add up to the whole. 
    all_ids = set()
    total_rows_of_binary_data_added = 0
    
    # do the same for continuous variables. Note that we filter for user_id_hashes in the continuous data 
    # which are contained in binary data. 
    # We need to do this or we will not have period start. 
    total_rows_of_continuous_data_added = {} # how many rows we've added for each continuous variable. 
    rows_of_continuous_data_we_need_to_add = {} # how many rows we NEED to add (ie, are contained in the binary data)
    set_of_unique_ids = set(unique_ids)
    for k in all_continuous_data:
        print("Making sure we add all the necessary rows for %s" % k)
        n_rows_in_original_id_set = all_continuous_data[k]['user_id_hash'].map(lambda x:x in set_of_unique_ids).sum()
        print("Number of rows in continuous data %s contained in the IDs in the original data: %i/%i" % (
            k,
            n_rows_in_original_id_set, 
            len(all_continuous_data[k])))
        total_rows_of_continuous_data_added[k] = 0
        rows_of_continuous_data_we_need_to_add[k] = n_rows_in_original_id_set
        
    for chunk in range(n_chunks):
        ids_in_chunk = set(unique_ids[chunk::n_chunks])
        all_ids = all_ids.union(ids_in_chunk)
        chunk_d = copy.deepcopy(d.loc[d['user_id_hash'].map(lambda x:x in ids_in_chunk)])
        total_rows_of_binary_data_added += len(chunk_d)
        # remove symptoms (eg bleeding) which are logged twice on one day and shouldn't be. 
        # we do this here because computing on only a single chunk will be faster. 
        chunk_d = remove_double_logged_symptoms(chunk_d) 
        
        # Now we need to annotate continuous data with the data we want to join with. 
        data_to_join_with = chunk_d[['user_id_hash', 
                                     'age', 
                                     'timezone', 
                                     'rounded_latitude', 
                                     'us_state']].drop_duplicates()
        data_to_join_with.index = range(len(data_to_join_with))
        assert len(data_to_join_with) == len(set(data_to_join_with['user_id_hash']))
        
        for k in all_continuous_data:
            chunk_idxs = all_continuous_data[k]['user_id_hash'].map(
                lambda x:x in ids_in_chunk)
            continuous_chunk = copy.deepcopy(all_continuous_data[k].loc[chunk_idxs])
            continuous_chunk.index = range(len(continuous_chunk))
            old_length = len(continuous_chunk)
            continuous_chunk = pd.merge(continuous_chunk, 
                                        data_to_join_with, 
                                        on='user_id_hash', 
                                        how='left')
            assert len(continuous_chunk) == old_length
            total_rows_of_continuous_data_added[k] += len(continuous_chunk)
            print("Number of rows in chunk %i from continuous data %s: %i" % 
                  (chunk, 
                   k, 
                   len(continuous_chunk)))
            assert sorted(continuous_chunk.columns) == sorted(chunk_d.columns)
            continuous_chunk = continuous_chunk[chunk_d.columns]
            chunk_d = pd.concat([chunk_d, continuous_chunk])
        chunk_d.index = range(len(chunk_d))
        
        print("Number of IDs in chunk %i: %i; number of rows %i" % (chunk, len(ids_in_chunk), len(chunk_d)))
        chunk_d.to_csv(os.path.join(chunked_tracking_dir, 'tracking_%i.csv' % chunk))
    
    # sanity checks to make sure we added the right number of rows. 
    for k in all_continuous_data:
        print("Total rows of continuous data: %i" % len(all_continuous_data[k]))
        print("Rows of continuous data contained in original id set (for binary features): %i" % 
              rows_of_continuous_data_we_need_to_add[k])
        print("Total rows of continuous data added: %i" % total_rows_of_continuous_data_added[k])
        assert rows_of_continuous_data_we_need_to_add[k] == total_rows_of_continuous_data_added[k]
    assert set(unique_ids) == all_ids
    assert total_rows_of_binary_data_added == len(d)
    print 'Done writing out into chunks!'

def read_original_tracking_data(chunk_number):
    """
    Reads in one chunk of the original tracking data. Checked. 
    """
    path = os.path.join(chunked_tracking_dir, 'tracking_%i.csv' % chunk_number)
    d = pd.read_csv(path, \
                    index_col=0, \
                    nrows = max_tracking_rows_to_read_in, \
                    dtype = {'us_state':str})
    print("Total number of tracking records read in from %s is %i" % 
          (path, 
           len(d)))
    old_length = len(d)
    no_duplicate_symptom_length = len(d[['user_id_hash', 'date', 'category', 'type']].drop_duplicates())
    assert old_length == no_duplicate_symptom_length

    d = d.drop_duplicates()
    new_length = len(d)
    print("Length prior to dropping duplicates: %i; after dropping duplicates: %i" % (old_length, new_length))
    assert old_length == new_length
    
    return d

def perform_sanity_checks(d):
    """
    Prints out column counts etc. Checked.
    """
    print 'Performing basic sanity checks on dataframe'
    assert len(d.drop_duplicates()) == len(d)
    for col in d.columns:
        non_missing_data = d[col].dropna()
        c = Counter(non_missing_data)
        print "In column %s, %2.3f%% of values are non-missing, %i unique values" % (col, \
                                                                                    100.0 * len(non_missing_data) / len(d), 
                                                                                    len(c))
        if len(c) < 500:
            for k in sorted(c.keys(), key = lambda x:c[x])[::-1]:
                print '%-50s %2.1f%%' % (k, 100.0 * c[k] / len(non_missing_data))
        if col == 'date':
            for k in sorted(c.keys()):
                print '%-50s %i' % (k, c[k])
        if col == 'user_id_hash':
            print 'Mean number of readings per user: %2.3f; median %2.3f; max: %i; min: %i' % (np.mean(c.values()), 
                                                                                               np.median(c.values()), 
                                                                                               np.max(c.values()), 
                                                                                               np.min(c.values()))
    print 'Done with sanity checks'

def no_date_x_days_before_date_to_check(date_to_check, datelist, min_gap):
    """
    Returns true if datelist contains a date less than or equal to min_gap days before date_to_check. Checked. 
    Ignores dates that are greater than or equal to date_to_check. 
    """
    for previous_date in datelist:
        if previous_date >= date_to_check:
            continue
        if (date_to_check - previous_date).days <= min_gap:
            return False
    return True

def extract_cycle_start_dates_from_tracking_data(chunk_number):
    """
    Reads in the tracking dictionary and extracts the dates of period starts. 
    returns a dictionary where key is a user id hash mapping to value: list of period starts. 
    Checked. 
    """
    min_gap = 7 # number of days we require between periods to define a cycle start. 
    period_terms = ['period*heavy', 
                    'period*medium', 
                    'period*light']
    d = read_original_tracking_data(chunk_number)
    all_original_users = sorted(list(set(d['user_id_hash'])))
    d = process_implausible_values_of_date_age_or_type(d, min_date='2012-01-01') # we need this here to remove dates that are out of range. 
    print("Computing period starts; number of users before filtering out implausible dates, %i, after filtering %i" % (len(all_original_users), len(set(d['user_id_hash']))))
    
    d['combined_type'] = d['category'] + '*' + d['type']
    grouped_d = d.groupby('user_id_hash')
    n_users_processed = 0
    n_users = len(set(d['user_id_hash'].dropna()))
    users_to_periods = {}

    # Initialize everyone to have no period starts. 
    # we do this for all_original_users because a very very small fraction get filtered out in 
    # process_implausible_values_of_date_age_or_type and this causes bugs. 
    for user_id in all_original_users:
        users_to_periods[user_id] = []

    for user_id, user_d in grouped_d:
        user_d = copy.deepcopy(user_d)
        user_d['on_period'] = user_d['combined_type'].map(lambda x:x in period_terms)
        n_users_processed += 1
        first_date = user_d['date'].min()
        dates_on_period = sorted(list(user_d['date'].loc[user_d['on_period']]))
        dates_on_period = [datetime.datetime.strptime(x, '%Y-%m-%d') for x in dates_on_period]
        period_start_dates = []
        for date in dates_on_period:
            if no_date_x_days_before_date_to_check(date, dates_on_period, min_gap = min_gap):
                period_start_dates.append(date)

        # important: remove the first period start date for each user. 
        # we do this because these dates may not be reliable for all users. 
        # per conversation with Daniel, when some users are onboarded they enter their first period
        # only to the nearest week, so we agree to throw this out to be safe. 

        period_start_dates = period_start_dates[1:]
        users_to_periods[user_id] = period_start_dates
        if n_users_processed % 1000 == 0:
            print 'period start dates computed for %i / %i users' % (n_users_processed, n_users)
    print 'Number of periods per user'
    period_counts = Counter([len(a) for a in users_to_periods.values()])
    for k in sorted(period_counts.keys()):
        print k, period_counts[k]
    return users_to_periods

def extract_symptom_date_relative_to_period(d):
    # extract the date on which a symptom occurs relative to a user's period. 
    # This will be negative if the date is prior to the period. 
    # For robustness checks, we also compute days_after_last_cycle_start (which is >= 0)
    # and days_before_next_cycle_start (which is <= 0). 
    # Checked. 

    print 'Computing date relative to period'
    dates_relative_to_period = []
    days_after_last_cycle_start = []
    days_before_next_cycle_start = []
    max_diff = 20
    for i in range(len(d)):
        if i % 10000 == 0:
            print i, len(d)
        row = d.iloc[i]
        diffs = [(row['date'] - a).days for a in row['period_dates']]

        # compute date relative to period -- this ranges from -20 to 20, aligns to closest period start. 
        date_relative_to_period_diffs = [a for a in diffs if np.abs(a) <= max_diff]
        if len(date_relative_to_period_diffs) > 0:
            date_relative_to_period_diffs = sorted(date_relative_to_period_diffs, key = np.abs)
            dates_relative_to_period.append(date_relative_to_period_diffs[0])
        else:
            dates_relative_to_period.append(None)

        # compute days since last cycle start. 
        diffs_from_last_cycle_start = [a for a in diffs if a >= 0 and a <= max_diff * 2]
        if len(diffs_from_last_cycle_start) > 0:
            days_after_last_cycle_start.append(min(diffs_from_last_cycle_start))
        else:
            days_after_last_cycle_start.append(None)

        # compute days until next cycle start
        diffs_from_next_cycle_start = [a for a in diffs if a <= 0 and a >= -max_diff * 2]
        if len(diffs_from_next_cycle_start) > 0:
            days_before_next_cycle_start.append(max(diffs_from_next_cycle_start))
        else:
            days_before_next_cycle_start.append(None)

    d['date_relative_to_period'] = dates_relative_to_period
    d['days_after_last_cycle_start'] = days_after_last_cycle_start
    d['days_before_next_cycle_start'] = days_before_next_cycle_start
    return d

def annotate_with_cycles(d, users_to_starts):
    # Method which does the heavy lifting. 
    # Adds period date, weekday, month, utc_hour, and local_hour to symptom dataframe. 
    # Checked. 
    max_days_between_session_and_date = 1 # this ensures that session_start_time and date occur on the same day (so the person is actually tracking data for the day when they're logging)
    # date and session_start_date should be identical after the filters we apply, so we use them interchangeably. 
    max_session_length_in_seconds = 3600
    print 'Total number of datapoints', len(d)
    d = copy.deepcopy(d)
    d = d.dropna(subset = ['session_start', 'session_end', 'date']) 
    print 'After dropping rows missing session start or session end or date, %i rows' % len(d)
    
    # per Daniel: All time stamps are UTC. 
    # map dates. And map session start time to local time
    d['date'] = d['date'].map(lambda x:datetime.datetime.strptime(x, '%Y-%m-%d'))
    d['session_start_utc_time'] = d['session_start'].map(lambda x:datetime.datetime.strptime(x.split('.')[0], '%Y-%m-%d %H:%M:%S'))
    d['session_end_utc_time'] = d['session_end'].map(lambda x:datetime.datetime.strptime(x.split('.')[0], '%Y-%m-%d %H:%M:%S'))
    # Filter out people with super-long sessions (could be weird)
    d['session_length'] = (d['session_end_utc_time'].subtract(d['session_start_utc_time'])).map(lambda x:x.total_seconds())
    assert((d['session_length'] >= 0).all())
    for session_length_in_seconds in [60, 120, 180, 300, 600, 1800, 3600]:
        print 'Proportion %2.3f of sessions had session length >= %i seconds' % ((d['session_length'] >= session_length_in_seconds).mean(), 
                                                                         session_length_in_seconds)
    d = d.loc[d['session_length'] <= max_session_length_in_seconds]
    print 'After filtering for sessions with length <= %i seconds, %i rows' % (max_session_length_in_seconds, len(d))

    # now need to map to local times using timezone. Spot-checked these look ok. 
    all_local_times = []
    n_mismatches_between_old_and_new_methods = 0
    for i in range(len(d)):
        # TZ conversion: annoying. 
        
        # this is the old method. Deprecated but we keep it around just to check that both ways are consistent. 
        old_tz = d['timezone'].iloc[i]
        old_utc_time = d['session_start_utc_time'].iloc[i].replace(tzinfo=pytz.utc)
        old_local_time = old_utc_time.astimezone(pytz.timezone(old_tz)) 
        old_local_time = old_local_time.replace(tzinfo = None)
        
        # this is the new method. 
        # source: 
        # https://stackoverflow.com/questions/4563272/convert-a-python-utc-datetime-to-a-local-datetime-using-only-python-standard-lib, 
        # "Using pytz (both Python 2/3)" answer
       
        # extract UTC time. make sure UTC is a datetime, not a pandas timestamp, just to be safe. 
        # these are generally interchangeable but deal with DST slightly differently (sigh). 

        utc_time = datetime.datetime.strptime(d['session_start'].iloc[i].split('.')[0], 
                                              '%Y-%m-%d %H:%M:%S').replace(tzinfo=pytz.utc)
        local_tz = pytz.timezone(d['timezone'].iloc[i])
        local_time = utc_time.replace(tzinfo=pytz.utc).astimezone(local_tz)
        local_time = local_tz.normalize(local_time).replace(tzinfo=None)

        if local_time != old_local_time:
            n_mismatches_between_old_and_new_methods += 1

        all_local_times.append(local_time)

    print 'Number of mismatches between old and new timezone conversion methods: %i' % n_mismatches_between_old_and_new_methods
    d['session_start_local_time'] = all_local_times
    
    # For the continuous features, there is no meaningful time information, 
    # so we just set the session start local time equal to midnight on the date.
    # This is the same as session_start_utc_time.
    continuous_feature_idxs = d['category'] == 'continuous_features'
    d.loc[continuous_feature_idxs, 'session_start_local_time'] = d.loc[continuous_feature_idxs, 
                                                                       'session_start_utc_time']
    
    # filter out all session start times which are too old. 
    # This ensures that the date will be equal to the date of session_start_local_time 
    # so the user is logging a symptom for the same day. 
    d['diff_between_session_start_and_date_logging'] = (d['session_start_local_time'].subtract(d['date']).map(lambda x:x.total_seconds()))
    print 'Proportion of sessions with login date after session start time (this should happen very rarely): %2.5f' % \
        (d['diff_between_session_start_and_date_logging'] < 0).mean()
        
    print 'Proportion for continuous features: %2.5f' % \
        (d.loc[continuous_feature_idxs, 'diff_between_session_start_and_date_logging'] < 0).mean()

    for n_days in [1, 1.05, 1.1, 1.2, 1.5, 2, 3, 4, 5]:
        print 'Proportion of sessions with login date more than %2.3f days before session start time: %2.3f' % \
            (n_days, \
            (d['diff_between_session_start_and_date_logging'] > 86400 * n_days).mean())
    d = d.loc[d['diff_between_session_start_and_date_logging'].map(lambda x:(x < 86400 * max_days_between_session_and_date) and (x >= 0))]
    assert((d['session_start_local_time'].map(lambda x:x.strftime('%Y-%m-%d')) == \
            d['date'].map(lambda x:x.strftime('%Y-%m-%d'))).all())
    print 'After dropping with login date more than %i days before session start, %i rows' % (max_days_between_session_and_date, len(d))
    d['period_dates'] = d['user_id_hash'].map(lambda x:users_to_starts[x])
    assert pd.isnull(d['period_dates']).sum() == 0
    #d = d.dropna(subset = ['period_dates'])
    d = extract_symptom_date_relative_to_period(d)
    d = d.dropna(subset = ['date_relative_to_period'])
    print 'After dropping rows overly far from a period start, %i rows' % (len(d))
    d['weekday'] = d['session_start_local_time'].map(lambda x:x.strftime('%A'))
    d['month'] = d['session_start_local_time'].map(lambda x:int(x.strftime('%m')))

    d['utc_hour'] = d['session_start_utc_time'].map(lambda x:int(x.strftime('%H')))
    d['local_hour'] = d['session_start_local_time'].map(lambda x:int(x.strftime('%H')))
    
    return d

def get_last_date_logged_any_binary_symptom(d):
    """
    Compute the last date when the user logged any binary symptom. 
    """
    all_users = sorted(list(set(d['user_id_hash'])))
    binary_d = d.loc[d['category'].map(lambda x:'continuous' not in x)]
    assert binary_d['type'].map(lambda x:WEIGHT_SUBSTRING in str(x)).sum() == 0
    assert binary_d['type'].map(lambda x:BBT_SUBSTRING in str(x)).sum() == 0
    assert binary_d['type'].map(lambda x:HEART_SUBSTRING in str(x)).sum() == 0
    binary_d = binary_d.loc[binary_d['date'].map(lambda x:x<= '2017-11-20')]
    grouped_d = binary_d[['user_id_hash', 'date']].groupby('user_id_hash').max()
    print 'Maximum date logged any binary symptom is', grouped_d['date'].max()
    results = dict(zip(grouped_d.index, grouped_d['date']))
    for user_id in all_users:
        if user_id not in results:
            print("Warning: user id %s has no binary symptoms, setting last binary symptom log date to None." % user_id)
            results[user_id] = None
    return results

def annotate_with_period_summary_statistics_for_each_user(d, users_to_starts):
    print("Computing period summary statistics like cycle length")
    summary_stats = {}
    for user_id in users_to_starts:
        period_starts = users_to_starts[user_id]
        n_starts = len(period_starts)
        cycle_lengths = []
        mean_cycle_length = None
        median_cycle_length = None
        std_cycle_length = None

        if n_starts >= 2:
            for i in range(len(period_starts) - 1):
                length = (period_starts[i + 1] - period_starts[i]).days
                assert length >= 0
                if length > 0: # sometimes lengths can be zero because users log eg light + medium bleeding on same day. In later versions of the data processing this should no longer happen, but it's a harmless filter. 
                    cycle_lengths.append(length)
            cycle_lengths = np.array(cycle_lengths)
            cycle_lengths = cycle_lengths[(cycle_lengths >= 15) & (cycle_lengths <= 45)] # these are the cutoffs used in the Chiazze paper. 
            if len(cycle_lengths) >= 2:
                # Only compute these stats for users for whom we have at least 2 lengths, so estimates aren't totally unreliable. 
                # (std can't be computed at all for fewer than 2 lengths)
                mean_cycle_length = np.mean(cycle_lengths)
                median_cycle_length = np.median(cycle_lengths)
                std_cycle_length = np.std(cycle_lengths)

        summary_stats[user_id] = {'n_cycle_starts_logged':n_starts, 
        'mean_cycle_length':mean_cycle_length, 
        'std_cycle_length':std_cycle_length, 
        'median_cycle_length':median_cycle_length}

    for stat in ['n_cycle_starts_logged', 'mean_cycle_length', 'std_cycle_length', 'median_cycle_length']:
        d[stat] = d['user_id_hash'].map(lambda user_id:summary_stats[user_id][stat])
        print("%s: mean %2.3f, median %2.3f; frac missing data %2.3f" % (stat, 
            np.mean(d[stat].dropna()), np.median(d[stat].dropna()), pd.isnull(d[stat]).mean()))
    return d

def make_big_annotated_symptom_d(chunk_number):
    # make big annotated symptom dataframe with weekday, month, and day of period. 
    # Checked. 
    d = read_original_tracking_data(chunk_number)
    perform_sanity_checks(d)
    users_to_starts = extract_cycle_start_dates_from_tracking_data(chunk_number)
    users_to_starts_filename = os.path.join(processed_chunked_tracking_dir, 'users_to_period_starts_chunk_%i.pkl' % chunk_number)
    cPickle.dump(users_to_starts, open(users_to_starts_filename, 'wb'))

    users_to_last_date_logged_any_binary_symptom = get_last_date_logged_any_binary_symptom(d)
    d['last_date_logged_any_binary_symptom'] = d['user_id_hash'].map(lambda x:users_to_last_date_logged_any_binary_symptom[x])

    d = annotate_with_period_summary_statistics_for_each_user(d, users_to_starts)

    d = annotate_with_cycles(d, users_to_starts)
    # filter for days after all symptoms have appeared: 2015-11-01
    # see this: https://helloclue.com/articles/about-clue/23-new-ios-tracking-categories-more-accurate-algorithm
    # confirmed that symptoms we care about have started being logged by this date. 
    d = process_implausible_values_of_date_age_or_type(d, min_date='2015-11-01')
    d = d[['user_id_hash', 'date', 'session_start', 'session_start_local_time', 'session_end', 'category', 'type', 'value', \
    'date_relative_to_period', 'days_after_last_cycle_start', 'days_before_next_cycle_start', \
    'weekday', 'month', 'utc_hour', 'local_hour', \
    'age', 'timezone', 'rounded_latitude', 'us_state', 'last_date_logged_any_binary_symptom', \
    'n_cycle_starts_logged', 'mean_cycle_length', 'std_cycle_length', 'median_cycle_length']]

    d.index = range(len(d))
    assert_a_lot_of_things_which_should_be_true(d)
    print 'Final outfile looks like'
    print d.head(n = 10)
    print 'Number of rows', len(d)

    # make sure dates are consistently formatted prior to writing out. 
    d['date'] = d['date'].map(lambda x:x.strftime('%Y-%m-%d'))
    for k in ['session_start_local_time']:
        d[k] = d[k].map(lambda x:x.strftime('%Y-%m-%d %H:%M:%S'))

    # make sure we don't use deprecated age column anywhere by recalculating it.    
    d = annotate_with_average_age_when_user_logged_symptoms(d)

    d.to_csv(big_annotated_symptom_filename(chunk_number))
    print("Finished writing out data for chunk %i" % chunk_number)

def annotate_with_average_age_when_user_logged_symptoms(d):
    """
    The age recorded in the original data is their age when they last logged a binary symptom. 
    Instead we recalculate it so it is their average age when they were logging. 
    This barely changes the final age estimate at all, because two things cancel out: 
    a) we add .5 to the age (since users are in expectation 16.5 if their age is 16)
    b) we subtract a correction from the age because we want to take average age during the logging period. 
    """
    print("Correcting for average age during logging period.")
    d['age_when_last_logged_binary_symptom'] = d['age'] + .5 # expected age when they last logged a binary symptom
    
    grouped_d = d[['user_id_hash', 'date', 'last_date_logged_any_binary_symptom', 'age_when_last_logged_binary_symptom']].groupby('user_id_hash')
    users_to_average_ages = {}
    all_corrections = []
    for user_id, user_d in grouped_d:
        unique_ages = list(set(user_d['age_when_last_logged_binary_symptom'].dropna()))
        if len(unique_ages) == 0:
            users_to_average_ages[user_id] = None
        else:
            assert len(unique_ages) == 1
            age_at_last_symptom_log = unique_ages[0]
            first_date_logging_in_log_period = datetime.datetime.strptime(user_d['date'].min(), '%Y-%m-%d')
            last_date_logging_in_log_period = datetime.datetime.strptime(user_d['date'].max(), '%Y-%m-%d')
            last_date_logging_any_binary_symptom = datetime.datetime.strptime(user_d['last_date_logged_any_binary_symptom'].iloc[0], '%Y-%m-%d')
            logging_interval = (last_date_logging_in_log_period - first_date_logging_in_log_period).days
            # the correction is: the gap between the last binary symptom logged and the start of the logging interval + half the logging interval
            # this is almost always greater than 0, but it doesn't have to be, because someone could continuous symptoms for longer than they logged
            # binary symptoms. 
            correction_in_days = ((last_date_logging_any_binary_symptom - first_date_logging_in_log_period).days - logging_interval / 2.)
            correction = correction_in_days / 365.
            all_corrections.append(correction)
            average_age = age_at_last_symptom_log - correction
            users_to_average_ages[user_id] = average_age
    print('Mean difference between last date of logging any symptom, and middle of logging period, in years: %2.3f; median %2.3f' % (np.mean(all_corrections), np.median(all_corrections)))
    d['average_age'] = d['user_id_hash'].map(lambda x:users_to_average_ages[x])
    amount_we_actually_changed_age = d['age'] - d['average_age']
    print('Amount age changed, averaged across logs: mean %2.3f; median %2.3f' % (np.mean(amount_we_actually_changed_age.dropna()), np.median(amount_we_actually_changed_age.dropna())))
    d['age'] = d['average_age']
    del d['average_age']
    return d
            

def process_implausible_values_of_date_age_or_type(d, min_date):
    """
    filter for implausible values of date or age, or people missing type data.
    Note that we call this twice with different minimum dates. 
    1. we call it for all the symptoms, with a min date of when all the symptoms are available (for platform consistency)
    2. we call it to filter the period start data, with an earlier min date (because period logging is consistent)
    latitude is already fine. 
    Checked. 
    """
    d = copy.deepcopy(d)
    # the early date range here is to make sure that all dates are after all symptoms have been rolled out. 
    good_date_idxs = d['date'].map(lambda x:(str(x) >= min_date) and (str(x) <= '2017-11-20'))
    assert np.isnan(good_date_idxs).sum() == 0
    print("After filtering out dates which are out of range, %2.5f%% of rows remain" % (100*good_date_idxs.mean()))
    d = d.loc[good_date_idxs]
    
    # for age filters, we don't want to throw out people with NaNs for age. We just want to set people
    # with totally implausible values to NA. 
    # these age ranges come from looking at the data: there are very few people over 60 or under 10. 
    # while menopause probably precedes 60, menopausal people might still be using the app and entering reliable data. 
    bad_age_idxs = d['age'].map(lambda x:(x > 60) or (x < 10)) 
    assert np.isnan(bad_age_idxs).sum() == 0
    d.loc[bad_age_idxs, 'age'] = np.nan
    print("After setting people with bad age values to NA for age, %2.3f%% of people set to NA." % (100 * bad_age_idxs.mean()))
    bad_age_idxs = d['age'].map(lambda x:(x > 60) or (x < 10))
    assert bad_age_idxs.sum() == 0
    assert np.isnan(bad_age_idxs).sum() == 0
    
    # filter out people missing type data. 
    old_length = 1.*len(d)
    d = d.dropna(subset=['type'])   
    print("After removing people with bad type data, %2.5f%% of rows remain." % (100.0*len(d) / old_length))
    return d

def assert_a_lot_of_things_which_should_be_true(d):
    """
    make sure every column except for age and us_state have no NaNs. 
    Ok to have NaNs in age and us_state because some people are missing age data and some people are not from the US. 
    For columns with predictable values, assert those values are reasonable.
    Checked. 
    """
    for c in d.columns:
        if c in ['age', 'us_state', 'days_before_next_cycle_start', 'days_after_last_cycle_start', 'mean_cycle_length', 'std_cycle_length', 'median_cycle_length']:
            continue
        print 'Asserting there are no nans in %s' % c
        assert len(d[c].dropna()) == len(d[c])
    
    print 'Asserting date_relative_to_period has appropriate values'
    assert set(d['date_relative_to_period']) == set(range(-20, 21))

    print "Asserting days_after_last_cycle_start has appropriate values"
    assert set(d['days_after_last_cycle_start'].dropna()) == set(range(0, 41))

    print "Asserting days_before_next_cycle_start has appropriate values"
    assert set(d['days_before_next_cycle_start'].dropna()) == set(range(-40, 1))
    
    print 'Asserting weekday has appropriate values'
    assert set(d['weekday']) == set({'Friday', 'Monday', 'Saturday', 'Sunday', 'Thursday', 'Tuesday', 'Wednesday'})
    
    print 'Asserting month has appropriate values'
    assert set(d['month']) == set(range(1, 13))
    
    print 'Asserting UTC hour has appropriate values'
    assert set(d['utc_hour']) == set(range(0, 24))
    
    print 'Asserting local hour has appropriate values'
    assert set(d['local_hour']) == set(range(0, 24))
    
    print 'Asserting rounded_latitude has appropriate values'
    assert d['rounded_latitude'].map(lambda x:np.abs(x) <= 90).all()
    
    print 'All assertions passed'
    
    
def load_processed_data():
    # Reads in a fraction of the processed data given by chunks_to_use. 
    # Checked. 
    print 'Reading in files for chunks', chunks_to_use
    d = pd.concat([pd.read_csv(big_annotated_symptom_filename(chunk_number), 
                              nrows = max_tracking_rows_to_read_in, index_col=0) 
                   for chunk_number in chunks_to_use])
    d.index = range(len(d))
    print 'Total rows read in: %i' % len(d)
    return d

def get_timezone_to_country_mapping():
    """
    This function returns a dictionary mapping timezones to countries
    using pytz. 

    Each timezone maps to a single country (with very rare exceptions we don't care about)
    https://en.wikipedia.org/wiki/Tz_database#Zones_covering_multiple_post-1970_countries

    Spot-checked that this correctly maps 25 most common timezones. 
    """
    print("Loading in timezone to country mapping from pytz")
    tzs_to_countries = {}
    for country in pytz.country_timezones:
        for tz in pytz.country_timezones[country]:
            assert tz not in tzs_to_countries
            tzs_to_countries[tz] = pytz.country_names[country]

    # Now we need to add some deprecated timezones that occur in our dataset
    # See eg http://www.timezoneconverter.com/cgi-bin/zoneinfo.tzc?s=default&tz=America/Buenos_Aires
    tzs_to_countries['Asia/Calcutta'] = tzs_to_countries['Asia/Kolkata']
    tzs_to_countries['America/Buenos_Aires'] = tzs_to_countries['America/Argentina/Buenos_Aires']
    tzs_to_countries['America/Cordoba'] = tzs_to_countries['America/Argentina/Cordoba']
    tzs_to_countries['America/Mendoza'] = tzs_to_countries['America/Argentina/Mendoza']
    tzs_to_countries['America/Catamarca'] = tzs_to_countries['America/Argentina/Catamarca']
    tzs_to_countries['Asia/Saigon'] = tzs_to_countries['Asia/Ho_Chi_Minh']
    tzs_to_countries['Asia/Katmandu'] = tzs_to_countries['Asia/Kathmandu']
    tzs_to_countries['America/Jujuy'] = tzs_to_countries['America/Argentina/Jujuy']

    for tz in tzs_to_countries:
        tzs_to_countries[tz] = tzs_to_countries[tz].encode('ascii', 'ignore')

    for i, k in enumerate(sorted(tzs_to_countries.keys())):
        print '%i. %-30s -> %s' % (i + 1, k, tzs_to_countries[k])
    print("Done loading timezone to country mapping")

    return tzs_to_countries

def annotate_with_user_features_to_stratify_by(d):
    """
    add user features to perform robustness checks + look for heterogeneity. 
    Checked.
    """
    d = copy.deepcopy(d)
    print("Annotating with user features")
    grouped_d = d.groupby('user_id_hash')
    print("Computing features for each user...")
    # compute the features for each user. 
    users_to_features = {}
    tzs_to_countries = get_timezone_to_country_mapping()

    def as_categorical(val, bins):
        val = pd.cut([val], bins, right=False).astype('str')[0] 
        if val == 'nan' or val == 'None':
            val = None
        return val
    def assert_and_return_single_val(arr):
        # make sure that array arr contains a single value, and return it. 
        assert type(arr) == np.ndarray
        unique_vals = set(arr)
        if len(unique_vals) > 1:
            if not np.isnan(list(unique_vals)).all():
                print(unique_vals)
                raise Exception("multiple values for a single user!")
        return list(unique_vals)[0]

    # compute largest timezones
    timezone_counts = Counter(d['timezone'].dropna())

    largest_timezones = sorted(timezone_counts, key = lambda x:timezone_counts[x])[::-1][:25]
    print 'Largest timezones are'
    for tz in largest_timezones:
        print '%s: %2.3f%% of observations' % (tz, 100.*timezone_counts[tz] / len(d))
    largest_timezones = set(largest_timezones)

    # get countries. 
    d['country'] = d['timezone'].map(lambda x:tzs_to_countries[str(x)] if str(x) in tzs_to_countries else None)
    country_counts = Counter(d['country'].dropna())
    largest_countries = sorted(country_counts, key = lambda x:country_counts[x])[::-1]
    print '\n\n\n****Country fractions'
    for i, country in enumerate(largest_countries):
        print '%-3i. %-50s: %2.3f%% of observations (approximately %i in full dataset)' % (i + 1,
            country, 
            100.*country_counts[country] / len(d), 
            country_counts[country] * n_chunks)

    assert sum(pd.isnull(d['country'])) == 0

    for user_id_hash, user_d in grouped_d:
        user_features = {}
        user_features['exact_total_symptoms_logged'] = len(user_d)
        user_features['total_symptoms_logged'] = as_categorical(len(user_d), [-1, 100, 250, 500, np.inf])
        user_features['start_date'] = user_d['date'].min()
        user_features['start_year'] = str(user_features['start_date']).split('-')[0]
        user_features['end_date'] = user_d['date'].max()
        user_features['exact_n_days_using_app'] = (datetime.datetime.strptime(user_features['end_date'].split()[0], '%Y-%m-%d') - datetime.datetime.strptime(user_features['start_date'].split()[0], '%Y-%m-%d')).days + 1
        assert user_features['exact_n_days_using_app'] >= 1
        user_features['n_days_using_app'] = as_categorical(user_features['exact_n_days_using_app'], [-1, 30, 60, 180, 365, np.inf])
        user_features['n_symptom_categories_used'] = as_categorical(len(set(user_d['category'].dropna())), [-1, 10, 15, 20, np.inf])
        user_features['logged_hormonal_birth_control'] = user_d['category'].map(lambda x:'_hbc' in str(x)).sum() > 0
        user_features['logged_iud'] = user_d['category'].map(lambda x:'iud' in str(x)).sum() > 0
        user_features['logged_birth_control_pill'] = user_d['category'].map(lambda x:'pill_hbc' in str(x)).sum() > 0
        user_features['proportion_pain_symptoms'] = as_categorical((user_d['category'] == 'pain').mean(), [-.01, .05, .1, .2, 1])
        user_features['proportion_mood_symptoms'] = as_categorical((user_d['category'] == 'emotion').mean(), [-.01, .05, .1, .2, 1])
        user_features['logged_any_cigarettes'] = (user_d['combined_type'] == 'party*cigarettes').sum() > 0
        user_features['logged_any_alcohol'] = (user_d['combined_type'].map(lambda x:x in ['party*drinks_party', 
                                                                           'party*hangover'])).sum() > 0
        user_features['logged_any_exercise'] = (user_d['category'] == 'exercise').sum() > 0
        # define very active loggers as people who log at least 12 unique months. 
        unique_months_logged = len(set(user_d['date'].map(lambda x:'-'.join(x.split('-')[:2]))))
        if unique_months_logged >= 12:
            user_features['very_active_loggers'] = True
        else:
            user_features['very_active_loggers'] = None
        user_features['age'] = assert_and_return_single_val(user_d['age'].values)
        user_features['categorical_age'] = as_categorical(
            assert_and_return_single_val(user_d['age'].values),
            [10, 15, 20, 25, 30, 35, np.inf])
        for period_feature in ['n_cycle_starts_logged','mean_cycle_length','std_cycle_length','median_cycle_length']:
            user_features[period_feature] = assert_and_return_single_val(user_d[period_feature].values)

        # various location features
        user_tz = assert_and_return_single_val(user_d['timezone'].values)
        user_features['country'] = assert_and_return_single_val(user_d['country'].values)
        user_features['largest_timezones'] = user_tz if user_tz in largest_timezones else None
        user_features['in_united_states'] = assert_and_return_single_val((~pd.isnull(user_d['us_state'])).values)
        rounded_latitude = assert_and_return_single_val(user_d['rounded_latitude'].values)
        user_features['categorical_latitude'] = as_categorical(rounded_latitude,  [-90, -25, 0, 25, 50, 90]) # too few people between -90 and -50 so we make it a single category. 
        user_features['hemisphere'] = get_hemisphere(rounded_latitude)

        # define active loggers in each hemisphere. Value is either True or None because we only want to analyze subset for which it's True. 
        user_features['very_active_northern_hemisphere_loggers'] = (user_features['very_active_loggers'] == True) and (rounded_latitude > 0)
        if user_features['very_active_northern_hemisphere_loggers'] == False:
            user_features['very_active_northern_hemisphere_loggers'] = None

        user_features['very_active_southern_hemisphere_loggers'] = (user_features['very_active_loggers'] == True) and (rounded_latitude < 0)
        if user_features['very_active_southern_hemisphere_loggers'] == False:
            user_features['very_active_southern_hemisphere_loggers'] = None

        users_to_features[user_id_hash] = user_features
    print("Annotating original dataframe with the features")
    
    print("Done annotating with user features")
    return users_to_features

def compute_emission_probability_conditional_on_logging(d, 
                                                        opposite_name, 
                                                        filter_for_users_who_log_at_least_once, 
                                                        group_by_hour_as_well=False, 
                                                        symptom_list_to_use_as_normalizer=None):
    """
    Checked. 
    for each user and each day, define it as 1 if the user logged a symptom in the symptom group
    and 0 if they did not. (Days in which a user did not log anything are simply missing data.)
    if filter_for_users_who_log_at_least_once, filter for users who log a symptom in the group at least once. 
    This is probably a good thing to do with sex and exercise because if the user never logs we don't know
    whether they actually never experienced the symptom or just never logged it. 

    If group_by_hour_as_well is True, the group_by is taken by hour as well, rather than randomizing hour. 

    If symptom_list_to_use_as_normalizer is not None, we only use these symptoms to normalize
    """
    d = copy.deepcopy(d)
    print("Computing emission probability conditional on logging with %s" % opposite_name)

    if opposite_name == 'exercise*exercised_versus_exercise*did_not_exercise':
        d['symptom_in_group'] = d['combined_type'].map(lambda x:'exercise*' in str(x))
        assert symptom_list_to_use_as_normalizer is None
    elif opposite_name == 'sex*had_sex_versus_sex*did_not_have_sex':
        # there is also sex*high_sex_drive but this doesn't imply they had sex. 
        d['symptom_in_group'] = d['combined_type'].map(lambda x:str(x) in ['sex*protected_sex',
                                                                           'sex*unprotected_sex', 
                                                                           'sex*withdrawal_sex'])
        assert symptom_list_to_use_as_normalizer is None 
    elif ('didnt*didnt_log_symptom' in opposite_name) and (opposite_name[:15] == 'logged_symptom_'):
        # eg, logged_symptom_emotion*happy_versus_didnt*didnt_log_symptom
        symptom_of_interest = opposite_name.split('logged_symptom_')[1].split('_versus')[0]
        if symptom_list_to_use_as_normalizer is not None:
            print("Prior to filtering for specific symptom normalization list, %i rows" % len(d))
            assert symptom_of_interest in symptom_list_to_use_as_normalizer
            d = d.loc[d['combined_type'].map(lambda x:x in symptom_list_to_use_as_normalizer)]
            print("After filtering for specific symptom normalization list, %i rows" % len(d))

        d['symptom_in_group'] = d['combined_type'] == symptom_of_interest
        if d['symptom_in_group'].sum() == 0:
            print('There are no users who logged %s' % symptom_of_interest)
            return None

    else:
        raise Exception("Not a valid symptom name")
                
    if filter_for_users_who_log_at_least_once:
        print 'Filtering for users who log at least once! prior to filtering %i users' % len(set(d['user_id_hash']))
        good_users = set(d['user_id_hash'].loc[d['symptom_in_group'] == True])
        d = d.loc[d['user_id_hash'].map(lambda x:x in good_users)]
        print 'After filtering, %i users' % len(set(d['user_id_hash']))
    d.index = range(len(d))

    # first compute the grouped dataframe. Group by day because we have no time information. 
    cols_to_group_by = ['user_id_hash', 
    'date', 
    'date_relative_to_period', 
    'days_before_next_cycle_start', 
    'days_after_last_cycle_start', 
    'weekday', 
    'month']

    if group_by_hour_as_well:
        cols_to_group_by.append('local_hour')
        # filter out continuous symptoms because they have fake local hours + will give misleading results. 
        continuous_symptom_idxs = d['combined_type'].map(lambda x:'continuous' in x).values
        if continuous_symptom_idxs.sum() > 0:
            exactly_midnight_idxs = d['session_start_local_time'].map(lambda x:x.split()[1] == '00:00:00').values
            print("Proportion of values which are exactly midnight among continuous symptoms: %2.7f; amid other symptoms: %2.7f" % 
                (exactly_midnight_idxs[continuous_symptom_idxs].mean(), 
                 exactly_midnight_idxs[~continuous_symptom_idxs].mean()))
            d = d.loc[~continuous_symptom_idxs]
            d.index = range(len(d))

    # This is truly a terrible hack to avoid dropping rows with missing values 
    # due to idiosyncrasies of pandas groupby :(
    cols_with_missing_vals = ['days_before_next_cycle_start', 'days_after_last_cycle_start']
    for col in cols_to_group_by:
        if col in cols_with_missing_vals:
            continue
        if pd.isnull(d[col]).sum() != 0:
            print "Warning! %s has %i null values out of %i" % (col, pd.isnull(d[col]).sum(), len(d))
            assert False
    HACKY_NA_VALUE = -999999999
    d = d.fillna(value=HACKY_NA_VALUE)
    original_n_unique_dates_and_users = len(d[['user_id_hash', 'date']].drop_duplicates())


    mood_d = d[cols_to_group_by + ['symptom_in_group']].groupby(cols_to_group_by).sum()
    mood_d = mood_d.reset_index() # https://stackoverflow.com/questions/20110170/turn-pandas-multi-index-into-column
    if not group_by_hour_as_well:
        assert len(mood_d[['user_id_hash', 'date']].drop_duplicates()) == len(mood_d)
    mood_d['good_mood'] = 1.*(mood_d['symptom_in_group'] > 0) # good mood if logged at least one symptom in group
    for col in cols_with_missing_vals:
        mood_d.loc[mood_d[col] == HACKY_NA_VALUE, col] = None

    # remove symptom_in_group column (don't need it anymore)
    mood_d = mood_d[[a for a in mood_d.columns if a != 'symptom_in_group']]  

    if not group_by_hour_as_well:
        assert 'local_hour' not in mood_d.columns
        # set hour to be something random so we don't run into errors. 
        # important to remember that hour is not meaningful! 
        mood_d['local_hour'] = [random.choice(range(24)) for a in range(len(mood_d))]
        # Make sure the number of dates hasn't changed (we haven't lost any user-days). 
        assert original_n_unique_dates_and_users == len(mood_d[['user_id_hash', 'date']].drop_duplicates())

    return mood_d

def generate_filename_for_symptom_group_and_chunk(symptom_group, chunk_number):
    filename = os.path.join(processed_chunked_tracking_dir, 
            'individual_symptoms/', 
            '%s_chunk_%i.csv' % ('_versus_'.join(symptom_group), chunk_number))
    return filename

def get_user_feature_filename(chunk_number):
    return os.path.join(processed_chunked_tracking_dir, 
        'user_features', 
        'user_features_chunk_%i.pkl' % chunk_number)


def break_into_individual_symptoms(full_d, all_symptom_groups, chunk_number):
    """
    Given a full dataframe, breaks it into individual pieces. Checked. 
    """
    for symptom_group in all_symptom_groups:
        if len(symptom_group) == 1:
            # continuous features case
            symptom_key = symptom_group[0]
            assert 'continuous_features' in symptom_key
            symptom_idxs = (full_d['category'] == 'continuous_features') & (full_d['type'] == symptom_key.split('*')[1])
            d = copy.deepcopy(full_d.loc[symptom_idxs])
            d['good_mood'] = d['value']

            # set hour to be something random so we don't run into errors. Hour is not meaningful for these symptoms. 
            d['local_hour'] = [random.choice(range(24)) for a in range(len(d))]
            good_symptom = symptom_key
            bad_symptom = 'continuous_features*null'
        elif len(symptom_group) == 2:
            good_symptom, bad_symptom = symptom_group
            symptom_key = '%s_versus_%s' % (good_symptom, bad_symptom)
            if good_symptom == 'sleep*slept_6_hours_or_more' and bad_symptom == 'sleep*slept_6_hours_or_less':
                symptom_idxs = full_d['combined_type'].map(lambda x:'sleep*' in str(x))
                assert np.isnan(symptom_idxs).sum() == 0
                d = copy.deepcopy(full_d.loc[symptom_idxs])
                d['good_mood'] = (d['combined_type'].map(lambda x:x in ['sleep*6-9', 'sleep*>9']))
            elif ((good_symptom == 'sex*had_sex' and bad_symptom == 'sex*did_not_have_sex') or 
                  (good_symptom == 'exercise*exercised' and bad_symptom == 'exercise*did_not_exercise')):
                d = compute_emission_probability_conditional_on_logging(full_d, 
                                                            '%s_versus_%s' % (good_symptom, bad_symptom), 
                                                            filter_for_users_who_log_at_least_once=True)
            else:
                symptom_idxs = full_d['combined_type'].map(lambda x:x in [good_symptom, bad_symptom])
                assert np.isnan(symptom_idxs).sum() == 0
                d = copy.deepcopy(full_d.loc[symptom_idxs])
                d['good_mood'] = d['combined_type'] == good_symptom
        else:
            raise Exception("Number of symptoms must be one or two!")
        d.index = range(len(d))
        d['good_mood'] = d['good_mood'] * 1.0
        d = d[['user_id_hash', 
        'date', 
        'date_relative_to_period', 
        'days_after_last_cycle_start', 
        'days_before_next_cycle_start',
         'weekday', 
         'month', 
         'local_hour', 
         'good_mood']]

        print("total number of rows for %s is %i" % (symptom_group, len(d)))
        filename = generate_filename_for_symptom_group_and_chunk(symptom_group, chunk_number)
        d.to_csv(filename)

def compute_user_features_and_split_into_individual_symptoms(chunk_number):
    """
    Checked. 
    1. Compute features for every user in a chunk
    2. Split the chunk into individual binary symptoms
    """
    d = pd.read_csv(big_annotated_symptom_filename(chunk_number),
        nrows = max_tracking_rows_to_read_in, 
        index_col=0)
    d['combined_type'] = d['category'] + "*" + d['type']
    print("Finished reading in chunk %i, with %i rows" % (chunk_number, len(d)))
    print("Computing user features for chunk %i" % chunk_number)
    users_to_features = annotate_with_user_features_to_stratify_by(d)
    user_feature_filename = get_user_feature_filename(chunk_number)
    cPickle.dump(users_to_features, open(user_feature_filename, 'wb'))
    print("Saved features for %i users in %s" % (len(users_to_features), user_feature_filename))

    print("Breaking chunk into individual symptoms")
    break_into_individual_symptoms(d, ALL_SYMPTOM_GROUPS, chunk_number)
    print("Data processing for chunk %i completed successfully" % chunk_number)

def load_dataframe_for_symptom_group(symptom_group, chunks_to_use, annotate_with_user_features=True, verbose=False):
    """
    Checked. Loops over all chunks in chunks_to_use; for each chunk, annotates with symptoms. 
    Then concatenates all chunks together. 
    """
    print("\n\n\n****Loading data for %s" % str(symptom_group))
    all_dataframes = []
    pd.set_option('max_columns', 500)
    chunks_added = 0
    for chunk_number in chunks_to_use:
        print("Loading chunk %i" % chunk_number)
        symptom_df = pd.read_csv(generate_filename_for_symptom_group_and_chunk(symptom_group, chunk_number), index_col=0)
        if annotate_with_user_features:
            user_feature_filename = get_user_feature_filename(chunk_number)
            user_features = cPickle.load(open(user_feature_filename, 'rb'))
            user_feature_names = sorted(list(user_features[user_features.keys()[0]].keys()))
            
            for feature in user_feature_names:
                symptom_df[feature] = symptom_df['user_id_hash'].map(lambda user_id:user_features[user_id][feature])
            sorted_cols = sorted(symptom_df.columns, key = lambda x:x in user_feature_names)
        else:
            sorted_cols = list(symptom_df.columns)

        if chunks_added == 0:
            original_sorted_cols = sorted_cols
        else:
            assert sorted_cols == original_sorted_cols
        symptom_df = symptom_df[sorted_cols]
        all_dataframes.append(symptom_df)
        chunks_added += 1

    full_df = pd.concat(all_dataframes)
    full_df.index = range(len(full_df))
    assert list(full_df.columns) == sorted_cols
    full_df['good_mood'] = full_df['good_mood'] * 1.0
    
    print("Total rows read in for %s, with %i chunks: %i. Final dataframe looks like" % (str(symptom_group), len(chunks_to_use), len(full_df)))
    if verbose:
        print(full_df.iloc[0])
    return full_df

if __name__ == '__main__':
    #split_original_tracking_data_into_manageable_pieces()
    #sldkfj
    if not (len(sys.argv) in [2, 3]):
        raise Exception("This function must be called with 1 or 2 args, where first arg is task to perform and second is chunk number")
    processing_step = sys.argv[1]
    assert processing_step in ['split_original_tracking_data_into_manageable_pieces', 'annotate_with_cycle_days', 'compute_user_features_and_individual_symptoms']

    if len(sys.argv) == 2: # if called with no arguments, process all chunks.
        if processing_step == 'split_original_tracking_data_into_manageable_pieces':
            split_original_tracking_data_into_manageable_pieces() # this is single-threaded. 
        else:
            for chunk_number in range(n_chunks):
                os.system('nohup python -u dataprocessor.py %s %i > %s/processing_%s_%i.out &' % (
                    processing_step,
                    chunk_number, 
                    base_processing_outfile_dir, 
                    processing_step,
                    chunk_number))
    elif len(sys.argv) == 3:
        chunk_number = int(sys.argv[2])
        if processing_step == 'annotate_with_cycle_days':
            make_big_annotated_symptom_d(chunk_number = chunk_number)
        elif processing_step == 'compute_user_features_and_individual_symptoms':
            compute_user_features_and_split_into_individual_symptoms(chunk_number)
        else:
            raise Exception("Invalid processing task")

  
