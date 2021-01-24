import os, datetime, platform

MAX_NUMPY_CORES = 1
print("Setting numpy cores to %i" % MAX_NUMPY_CORES)
os.environ["MKL_NUM_THREADS"] = str(MAX_NUMPY_CORES)  # this keeps numpy from using every available core. We have to do this BEFORE WE import numpy for the first time.
os.environ["NUMEXPR_NUM_THREADS"] = str(MAX_NUMPY_CORES)
os.environ["OMP_NUM_THREADS"] = str(MAX_NUMPY_CORES)
os.environ["NUMEXPR_MAX_THREADS"] = str(MAX_NUMPY_CORES)

from scipy.stats import scoreatpercentile
import numpy as np

print("Warning: basemap is not currently installed. If you want to make maps, you will have to figure out how to install it.")
COMPUTER_WE_ARE_USING = platform.node()



assert COMPUTER_WE_ARE_USING in ['rambo.stanford.edu', 'trinity.stanford.edu']

if COMPUTER_WE_ARE_USING == 'mbcomp1.stanford.edu':
    BASE_PATH_FOR_EVERYTHING = '/remote/emmap1/clue_data/november_2017_clue_data_update/'
else:
    BASE_PATH_FOR_EVERYTHING = '/dfs/scratch2/emmap1/clue_data/november_2017_clue_data_update/'

print 'Importing constants and util'
USE_SIMULATED_DATA = False
if USE_SIMULATED_DATA:
    base_data_dir = os.path.join(BASE_PATH_FOR_EVERYTHING, 'simulated_data/')
    chunked_tracking_dir = os.path.join(BASE_PATH_FOR_EVERYTHING, 'simulated_data', 'chunked_tracking_data/')
    processed_chunked_tracking_dir = os.path.join(BASE_PATH_FOR_EVERYTHING, 'simulated_data', 'chunked_processed_data/')
    base_results_dir = os.path.join(BASE_PATH_FOR_EVERYTHING, 'simulated_data', 'analysis_results/')
    base_processing_outfile_dir = os.path.join(BASE_PATH_FOR_EVERYTHING, 'simulated_data', 'processing_outfiles/')
    original_tracking_csv = os.path.join(base_data_dir, 'simulated_tracking.csv')
    n_chunks = 5
    n_chunks_to_use = n_chunks
    chunks_to_use = range(n_chunks_to_use)
else:
    base_data_dir = BASE_PATH_FOR_EVERYTHING
    chunked_tracking_dir = os.path.join(BASE_PATH_FOR_EVERYTHING, 'chunked_tracking_data')
    processed_chunked_tracking_dir = os.path.join(BASE_PATH_FOR_EVERYTHING, 'chunked_processed_data/')
    base_results_dir = os.path.join(BASE_PATH_FOR_EVERYTHING, 'analysis_results/')
    base_processing_outfile_dir = os.path.join(BASE_PATH_FOR_EVERYTHING, 'processing_outfiles/')
    original_tracking_csv = os.path.join(base_data_dir, 'tracking.csv')
    n_chunks = 64
    n_chunks_to_use = 64
    chunks_to_use = range(n_chunks_to_use)

N_BOOTSTRAP_ITERATES = 1000
COLS_TO_STRATIFY_BY = ['country',
                       'very_active_northern_hemisphere_loggers',
                       'very_active_southern_hemisphere_loggers',
                       'logged_hormonal_birth_control',
                       'logged_birth_control_pill',
                       'logged_iud', 
                       'start_year',
                       'total_symptoms_logged', 
                       'n_days_using_app', 
                       'n_symptom_categories_used', 
                       'categorical_age', 
                       'categorical_latitude', 
                       #'proportion_pain_symptoms',
                       #'proportion_mood_symptoms', 
                       'hemisphere'] 
                       #'logged_any_cigarettes', 
                       #'logged_any_alcohol', 
                       #'logged_any_exercise', 
                       #'in_united_states', 
                       #'largest_timezones']   

MIN_USERS_FOR_SUBGROUP = 100
MIN_OBS_FOR_SUBGROUP = 1000

MOOD_SYMPTOM_CATEGORIES = ['emotion*', 'energy*', 'mental*', 'motivation*', 'social*']
                       
FILTER_FOR_VERY_ACTIVE_LOGGERS_IN_ALL_ANALYSIS = True

CONTINUOUS_VARIABLE_FILENAMES = ['weight.csv.gzip', 'bbt.csv.gzip', 'fitbit_resting_heartrates.csv.gzip']

# mood symptoms
ALL_SYMPTOM_GROUPS = [line.strip().split(',') for line in open('opposite_emotional_features.csv')]

PERIOD_CYCLE_COLOR = '#d62728'
HOUR_CYCLE_COLOR = '#9467bd'
WEEKDAY_CYCLE_COLOR = '#2ca02c'
SEASONAL_CYCLE_COLOR = '#1f77b4'

PRETTY_CYCLE_NAMES = {'month':'Seasonal', 
                             'weekday':'Weekly', 
                              'local_hour':'Daily', 
                              'date_relative_to_period':'Menstrual', 
                              'week_of_year':'Seasonal'}

# add in behavioral symptoms
BBT_SUBSTRING = 'bbt'
HEART_SUBSTRING = 'heart_rate'
WEIGHT_SUBSTRING = 'weight'
ALL_SYMPTOM_GROUPS = [['continuous_features*%s' % WEIGHT_SUBSTRING], 
                      ['continuous_features*%s' % BBT_SUBSTRING], 
                     ['continuous_features*%s' % HEART_SUBSTRING],
                    ['sleep*slept_6_hours_or_more', 'sleep*slept_6_hours_or_less'], 
                    ['exercise*exercised', 'exercise*did_not_exercise'], 
                     ['sex*had_sex', 'sex*did_not_have_sex']] + ALL_SYMPTOM_GROUPS

CANONICAL_PRETTY_SYMPTOM_NAMES = {'sex*had_sex_versus_sex*did_not_have_sex':'had sex', 
'exercise*exercised_versus_exercise*did_not_exercise':'exercised', 
 'motivation*motivated_versus_motivation*unmotivated':'motivated\nversus\nunmotivated', 
 'energy*energized_versus_energy*exhausted':'energized\nversus\nexhausted', 
 'mental*focused_versus_mental*distracted':'focused\nversus\ndistracted',
 'emotion*happy_versus_emotion*sensitive_emotion':'happy\nversus\nsensitive', 
 'emotion*happy_versus_emotion*sad':'happy\nversus\nsad', 
 'continuous_features*weight_versus_continuous_features*null':'Weight\n(LB)', 
 'continuous_features*heart_rate_versus_continuous_features*null':'RHR\n(BPM)', 
 'motivation*productive_versus_motivation*unproductive':'productive\nversus\nunproductive', 
 'social*sociable_versus_social*withdrawn_social':'social\nversus\nwithdrawn', 
 'continuous_features*bbt_versus_continuous_features*null':'BBT\n(deg F)', 
 'social*supportive_social_versus_social*conflict_social':'supportive\nversus\nconflict', 
 'mental*calm_versus_mental*stressed':'calm\nversus\nstressed',
 'sleep*slept_6_hours_or_more_versus_sleep*slept_6_hours_or_less':'slept\n>6 hours'}

ORDERED_SYMPTOM_NAMES = ['emotion*happy_versus_emotion*sad',
 'emotion*happy_versus_emotion*sensitive_emotion',
 'energy*energized_versus_energy*exhausted',
 'mental*calm_versus_mental*stressed',
 'mental*focused_versus_mental*distracted',
 'motivation*motivated_versus_motivation*unmotivated',
 'motivation*productive_versus_motivation*unproductive',
 'social*sociable_versus_social*withdrawn_social',
 'social*supportive_social_versus_social*conflict_social', 
 'exercise*exercised_versus_exercise*did_not_exercise',
 'sex*had_sex_versus_sex*did_not_have_sex',
 'sleep*slept_6_hours_or_more_versus_sleep*slept_6_hours_or_less',
 'continuous_features*bbt_versus_continuous_features*null',
 'continuous_features*heart_rate_versus_continuous_features*null',
 'continuous_features*weight_versus_continuous_features*null']

SHORT_NAMES_TO_REGRESSION_COVS = {'country+age+behavior+app usage':'near_period*C(country)+near_period*C(categorical_age)+near_period*C(logged_any_alcohol)+near_period*C(logged_any_cigarettes)+near_period*C(logged_any_exercise)+near_period*C(logged_birth_control_pill)+near_period*C(logged_hormonal_birth_control)+near_period*C(logged_iud)+near_period*C(n_symptom_categories_used)+near_period*C(start_year)+near_period*C(total_symptoms_logged)', 
                       'country+age':'near_period*C(country)+near_period*C(categorical_age)', 
                       'country+age+behavior':'near_period*C(country)+near_period*C(categorical_age)+near_period*C(logged_any_alcohol)+near_period*C(logged_any_cigarettes)+near_period*C(logged_any_exercise)+near_period*C(logged_birth_control_pill)+near_period*C(logged_hormonal_birth_control)+near_period*C(logged_iud)', 
                       'age':'near_period*C(categorical_age)',
                       'country':'near_period*C(country)', 
                       'baseline effect':'near_period'}
def get_hemisphere(latitude):
    # Returns the hemisphere. If you're too close to the equator, returns None. Checked. 
    if latitude >= 30:
        return 'northern_above_30_lat'
    if latitude <= -30:
        return 'southern_below_30_lat'
    return None

def determine_mood_behavior_or_vital_sign(x):
    assert x in CANONICAL_PRETTY_SYMPTOM_NAMES.keys()
    if 'continuous' in x:
        return 'Vital sign'
    if ('sex' in x) or ('exercise' in x) or ('sleep' in x):
        return 'Behavior'
    else:
        return 'Mood' 


def date_near_period(x):
    """
    DEPRECATED. 
    """
    raise Exception("This method is deprecated and should not be used. See add_binary_annotations for preferred method.")

def get_season(month):
    # Checked. 
    if month in [6, 7, 8]:
        return 'summer'
    if month in [12, 1, 2]:
        return 'winter'
    return None

def in_middle_of_night(local_hour):
    # Somewhat arbitrarily, midnight to 6 AM local time (ie, local_hour >= 0, local_hour <= 5). 
    # This is a reasonable binarization of the period where mood dips.
    # Checked. 
    if (local_hour >= 0) and (local_hour <= 5):
        return True
    return False

def big_annotated_symptom_filename(chunk_number):
    # Gets the path to a given chunk. Checked. 
    return os.path.join(processed_chunked_tracking_dir, 'big_annotated_symptoms_%i.csv' % chunk_number)

def bootstrap_CI(bootstrap_iterates, original_estimate, verbose=False):
    """
    Use 2.5 and 97.5 percentiles of bootstrap_iterates. Previously we were using using 1.96 * std - confirmed that using either quantiles or 1.96 * std yield essentially identical estimates.
    """
    assert original_estimate < max(bootstrap_iterates)
    assert original_estimate > min(bootstrap_iterates)

    bootstrap_mean = np.mean(bootstrap_iterates)
    bootstrap_std = np.std(bootstrap_iterates, ddof=1)
    b = bootstrap_mean - original_estimate

    range_using_quantiles = scoreatpercentile(bootstrap_iterates, 97.5) - scoreatpercentile(bootstrap_iterates, 2.5)
    assert len(bootstrap_iterates) == N_BOOTSTRAP_ITERATES
    bootstrap_CI = {'lower_95_CI':scoreatpercentile(bootstrap_iterates, 2.5),#original_estimate - bootstrap_std * 1.96, 
                    'upper_95_CI':scoreatpercentile(bootstrap_iterates, 97.5)}#}original_estimate + bootstrap_std * 1.96}


    if verbose:
        print('Relative difference between mean of bootstrapped iterates and original estimate %2.6f: %2.1f%% (absolute difference: %2.6f); 1.96*std in bootstrapped estimates is %2.6f; corresponding estimate based on quantiles is %2.6f; 95%% CI based on std is %2.6f - %2.6f' % (
          original_estimate, 
          100*b / original_estimate, 
          b, 
          1.96*bootstrap_std,
          range_using_quantiles / 2., 
          bootstrap_CI['lower_95_CI'], 
          bootstrap_CI['upper_95_CI']))
    return bootstrap_CI
