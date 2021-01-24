from constants_and_util import *
import pandas as pd
from traceback import print_exc
import random
import numpy as np
from scipy.signal import argrelextrema
import statsmodels.api as sm
import warnings
import statsmodels.formula.api as smf
from copy import deepcopy
import json
from IPython import embed
from collections import Counter
from scipy.stats import pearsonr, linregress
import time
import string
import cPickle
import math
import dataprocessor
from scipy.special import expit
from multiprocessing import Process, Manager
from copy import deepcopy
import datetime
import matplotlib.pyplot as plt
from scipy.stats import scoreatpercentile
import sys

#from mpl_toolkits.basemap import Basemap
import matplotlib as mpl
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from matplotlib.patches import PathPatch

import matplotlib.gridspec as gridspec
import gc

def load_all_results(continuous_only=False):
    """
    Loads in a dictionary of results. 
    (results for each mood pair are saved in their own file; this combines them). 
    checked. 
    """
    print("Loading in analysis results from %s" % base_results_dir)
    results = {}
    for filename in os.listdir(base_results_dir):
        if continuous_only and 'continuous' not in filename:
            continue
        if FILTER_FOR_VERY_ACTIVE_LOGGERS_IN_ALL_ANALYSIS and 'VERY_ACTIVE_LOGGERS_' not in filename:
            continue

        if ('versus' in filename) and (('n_chunks_to_use_%i' % n_chunks_to_use) in filename):
            t0 = time.time()
            print 'Adding %s to results' % filename
            d = cPickle.load(open(os.path.join(base_results_dir, filename), 'rb'))
            assert len(d.keys()) == 1
            results.update(d)
            print("Time to load %s: %2.3f seconds" % (filename, time.time() - t0))
    for i in range(len(results)):
        print '%i. %s' % (i + 1, results.keys()[i])
    return results

def get_amplitude_standard_error_from_regression_data(regression_data, cycle):
    """
    compute the standard error in the cycle amplitude given a linear regression of coefficients
    by resampling the coefficients, taking coef covariances into account. 
    """
    raise Exception("Not using at present because it seems potentially sketchy, using bootstraps. If you use this, need to check it.")
    regression_data = deepcopy(regression_data)
    
    # computing amplitude the old way. Verified this yields the same results if covariance = 0.
    simple_mean_format = convert_regression_format_to_simple_mean_format(regression_data, 
                                                                         'linear_regression')
    amplitude = get_cycle_amplitude(simple_mean_format, 
                                    cycle, 
                                    metric_to_use='max_minus_min', 
                                    hourly_period_to_exclude=None)
    
    assert cycle in ['date_relative_to_period', 'month', 'weekday', 'local_hour']
    param_names = regression_data['params'].index
    if cycle == 'date_relative_to_period':
        cycle_idxs = param_names.map(lambda x:('date_relative_to_period' in x) and
                                      (np.abs(float(x.split('[T.')[1].replace(']', ''))) <= 14))
        assert cycle_idxs.sum() == 28                       
    else:
        cycle_idxs = param_names.map(lambda x:cycle in x)
        if cycle == 'local_hour':
            assert cycle_idxs.sum() == 23
        elif cycle == 'month':
            assert cycle_idxs.sum() == 11
        elif cycle == 'weekday':
            assert cycle_idxs.sum() == 6
    cov_matrix = deepcopy(regression_data['covariance_matrix'])
    beta_hat = deepcopy(regression_data['params'])

    assert (cov_matrix.index == beta_hat.index).all()
    
    cov_matrix = cov_matrix.values
    beta_hat = beta_hat.values
    n_samples = 10000
    sample = np.random.multivariate_normal(beta_hat, cov_matrix, size=[n_samples,])
    assert sample.shape[1] == len(cycle_idxs)
    sample_for_cycle = sample[:, cycle_idxs]
    sample_for_cycle = np.hstack([sample_for_cycle, np.zeros([n_samples,1])])
    max_vals = sample_for_cycle.max(axis=1)
    min_vals = sample_for_cycle.min(axis=1)
    sampled_amplitudes = max_vals - min_vals
    assert (sampled_amplitudes > 0).all()
    assert min(sampled_amplitudes) < amplitude
    assert max(sampled_amplitudes) > amplitude

    err_95 = 1.96 * np.std(sampled_amplitudes)
    #lower_CI = scoreatpercentile(sampled_amplitudes, 2.5)
    #upper_CI = scoreatpercentile(sampled_amplitudes, 97.5)
    #print 'Lower CI: %2.3f; upper CI %2.3f' % (lower_CI, upper_CI)
    #print amplitude, sampled_amplitudes.mean()
    return err_95, err_95 # lower and upper errorbars. 


def get_cycle_amplitude(data, cycle, metric_to_use, hourly_period_to_exclude):
    """
    given data (eg results[opposite_pair]
    [substratification][
    substratification_level]
    ['take_simple_means_by_group_no_individual_mean'])
    and a cycle and a metric to use (max_minus_min or average_absolute_difference_from_mean)
    computes the cycle amplitude. 
    """
    data = deepcopy(data)
    assert metric_to_use in ['max_minus_min' ,'average_absolute_difference_from_mean']
    assert cycle in ['date_relative_to_period', 'local_hour', 'weekday', 'month', 'week_of_year']
    if cycle == 'date_relative_to_period':
        data[cycle] = data[cycle].loc[data[cycle].index.map(lambda x:np.abs(x) <= 14)]
        assert list(data[cycle].index) == list(range(-14, 15))
    if cycle == 'local_hour':
        if hourly_period_to_exclude is None:
            assert list(data[cycle].index) == list(range(24))
        else:
            assert len(hourly_period_to_exclude) == 2
            assert hourly_period_to_exclude[0] < hourly_period_to_exclude[1]
            data[cycle] = data[cycle].loc[data[cycle].index.map(lambda x:(x < hourly_period_to_exclude[0]) or (x > hourly_period_to_exclude[1]))]
            assert list(data[cycle].index) == [a for a in list(range(24)) if a < hourly_period_to_exclude[0] or a > hourly_period_to_exclude[1]]
    if cycle == 'weekday':
        assert list(data[cycle].index) == list(['Friday', 'Monday', 'Saturday', 'Sunday', 'Thursday', 'Tuesday',
'Wednesday'])
    if cycle == 'month':
        assert list(data[cycle].index) == list(range(1, 13))
    if cycle == 'week_of_year':
        assert list(data[cycle].index) == list(range(52))
        
    y = np.array(data[cycle]['mean']) 
    
    y_mu = y.mean()
    average_absolute_difference_from_mean = np.mean(np.abs(y - y_mu))
    largest_difference = y.max() - y.min()
    if metric_to_use == 'max_minus_min':
        metric_val = largest_difference
    else:
        metric_val = average_absolute_difference_from_mean
    return metric_val

def make_basemap(country_vals, bin_edges, bin_edge_labels, title_string, filename=None, plot_colorbar=True):
    """
    Given a dictionary which maps country names to values
    a list of bin edges (numbers) (which must span the min and max values)
    a list of bin edge labels (strings)
    and a title, 
    makes a map. 
    Checked. 
    """

    fig     = plt.figure(figsize=[7, 5])
    ax      = fig.add_subplot(111)

    from matplotlib.colors import LinearSegmentedColormap
    colors = ['#023eff', '#ffffff', '#e8000b']#[(1, 0, 0), (1, 1, 1), (0, 0, 1)]
    cm_fxn = LinearSegmentedColormap.from_list(
        'mycmap', colors, N=256)

    #cm_fxn = plt.get_cmap(cmap_coloring)
    num_bins = len(bin_edges) - 1
    assert len(bin_edges) == len(bin_edge_labels)
    color_scheme = [cm_fxn((i + 1) / (1.*num_bins + 1)) for i in range(num_bins)] # normalization is + 1 so that the middle bin is .5. 
    cmap_for_legend = mpl.colors.ListedColormap(color_scheme) # we use this to make the color scheme in the legend. 

    m = Basemap(llcrnrlat = -60, urcrnrlat = 85, llcrnrlon = -180, urcrnrlon=180, resolution='l', fix_aspect = True)
    m.drawmapboundary(fill_color='white')
    m.fillcontinents(color='#dddddd',lake_color='white')
    m.readshapefile('country_shapefile/UIA_World_Countries_Boundaries', 
                      'UIA_World_Countries_Boundaries', drawbounds = False)

    patches   = []
    facecolors = []
    countries_not_found = list(set(country_vals.keys()))
    for info, shape in zip(m.UIA_World_Countries_Boundaries_info, m.UIA_World_Countries_Boundaries):
        # loop over the patches which make up countries. 
        # For each patch, if it's in the list of countries we passed in, color it appropriately. 
        def map_to_my_country_names(x):
            # small helper method to render names consistent. 
            if x == 'United Kingdom':
                return 'Britain (UK)'
            elif x == 'South Korea':
                return 'Korea (South)'
            elif x == 'Russian Federation':
                return 'Russia'
            elif x == 'Trinidad and Tobago':
                return 'Trinidad & Tobago'
            elif x == "C\xc3\xb4te d'Ivoire":
                return "Cte d'Ivoire"
            elif x == 'Bosnia and Herzegovina':
                return 'Bosnia & Herzegovina'
            elif x == 'Brunei Darussalam':
                return 'Brunei'
            elif x == 'The Former Yugoslav Republic of Macedonia':
                return 'Macedonia'
            return x
        country_name = map_to_my_country_names(info['Country'])
        if country_name in country_vals:
            country_val = country_vals[country_name]
            if not ((country_val < np.max(bin_edges)) and (country_val > np.min(bin_edges))):
                raise Exception("%s, value %2.3f is out of range of bins" % (country_name, country_val))
            bin_idx = int(np.digitize(country_val, bin_edges)) 
            # digitize documentation: Each index i returned is such that bins[i-1] <= x < bins[i] if bins is monotonically increasing
            bin_idx = bin_idx - 1 # smallest value of bin_idx will be 0; largest will be num_bins - 1. 
            assert bin_idx >= 0
            # So bin_idx = 0 means you're in color_scheme[0].  
            # This is sort of hard to check by hand so I just made maps with fake data to check it. 
            facecolors.append(color_scheme[bin_idx])
            patches.append(Polygon(np.array(shape), True) )
            #facecolors.append('red')
            if country_name in countries_not_found:
                countries_not_found.remove(country_name)
        #else:
        #    facecolors.append('lightgrey')
        #    patches.append(Polygon(np.array(shape), True) )
    print("Warning: the following countries could not be mapped because they were not found in mapping data")
    print countries_not_found
    ax.add_collection(PatchCollection(patches, facecolor=facecolors, edgecolor='k', linewidths=1., zorder=2))
    plt.title(title_string, fontsize=16, fontweight='bold')
    
    

    if plot_colorbar:
        # Add legend
        ax_legend = fig.add_axes([0.2, 0.15, 0.6, 0.03], zorder=3)
        
        cb = mpl.colorbar.ColorbarBase(ax_legend, cmap=cmap_for_legend, ticks=range(num_bins + 1), boundaries=range(num_bins + 1), orientation='horizontal')
        cb.ax.set_xticklabels(bin_edge_labels, fontsize=14, fontweight='bold')
    fig.subplots_adjust(top=.95, left=.02, right=.98, bottom=.05)
    
    if filename is not None:
        fig.savefig(filename, dpi=300)
    plt.show()

def analyze_individual_specific_interactions_between_cycles(results):
    """
    Checked. Analyzes whether individuals who have larger effects in one cycle have larger effects in another cycle. 
    mu here is when the cycle indicator is TRUE - when the cycle indicator is FALSE. 
    """
    results = deepcopy(results)
    all_pvals = []
    for emotion_pair in results:
        person_specific_interactions = results[emotion_pair]['no_substratification']['person_specific_interaction_between_cycles']
        for cycle_pair in person_specific_interactions.keys():
            all_pvals.append({'emotion':emotion_pair, 
                              'cycle_pair':cycle_pair, 
                              'p':person_specific_interactions[cycle_pair]['p'], 
                              'r':person_specific_interactions[cycle_pair]['r'],
                              'abs_r':np.abs(person_specific_interactions[cycle_pair]['r']), 
                              'n':person_specific_interactions[cycle_pair]['n'], 
                              'mu_1':person_specific_interactions[cycle_pair]['mu_1'], 
                              'mu_2':person_specific_interactions[cycle_pair]['mu_2']})

            if person_specific_interactions[cycle_pair]['p'] < 1e-5:
                plt.figure()
                plt.title(cycle_pair + '\n' + emotion_pair + '\nr = %2.3f, p = %2.3e' % (person_specific_interactions[cycle_pair]['r'], 
                                                                                       person_specific_interactions[cycle_pair]['p']))
                plt.scatter(person_specific_interactions[cycle_pair]['full_effects_1'], 
                            person_specific_interactions[cycle_pair]['full_effects_2'])
                plt.xlim(-1, 1)
                plt.ylim(-1, 1)
                plt.show()
    all_pvals = pd.DataFrame(all_pvals).sort_values(by = 'p')
    return all_pvals
    
def analyze_non_individual_specific_interactions_between_cycles(results):
    """
    Checked. Analyzes non individual-specific interactions between cycles.  
    """
    results = deepcopy(results)
    all_pvals = []
    pd.set_option('display.width', 500)

    for emotion_pair in results:
        regression = results[emotion_pair]['no_substratification']['interaction_between_cycles']['linear_regression']
        raw_means = results[emotion_pair]['no_substratification']['interaction_between_cycles']['raw_means']

        for pair_of_cycles in regression:
            # loop over pairs of cycles
            # get the interaction coefficient for each one
            interaction_coef = [a for a in regression[pair_of_cycles]['pvalues'].index if ':' in a][0]
            pval = float(regression[pair_of_cycles]['pvalues'].loc[interaction_coef])
            beta = float(regression[pair_of_cycles]['betas'].loc[interaction_coef])
            raw_mean_vals = raw_means[pair_of_cycles].reset_index()
            assert np.all(raw_mean_vals[raw_mean_vals.columns[0]] == [False, False, True, True])
            assert np.all(raw_mean_vals[raw_mean_vals.columns[1]] == [False, True, False, True])
            if pval < 1e-5:
                # make a plot for all statistically significant interactions
                plt.figure(figsize = [10, 5])
                xticks = []
                vals = list(raw_mean_vals['good_mood'].values)
                assert len(vals) == 4
                col0 = raw_mean_vals.columns[0]
                col1 = raw_mean_vals.columns[1]
                assert col0 in ['summer', 'winter', 'weekend', 'middle_of_night', 'near_period']
                assert col1 in ['summer', 'winter', 'weekend', 'middle_of_night', 'near_period']
                for i in range(4):
                    xticks.append(str(col0) + ':' + str(raw_mean_vals[col0].iloc[i]) + '\n' +
                                  str(col1) + ':' + str(raw_mean_vals[col1].iloc[i]))
                x_positions = [0, 1, 3, 4]
                plt.bar(x_positions, vals, width = 1) # space out bars. 
                plt.title('%s\n%s\n%2.3e' % (emotion_pair, pair_of_cycles, pval))
                plt.xticks(x_positions, xticks)
                plt.ylim([-.05, .05])
                plt.show()

            all_pvals.append({'emotion':emotion_pair, 
                              'coef':interaction_coef, 
                              'pval':pval, 
                              'beta':beta})
  
    all_pvals = pd.DataFrame(all_pvals)
    all_pvals = all_pvals.loc[all_pvals['pval'] < 1e-5]
    pd.set_option('precision', 2)
    all_pvals['coef'] = all_pvals['coef'].map(lambda x:x.replace('[T.True]', ''))
    all_pvals.sort_values(by = 'pval')[['emotion', 'coef', 'beta', 'pval']]
    return all_pvals

def extract_most_dramatic_period_bin(data, bin_size):
    """
    data should be, eg, results[opposite_pair]['no_substratification']['take_simple_means_by_group_no_individual_mean']
    Looks for the bin between -7 days before period and start of period which has the most dramatic discrepancy. 
    """
    data = deepcopy(data)
    assert sorted(data.columns) == ['err', 'mean', 'size']

    data = data.loc[data.index.map(lambda x:np.abs(x) <= 14)]
    assert list(data.index) == range(-14, 15)
    # allow the bin to start up to two weeks before the period to the start of the period. 
    min_day = -14
    max_day = 0
    max_effect = 0
    for start_day in range(min_day, max_day + 1):
        idxs = data.index.map(lambda day:day >= start_day and day < start_day + bin_size)
        idxs = np.array(idxs)
        assert np.sum(idxs) == bin_size
        effect = data.loc[idxs, 'mean'].mean() - data.loc[~idxs, 'mean'].mean()
        if np.abs(effect) > np.abs(max_effect):
            max_effect = effect
            best_start_day = start_day
    print 'maximum effect between days %i and %i (inclusive) is %2.3f, %i<=day<%i' % (min_day, 
                                                  max_day,
                                                  max_effect, 
                                                   best_start_day, 
                                                   best_start_day + bin_size)
    return best_start_day

def convert_regression_format_to_simple_mean_format(regression_data, type_of_regression, seasonal_variable='month'):
    """
    Given data in the regression format -- ie, a dictionary of params and p-values -- 
    converts to the data format make_four_cycle_plots requires. 
    This is convenient because otherwise we have to have make_four_cycle_plots and make_four_cycle_regression_plots
    The size column is not meaningful but is included for compatibility with the other format. 
    Checked. 
    """

    assert seasonal_variable in ['month', 'week_of_year']
    assert type_of_regression in ['linear_regression', 'mixed_model_regression']
    if type_of_regression == 'linear_regression':
        assert sorted(regression_data.keys()) == sorted(['95_CI', 'covariance_matrix', 'params', 'pvalues'])
        params = regression_data['params']
        lower_CI = regression_data['95_CI'][0]
        upper_CI = regression_data['95_CI'][1]
        assert np.allclose(upper_CI.values - params.values, params.values - lower_CI.values)
        assert list(params.index) == list(upper_CI.index)
        err = upper_CI - params
    else:
        print sorted(regression_data.keys())
        assert sorted(regression_data.keys()) == ['fixed_effects_coefficients', 
        'fixed_effects_standard_errors', 'random_effects_covariance', 'random_effects_covariance_errors', 'ranef']
        params = regression_data['fixed_effects_coefficients']
        err = regression_data['fixed_effects_standard_errors'] * 1.96

    cycles_to_loop_over = ['weekday', seasonal_variable, 'local_hour', 'date_relative_to_period']
    no_hourly_data = np.sum(params.index.map(lambda x:'local_hour' in x)) == 0
    if no_hourly_data:
        cycles_to_loop_over.remove('local_hour')
    new_data = {}
    for cycle in cycles_to_loop_over:
        beta = deepcopy(params.loc[params.index.map(lambda x:cycle in x)])
        beta_err = deepcopy(err.loc[err.index.map(lambda x:cycle in x)])

        if cycle == 'weekday':
            missing_val = 'Friday'
            weekday_processing_fxn = lambda x:x.replace('C(weekday)[T.', '').replace(']', '')
            beta.index = beta.index.map(weekday_processing_fxn)
            beta_err.index = beta_err.index.map(weekday_processing_fxn)
        elif cycle == 'date_relative_to_period':
            # this has to deal with both the mixed model and OLS formulations
            # which have slightly different coefficient formats. 
            # the missing value is 0. 
            processing_fxn = lambda x:x.replace('C(date_relative_to_period, Treatment(reference=0))[T.', '').replace('C(date_relative_to_period)[T.', '').replace(']', '')
            beta.index = beta.index.map(processing_fxn).astype('float')
            beta_err.index = beta_err.index.map(processing_fxn).astype('float')
            missing_val = 0
        else:
            non_weekday_processing_fxn = lambda x:x.replace('C(%s)[T.' % cycle, '').replace(']', '')
            beta.index = beta.index.map(non_weekday_processing_fxn).astype('float')
            beta_err.index = beta_err.index.map(non_weekday_processing_fxn).astype('float')
            missing_val = min(beta.index) - 1
        # insert coefficient for the missing value, which is 0. 
        beta = beta.append(pd.Series([0], index=[missing_val]))
        assert len(set(beta.index)) == len(beta)
        beta = beta.sort_index()

        beta_err = beta_err.append(pd.Series([0], index=[missing_val]))
        assert len(set(beta_err.index)) == len(beta_err)
        beta_err = beta_err.sort_index()
        assert list(beta_err.index) == list(beta.index)

        new_df = pd.DataFrame({'mean':beta.values, 
                              'size':0.0,
                              'err':beta_err.values})
        

        new_df.index = beta.index
        new_df = new_df[['mean', 'size', 'err']]
        new_df.index.name = cycle
        if cycle == 'date_relative_to_period':
            new_df = new_df.loc[new_df.index.map(lambda x:np.abs(x) <= 14)]
        new_data[cycle] = new_df  

    # if there's no hourly data in the regression (eg, because we have no hourly info) add in zero values so nothing crashes. 
    if no_hourly_data:
        new_data['local_hour'] = pd.DataFrame({'mean':0.0, 
                              'size':0.0,
                              'err':0.0}, 
                              index=range(24))
        new_data['local_hour'].index.name = 'local_hour'

    for cycle in new_data:
        for k in new_data[cycle].columns:
            assert np.isnan(new_data[cycle][k].values).sum() == 0
        index_labels = list(new_data[cycle][k].index)
        if cycle == 'date_relative_to_period':
            assert index_labels == range(-14, 15)
        elif cycle == 'local_hour':
            assert index_labels == range(24)
        elif cycle == 'weekday':
            # sorted weekday names. 
            assert index_labels == ['Friday', 'Monday', 'Saturday', 'Sunday', 'Thursday', 'Tuesday', 'Wednesday']
        elif cycle == 'month':
            assert index_labels == range(1, 13) or index_labels == range(3, 13) # heart rate only has partial data. 
        elif cycle == 'week_of_year':
            assert index_labels == range(52) or index_labels == range(12, 52) #  heart rate only has partial data. 
    return new_data

def make_four_cycle_plots(results, 
                          substratifications, 
                          emotion_pairs, 
                          use_20_day_menstrual_cycle=False,
                         data_to_use='linear_regression', 
                         ylimits_by_pair=None, 
                         figname=None, 
                         show_errorbars=True, 
                         substratification_levels_to_skip=None, 
                         hourly_period_to_exclude=None, 
                         colors_for_lines=None, 
                         suptitle=True, 
                         include_amplitudes_in_title=True, 
                         different_colors_for_each_cycle=False, 
                         use_expanded_seasonal_x_axis=False):
    """
    loop over all substratifications (eg, [by_categorical_latitude]) and all emotion pairs.
    plot all four cycles for each (just raw means, not regression adjusted estimates). 
    If use_20_day_menstrual_cycle = True, extends range slightly for menstrual cycle plot to make sure there
    are no super-weird tail effects. 
    Checked. 
    """
    assert data_to_use in ['take_simple_means_by_group_no_individual_mean', 
                           'take_simple_means_by_group', 
                           'derivative_estimates_with_no_looparound', 
                           'derivative_estimates_with_looparound', 
                           'linear_regression']

    # an annoying warning about future deprecation. 
    warnings.filterwarnings("ignore", message="Adding an axes using the same arguments as a previous axes currently reuses the earlier instance.  In a future version, a new instance will always be created and returned.  Meanwhile, this warning can be suppressed, and the future behavior ensured, by passing a unique label to each axes instance.")

    if ylimits_by_pair is not None:
        assert sorted(ylimits_by_pair.keys()) == sorted(emotion_pairs)
    else:
        ylimits_by_pair = {}
        for pair in emotion_pairs:
            if BBT_SUBSTRING in pair:
                ylimits_by_pair[pair] = .5
            elif HEART_SUBSTRING in pair:
                ylimits_by_pair[pair] = 2
            elif WEIGHT_SUBSTRING in pair:
                ylimits_by_pair[pair] = .4
            elif 'sex' in pair or 'exercise' in pair or 'sleep' in pair:
                ylimits_by_pair[pair] = 15
            else:
                ylimits_by_pair[pair] = 10

    for substratification in substratifications:
        for emotion_pair in emotion_pairs:
            plt.figure(figsize = [20, 6])
            lines_plotted = 0
            n_lines_to_plot = len(results[emotion_pair][substratification].keys())
            
            # extract the levels we want to loop over for the substratification. 
            if substratification == 'no_substratification':
                sorted_levels = [None]
            elif all([a in [False, True] for a in results[emotion_pair][substratification].keys()]):
                sorted_levels = sorted(results[emotion_pair][substratification].keys())
            elif substratification in ['normalization_procedure', 'by_hemisphere', 'by_largest_timezones', 'by_country', 'all_regression_specifications', 'by_any_filtering']:
                sorted_levels = sorted(results[emotion_pair][substratification].keys())
            else:
                # if not boolean substratifications or hemisphere, levels are are numeric; sort them in ascending order. 
                sorted_levels = sorted(results[emotion_pair][substratification].keys(), 
                                       key = lambda x:float(x.split()[0].replace('[', '').replace(',', '')))
            if substratification_levels_to_skip is not None:
                for level in substratification_levels_to_skip:
                    assert level in sorted_levels
                    n_lines_to_plot = n_lines_to_plot - 1
                    sorted_levels.remove(level)
            for substratification_level in sorted_levels:
                # first extract the data for this substratification and level
                if substratification == 'no_substratification':
                    mean_data = deepcopy(results[emotion_pair][substratification][data_to_use])
                else:
                    mean_data = deepcopy(results[emotion_pair][substratification][substratification_level][data_to_use])

                if data_to_use == 'linear_regression':
                    mean_data = convert_regression_format_to_simple_mean_format(mean_data, 'linear_regression')
                # now loop over all four cycles  

                if 'month' in mean_data:
                    seasonal_variable = 'month'
                elif 'week_of_year' in mean_data:
                    seasonal_variable = 'week_of_year'
                else:
                    raise Exception("Either month or week of year needs to be in data")
                cycles_to_plot = ['date_relative_to_period', 'local_hour', 'weekday', seasonal_variable]
                for cycle in cycles_to_plot:
                    if (cycle == 'local_hour' and 
                            ('sex*' in emotion_pair or 
                             'sleep*' in emotion_pair or 
                             'exercise*' in emotion_pair or 
                             'continuous' in emotion_pair)):
                            # no reliable hourly information for these features. 
                            cycles_to_plot.remove(cycle)
                    if cycle in ['month', 'week_of_year'] and HEART_SUBSTRING in emotion_pair:
                            # don't have a full year of data for this. 
                            cycles_to_plot.remove(cycle)

                for cycle_idx, cycle in enumerate(cycles_to_plot):
                    
                    plt.subplot(1, 4, cycle_idx + 1)
                    
                    
                    if (cycle == 'date_relative_to_period') and \
                    ('derivative' in data_to_use) and \
                    use_20_day_menstrual_cycle:
                        # rare and annoying special case: if we're using the derivative estimates, 
                        # we save the 20-day period cycle in a separate df from the 14-day period one, 
                        # since the estimates are not identical even for overlapping time periods. 
                        means_by_cycle_point = deepcopy(mean_data['period_20'])
                    else:                                             
                        means_by_cycle_point = deepcopy(mean_data[cycle])
                    if data_to_use == 'take_simple_means_by_group_no_individual_mean':
                        # if data is supposed to already be zero-meaned, assert that it is. 
                        assert(np.abs(np.sum(means_by_cycle_point['mean'] * means_by_cycle_point['size'])) < 1e-5)
                     
                    xtick_kwargs = {'fontsize':20, 'rotation':90} # 'fontweight':'bold'
                    plt.tick_params(axis='x', length=15)
                    if cycle == 'weekday': 
                        # if weekday cycle, sort and truncate names. 
                        means_by_cycle_point = means_by_cycle_point.loc[['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']]
                        xs = range(7)
                        short_weekday_names = list(means_by_cycle_point.index.map(lambda x:x[:3]))
                        plt.xticks(range(7), short_weekday_names, **xtick_kwargs)
                    elif cycle == 'date_relative_to_period': 
                        # plot two weeks before and after, so filter data down slightly
                        if use_20_day_menstrual_cycle:
                            xs = list(means_by_cycle_point.index)
                            assert xs == range(-20, 21)
                            plt.xticks([-20, -15, -10, -5, 0, 5, 10, 15, 20],  **xtick_kwargs)
                        else:
                            means_by_cycle_point = means_by_cycle_point.loc[means_by_cycle_point.index.map(lambda x:np.abs(x) <= 14)]
                            xs = list(means_by_cycle_point.index)
                            assert xs == range(-14, 15)
                            plt.xticks([-14, -7, 0, 7, 14],  ['Day -14', 'Day -7', 'Day 0:\nperiod start', 'Day 7', 'Day 14'],
                                **xtick_kwargs)
                    elif cycle == 'local_hour':
                        if hourly_period_to_exclude is not None:
                            assert hourly_period_to_exclude[0] == 0
                            assert hourly_period_to_exclude[1] > hourly_period_to_exclude[0]
                            means_by_cycle_point = means_by_cycle_point.loc[means_by_cycle_point.index.map(lambda x:x > hourly_period_to_exclude[1])]
                            xs = list(means_by_cycle_point.index)
                            assert xs == range(hourly_period_to_exclude[1] + 1, 24)
                            plt.xticks(range(hourly_period_to_exclude[1] + 1, 24, 3),  **xtick_kwargs)
                        else:   
                            xs = list(means_by_cycle_point.index)
                            assert xs == range(24)
                            plt.xticks([0, 6, 12, 18], ['12AM', '6AM', '12PM', '6PM'], **xtick_kwargs)
                    elif cycle in ['month', 'week_of_year']:
                        # this is sort of gross. We parameterize seasonal cycles in two different ways
                        # and sometimes put them on the same plot.
                        # consequently, we can't just use the index as the xticks. 
                        # Rather, we convert to day of year, so it's consistent. 
                        if cycle == 'month':
                            xs = [int(datetime.datetime(2016, i, 15).strftime('%j')) for i in range(1, 13)]
                            # little bit of missing month data for 2017, since our data ends before year end. 
                            if (substratification == 'by_start_year') and (substratification_level == '2017'):
                                assert list(means_by_cycle_point.index) == range(1, 12)
                            else:
                                assert list(means_by_cycle_point.index) == range(1, 13)
                        else:
                            xs = list(means_by_cycle_point.index * 7.)
                            assert xs == range(0, 52 * 7, 7)
                        plt.xticks([int(datetime.datetime(2016, a, 15).strftime('%j')) for a in [1, 4, 7, 10]], 
                                ['Jan', 'Apr', 'Jul', 'Oct'],  **xtick_kwargs)

                    else:
                        raise Exception("Not a valid cycle!!")
                                                
                    ys = means_by_cycle_point['mean'].values
                    errs = means_by_cycle_point['err'].values 

                    if np.abs(np.mean(ys)) > 1e-8:
                        # if data is not already zero-meaned, zero-mean it. 
                        ys = ys - ys.mean()
                    
                    if 'continuous' not in emotion_pair:
                        ys = ys * 100
                        errs = errs * 100
                    if different_colors_for_each_cycle:
                        if cycle == 'date_relative_to_period':
                            color_for_line = PERIOD_CYCLE_COLOR
                        elif cycle == 'local_hour':
                            color_for_line = HOUR_CYCLE_COLOR
                        elif cycle == 'weekday':
                            color_for_line = WEEKDAY_CYCLE_COLOR
                        else:
                            color_for_line = SEASONAL_CYCLE_COLOR
                    else:
                        if colors_for_lines is not None:
                            color_for_line = colors_for_lines[substratification_level]
                        else:
                            color_for_line = [(lines_plotted + 1.0) / n_lines_to_plot, 0, 0]
                    plt.errorbar(xs, 
                             ys, 
                             yerr = errs if show_errorbars else None,
                             label = substratification_level, 
                             color = color_for_line)
                    if (use_expanded_seasonal_x_axis) and cycle in ['month', 'week_of_year']:
                        xlims = [0, 51*7]
                    else:
                        xlims = [min(xs), max(xs)]
                    plt.plot(xlims, [0, 0], color = 'black')
                    
                    if len(sorted_levels) == 1:
                        maximum_delta = ys.max() - ys.min()
                        # if there is only one line being plotted, print out the maximum change.
                        if include_amplitudes_in_title:
                            if 'continuous' in emotion_pair:
                                plt.title(PRETTY_CYCLE_NAMES[cycle] + ' $\Delta$: %2.2f' % maximum_delta, fontsize = 20) # , fontweight='bold'
                            else:
                                plt.title(PRETTY_CYCLE_NAMES[cycle] + ' $\Delta$: %2.1f%%' % maximum_delta, fontsize = 20) # , fontweight='bold'
                        else:
                            plt.title(PRETTY_CYCLE_NAMES[cycle], fontsize = 20)
                    else:
                        plt.title(PRETTY_CYCLE_NAMES[cycle], fontsize=20)
                    plt.xlim(xlims)
                    ylimit = ylimits_by_pair[emotion_pair]                                                
                    plt.ylim([-ylimit, ylimit])
                    
                    if cycle_idx == 0:
                        good_symptom, bad_symptom = emotion_pair.split('_versus_')
                        good_symptom = good_symptom.split('*')[1].replace('_', ' ').replace('didnt', "didn't")
                        bad_symptom = bad_symptom.split('*')[1].replace('_', ' ').replace('didnt', "didn't")
                        #plt.ylabel("Change relative to baseline", fontsize = 16)
                        if BBT_SUBSTRING in emotion_pair:
                            plt.yticks([-ylimit, 0, ylimit], fontsize=20) # , fontweight='bold'
                            plt.ylabel("Change in BBT (deg F)", fontsize=20)
                        elif HEART_SUBSTRING in emotion_pair:
                            plt.yticks([-ylimit, 0, ylimit], fontsize=20)
                            plt.ylabel("Change in RHR (BPM)", fontsize=20)
                        elif WEIGHT_SUBSTRING in emotion_pair:
                            plt.yticks([-ylimit, 0, ylimit], fontsize=20)
                            plt.ylabel("Change in weight (LB)", fontsize=20)

                        else:
                            plt.yticks([-ylimit, 0, ylimit], 
                                       ['%i%% more %s\n' % (ylimit, bad_symptom), 
                                        'Baseline', 
                                        '%i%% more %s' % (ylimit, good_symptom)], 
                                       fontsize=20)
                    else:
                        plt.yticks([])
                    if cycle_idx == len(cycles_to_plot) - 1:
                        if substratification != 'no_substratification' and (len(sorted_levels) > 1):
                            plt.legend(prop={'size':16}, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
                lines_plotted += 1
            if substratification != 'no_substratification' and (len(sorted_levels) > 1):
                plt.subplots_adjust(top=.92, bottom=.35, left=.28, right=.8)
            else:
                plt.subplots_adjust(top=.92, bottom=.35, left=.28)
            suptitle_string = 'emotion pair: %s\nsubstratification: %s\ndata source: %s, 20-day-cycle: %s' % (emotion_pair, 
                                                                                      substratification, 
                                                                                      data_to_use,
                                                                                      use_20_day_menstrual_cycle)
            if suptitle:
                plt.suptitle(suptitle_string, fontsize=20)
            else:
                print(suptitle_string)

            if figname is not None:
                plt.savefig(figname, dpi=300)
            plt.show()
    warnings.resetwarnings()


            
def get_hemisphere_of_timezone(tz):
    """
    Checked. Returns the hemisphere of the most common timezones or raises an exception if we haven't figured out the hemisphere. 
    some of the Southern TZs are right on the equator
    broadly the division is: South America + Australia = southern list
    Europe, USA, Canada, Mexico = Northern List
    exception is Caracas, which is in South America but is in the Northern hemisphere. 
    Some of the Brazilian TZs are quite close to the equator, as is America/Guayaquil. So in general Northern tzs are farther North 
    than Southern TZs are South. 
    """
    northern_timezones = ['Europe/Berlin', 
                          'Europe/Lisbon',
                          'Europe/Paris',
                          'Europe/Rome',
                          'Europe/London', 
                          'Europe/Copenhagen',
                          'America/Denver',
                          'Europe/Moscow', 
                          'America/Chicago', 
                          'Europe/Madrid',
                          'America/Los_Angeles', 
                          'America/New_York', 
                          'America/Vancouver', 
                          'America/Toronto', 
                          'America/Mexico_City', 
                          'America/Caracas']
    
    southern_timezones = ['America/Buenos_Aires', 
                          'Australia/Melbourne', 
                          'Australia/Sydney',
                          'America/Lima',
                          'America/Recife',
                          'America/Santiago',
                          'America/Fortaleza',
                          'America/Sao_Paulo',
                          'America/Guayaquil']
    if tz in northern_timezones:
        return 'Northern'
    if tz in southern_timezones:
        return 'Southern'
    raise Exception("Not a valid timezone")
    
def order_subcategories(l, category_type):
    """
    small helper method used by make_plots_stratified_by_category_type to put things in order. Checked.
    """
    if category_type == 'by_largest_timezones':
        # sort by which hemisphere and then sort alphabetically. 
        return sorted(l, key = lambda x:get_hemisphere_of_timezone(x) + ' ' + x)[::-1]
    elif str(l[0])[0] == '[':
        return sorted(l, key = lambda x:float(x.split(',')[0].replace('[', '')))
    else:
        return sorted(l)


                          

def make_plots_stratified_by_category_type(results, 
                                           category_type, 
                                           emotion_pairs=None,
                                           cycle_types_to_plot = ['near_period',
                                                                  'middle_of_night', 
                                                                  'weekend', 
                                                                  'summer', 
                                                                  'winter'], 
                                          data_to_use='binary_analysis_no_individual_mean', 
                                          top_margin=.8, 
                                          axis_fontsize=11): 
    """
    Checked. 
    category_type is what category to substratify by. 
    analysis = binary_analysis_no_individual_mean
    plots results broken down by category. 
    """
    assert data_to_use in ['binary_analysis_no_individual_mean', 'binary_analysis']
    print 'USING DATA TYPE %s' % data_to_use
    if emotion_pairs is None:
        emotion_pairs = results.keys()
        
    for emotion_pair in emotion_pairs:
        good_symptom, bad_symptom = emotion_pair.split('_versus_')
        results_by_cat = results[emotion_pair][category_type]
        if category_type == 'no_substratification':
            plt.figure(figsize = [5, 2])
        elif category_type == 'by_largest_timezones': # need extra room because so many categories. 
            plt.figure(figsize = [15, 5])
        else:
            plt.figure(figsize = [len(cycle_types_to_plot) * 5, 2 + .5 * len(results_by_cat.keys())])
        
        if category_type != 'no_substratification':
            # create one subplot for each cycle type. 
            subplot_idx = 1
            for cycle_type in cycle_types_to_plot:
                plt.subplot(1, len(cycle_types_to_plot), subplot_idx)

                
                # we want to plot the differences by subcategory for each cycle. 
                diffs = []
                cat_levels = []
                for level in order_subcategories(results_by_cat.keys(), category_type):

                    cat_levels.append(level)
                    if cycle_type in ['summer', 'winter'] and HEART_SUBSTRING in emotion_pair:
                        # no reliable data
                        diffs.append(0)
                    else:
                        diffs.append(results_by_cat[level][data_to_use]['%s_mean' % cycle_type] - 
                                     results_by_cat[level][data_to_use]['not_%s_mean' % cycle_type])
                if subplot_idx == 1:
                    # make sure category levels are in same order across subplots, if not something is very weird
                    original_cat_levels = cat_levels
                else:
                    assert cat_levels == original_cat_levels
                assert sum(np.isnan(diffs)) == 0
                assert len(diffs) == len(results_by_cat.keys())
                plt.barh(range(len(diffs)), 
                         diffs,
                         color = ['blue' if x < 0 else 'red' for x in diffs])
                if subplot_idx == 1:
                    plt.yticks(range(len(diffs)),
                               [str(a).replace('_', ' ').replace('America/', '').replace('Europe/', '') for a in cat_levels], 
                               fontsize=axis_fontsize)
                
                    plt.ylabel(category_type.replace('by_', '').replace('_', ' '), fontsize=axis_fontsize)
                else:
                    plt.yticks([])
                if HEART_SUBSTRING in emotion_pair:
                    plt.xticks([-3, 3], 
                                   ['-3\nBPM', '+3\nBPM'], 
                                  fontweight = 'bold', 
                                  fontsize=axis_fontsize)
                    plt.xlim([-3, 3])
                elif BBT_SUBSTRING in emotion_pair:
                    plt.xticks([-.5, .5], 
                                   ['-0.5\ndeg F', '+0.5\ndeg F'], 
                                  fontweight = 'bold', 
                                  fontsize=axis_fontsize)
                    plt.xlim([-.2, .2])
                elif WEIGHT_SUBSTRING in emotion_pair:
                    plt.xticks([-.5, .5], 
                        ['-.5 LBS', '+.5 LBS'], 
                                  fontweight = 'bold', 
                                  fontsize=axis_fontsize)
                else:
                    plt.xticks([-.1, -.05, 0, .05, .1], 
                                   ['10%', '5%', '0%', '5%', '10%'], 
                                  fontweight = 'bold', 
                                  fontsize=axis_fontsize)
                    plt.xlim([-.1, .1])
                plt.title('%s effect' % cycle_type.replace('_' , ' '), fontweight='bold', fontsize=16)
                pretty_bad_symptom_name = bad_symptom.split('*')[1].replace('_', ' ').replace('emotion', '').replace('6 hours or less', '<6 hrs')
                pretty_good_symptom_name = good_symptom.split('*')[1].replace('_', ' ').replace('emotion', '').replace('6 hours or more', '>6 hrs')
                    
                        
                # put bad symptom first because it's on the left in the plot. 
                plt.suptitle('<-%s vs. %s->' % (pretty_bad_symptom_name, 
                                               pretty_good_symptom_name), 
                             fontweight='bold', 
                             fontsize=16)
                
                subplot_idx += 1
        else:
            # if we're not substratifying, 
            # we just want to make a simple plot with one bar for each type of cycle.
            diffs_by_cycle_type = []
            for cycle_type in cycle_types_to_plot:
                diffs_by_cycle_type.append(results_by_cat[data_to_use]['%s_mean' % cycle_type] - 
                                 results_by_cat[data_to_use]['not_%s_mean' % cycle_type])
            assert sum(np.isnan(diffs_by_cycle_type)) == 0
            barwidth = .8
            plt.barh(range(len(diffs_by_cycle_type)), 
                     diffs_by_cycle_type,
                     color = ['blue' if x < 0 else 'red' for x in diffs_by_cycle_type], height = barwidth)
            plt.yticks(range(len(diffs_by_cycle_type)), 
                       [str(a).replace('_', ' ') for a in cycle_types_to_plot],
                       fontweight = 'bold')
            plt.xlim([-.18, .18]) # we put the positive negative emotion labels as xticks, 
            plt.xticks([-.18, -.1, 0, .1, .18], 
                   [bad_symptom.split('*')[1].replace('_', ' '), 
                    '10%', '0%', '10%', 
                    good_symptom.split('*')[1].replace('_', ' ')], fontweight = 'bold')
            ylimits = [-.2 - barwidth / 2, 4 + barwidth / 2 + .2]
            plt.plot([0, 0], ylimits, color = 'black')
            plt.ylim(ylimits)
            plt.subplots_adjust(left = .2)
            plt.title(data_to_use)
    
        plt.subplots_adjust(wspace = .15, hspace = .4, top = top_margin)
        plt.show()
                
def make_simple_histograms_of_individual_effects(individual_effects_df, plot_titlestring, min_obs):
    """
    Make histograms of individual-level effects for people who have at least min_obs observations in both binary bins.
    Checked. 
    Main argument is individual_effects_df, which is a dataframe with individual effects. 
    """
    data = deepcopy(individual_effects_df)
    binary_variables = ['near_period', 'middle_of_night', 'weekend', 'winter', 'summer']

    plt.figure(figsize=[20, 3])
    for i, binary_variable in enumerate(binary_variables):
        plt.subplot(1, 5, i + 1)
        # filter for people who have a minimum of n_obs with binary_variable = True and n_obs with binary_variable = False
        have_enough_obs = (data['%s_n_obs' % binary_variable] >= min_obs) & (data['%s_n_obs' % binary_variable] <= data['n_obs'] - min_obs)
        percent_with_enough_obs = 100 * have_enough_obs.mean()
        individual_effects = 100 * data['%s_cycle_effect' % binary_variable].loc[have_enough_obs].values
        mu = individual_effects.mean()
        absolute_mu = np.abs(individual_effects).mean()
        std = individual_effects.std()
        frac_feeling_at_least_20_percent_better = 100*(individual_effects >= 20).mean()
        frac_feeling_at_least_20_percent_worse = 100*(individual_effects <= -20).mean()
        plt.hist(individual_effects, bins=50, range=[-100, 100])
        plt.xticks([-60, -30, 0, 30, 60], ['-60%', '-30%', '0%', '30%', '60%'])
        plt.xlim([-60, 60])
        plt.xlabel("Change in mood")
        if i == 0:
            plt.ylabel("Number of people\n with this mood change")
        plt.title('%s\n%2.1f%% have at least %i obs\nmu: %2.1f%%; |mu| %2.1f%%; std %2.1f%%\n\
        at least 20%% better: %2.1f%%\nat least 20%% worse: %2.1f%%;\n%s' % 
                  (plot_titlestring,
                   percent_with_enough_obs,
                   min_obs, 
                   mu, 
                   absolute_mu, 
                   std, 
                   frac_feeling_at_least_20_percent_better,
                   frac_feeling_at_least_20_percent_worse,
                  binary_variable))
    plt.show()
                
def plot_ccdf_of_individual_effects(individual_effects_df, plot_titlestring):
    """
    plot the ccdf of individual effect sizes so we can see how many people have MORE extreme cycles than a given cutoff. 
    individual_effects_df should be a dataframe of individual effects. 
    Checked. 
    """
    data = deepcopy(individual_effects_df)
    absolute_cutoffs = range(5, 100, 5) 
    colors = ['red', 'black', 'blue', 'green', 'magenta']
    binary_variables = ['near_period', 'middle_of_night', 'weekend', 'winter', 'summer']

    # because we plot all binary effects on the same graph
    # we need to make sure they're computed on the same set of people. 
    # so we filter for people for whom all five binary effects can be estimated. 
    for i, binary_variable in enumerate(binary_variables):
        is_nan = np.isnan(data['%s_cycle_effect' % binary_variable])
        have_zero_obs = (data['%s_n_obs' % binary_variable] == data['n_obs']) | (data['%s_n_obs' % binary_variable] == 0)
        assert (is_nan == have_zero_obs).all() # make sure people with no observations are missing data. 

        if i == 0:
            good_idxs = ~is_nan
        else:
            good_idxs = good_idxs & (~is_nan)
        print("For %s, %2.1f%% of people have no cycle effect estimates; now have cycle estimates for %2.1f%% of people." % (
            binary_variable, is_nan.mean() * 100, good_idxs.mean() * 100))

    plt.figure(figsize=[12, 4])
    # make two subplots: one for people who feel better, one for people who feel worse. 
    for feel_better in [False, True]:
        plt.subplot(1, 2, 1 + feel_better)
        for i, binary_variable in enumerate(binary_variables):

            individual_effects = data.loc[good_idxs, '%s_cycle_effect' % binary_variable].values
            individual_effects = 100.*np.array(individual_effects)
            average_effect = np.mean(individual_effects)
            assert sum(np.isnan(individual_effects)) == 0
            percentages_whose_mood_changes_at_least_that_much = []
            for abs_cutoff in absolute_cutoffs:
                if feel_better:
                    p = 100.*np.mean(individual_effects >= abs_cutoff)
                else:
                    p = 100.*np.mean(individual_effects <= -abs_cutoff)
                percentages_whose_mood_changes_at_least_that_much.append(p)
            plt.plot(absolute_cutoffs, 
                     percentages_whose_mood_changes_at_least_that_much, 
                     label='%s (n=%i, mean=%2.1f%%)' % (binary_variable,
                                                        len(individual_effects),
                                                        average_effect),
                     color=colors[i])
        plt.xlim([0, 60])
        plt.ylim([0, 50])
        plt.xlabel("Percentage change in mood")
        if feel_better:
            plt.title('%s\nFeel BETTER during this period' % plot_titlestring)
        else:
            plt.title('%s\nFeel WORSE during this period' % plot_titlestring)
            plt.ylabel("Percentage of population\nwhose mood changes this dramatically")

        plt.legend()

    plt.show()


    
def plot_means_by_date(results, 
                       emotion_pairs,
                       data_to_use, 
                       min_date, 
                       max_date,
                       min_obs=500, 
                      substratification='no_substratification', 
                      substratification_level=None, 
                      ylimit=10, 
                      outliers_to_highlight=None, 
                      plot_the_weekend=False, 
                      period_effects_vector=None, 
                      figname=None, 
                      print_outlier_dates=True):
    """
    Plots the means over time. Plot is zero-centered. Checked. 
    Currently only set up to do a single substratification level. 
    """
    assert type(ylimit) is int
    assert data_to_use in ['means_by_date', 'means_by_date_no_individual_mean']
    for emotion_pair in emotion_pairs:
        if substratification == 'no_substratification':
            assert substratification_level is None
            data = deepcopy(results[emotion_pair][substratification][data_to_use])
        else:   
            assert substratification_level is not None
            data = deepcopy(results[emotion_pair][substratification][substratification_level][data_to_use])
        data['date'] = data['date'].map(lambda x:x.split()[0]) # remove 00:00:00 at end
        data = data.loc[(data['date'] >= min_date) & (data['date'] < max_date)]
        print '%i dates in data between %s and %s' % (len(data), min_date, max_date)
        data = data.loc[data['n_obs'] >= min_obs]
        data.index = range(len(data))
        print("After filtering for min_obs=%i, %i dates" % (min_obs, len(data)))
        assert list(data['date']) == sorted(list(data['date'])) # make sure dates are in correct order, if not something is wrong. 
        # make sure dates are correctly formatted. 
        assert data['date'].map(lambda x:(len(x.split('-')[0]) == 4) and
                                (len(x.split('-')[1]) == 2) and
                                (len(x.split('-')[2].split()[0]) == 2)).all()
        data['date'] = data['date'].map(lambda x:datetime.datetime.strptime(x, '%Y-%m-%d'))
        data['mean'] = data['mean'] - data['mean'].mean()
        if 'continuous' not in emotion_pair:
            data['mean'] = data['mean'] * 100.
            print("means are in units of percent!!")
        ## print out high and low dates.
        if print_outlier_dates:
            print 'Highest dates'
            print data.sort_values(by='mean')[::-1].head(n=10)
            print 'Lowest dates'
            print data.sort_values(by='mean').head(n=10)

            def moving_average(a, n=31) :
                ret = np.cumsum(a, dtype=float)
                ret[n:] = ret[n:] - ret[:-n]
                return ret[n - 1:] / n

            moving_average = (15 * [None]) + list(moving_average(data['mean'].values)) + (15 * [None])
            data['rolling_average'] = moving_average
            data['mean_minus_rolling_average'] = data['mean'] - data['rolling_average']

            print 'Highest dates after removing rolling average'
            print data.dropna(subset=['mean_minus_rolling_average']).sort_values(by='mean_minus_rolling_average')[::-1].head(n=10)
            print 'Lowest after removing rolling average'
            print data.dropna(subset=['mean_minus_rolling_average']).sort_values(by='mean_minus_rolling_average')[::-1].tail(n=10)

            print 'Special dates'
            for date in ['2016-01-01', '2016-02-14', '2016-11-09', '2016-12-25', '2017-01-01', '2017-02-14']:
                print date
                print data.loc[data['date'] == date]

        # check for a time-trend (which could complicate analysis)
        # this is actually slightly off because it assumes we have data for every single day, and that's not quite true.
        # but we're only missing ~4 days so it should be okay.
        slope, intercept, r_value, p_value, std_err = linregress(range(len(data)), data['mean'])
        increase_per_year_in_percent = slope * 365
        sorted_data = data.sort_values(by='mean')

        min_val_string = 'Minimum value is %2.1f%%, occurring on %s' % (sorted_data['mean'].iloc[0], 
                                                        sorted_data['date'].iloc[0])
        slope_string = 'Linear trend: increase per year %2.1f%%, r^2 %2.3f' % (increase_per_year_in_percent, r_value**2)
        if data_to_use == 'means_by_date':
            individual_mean_string = "individual mean NOT removed"
        else:
            individual_mean_string = "individual mean removed"

        fig = plt.figure(figsize=[24, 5])                                           
        
        if period_effects_vector is None:                                                            
            plt.plot_date(data['date'], data['mean'], '-', label='Currently observed signal', color='black', linewidth=3)
        else:
            data['days_since_beginning'] = data['date'].map(lambda x:(x - data['date'].min()).days)
            data['period_effect'] = data['days_since_beginning'].map(lambda x:period_effects_vector[x % len(period_effects_vector)])
            data['counterfactual_signal'] = data['mean'] + data['period_effect']
            plt.plot_date(data['date'], data['counterfactual_signal'], '-', label='If menstrual effect were observed', color=PERIOD_CYCLE_COLOR, linewidth=3)
            plt.plot_date(data['date'], data['mean'], '-', label='Currently observed signal', color='black', linewidth=3)

        
        
        if plot_the_weekend:
            data['is_weekend'] = data['date'].map(lambda x:x.strftime('%A') in ['Saturday', 'Sunday'])
            plt.plot_date(data['date'].loc[data['is_weekend']], 
                data['mean'].loc[data['is_weekend']], 
                label='', 
                color='black')
        #for year in [2016, 2017]:
        #    plt.plot_date([datetime.datetime(year, 1, 1), datetime.datetime(year, 1, 1)], [-ylimit, ylimit], '-', color='black')
        outliers_to_nice_names = {'2016-11-09':'Day after 2016 US Election', 
                              '2016-12-25':'Christmas', 
                              '2015-12-25':None, 
                              '2017-06-25':'Eid al-Fitr, Saudi Arabia', 
                              '2017-02-11':'Chinese New Year Lantern Festival', 
                              '2017-10-12':'Lady of Aparecida Day', 
                              '2017-09-07':'Brazilian Independence', 
                              '2016-10-12':None}
        outliers_to_colors = {'2016-11-09':'#9467bd', 
                              '2016-12-25':'#2ca02c', 
                              '2015-12-25':'#2ca02c', 
                              '2017-06-25':'#2ca02c', 
                              '2017-02-11':'#2ca02c',
                              '2017-10-12':'#2ca02c', 
                              '2017-09-07':'#2ca02c', 
                              '2016-10-12':'#2ca02c'}
        warning_string =''
        if outliers_to_highlight is not None:
            for outlier in outliers_to_highlight:
                assert outlier in outliers_to_nice_names
                outlier_idxs = data['date'] == datetime.datetime(*[int(a) for a in outlier.split('-')])
                assert outlier_idxs.sum() == 1
                val_to_plot = float(data['mean'].loc[outlier_idxs].iloc[0])
                most_extreme_val_to_plot = ylimit * .95
                if val_to_plot < -most_extreme_val_to_plot:
                    val_to_plot = -most_extreme_val_to_plot 
                    warning_string += '\nWARNING: EXTREME VALUE %s has been truncated' % outlier
                elif val_to_plot > most_extreme_val_to_plot:
                    val_to_plot = most_extreme_val_to_plot
                    warning_string += '\nWARNING: EXTREME VALUE %s has been truncated' % outlier
                plt.plot_date(data['date'].loc[outlier_idxs], 
                              [val_to_plot], 
                              color=outliers_to_colors[outlier], 
                              label=outliers_to_nice_names[outlier],
                              markersize=13)
                #plt.plot_date([data['date'].loc[outlier_idxs].values[0], 
                #    data['date'].loc[outlier_idxs].values[0]],
                ##    [-ylimit, ylimit],
                #    color=outliers_to_colors[outlier], 
                #    label=outliers_to_nice_names[outlier], 
                #    linestyle='--')


        
        plt.title('%s\n%s\n%s\n%s\nsubstratification: %s, val: %s%s\n' % (emotion_pair, 
                                        min_val_string, 
                                        slope_string, 
                                        individual_mean_string,
                                        substratification, 
                                        substratification_level, 
                                        warning_string))
                
        plt.legend(prop={'size':20}, bbox_transform=fig.transFigure, bbox_to_anchor=(.68, .7)) #'weight':'bold'
        plt.subplots_adjust(right=.68)
        #plt.legend(prop={'size':18, 'weight':'bold'})
        assert ylimit > 0
        plt.ylim([-ylimit, ylimit])
        plt.xlim([data['date'].min(), data['date'].max()])
        #plt.plot_date([data['date'].min(), data['date'].max()], [0, 0], linestyle='--', color='grey')
        good_symptom, bad_symptom = emotion_pair.split('_versus_')
        good_symptom = good_symptom.split('*')[1]
        bad_symptom = bad_symptom.split('*')[1]
        if 'continuous' not in emotion_pair:
            assert ylimit % 2 == 0
            plt.yticks([-ylimit / 2, 0, ylimit / 2], 
                       ['%i%% more %s' % (ylimit / 2, bad_symptom), 
                        'Baseline', 
                        '%i%% more %s' % (ylimit / 2, good_symptom)], 
                       fontsize=22)


        date_ticks = [datetime.datetime(2016, 1, 1), 
            datetime.datetime(2016, 4, 1), 
            datetime.datetime(2016, 7, 1), 
            datetime.datetime(2016, 10, 1), 
            datetime.datetime(2017, 1, 1), 
            datetime.datetime(2017, 4, 1), 
            datetime.datetime(2017, 7, 1), 
            datetime.datetime(2017, 10, 1)]
        date_ticks = [tick for tick in date_ticks if tick >= data['date'].min() and tick <= data['date'].max()]
        plt.xticks(date_ticks, [x.strftime('%Y-%m') for x in date_ticks], fontsize=22)
        if figname is not None:
            plt.savefig(figname, dpi=300)

        plt.show()
            
def add_binary_annotations(d, start_day_for_period_individual_effect, end_day_for_period_individual_effect):
    # Checked. 
    assert start_day_for_period_individual_effect is not None
    assert end_day_for_period_individual_effect is not None
    d['weekend'] = d['weekday'].map(lambda x:x in ['Saturday', 'Sunday'])
    d['summer'] = d['month'].map(lambda x:get_season(x) == 'summer')
    d['winter'] = d['month'].map(lambda x:get_season(x) == 'winter')
    d['near_period'] = d['date_relative_to_period'].map(lambda x:(x >= start_day_for_period_individual_effect) and
                                                                 (x < end_day_for_period_individual_effect))
    d['middle_of_night'] = d['local_hour'].map(in_middle_of_night)
    return d

def remove_individual_means_from_df(d):
    """
    Checked. For each individual, removes their mean value, consistent with Golder + Macy Science paper. 
    """
    d = deepcopy(d)
    individual_means = d[['user_id_hash', 'good_mood']].groupby('user_id_hash').mean()
    individual_means = dict(zip(individual_means.index, individual_means['good_mood']))
    d['individual_mean'] = d['user_id_hash'].map(lambda x:individual_means[x])
    d['good_mood'] = d['good_mood'] - d['individual_mean']
    assert abs(d[['user_id_hash', 'good_mood']].groupby('user_id_hash').mean()).values.max() < 1e-8
    return d

def compute_means_by_date(d, remove_individual_means):
    """
    Checked. 
    Returns a dataframe with the fraction of people who report positive emotion on each date
    along with the number of observations on that date. 
    """
    d = deepcopy(d)
    if remove_individual_means:
        d = remove_individual_means_from_df(d)
    means_by_date = d[['good_mood', 'date']].groupby('date').agg(['mean', 'size'])
    
    # Reformat data so it doesn't have a weird multi-index and make sure it's in sorted order. 
    means_by_date.columns = ['mean', 'n_obs']
    means_by_date['date'] = means_by_date.index
    del means_by_date.index.name
    means_by_date = means_by_date.sort_values(by='date')
    means_by_date.index = range(len(means_by_date))
    
    return means_by_date

def compute_diffs_in_binary_variable(d, remove_individual_means, start_day_for_period_individual_effect, end_day_for_period_individual_effect):
    """
    prints the differences in a binary variable between weekend and weekday, summer and winter, and period / not period.       
    Checked. 
    """
    d = deepcopy(d)
    d = add_binary_annotations(d, start_day_for_period_individual_effect, end_day_for_period_individual_effect)
    if remove_individual_means:
        d = remove_individual_means_from_df(d)
    
    results = {}
    for time_period in ['summer', 'winter', 'weekend', 'middle_of_night', 'near_period']:
        results['%s_mean' % time_period] = d['good_mood'].loc[d[time_period] == 1].mean()
        results['not_%s_mean' % time_period] = d['good_mood'].loc[d[time_period] == 0].mean()
    results['n'] = len(d)
    return results

def take_simple_means_by_group(d, remove_individual_means):
    """
    rather than doing a regression or anything fancy, just does a groupby to get counts and means for each group. 
    Checked. 
    """
    
    d = deepcopy(d)
    if remove_individual_means:
        d = remove_individual_means_from_df(d)
    results = {}
    for grouping_variable in ['month', 'weekday', 'local_hour', 'date_relative_to_period']:
        summary_stats = ['mean', 'size', 'std']
        grouped_d = d[[grouping_variable, 'good_mood']].groupby(grouping_variable).agg(summary_stats)
        grouped_d.columns = summary_stats
        grouped_d['err'] = 1.96 * grouped_d['std']/np.sqrt(grouped_d['size']) 
        grouped_d = grouped_d[['mean', 'size', 'err']]
        results[grouping_variable] = grouped_d
    return results

def find_interaction_between_cycles(d, remove_individual_means, start_day_for_period_individual_effect, end_day_for_period_individual_effect):
    """
    assesses whether there are interactions between cycles (not user-specific). Checked. 
    """
    results = {'linear_regression':{}, 
               'raw_means':{}}
    print "assessing non-user-specific interaction between cycles"
    d = deepcopy(d)  
    d = add_binary_annotations(d, start_day_for_period_individual_effect, end_day_for_period_individual_effect)
    d['good_mood'] = 1.0 * d['good_mood'] 
    if remove_individual_means:
        d = remove_individual_means_from_df(d)
    boolean_cols = ['near_period', 'middle_of_night', 'weekend', 'summer', 'winter']
    for i in range(len(boolean_cols)):
        for j in range(i):
            if boolean_cols[i] == 'winter' and boolean_cols[j] == 'summer':
                # singular matrix (these are mutually exclusive)
                continue
            key = '%s*%s' % (boolean_cols[i], boolean_cols[j])
            raw_means = d[['good_mood', boolean_cols[i], boolean_cols[j]]].groupby([boolean_cols[i], boolean_cols[j]]).mean()
            print 'results for', key
            print raw_means
            model = sm.OLS.from_formula('good_mood ~ %s*%s' % (boolean_cols[i], boolean_cols[j]), data = d).fit()
            print model.summary()
            results['linear_regression'][key] = {'pvalues':model.pvalues, 'betas':model.params}
            results['raw_means'][key] = raw_means
    return results

def compute_individual_level_cycle_effects(d, start_day_for_period_individual_effect, end_day_for_period_individual_effect):
    """
    compute the individual cycle effects by person. Checked
    Does not remove the mean for each individual prior to computing individual cycle-level effects (but this shouldn't matter). 
    """
    raise Exception("This is deprecated!")
    print("Computing individual-level cycle effects.")
    boolean_cols = ['near_period', 'middle_of_night', 'weekend', 'summer', 'winter']
    d = deepcopy(d)  
    d = add_binary_annotations(d, start_day_for_period_individual_effect, end_day_for_period_individual_effect)
    grouped_d = d.groupby('user_id_hash')
    print("Done grouping data")
    # results data format is a little odd here. one list entry for each user. 
    # each list entry is a dictionary: keys are number of observations and cycle effects. 
    results = []
    total_users = len(set(d['user_id_hash']))
    n_users_examined = 0
    for user, user_d  in grouped_d:
        n_users_examined += 1
        user_results = {}
        n = len(user_d)
        user_results['n_obs'] = n
        user_results['user_mean'] = user_d['good_mood'].mean()
        for boolean_col in boolean_cols:
            n_in_cycle = user_d[boolean_col].sum()
            if (n_in_cycle == 0) or (n_in_cycle == n):
                cycle_effect = np.nan
            else:
                cycle_effect = user_d.loc[user_d[boolean_col] == True, 'good_mood'].mean() - \
                               user_d.loc[user_d[boolean_col] == False, 'good_mood'].mean() 
                assert ~np.isnan(cycle_effect)
            user_results[boolean_col + '_n_obs'] = n_in_cycle
            user_results[boolean_col + '_cycle_effect'] = cycle_effect
        for covariate_col in COLS_TO_STRATIFY_BY:
            user_col_vals = list(set(user_d[covariate_col].dropna()))
            if len(user_col_vals) != 1:
                user_results[covariate_col] = None
            else:
                user_results[covariate_col] = user_col_vals[0]
            
        results.append(user_results)
        if len(results) % 10000 == 0:
            print '%i / %i users examined' % (len(results), total_users)
    results_dataframe = pd.DataFrame(results)
    for c in results_dataframe.columns:
        print 'Column %s has %2.3f%% good values' % (c, 100.*len(results_dataframe[c].dropna()) / len(results_dataframe))
    return results    

def fast_compute_individual_level_cycle_effects(d, start_day_for_period_individual_effect, end_day_for_period_individual_effect):
    """
    Computes individual-level cycle effects using a group_by as opposed to being really really slow. 
    Checked. 
    """
    print("********")
    print("Computing individual level cycle effects QUICKLY for %i rows and %i users" % (len(d), 
                                                                                 len(set(d['user_id_hash']))))
    boolean_cols = ['near_period', 'middle_of_night', 'weekend', 'summer', 'winter']
    d = deepcopy(d)  
    d = add_binary_annotations(d, start_day_for_period_individual_effect, end_day_for_period_individual_effect)
    
    overall_means = d[['user_id_hash', 'good_mood']].groupby('user_id_hash').agg(['mean', 'size'])
    overall_means = overall_means.reset_index()
    overall_means.columns = ['user_id_hash', 'user_mean', 'n_obs']


    for c in boolean_cols:
        print(c)
        on_cycle_means = d.loc[d[c] == True, ['user_id_hash', 'good_mood']].groupby('user_id_hash').mean()

        on_cycle_means = dict(zip(on_cycle_means.index, on_cycle_means['good_mood']))

        on_cycle_counts = d.loc[d[c] == True, ['user_id_hash']].groupby('user_id_hash').size()
        on_cycle_counts = dict(zip(on_cycle_counts.index, on_cycle_counts.values))

        off_cycle_means = d.loc[d[c] == False, ['user_id_hash', 'good_mood']].groupby('user_id_hash').mean()
        off_cycle_means = dict(zip(off_cycle_means.index, off_cycle_means['good_mood']))

        for k in overall_means['user_id_hash']:
            if k not in on_cycle_means:
                on_cycle_means[k] = np.nan
            if k not in off_cycle_means:
                off_cycle_means[k] = np.nan
            if k not in on_cycle_counts:
                on_cycle_counts[k] = 0
        overall_means['%s_cycle_effect' % c] = overall_means['user_id_hash'].map(lambda x:on_cycle_means[x] - 
                                                                                 off_cycle_means[x])
        overall_means['%s_n_obs' % c] = overall_means['user_id_hash'].map(lambda x:on_cycle_counts[x])
    return overall_means

def compute_individual_level_cycle_effects_for_period_by_splitting_into_two_and_taking_median(d):
    """
    Used for computing period effects: dichotomizes at a cutoff day for each user 
    which maximizes the difference between pre and post 
    Checked. 
    """
    d = deepcopy(d)
    print("Computing individual-level effects using median method")
    grouped_d = d.groupby('user_id_hash')
    print("Done grouping data")
    results = []
    n_users = len(set(d['user_id_hash']))
    n_analyzed = 0
    total_users_without_cutoffs = 0
    for user, user_d in grouped_d:
        user_d = deepcopy(user_d).dropna(subset=['days_after_last_cycle_start'])
        if n_analyzed % 100 == 0:
            print("Computed effects for %i/%i users" % (n_analyzed, n_users))
        n_analyzed += 1
        user_d = user_d.sort_values(by='date')
        assert (user_d['days_after_last_cycle_start'] >= 0).all()
         
        possible_cutoff_days = range(7, 22)
        max_diff = 0
        no_valid_cutoffs_found = True
        for cutoff_day in possible_cutoff_days:
            pre_cutoff_vals = user_d.loc[user_d['days_after_last_cycle_start'] <= cutoff_day, 'good_mood'].values
            post_cutoff_vals = user_d.loc[user_d['days_after_last_cycle_start'] > cutoff_day, 'good_mood'].values
            if len(pre_cutoff_vals) < 10 or len(post_cutoff_vals) < 10:
                continue
            diff = np.median(post_cutoff_vals) - np.median(pre_cutoff_vals)
            if np.abs(diff) >= np.abs(max_diff):
                no_valid_cutoffs_found = False
                max_diff = diff
                max_day = cutoff_day

        total_users_without_cutoffs += no_valid_cutoffs_found
        if no_valid_cutoffs_found:
            print("Median effect could not be computed for this user:")
            print user_d[['days_after_last_cycle_start', 'good_mood']].to_string()
            continue
        results.append({'user_id_hash':user, 
            'period_cycle_effect':max_diff,
            'n_obs':len(user_d), 
            'user_mean':user_d['good_mood'].mean(),
            'cutoff_day':max_day})
    print("Total users without valid cutoffs (for whom no median effect could be computed: %i" % total_users_without_cutoffs)
    return results



def find_person_specific_interactions_between_cycles(d, min_n = 20, min_in_group = 3, start_day_for_period_individual_effect=None, end_day_for_period_individual_effect=None):
    """
    Looks for correlations between cycle strength across people: eg, do people with larger PMS 
    swings have larger midnight swings? 
    Do not need to remove individual mean because we are comparing individuals to themselves anyway.
    Checked. 
    """
    boolean_cols = ['near_period', 'middle_of_night', 'weekend', 'summer', 'winter']
    d = deepcopy(d)  
    d = add_binary_annotations(d, start_day_for_period_individual_effect, end_day_for_period_individual_effect)
    grouped_d = deepcopy(d).groupby('user_id_hash')
    results = {}
    for i in range(len(boolean_cols)):
        for j in range(i):
            key = '%s*%s' % (boolean_cols[i], boolean_cols[j])
            cycle_i_effects = []
            cycle_j_effects = []
            if boolean_cols[i] == 'winter' and boolean_cols[j] == 'summer':
                # singular matrix (these are mutually exclusive)
                continue
            for user, user_d  in grouped_d:
                n = len(user_d)
                # identify number of observations we have for each cycle. 
                i_true = user_d[boolean_cols[i]].sum()
                j_true = user_d[boolean_cols[j]].sum()
                # skip people who don't have enough good data (total count or count in each cycle group)
                if n < min_n:
                    continue
                if (i_true < min_in_group) or (n - i_true < min_in_group):
                    continue
                if (j_true < min_in_group) or (n - j_true < min_in_group):
                    continue
                cycle_i_effect = user_d.loc[user_d[boolean_cols[i]] == True, 'good_mood'].mean() - user_d.loc[user_d[boolean_cols[i]] == False, 'good_mood'].mean() 
                cycle_j_effect = user_d.loc[user_d[boolean_cols[j]] == True, 'good_mood'].mean() - user_d.loc[user_d[boolean_cols[j]] == False, 'good_mood'].mean()
                cycle_i_effects.append(cycle_i_effect)
                cycle_j_effects.append(cycle_j_effect)
            r, p = pearsonr(cycle_i_effects, cycle_j_effects)
            print '%s mean: %2.3f; %s mean %2.3f; correlation %2.3f; p %2.3e, n = %i' % (boolean_cols[i], \
                                                                             np.mean(cycle_i_effects), \
                                                                            boolean_cols[j], \
                                                                             np.mean(cycle_j_effects), \
                                                                             r, p, \
                                                                            len(cycle_i_effects))
            results[key] = {'mu_1':np.mean(cycle_i_effects), \
                            'mu_2':np.mean(cycle_j_effects), \
                            'r':r, \
                            'p':p, \
                            'n':len(cycle_i_effects), \
                            'full_effects_1':cycle_i_effects, \
                            'full_effects_2':cycle_j_effects}
    return results

def compute_most_likely_derivatives(user_data, max_timestep):
    """
    Given user data (a list of dictionaries with 'y' and 't' for each user)
    and a maximum timestep, returns estimates of the derivatives. 
    Checked. Works on a bunch of simulated datasets. 
    Minimizes a least-squares objective ||A*delta - (y1 - y0)||^2 where A is a design matrix
    delta is the vector of discrete derivatives we're solving for
    and (y1 - y0) is the change in y. 
    """
    user_data = deepcopy(user_data)
    user_data = [a for a in user_data if len(a['y']) >= 2] # can only use data with at least two timepoints. 
    total_datapoints = sum([len(user_data_i['y']) for user_data_i in user_data]) # number of rows in A matrix. 
    
    # the jth column corresponds to delta_j, ie, the derivative from t_j to t_j + 1, starting at j=0.
    # If j = max_timestep, this is derivative from t_max_timestep to 0 (ie, loop back around).  
    A = np.zeros([total_datapoints, max_timestep + 1]) 
    print 'Shape of original user data is', A.shape
    row_idx = 0
    y1_minus_y0 = [] # this is the target we're trying to hit: the change in y. 
    # Each entry is the user's value at the end of the time period minus their value at the beginning. 
    
    loop_around_rows = [] # these rows correspond to looping back and completing the cycle. Each user has one row. 
    # Not sure we want to include these rows in the computation. Including yields similar but non-identical results
    # for noisy data (for perfect data, results are identical). We return computations performed both ways.  
    all_unique_timesteps = set()
    for i in range(len(user_data)):
        y_i = np.array(user_data[i]['y'])
        t_i = np.array(user_data[i]['t'])
        
        for t_ij in t_i:
            all_unique_timesteps.add(t_ij)
        assert len(y_i) == len(t_i) # timestamps should be equal in length to observations
        assert len(set(t_i)) == len(t_i) # timestamps should be unique. 
        user_start_row = row_idx
        for j in range(len(y_i) - 1): # -1 because we compare each timestep to the next timestep.
            col_start_idx = t_i[j]
            col_stop_idx = t_i[j + 1]
            assert  col_stop_idx > col_start_idx # timesteps should be in sorted order. 
            A[row_idx, col_start_idx:col_stop_idx] = 1
            y1_minus_y0.append(y_i[j + 1] - y_i[j])
            row_idx += 1
            loop_around_rows.append(False)
        # add loop around row, one for each user. 
        A[row_idx, t_i[-1]:] = 1
        A[row_idx, :t_i[0]] = 1
        assert A[user_start_row:(row_idx + 1), :].sum() == (max_timestep + 1) # each user should have exactly this many nonzero entries.
        y1_minus_y0.append(y_i[0] - y_i[-1])
        loop_around_rows.append(True)
        row_idx += 1
    assert set(all_unique_timesteps) == set(range(max_timestep + 1)) # we should have observations for every timestep. 
    assert (A.sum(axis=1) > 0).all() # all rows of A should be nonzero. 

    assert set(A.sum(axis=0)) == set([len(user_data)]) # all columns of A should add up to the number of users. 
    assert len(A) == len(y1_minus_y0) # make sure shapes are correct. 
    
    assert len(loop_around_rows) == len(y1_minus_y0)
    loop_around_rows = np.array(loop_around_rows)
    y1_minus_y0 = np.atleast_2d(np.array(y1_minus_y0)).transpose()
    
    beta_with_looparound_rows, _, _, _ = np.linalg.lstsq(A, y1_minus_y0)
    beta_without_looparound_rows, _, _, _ = np.linalg.lstsq(A[~loop_around_rows, :], y1_minus_y0[~loop_around_rows])
    
    # remove looparound entry since we wouldn't use it anyway
    beta_with_looparound_rows = beta_with_looparound_rows[:-1] 
    beta_without_looparound_rows = beta_without_looparound_rows[:-1]
    print 'absolute difference in derivative estimates'
    print np.abs(beta_with_looparound_rows - beta_without_looparound_rows)
    return beta_with_looparound_rows, beta_without_looparound_rows

def do_derivative_estimation_on_real_data(d, cycle_column):
    """
    runs the above method (which is designed for general, abstract data) on real data. 
    Checked. 
    """
    assert cycle_column in ['local_hour', 'weekday', 'date_relative_to_period', 'month']
    n_unique_users = len(set(d['user_id_hash']))
    user_data = []
    d = deepcopy(d)
    if cycle_column == 'weekday':
        weekdays = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        weekdays_to_numbers = dict(zip(weekdays, range(len(weekdays))))
        d['numeric_weekday'] = d['weekday'].map(lambda x:weekdays_to_numbers[x])
        cycle_column = 'numeric_weekday'
    cycle_start = d[cycle_column].min()
    cycle_end = d[cycle_column].max()
    all_cycle_vals = sorted(list(set(d[cycle_column])))
    assert np.isnan(d[cycle_column]).sum() == 0
    
    if cycle_column == 'local_hour':
        assert (cycle_start == 0) and (cycle_end == 23)
        assert all_cycle_vals == range(24)
    elif cycle_column == 'weekday':
        assert (cycle_start == 0) and (cycle_end == 6)
        assert all_cycle_vals == range(7)
    elif cycle_column == 'date_relative_to_period':
        # two different filters might potentially be used. 
        assert (cycle_start == -14 and cycle_end == 14) or (cycle_start == -20 and cycle_end == 20)
        assert (all_cycle_vals == range(-14, 15)) or (all_cycle_vals == range(-20, 21))
    elif cycle_column == 'month':
        assert (cycle_start == 1) and (cycle_end == 12)
        assert (all_cycle_vals == range(1, 13))
        
    print 'Original data for cycle %s ranges from %i to %i' % (cycle_column, cycle_start, cycle_end)
    d['cycle_timestep'] = d[cycle_column].map(lambda x:int(x - cycle_start))
    min_timestep = d['cycle_timestep'].min()
    max_timestep = d['cycle_timestep'].max()
    print 'New cycle timestep range is %i to %i' % (min_timestep, max_timestep)
    grouped_by_user = d[['user_id_hash', 'cycle_timestep', 'good_mood']].groupby(['user_id_hash'])
    n_users_read = 0
    for user_id, user_d in grouped_by_user:
        means_by_user_and_cycle_timestep = user_d.groupby('cycle_timestep').mean() * 1.0
        user_data.append({'t':list(means_by_user_and_cycle_timestep.index), 
                          'y':list(means_by_user_and_cycle_timestep['good_mood'].values)})
        n_users_read += 1
        if n_users_read % 10000 == 0:
            print '%i / %i users read' % (n_users_read, n_unique_users)

    derivative_with_looparound, derivative_without_looparound = compute_most_likely_derivatives(
        user_data=user_data,
        max_timestep=max_timestep)
    
    def get_function_from_derivative(derivative_vector):
        """
        small helper method to integrate a derivative. Starts at 0, adds the derivative at each timestep. 
        Y will end up being 1 longer than the derivative vector, which makes sense. So eg if max_timestep=3, 
        the derivative will have 3 entries (delta_0, delta_1, delta_2) and Y will have 4 (Y_0, Y_1, Y_2, Y_3). 
        Checked. 
        """
        y = [0]
        for i in range(len(derivative_vector)):
            y0 = y[-1]
            y.append(derivative_vector[i] + y0)
        y = np.array(y)
        y = y - np.mean(y)
        return y
    
    looparound_estimated_cycle_values = get_function_from_derivative(derivative_with_looparound)
    no_looparound_estimated_cycle_values = get_function_from_derivative(derivative_without_looparound)
    
    if cycle_column != 'numeric_weekday':
        true_cycle_timesteps = all_cycle_vals
    else:
        true_cycle_timesteps = weekdays
    
    # put data into format for four cycle plots. 
    looparound_df = pd.DataFrame({'mean':looparound_estimated_cycle_values, 'size':1e12,'std':0,'err':0, 'looparound':True})
    looparound_df.index = true_cycle_timesteps
    
    no_looparound_df = pd.DataFrame({'mean':no_looparound_estimated_cycle_values, 'size':1e12,'std':0,'err':0, 'looparound':False})
    no_looparound_df.index = true_cycle_timesteps
    
    return looparound_df, no_looparound_df

def estimate_derivatives_for_all_four_types_of_cycles(d):
    """
    Loops over all four types of cycles and computes derivatives for each. Checked. 
    """
    print 'Estimating derivatives for all four types of cycles'
    d = deepcopy(d)
    looparound_results = {}
    no_looparound_results = {}
    print 'Computing derivatives for period with 14-day span'
    looparound_results['date_relative_to_period'], no_looparound_results['date_relative_to_period']  = do_derivative_estimation_on_real_data(
        d.loc[d['date_relative_to_period'].map(lambda x:np.abs(x) <= 14)],
        cycle_column = 'date_relative_to_period')
    
    print 'Computing derivatives for period with 20-day span'
    looparound_results['period_20'], no_looparound_results['period_20']  = do_derivative_estimation_on_real_data(d, cycle_column = 'date_relative_to_period')
    
    for cycle_column in ['weekday', 'month', 'local_hour']:
        print 'Computing derivatives for %s' % cycle_column
        looparound_results[cycle_column], no_looparound_results[cycle_column] = do_derivative_estimation_on_real_data(d,cycle_column = cycle_column)
    return looparound_results, no_looparound_results

def fit_linear_regression(d, covariates_for_regression):
    """
    checked. Performs a linear regression on day of year, weekday, and period date. 
    does this after removing individual means. This regression is NOT analogous to the logistic regression below. 
    """
    for c in covariates_for_regression:
        if c != 'days_since_start':
            assert c[:2] == 'C('
        if c == 'C(date_relative_to_period)':
            raise Exception("This is deprecated; you want to set the base level as 0. Correct covariate is C(date_relative_to_period, Treatment(reference=0))")
    t0 = time.time()
    print("Performing linear regression. First removing individual means.")
    d = deepcopy(d)
    d['year'] = d['date'].map(lambda x:x.split('-')[0])
    assert d['year'].map(lambda x:x in ['2015', '2016', '2017']).all()
    d = remove_individual_means_from_df(d)
    d = d.dropna(subset=[a.replace(')', '').replace('C(', '').split(', Treatment')[0] for a in covariates_for_regression])
    model = sm.OLS.from_formula('good_mood ~ %s' % ('+'.join(covariates_for_regression)), data = d).fit(cov_type='cluster', cov_kwds={'groups':d['user_id_hash']})
    print model.summary()
    model_results = {'params':model.params, 'pvalues':model.pvalues, '95_CI':model.conf_int(), 'covariance_matrix':model.cov_params()}
    print "Total time to perform linear regression: %2.3f seconds" % (time.time() - t0)
    return model_results

def replace_elem_in_list(l, old, new):
    new_l = []
    assert new not in l
    for i in range(len(l)):
        if l[i] == old:
            new_l.append(new)
        else:
            new_l.append(l[i])
    return new_l

def fit_regression_with_alternate_period_specifications(d, covariates_for_regression):
    # add extra column for how far someone is through their cycle. 
    print("Computing alternate period regression specifications")
    print("Original covariates were")
    print(covariates_for_regression)
    d = deepcopy(d)
    print d[['mean_cycle_length']].describe()
    assert (d['mean_cycle_length'] < 7).sum() == 0
    d['frac_through_cycle'] = d['date_relative_to_period'] / d['mean_cycle_length'].map(lambda x:x if x is not None else 1)
    d.loc[pd.isnull(d['mean_cycle_length']), 'frac_through_cycle'] = None
    d['frac_through_cycle'] = d['frac_through_cycle'].map(lambda x:math.floor(x * 28) / 28.)

    d.loc[d['frac_through_cycle'].map(lambda x:np.abs(x) > .5), 'frac_through_cycle'] = None
    print("Fraction of binned_frac cycle set to None: %2.3f" % np.mean(pd.isnull(d['frac_through_cycle'])))
    val_counts = Counter(d['frac_through_cycle'].dropna())
    assert len(val_counts) == 29
    for val in sorted(val_counts.keys())[::-1]:
        print("%s: %2.1f%% of non-missing values (%i)" % (val, 100.*val_counts[val]/len(d['frac_through_cycle'].dropna()), val_counts[val]))

    # actually fit regressions 
    results = {}
    for alternate_specification in ['days_after_last_cycle_start', 'days_before_next_cycle_start', 'frac_through_cycle']:
        if alternate_specification == 'days_after_last_cycle_start':
            alternate_specification_and_ref_level = 'C(days_after_last_cycle_start, Treatment(reference=0))'
        elif alternate_specification == 'days_before_next_cycle_start':
            alternate_specification_and_ref_level = 'C(days_before_next_cycle_start, Treatment(reference=-40))'
        else:
            alternate_specification_and_ref_level = 'C(frac_through_cycle, Treatment(reference=-0.5))'
        results['with_period_parameterization_%s' % alternate_specification] = fit_linear_regression(d, 
        replace_elem_in_list(covariates_for_regression, 
            'C(date_relative_to_period, Treatment(reference=0))',
             alternate_specification_and_ref_level))
    return results


def fit_regression_with_alternate_seasonal_specifications(d, covariates_for_regression):
    # Add some extra columns. 
    print("Computing alternate seasonal regression specifications")
    print("Original covariates were")
    print(covariates_for_regression)
    d = deepcopy(d)
    d['datetime'] = d['date'].map(lambda x:datetime.datetime.strptime(x, '%Y-%m-%d'))

    # annotate with a grouped variable based on day of year. Eg, if bandwidth = 7
    # this roughly corresponds to week of year. 
    def get_day_grouping(x, bandwidth):
        day_of_year = int(x.strftime('%j'))
        grouping = int(math.floor(1.*day_of_year / bandwidth))
        return grouping
    d['week_of_year'] = d['datetime'].map(lambda x:get_day_grouping(x, 7))
    max_week_val = d['week_of_year'].max()
    # keep from having super-short weeks
    d.loc[d['week_of_year'] == max_week_val, 'week_of_year'] = max_week_val - 1
    print("Week of year ranges from %i-%i" % (d['week_of_year'].min(), d['week_of_year'].max()))

    # annotate with start date so we can fit a linear trend. 
    start_date = d['datetime'].min()
    d['days_since_start'] = d['datetime'].map(lambda x:(x - start_date).days)
    assert (d['days_since_start'] >= 0).all()
    print("Days since start ranges from %i-%i" % (d['days_since_start'].min(), d['days_since_start'].max()))

    # actually fit regressions 
    results = {}

    results['with_week_of_year_instead_of_month'] = fit_linear_regression(d, 
        replace_elem_in_list(covariates_for_regression, 'C(month)', 'C(week_of_year)'))
    results['with_days_since_start_instead_of_year'] = fit_linear_regression(d, 
        replace_elem_in_list(covariates_for_regression, 'C(year)', 'days_since_start'))
    results['swap_out_both'] = fit_linear_regression(d, 
        replace_elem_in_list(
            replace_elem_in_list(covariates_for_regression, 'C(year)', 'days_since_start'), 
            'C(month)', 'C(week_of_year)'))

    return results

def bootstrapped_analyses_one_process(d, 
                                 bootstrap_regression_amplitude_errorbars, 
                                 bootstrap_regressions_of_period_effects, 
                                 bootstrap_daily_time_series,
                                 regression_amplitude_kwargs, 
                                 period_effects_kwargs, 
                                 bootstrap_seeds, 
                                 return_dict):
    """
    Parallelize bootstrapping - this is single-process helper method. 
    bootstrap_regression_amplitude_errorbars, bootstrap_regressions_of_period_effects, bootstrap_daily_time_series are 
    boolean flags which specify which analyses we actually have to run. kwargs specify kwargs for these analyses. 
    bootstrap_seeds specifies which seeds to use to select user_id_hases, and return_dict lets us store the data for multiprocessing. 
    """
    grouped_d = d.groupby('user_id_hash')
    unique_user_ids = sorted(list(set(d['user_id_hash'])))
    for iterate in range(len(bootstrap_seeds)):
        seed = bootstrap_seeds[iterate]
        reproducible_random_sampler = random.Random(seed)
        results_for_iterate = {'linear_regression_bootstrapped_iterates':None, 
                               'period_effects_with_covariates':None, 
                               'daily_time_series_no_individual_mean':None}
        print("Multithreaded bootstrap iterate %i/%i, seed %i" % (iterate + 1, len(bootstrap_seeds), seed))
        t0 = time.time()
        sampled_ids = [reproducible_random_sampler.choice(unique_user_ids) for i in unique_user_ids] # sample with replacement
        bootstrapped_d = pd.concat([grouped_d.get_group(user) for user in sampled_ids])
        bootstrapped_d.index = range(len(bootstrapped_d))
        t1 = time.time()
        
        if bootstrap_regression_amplitude_errorbars:
            results_for_iterate['linear_regression_bootstrapped_iterates'] = fit_linear_regression(bootstrapped_d, **regression_amplitude_kwargs)
        if bootstrap_regressions_of_period_effects:
            results_for_iterate['period_effects_with_covariates'] = fit_regression_of_period_effects(bootstrapped_d, **period_effects_kwargs)
        if bootstrap_daily_time_series:
            results_for_iterate['daily_time_series_no_individual_mean'] = compute_means_by_date(bootstrapped_d, remove_individual_means=True)
        print("Time to compute bootstrapped dataframe: %2.3f seconds; time to do analysis %2.3f seconds" % (t1 - t0, time.time() - t1))
        return_dict[seed] = results_for_iterate
        gc.collect()

def do_all_bootstrapped_analyses(d, 
                                 bootstrap_regression_amplitude_errorbars, 
                                 bootstrap_regressions_of_period_effects, 
                                 bootstrap_daily_time_series,
                                 n_iterates,
                                 regression_amplitude_kwargs, 
                                 period_effects_kwargs, 
                                 n_processes_to_use=25):
    """
    Manager job for parallelizing bootstrap analyses. 
    """
    manager = Manager()
    return_dict = manager.dict()
    processes = []
    chunk_size = int(n_iterates / n_processes_to_use)
    all_bootstrap_seeds = []
    for process_idx in range(n_processes_to_use):
        if process_idx < n_processes_to_use - 1:
            bootstrap_seeds = range(process_idx * chunk_size, (process_idx + 1) * chunk_size)
        else:
            bootstrap_seeds = range(process_idx * chunk_size, n_iterates)
        all_bootstrap_seeds += bootstrap_seeds
        args = [d, 
                bootstrap_regression_amplitude_errorbars, 
                bootstrap_regressions_of_period_effects, 
                bootstrap_daily_time_series,
                regression_amplitude_kwargs, 
                period_effects_kwargs, 
                bootstrap_seeds, 
                return_dict]
        p = Process(target=bootstrapped_analyses_one_process, args=tuple(args))
        processes.append(p)
        print("Process %i has %i seeds (%i-%i)" % (process_idx, len(bootstrap_seeds), min(bootstrap_seeds), max(bootstrap_seeds)))
    time.sleep(5)
    for process in processes:
        process.start()
    for process in processes:
        process.join()
    assert all_bootstrap_seeds == range(n_iterates)
    assert len(return_dict.keys()) == n_iterates
    bootstrapped_results = []
    for i in range(n_iterates):
        bootstrapped_results.append(return_dict[i])

    print("Completed all processes")
    return bootstrapped_results

def fit_regression_of_period_effects(d, start_day_for_period_individual_effect, end_day_for_period_individual_effect, min_obs_for_group, min_users_for_group, specifications=None):
    """
    Examines how much the period effect varies based on various covariates. 
    In the simple univariate case (eg, good_mood ~ near_period * C(country)) the estimates are the same as the 
    binary effects estimated by binary_analysis_no_individual_mean.
    """
    print("Performing linear regression of binarized period effects. First removing individual means.")
    d = deepcopy(d)
    d['year'] = d['date'].map(lambda x:x.split('-')[0])
    assert d['year'].map(lambda x:x in ['2015', '2016', '2017']).all()
    d = remove_individual_means_from_df(d)
    d = add_binary_annotations(d, start_day_for_period_individual_effect, end_day_for_period_individual_effect)

    assert pd.isnull(d['country']).sum() == 0

    good_vals_for_group = {}
    for grouping in ['categorical_age', 'country']:
        good_vals_for_group[grouping] = set()
        grouped_d = d.groupby(grouping)
        for group_id, group_df in grouped_d:
            n_obs = len(group_df)
            n_users = len(set(group_df['user_id_hash']))
            if n_obs >= min_obs_for_group and n_users >= min_users_for_group:
                good_vals_for_group[grouping].add(group_id)
            else:
                print("Not producing estimates for group %s=%s because too few users (%i) or observations (%i)" % (grouping, 
                    group_id, n_users, n_obs))
    #good_country_idxs = d['country'].map(lambda x:x in good_vals_for_group['country'])
    #good_age_idxs = pd.isnull(d['categorical_age']) | d['categorical_age'].map(lambda x:x in good_vals_for_group['categorical_age'])
    #print("Prior to removing very rare values of country and group, %i values" % len(d))
    #d = d.loc[good_age_idxs & good_country_idxs]
    #d.index = range(len(d))
              
    #print("After removing very rare values of country and group, %i values" % len(d))

    # While we use all datapoints for fitting, we only generate estimates for an age group or country if we have enough datapoints in the group. 
    all_age_groups_to_generate_estimates_for = [a for a in sorted(list(set(d['categorical_age'].dropna()))) if a in good_vals_for_group['categorical_age']]
    all_country_groups_to_generate_estimates_for = [a for a in sorted(list(set(d['country']))) if a in good_vals_for_group['country']]

    all_results = {}

    behavior_controls = ['logged_any_alcohol', 
    'logged_any_cigarettes', 
    'logged_any_exercise', 
    'logged_birth_control_pill', 
    'logged_hormonal_birth_control', 
    'logged_iud']
    app_usage_controls = ['n_symptom_categories_used', 'start_year', 'total_symptoms_logged'] 

    if specifications is None:
        specifications = ['near_period']
        for cov_set in [['country'], 
        ['categorical_age'], 
        ['country', 'categorical_age'], 
        ['country', 'categorical_age'] + behavior_controls, 
        ['country', 'categorical_age'] + behavior_controls + app_usage_controls]:
            specifications.append('+'.join(['near_period*C(%s)' % cov for cov in cov_set]))

    # sample a small subset to predict on. 100000 should give us values very close to true one but this is much faster. 
    # We fit the model on the whole dataset; we just use the small sample for computing the true 
    n_points_to_sample = min(len(d), 100000)
    random_idxs = random.sample(range(len(d)), n_points_to_sample)
    data_to_predict_on = deepcopy(d.iloc[random_idxs])
    data_to_predict_on.index = range(len(data_to_predict_on))
    original_ages = deepcopy(data_to_predict_on['categorical_age'].values)
    original_countries = deepcopy(data_to_predict_on['country'].values)
    for specification in specifications:
        print("Results for %s" % specification)
        t0 = time.time()
        model = sm.OLS.from_formula('good_mood ~ %s' % specification, data = d).fit()
        model_results = {'params':model.params, 'pvalues':model.pvalues, '95_CI':model.conf_int()}
        all_results[specification] = {}
        all_results[specification]['model_results'] = model_results
        
        # loop over countries + ages and make predictions for each. 
        for col_to_alter in ['categorical_age', 'country']:
            print("Computing expected period effects for %s" % col_to_alter)
            predicted_effects_by_group = {}
            if col_to_alter == 'categorical_age':
                groups_to_loop_over = all_age_groups_to_generate_estimates_for
                original_values = deepcopy(original_ages)
            else:
                groups_to_loop_over = all_country_groups_to_generate_estimates_for
                original_values = deepcopy(original_countries)
            for group in groups_to_loop_over:
                data_to_predict_on[col_to_alter] = group
                data_to_predict_on['near_period'] = False
                off_period_predictions = model.predict(data_to_predict_on)
                data_to_predict_on['near_period'] = True
                on_period_predictions = model.predict(data_to_predict_on)
                predicted_effect = on_period_predictions.mean() - off_period_predictions.mean()
                print("Predicted average period effect for specification %s, %s=%s: %2.3f" % (specification, 
                    col_to_alter,
                    group, 
                    predicted_effect))
                predicted_effects_by_group[group] = predicted_effect
                data_to_predict_on[col_to_alter] = deepcopy(original_values) # put things back the way they were before
            all_results[specification]['predicted_effects_by_%s' % col_to_alter] = predicted_effects_by_group



        print("Time to run regression: %2.3f seconds" % (time.time() - t0))

    return all_results

def extract_results_from_statsmodels_ranef_model(model):
    # do not want model.params. https://github.com/statsmodels/statsmodels/issues/3532
    print(model.summary())
    results = {'random_effects_covariance':model.cov_re,
            'random_effects_covariance_errors':model.bse_re,
            'fixed_effects_coefficients':model.fe_params,
            'fixed_effects_standard_errors':model.bse_fe,
            'ranef':model.random_effects}
    for k in ['fixed_effects_coefficients', 
    'fixed_effects_standard_errors', 
    'random_effects_covariance', 
    'random_effects_covariance_errors']:
        if k != 'ranef':
            print k, '\n', results[k].to_string()
    return results

def fit_mixed_model_regression(d, n_people_to_fit_on, covariates_for_regression, min_obs_per_user=1, use_lme4=False):
    """
    subsamples a dataset to fit a mixed model regression on. Robustness check. 
    """
    
    for c in covariates_for_regression:
        assert c[:2] == 'C('
    t0 = time.time()
    print("Performing mixed model regression (with individual fixed effects) for %i people" % n_people_to_fit_on)
    user_obs_counts = Counter(d['user_id_hash'])
    potential_users_to_sample = [user for user in user_obs_counts.keys() if user_obs_counts[user] >= min_obs_per_user]
    print("After filtering out users with fewer than %i obs, %i/%i users remain" % (min_obs_per_user, 
        len(potential_users_to_sample), 
        len(user_obs_counts.keys())))


    n_people_to_fit_on = min(len(potential_users_to_sample), n_people_to_fit_on)
    random_users = set(random.sample(list(potential_users_to_sample), n_people_to_fit_on))

    small_d = deepcopy(d.loc[d['user_id_hash'].map(lambda x:x in random_users)])

    print("Total number of users: %i; total number of rows %i" % (len(set(small_d['user_id_hash'])), len(small_d)))
    small_d['year'] = small_d['date'].map(lambda x:x.split('-')[0])
    assert small_d['year'].map(lambda x:x in ['2015', '2016', '2017']).all()
    
    cols_to_examine = ['good_mood', 'year', 'month', 'weekday', 'local_hour', 'date_relative_to_period', 'user_id_hash']
    assert len(small_d[cols_to_examine].dropna()) == len(small_d)
    assert np.isfinite(small_d['good_mood']).all()

    print 'fitting OLS model as a test'
    ols_model_to_test = sm.OLS.from_formula('good_mood ~ %s' % ('+'.join(covariates_for_regression)), data=small_d).fit()
    print 'Params that do not have user id hash'
    print ols_model_to_test.params.loc[ols_model_to_test.params.index.map(lambda x:'user_id_hash' not in x)]
    print("Maximum absolute value of the parameters in the ols model: %2.3f" % np.abs(ols_model_to_test.params).max())

    print small_d[cols_to_examine].head()
    for c in cols_to_examine:
        assert pd.isnull(small_d[c]).sum() == 0
        print '%s ranges from %s to %s, %i unique values' % (c, small_d[c].min(), small_d[c].max(), len(set(small_d[c])))

    if not use_lme4:
        #raise Exception("This is deprecated and not guaranteed to work.")
        print("Now fitting mixed model using statsmodels!")
        model = smf.mixedlm('good_mood ~ %s' % ('+'.join(covariates_for_regression)), 
                small_d, 
                groups = small_d["user_id_hash"])
        fit_results = model.fit(method='cg')
        print fit_results.summary()
        print 'Model.score applied to params_object'
        print model.score(fit_results.params_object)
        model_results = extract_results_from_statsmodels_ranef_model(fit_results)
    else:
        print("Fitting mixed model regression using R!")


        # confirmed that the fixed effects look similar to statsmodels and the extracted pandas dataframe 
        # look similar to the R summary.  
        import rpy2.robjects as robjects
        from rpy2.robjects.packages import importr
        from rpy2.robjects import pandas2ri
        import warnings
        #from rpy2.robjects.conversion import localconverter
        raise Exception("Not working at present. Maybe see here: https://stackoverflow.com/questions/42127093/how-to-use-the-r-with-operator-in-rpy2/42127457")
        print("Doing R imports")
        base = importr('base')
        nlme = importr('nlme')
        lme4 = importr('lme4')

        print("Activating pandas2ri")
        pandas2ri.activate()
        print("Turning warnings on")
        robjects.r('options(warn=1)') # we want to know about it if R throws warnings, so we set the warning flag high. 

        small_d.index = range(len(small_d))

        # put strings into R format. This is a pain, because I don't know how to properly deal with factors in R to Python conversion.
        
        for i in range(len(covariates_for_regression)):
            covariates_for_regression[i] = covariates_for_regression[i].replace('C(', '').replace(')', '').split(', Treatment')[0] # split is for date_relative_to_period, we want to remove it. 
            if covariates_for_regression[i] == 'date_relative_to_period':
                desired_period_vals = range(-20, 21)
                assert sorted(list(set(small_d[covariates_for_regression[i]]))) == desired_period_vals
                # truly terrible hack: map original values to ascii strings
                # the reason we do this is to make sure the base level is correct. 
                # we want it to be 0. 
                # so base level must be lexicographically first.  
                # so we take the first 21 letters of the alphabet in reverse order
                # t, s, ..., a and then add on A B C D
                # the key point is that the 21st letter must be lexicographically first.
                # because this corresponds to date_relative_to_period = 0
                period_ascii_mapping = (string.ascii_lowercase[:21][::-1] + string.ascii_uppercase)[:41]
                assert period_ascii_mapping[20] == 'a'
                small_d[covariates_for_regression[i]] = small_d[covariates_for_regression[i]].map(lambda x:period_ascii_mapping[int(x) + 20])
            else:
                small_d[covariates_for_regression[i]] = small_d[covariates_for_regression[i]].map(lambda x:'STRING_' + str(x))
        print("the covariates being used in a mixed model regression are %s" % 'good_mood~%s+(1|user_id_hash)' % ('+'.join(covariates_for_regression)))
        print("the head of the dataframe is")
        print(small_d.head())
        dfr = pandas2ri.py2ri(small_d)
        print("done converting to R dataframe")
        formula = robjects.Formula('good_mood~%s+(1|user_id_hash)' % ('+'.join(covariates_for_regression)))
        print("done converting to R formula")
        r_regression_results=lme4.lmer(formula,data=dfr)
        print("done fitting regression")
        regression_summary = base.summary(r_regression_results)
        print("done summarizing regression")
        
        coefs = regression_summary.rx2('coefficients')
        pd_coefs = pd.DataFrame(np.array(coefs), columns=list(coefs.colnames), index=list(coefs.rownames))
        model_results = {'random_effects_covariance':None,
            'random_effects_covariance_errors':None,
            'fixed_effects_coefficients':pd_coefs['Estimate'],
            'fixed_effects_standard_errors':pd_coefs['Std. Error'],
            'ranef':None, 
            'n_obs':len(small_d), 
            'n_groups':len(set(small_d['user_id_hash']))}
        for k in ['fixed_effects_coefficients', 'fixed_effects_standard_errors']:
            print(k)
            def process_R_regression_coefs_into_standard_format(x):
                if 'Intercept' in x:
                    return x
                elif 'date_relative_to_period' in x:
                    reverse_mapping = dict(zip(period_ascii_mapping, desired_period_vals))
                    return 'C(date_relative_to_period)[T.%.1f]' % reverse_mapping[x.replace('date_relative_to_period', '')]
                return 'C(' + x.replace('STRING_', ')[T.') + ']'
            model_results[k].index = model_results[k].index.map(process_R_regression_coefs_into_standard_format)
            for date_relative_to_period in range(-20, 21):
                coef_name = 'C(date_relative_to_period)[T.%2.1f]' % date_relative_to_period 
                if date_relative_to_period == 0:
                    assert coef_name not in list(model_results[k].index)
                else:
                    assert coef_name in list(model_results[k].index)
            print model_results[k].to_string()
        
    print "Successfully performed mixed model regression. Total time taken %2.3f seconds" % (time.time() - t0)

    return model_results

def fit_logistic_regression(d):
    """
    checked. Performs a logistic regression on day of year, weekday, and period date. 
    """
    d = deepcopy(d)
    polynomial_order = 5
    d['day_of_year'] = d['date'].map(lambda x:datetime.datetime.strptime(x.split()[0], '%Y-%m-%d').timetuple().tm_yday)
    d['good_mood'] = d['good_mood'] * 1.

    formula_string = 'good_mood ~ weekday'
    for poly in range(1, polynomial_order + 1):
        d['period_x_%i' % poly] = d['date_relative_to_period'] ** poly
        d['day_of_year_%i' % poly] = d['day_of_year'] ** poly
        d['utc_hour_%i' % poly] = d['utc_hour'] ** poly
        formula_string = formula_string + ' + period_x_%i + day_of_year_%i + utc_hour_%i' % (poly, poly, poly)

    model = sm.Logit.from_formula(formula_string, data = d).fit()
    print model.summary()
    model_results = {'params':model.params, 'pvalues':model.pvalues}
    return model_results

def get_covariates_for_regression(symptom_key):
    assert type(symptom_key) is str
    assert '_versus_' in symptom_key
    covariates_for_regression = ['C(year)', 
    'C(month)',  
    'C(weekday)', 
    'C(local_hour)', 
    "C(date_relative_to_period, Treatment(reference=0))"]

    if HEART_SUBSTRING in symptom_key:
        # don't need year covariates for heart data. 
        covariates_for_regression.remove('C(year)')
    no_hourly_list = [WEIGHT_SUBSTRING, HEART_SUBSTRING, BBT_SUBSTRING, 'sex*', 'sleep*', 'exercise*']
    for substring in no_hourly_list:
        if substring in symptom_key:
            covariates_for_regression.remove('C(local_hour)')
    print 'The regression covariates for %s are' % symptom_key
    print covariates_for_regression

    return covariates_for_regression

def return_results_of_analyses_as_dictionary(d, 
                                             symptom_key,
                                             compute_interactions, 
                                             do_regression, 
                                             compute_individual_level_effects, 
                                            compute_daily_means, 
                                            compute_derivative_estimates, 
                                            compute_individual_effects_using_median_method, 
                                            do_mixed_model_regression, 
                                            do_binary_analysis,
                                            do_regressions_of_period_effects,
                                            do_regression_with_alternate_seasonal_specifications,
                                            bootstrap_regression_amplitude_errorbars,
                                            do_regression_with_alternate_period_specifications,
                                            start_day_for_period_individual_effect=None, 
                                            end_day_for_period_individual_effect=None, 
                                            bootstrap_daily_time_series=False):
    """
    performs the basic analyses on a dataframe and returns the results in a dictionary. Checked. 
    """
    covariates_for_regression = get_covariates_for_regression(symptom_key) # Eg, we do not want to regress on year for HR. 

    results = {
            'take_simple_means_by_group':take_simple_means_by_group(d, remove_individual_means = False), 
            'take_simple_means_by_group_no_individual_mean':take_simple_means_by_group(d, remove_individual_means = True),
            'overall_positive_frac':d['good_mood'].mean(), 
            'overall_n_obs':len(d), 
            'overall_n_users':len(set(d['user_id_hash'])), 
            'unique_user_ids':set(d['user_id_hash']),
            'start_day_for_period_individual_effect':start_day_for_period_individual_effect, 
            'end_day_for_period_individual_effect':end_day_for_period_individual_effect
    }
    if do_regressions_of_period_effects:
        results['period_effects_with_covariates'] = fit_regression_of_period_effects(d, start_day_for_period_individual_effect, end_day_for_period_individual_effect, 
            min_users_for_group=MIN_USERS_FOR_SUBGROUP, 
            min_obs_for_group=MIN_OBS_FOR_SUBGROUP)

    # put all our bootstrapping in one place. 
    if bootstrap_regression_amplitude_errorbars or do_regressions_of_period_effects or bootstrap_daily_time_series:
        bootstrap_regressions_of_period_effects = do_regressions_of_period_effects
        bootstrapped_results = do_all_bootstrapped_analyses(d, 
            bootstrap_regression_amplitude_errorbars, 
            bootstrap_regressions_of_period_effects, 
            bootstrap_daily_time_series,
            n_iterates=N_BOOTSTRAP_ITERATES,
            regression_amplitude_kwargs={'covariates_for_regression':covariates_for_regression}, 
            period_effects_kwargs={'start_day_for_period_individual_effect':start_day_for_period_individual_effect, 
                                   'end_day_for_period_individual_effect':end_day_for_period_individual_effect, 
                                   'min_users_for_group':MIN_USERS_FOR_SUBGROUP, 
                                   'min_obs_for_group':MIN_OBS_FOR_SUBGROUP, 
                                   'specifications':['near_period*C(categorical_age)']})
        if do_regressions_of_period_effects:
            results['period_effects_with_covariates']['bootstrapped_iterates'] = [a['period_effects_with_covariates'] for a in bootstrapped_results]
        results['linear_regression_bootstrapped_iterates'] = [a['linear_regression_bootstrapped_iterates'] for a in bootstrapped_results]
        results['bootstrapped_daily_time_series'] = [a['daily_time_series_no_individual_mean'] for a in bootstrapped_results]
    if do_regression:
        results['linear_regression'] = fit_linear_regression(d, covariates_for_regression)
        
    if do_regression_with_alternate_seasonal_specifications:
        results['linear_regression_with_alternate_seasonal_specifications'] = fit_regression_with_alternate_seasonal_specifications(d, covariates_for_regression)
    if do_regression_with_alternate_period_specifications:
        results['linear_regression_with_alternate_period_specifications'] = fit_regression_with_alternate_period_specifications(d, covariates_for_regression)
    if compute_interactions:
        results['interaction_between_cycles'] = find_interaction_between_cycles(d, 
            remove_individual_means = True, 
            start_day_for_period_individual_effect=start_day_for_period_individual_effect, 
            end_day_for_period_individual_effect=end_day_for_period_individual_effect)
        #results['person_specific_interaction_between_cycles'] = find_person_specific_interactions_between_cycles(d)
    if do_mixed_model_regression:
        results['mixed_model_regression'] = fit_mixed_model_regression(d, 100000, covariates_for_regression)
    if compute_individual_level_effects:
        results['individual_level_cycle_effects'] = fast_compute_individual_level_cycle_effects(d, 
            start_day_for_period_individual_effect, 
            end_day_for_period_individual_effect)
    if compute_daily_means:
        results['means_by_date'] = compute_means_by_date(d, remove_individual_means=False)
        results['means_by_date_no_individual_mean'] = compute_means_by_date(d, remove_individual_means=True)
    if compute_derivative_estimates:
        looparound_derivative_estimates, no_looparound_derivative_estimates = \
        estimate_derivatives_for_all_four_types_of_cycles(d)
        results['derivative_estimates_with_looparound'] = looparound_derivative_estimates
        results['derivative_estimates_with_no_looparound'] = no_looparound_derivative_estimates
    if compute_individual_effects_using_median_method:
        results['individual_effects_using_median_method'] = compute_individual_level_cycle_effects_for_period_by_splitting_into_two_and_taking_median(d)
    if do_binary_analysis:
        results['binary_analysis'] = compute_diffs_in_binary_variable(d, 
            remove_individual_means = False, 
            start_day_for_period_individual_effect=start_day_for_period_individual_effect,
            end_day_for_period_individual_effect=end_day_for_period_individual_effect)
        results['binary_analysis_no_individual_mean'] = compute_diffs_in_binary_variable(d, 
            remove_individual_means=True,
            start_day_for_period_individual_effect=start_day_for_period_individual_effect,
            end_day_for_period_individual_effect=end_day_for_period_individual_effect)
        
    return results

def do_analyses_on_big_symptom_dataframe(all_symptom_groups):
    """
    Do regression and simple mean analysis on all symptom data. 
    Checked. 
    """
    period_effect_bin_size = 7

    
    # outer loop: symptoms to look at. Save each result in a separate file. 
    for symptom_group in all_symptom_groups:
        # note that symptom_group is a list here, not a string. 
        all_results = {}
        d = dataprocessor.load_dataframe_for_symptom_group(symptom_group, chunks_to_use)
        d['season'] = d['month'].map(get_season)

        if len(symptom_group) == 1:
            assert 'continuous_features' in symptom_group[0]
            good_symptom = symptom_group[0]
            bad_symptom = 'continuous_features*null'
            compute_derivative_estimates = False
            if WEIGHT_SUBSTRING in good_symptom:
                compute_individual_effects_using_median_method = False
            else:
                compute_individual_effects_using_median_method = True
        else:
            good_symptom, bad_symptom = symptom_group
            compute_derivative_estimates = False
            compute_individual_effects_using_median_method = False
        symptom_key = '%s_versus_%s' % (good_symptom, bad_symptom)
        all_results[symptom_key] = {}

        # compute basic statistics on the users we remove prior to any filtering (this is just a robustness check to show we're not hiding anything and they have cycles too)
        all_results[symptom_key]['ALL_OMITTED_USERS_ROBUSTNESS_CHECK_ONLY'] = {}
        omitted_data = d.loc[pd.isnull(d['very_active_loggers'])]
        print 'Doing analysis on %i rows of omitted data' % len(omitted_data)
        all_results[symptom_key]['ALL_OMITTED_USERS_ROBUSTNESS_CHECK_ONLY'][True] = return_results_of_analyses_as_dictionary(omitted_data,
            symptom_key=symptom_key,
            compute_interactions=False,
            do_regression=True,
            compute_individual_level_effects=False,
            compute_daily_means=True,
            compute_derivative_estimates=False,
            compute_individual_effects_using_median_method=False,
            do_mixed_model_regression=False,
            do_binary_analysis=False,
            do_regressions_of_period_effects=False, 
            do_regression_with_alternate_seasonal_specifications=False, 
            bootstrap_regression_amplitude_errorbars=False, 
            do_regression_with_alternate_period_specifications=False)
              


        if FILTER_FOR_VERY_ACTIVE_LOGGERS_IN_ALL_ANALYSIS:
            print("Warning! Filtering for very active loggers.")
            print("Prior to filtering, %i rows, %i users." % (len(d), len(set(d['user_id_hash']))))
            d = d.loc[d['very_active_loggers'] == True]
            d.index = range(len(d))
            print("After filtering, %i rows, %i users." % (len(d), len(set(d['user_id_hash']))))

        
        # First do all results (no substratification)
        # first we have to compute the most dramatic period bin. 
        print("Now computing bin with most dramatic period effect using regression data + bin size of %i" % period_effect_bin_size)
        regression_covariates = get_covariates_for_regression(symptom_key)
        regression_results = fit_linear_regression(d, regression_covariates)


        data = deepcopy(convert_regression_format_to_simple_mean_format(regression_results, 'linear_regression'))

        data = data['date_relative_to_period']
        start_day_for_period_individual_effect = extract_most_dramatic_period_bin(data, bin_size=period_effect_bin_size)
        end_day_for_period_individual_effect = start_day_for_period_individual_effect + period_effect_bin_size

        all_results[symptom_key]['no_substratification'] = return_results_of_analyses_as_dictionary(d, 
                                                           symptom_key=symptom_key,
                                                           compute_interactions = True, 
                                                           do_regression = True, 
                                                           compute_individual_level_effects = True, 
                                                           compute_daily_means=True, 
                                                           compute_derivative_estimates=compute_derivative_estimates, 
                                                           compute_individual_effects_using_median_method=compute_individual_effects_using_median_method, 
                                                           do_mixed_model_regression=True,
                                                           do_binary_analysis=True,
                                                           start_day_for_period_individual_effect=start_day_for_period_individual_effect,
                                                           end_day_for_period_individual_effect=end_day_for_period_individual_effect,
                                                           do_regressions_of_period_effects=True, 
                                                           do_regression_with_alternate_seasonal_specifications=True, 
                                                           bootstrap_regression_amplitude_errorbars=False, 
                                                           do_regression_with_alternate_period_specifications=True)



        # Now look at results broken down by substratification. 
        for col in COLS_TO_STRATIFY_BY:
            substratifications_to_results = {} # dictionary that maps substratification levels to results. 
            c = Counter(d[col].dropna())
            for val in c.keys():
                if c[val] < 1000:
                    print("Skipping %s=%s because too few values (%i)" % (col, val, c[val]))
                    continue
                idxs = d[col] == val
                assert np.isnan(idxs).sum() == 0
                substratification_d = deepcopy(d.loc[idxs])
                print '\nAnalyzing substratification %s = %s (%i points)' % (col, val, idxs.sum())
                if col == 'very_active_northern_hemisphere_loggers':
                    substrat_results = return_results_of_analyses_as_dictionary(substratification_d, 
                                                           symptom_key=symptom_key,
                                                           compute_interactions = True, 
                                                           do_regression = True, 
                                                           compute_individual_level_effects = True, 
                                                           compute_daily_means=True, 
                                                           compute_derivative_estimates=compute_derivative_estimates, 
                                                           compute_individual_effects_using_median_method=compute_individual_effects_using_median_method, 
                                                           do_mixed_model_regression=True,
                                                           do_binary_analysis=True,
                                                           start_day_for_period_individual_effect=start_day_for_period_individual_effect,
                                                           end_day_for_period_individual_effect=end_day_for_period_individual_effect,
                                                           do_regressions_of_period_effects=False, 
                                                           do_regression_with_alternate_seasonal_specifications=True, 
                                                           bootstrap_regression_amplitude_errorbars=True, 
                                                           do_regression_with_alternate_period_specifications=True, 
                                                           bootstrap_daily_time_series=True)
                else:
                    substrat_results = return_results_of_analyses_as_dictionary(substratification_d,
                                                                                symptom_key=symptom_key,
                                                                                compute_interactions = False,
                                                                                do_regression = True,
                                                                                compute_individual_level_effects = False,
                                                                                compute_daily_means=True, 
                                                                                compute_derivative_estimates=False, 
                                                                                compute_individual_effects_using_median_method=False, 
                                                                                do_mixed_model_regression=False,
                                                                                start_day_for_period_individual_effect=start_day_for_period_individual_effect,
                                                                                end_day_for_period_individual_effect=end_day_for_period_individual_effect,
                                                                                do_binary_analysis=True, 
                                                                                do_regressions_of_period_effects=False, 
                                                                                do_regression_with_alternate_seasonal_specifications=False, 
                                                                                bootstrap_regression_amplitude_errorbars=False, 
                                                                                do_regression_with_alternate_period_specifications=False)
                    
                substratifications_to_results[val] = substrat_results
                    
            all_results[symptom_key]['by_' + col] = substratifications_to_results
        very_active_logger_string = 'VERY_ACTIVE_LOGGERS_' if FILTER_FOR_VERY_ACTIVE_LOGGERS_IN_ALL_ANALYSIS else ''
        file_handle = open(os.path.join(base_results_dir, '%s%s_versus_%s_n_chunks_to_use_%i.pkl' % (very_active_logger_string, good_symptom, bad_symptom, n_chunks_to_use)), 'wb')
        cPickle.dump(all_results, file_handle)
        file_handle.close()
    print("Successfully completed analysis of symptoms.")

    
def multithreaded_do_analyses_on_big_symptom_dataframe():
    """
    Checked. Runs multiple symptom pairs at one time. 
    """
    raise Exception("This is deprecated!")
    n_processes = len(ALL_SYMPTOM_GROUPS)
    processes = []
    
    for process_idx in range(n_processes):
        pairs_for_process = deepcopy(ALL_SYMPTOM_GROUPS[process_idx::n_processes])
        print(pairs_for_process)
        p = Process(target=do_analyses_on_big_symptom_dataframe, 
                    kwargs={'all_symptom_groups':pairs_for_process})
        processes.append(p)
    for process in processes:
        process.start()
    for process in processes:
        process.join()
    print("All processes completed successfully")

def get_jobs_that_crashed():
    jobs_to_rerun = []
    for i in range(len(ALL_SYMPTOM_GROUPS)):
        completed_successfully = False
        f = open('%s/compare_to_seasonal_cycles_symptom_group_%i.out' % (base_processing_outfile_dir, i))
        for line in f:
            if 'Successfully completed analysis of symptoms.' in line:
                completed_successfully = True
        if not completed_successfully:
            print("Job %i must be rerun." % i)
            jobs_to_rerun.append(i)
        else:
            print("Job %i finished successfully." % i)
    return jobs_to_rerun

def run_single_symptom_group(i):
    os.system('nohup python -u compare_to_seasonal_cycles.py %i > %s/compare_to_seasonal_cycles_symptom_group_%i.out &' % (i, base_processing_outfile_dir, i))

if __name__ == '__main__':
    os.environ["OMP_NUM_THREADS"]="5"
    if len(sys.argv) == 1:
        for i in range(len(ALL_SYMPTOM_GROUPS)):
            run_single_symptom_group(i)
    elif len(sys.argv) == 2:
        if sys.argv[1] == 'just_jobs_that_didnt_finish':
            jobs_to_rerun = get_jobs_that_crashed()
            print "Rerunning jobs"
            print(jobs_to_rerun)
            for i in jobs_to_rerun:
                run_single_symptom_group(i)
        else:
            i = int(sys.argv[1])
            do_analyses_on_big_symptom_dataframe([ALL_SYMPTOM_GROUPS[i]])
    else:
        raise Exception("must be called with 0 or 1 arguments.")

    """
    opposite_symptom_list = [line.split(',') for line in open('clue_opposite_symptom_list.csv').read().split('\n')]
    d = pd.read_csv(os.path.join(base_clue_data_dir, 'medium_symptoms.csv'))
    users_to_starts = make_users_to_starts()
    processes = []
    for process_idx in range(len(opposite_symptom_list)):
        args = [d, users_to_starts, opposite_symptom_list[process_idx][0], opposite_symptom_list[process_idx][1]]
        p = Process(target=fit_model, args=tuple(args))
        processes.append(p)
    for process in processes:
        process.start()
    for process in processes:
        process.join()
    """



