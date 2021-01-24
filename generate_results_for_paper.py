
import pandas as pd
from traceback import print_exc
import random
import numpy as np
from scipy.signal import argrelextrema
import statsmodels.api as sm
import warnings
import statsmodels.formula.api as smf
from constants_and_util import *
from matplotlib.colors import LinearSegmentedColormap
from copy import deepcopy
from scipy.stats import scoreatpercentile, norm
import json
from IPython import embed
from collections import Counter
from scipy.stats import pearsonr, linregress
import time
import string
import cPickle
import math
import seaborn as sns
import dataprocessor
from scipy.special import expit
from multiprocessing import Process
from copy import deepcopy
import datetime
import matplotlib.pyplot as plt
import sys

#from mpl_toolkits.basemap import Basemap
import matplotlib as mpl
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from matplotlib.patches import PathPatch

import matplotlib.gridspec as gridspec
import compare_to_seasonal_cycles

"""
This file makes custom plots and tables for the paper. 
Some plots in the paper are also made using compare_to_seasonal_cycles 
because they are more general plotting utilities. 

The functions in this file are called by IPython notebooks. 
"""

def make_maps_of_countries_with_clue_data():
    country_counts = cPickle.load(open(os.path.join(base_results_dir, 
                                                 'country_counts_in_processed_data.pkl')))

    vals_to_plot = {}
    for c in country_counts:
        if (country_counts[c]['n_users'] >= MIN_USERS_FOR_SUBGROUP) & (country_counts[c]['n_obs'] >= MIN_OBS_FOR_SUBGROUP):
            vals_to_plot[c] = 1
    compare_to_seasonal_cycles.make_basemap(vals_to_plot, 
                                            bin_edges=[-2, 0, 2], 
                                            bin_edge_labels=[-2, 0, 2], 
                                            title_string='%i countries with at least %i users and %i obs' % (len(vals_to_plot), MIN_USERS_FOR_SUBGROUP, MIN_OBS_FOR_SUBGROUP), 
                                            filename='figures_for_paper/countries_with_clue_data.png', 
                                            plot_colorbar=False)

def make_figure_to_illustrate_data_for_one_user():
    """
    Use purely synthetic data to illustrate what the data looks like. 
    """
    symptom_categories = ['mental', 'social', 'sex', WEIGHT_SUBSTRING, HEART_SUBSTRING, 'period'] # BBT_SUBSTRING # 'sleep', 'exercise'
    symptom_counts = cPickle.load(open(os.path.join(base_results_dir, 'symptom_counts_in_processed_data.pkl'), 'rb'))
    fig, all_axes = plt.subplots(len(symptom_categories), 1, figsize=[7.5, 10]) # [7.5, 15]

    fontsize = 16

    rng = random.Random(28)

    colors_for_cmap = [SEASONAL_CYCLE_COLOR, PERIOD_CYCLE_COLOR]
    cm_fxn = LinearSegmentedColormap.from_list(
        'mycmap', colors_for_cmap, N=256)
    period_start_days = [28, 56]
    xticks = range(0, 85, 14)

    for subplot_idx, symptom_category in enumerate(symptom_categories):
        symptoms_in_category = sorted([make_name_pretty_for_table(a.split('*')[1]) for a in symptom_counts.keys() if (symptom_category + '*') in a])
        if symptom_category == 'period':
            symptoms_in_category = ['spotting', 'light', 'medium', 'heavy']
        xs = []
        ys = []
        max_day = 84
        
        if symptom_category not in [WEIGHT_SUBSTRING, HEART_SUBSTRING, BBT_SUBSTRING]:
            assert len(symptoms_in_category) == 4
            if symptom_category == 'period':
                xs = [28, 29, 30, 31, 32, 38, 45, 56, 57, 58, 59]
                ys = [1, 1, 2, 3, 2, 0, 0, 1, 2, 3, 1]
                period_mapping_colors = ['salmon', 'red', 'crimson', 'darkred']
                colors = [period_mapping_colors[y] for y in ys]
                all_axes[subplot_idx].set_xlabel("Day", fontsize=fontsize, fontweight='bold')
                all_axes[subplot_idx].set_xticklabels(xticks, fontsize=fontsize, 
                    fontweight='bold')
            else:
                colors = HOUR_CYCLE_COLOR
                for i in range(1, max_day):
                    n_plotted_for_day = 0
                    for j in range(len(symptoms_in_category)):
                        if symptom_category == 'sleep' and n_plotted_for_day > 0:
                            # only one sleep log per day allowed. 
                            continue
                        if symptom_category == 'sleep' and j == 0:
                            # no one really sleeps 0-3 hours, do they?
                            continue
                        if n_plotted_for_day > 1:
                            # this just looks weird + unrealistic
                            continue
                        if (i % 28) <= 3 or (i % 28) >= 25:
                            # people log more on their period
                            logging_prob = .3
                        else:
                            logging_prob = .05
                        if rng.random() < logging_prob:
                            xs.append(i)
                            ys.append(j)
                            n_plotted_for_day += 1
                all_axes[subplot_idx].set_xticklabels([])
            
            all_axes[subplot_idx].set_ylim([-.5, 3.5])
            all_axes[subplot_idx].set_yticks(range(len(symptoms_in_category)))
            all_axes[subplot_idx].set_yticklabels(symptoms_in_category, fontsize=fontsize,fontweight='bold')
            all_axes[subplot_idx].grid(b=True, which='major')
            all_axes[subplot_idx].scatter(xs, ys, color=colors)
            all_axes[subplot_idx].set_ylabel(symptom_category, fontsize=fontsize, fontweight='bold')
            for period_day in period_start_days:
                all_axes[subplot_idx].plot([period_day, period_day], [-.5, 3.5], color='black')
        else:
            if symptom_category == WEIGHT_SUBSTRING:
                all_axes[subplot_idx].set_ylabel('Weight\n(LB)', fontsize=fontsize,fontweight='bold')
                xs = rng.sample(range(1, max_day), 30)
                ys = [((x % 28) > 20) * .4 + 135.2 + rng.choice([.1, .2, .3]) for x in xs]
                yticks = [135, 135.5, 136]
            elif symptom_category == BBT_SUBSTRING:
                all_axes[subplot_idx].set_ylabel('BBT\n(deg F)', fontsize=fontsize,fontweight='bold')
                xs = rng.sample(range(1, max_day), 30)
                ys = [((x % 28) > 14) * .4 + 97.6 + rng.choice([.1, .2, .3]) for x in xs]
                yticks = [97.5, 98, 98.5]
            elif symptom_category == HEART_SUBSTRING:
                all_axes[subplot_idx].set_ylabel('RHR\n(BPM)', fontsize=fontsize,fontweight='bold')
                xs = range(1, max_day)
                ys = [((x % 28) > 22) * 2 + 65 + rng.choice(range(3)) for x in xs]
                yticks = [64, 67, 70]
            all_axes[subplot_idx].set_yticks(yticks)
            all_axes[subplot_idx].set_yticklabels(yticks, fontsize=fontsize, fontweight='bold')
            y_min = min(yticks)
            y_max = max(yticks)
            for period_day in period_start_days:
                all_axes[subplot_idx].plot([period_day, period_day], [y_min, y_max], color='black')

            all_axes[subplot_idx].set_ylim([y_min, y_max])
            colors = [cm_fxn((1.0*y - y_min) / (y_max - y_min)) for y in ys]
            all_axes[subplot_idx].grid(b=True, which='major')
            all_axes[subplot_idx].scatter(xs, ys, color=colors)
            all_axes[subplot_idx].set_xticklabels([])

        all_axes[subplot_idx].set_xlim([0, max_day])
        all_axes[subplot_idx].set_xticks(xticks)
        
    fig.align_ylabels(all_axes)
    fig.subplots_adjust(left=.35)
    plt.savefig('figures_for_paper/single_user_data_plot.png', dpi=300)

def compare_christmas_and_trump_effects_to_period_effect(period_effects_data, means_by_date):
    period_effects = compare_to_seasonal_cycles.convert_regression_format_to_simple_mean_format(period_effects_data, 'linear_regression')
    period_effects = period_effects['date_relative_to_period']['mean']
    period_effects = period_effects.loc[period_effects.index.map(lambda x:np.abs(x) <= 14)].values
    period_effects = (period_effects - period_effects.mean()) * 100
    period_amp = (period_effects.max() - period_effects.min())
    #print 'Amplitude of period effect: %2.3f' % period_amp

    # trump effect: the difference between Nov 9 and other dates in Nov 2016. 
    trump_effect = 100 * (means_by_date.loc[means_by_date['date'] == '2016-11-09', 'mean'].iloc[0] - 
                    means_by_date.loc[means_by_date['date'].map(lambda x:'2016-11' in x), 'mean'].mean())

    christmas_effects = []
    for year in ['2015', '2016']:
        christmas_effects.append(100 * (means_by_date.loc[means_by_date['date'] == '%s-12-25' % year, 'mean'].iloc[0] - 
                    means_by_date.loc[means_by_date['date'].map(lambda x:'%s-12' % year in x), 'mean'].mean()))

    #print "Christmas effect %s is %2.5f, %2.5fx period amplitude" % (year, christmas_effect, christmas_effect/period_amp)
    #print "Trump effect is %2.5f, %2.5fx period amplitude" % (trump_effect, np.abs(trump_effect/period_amp))

    return {'christmas_effect':np.mean(christmas_effects), 
            'period_amp':period_amp, 
            'trump_effect':trump_effect, 
            'period_effects':period_effects, 
            'period_trump_ratio':period_amp/trump_effect, 
            'period_christmas_ratio':period_amp/np.mean(christmas_effects)}


def make_happiness_by_date_date_trump_effects_plot_for_figure_1(results, plot_red_line=True, opposite_pair=None):
    """
    Make figure of happiness by date that shows outliers like Christmas + Trump effect. 
    """
    # throughout we use very active Northern hemisphere loggers for consistency with other plots. 
    # and we plot happy vs sad. 
    substratification = 'by_very_active_northern_hemisphere_loggers'
    substratification_level = True
    if opposite_pair is None:
        opposite_pair = 'emotion*happy_versus_emotion*sad'

    # first get period effects for counter-factual.
    # use linear regression for consistency with other analyses.  



    period_effects_data = (results[opposite_pair]
     [substratification]
     [substratification_level]
     ['linear_regression'])
    

    # now plot signal over time. 
    means_by_date = (results[opposite_pair]
                    [substratification]
                    [substratification_level]
                    ['means_by_date_no_individual_mean'])

    original_estimates = compare_christmas_and_trump_effects_to_period_effect(period_effects_data, means_by_date)
    bootstrapped_estimates = []
    for i in range(N_BOOTSTRAP_ITERATES):
        bootstrapped_estimates.append(compare_christmas_and_trump_effects_to_period_effect(
            period_effects_data=results[opposite_pair][substratification][substratification_level]['linear_regression_bootstrapped_iterates'][i], 
            means_by_date=results[opposite_pair][substratification][substratification_level]['bootstrapped_daily_time_series'][i]))
    for k in original_estimates.keys():
        if k != 'period_effects':
            bootstrapped_CIs = bootstrap_CI([a[k] for a in bootstrapped_estimates], original_estimates[k])
            print("Original estimate for %s: %2.5f; 95%% CI, (%2.5f, %2.5f)" % (
                k, 
                original_estimates[k], 
                bootstrapped_CIs['lower_95_CI'], 
                bootstrapped_CIs['upper_95_CI']))
            if 'ratio' in k:
                print("Fraction of bootstrapped iterates > 1: %2.5f" % np.mean([a[k] > 1 for a in bootstrapped_estimates]))

    #lksjdf

    if plot_red_line:
        figname = 'figures_for_paper/means_by_date.png'
        period_effects_vector = original_estimates['period_effects']
    else:
        figname = 'figures_for_paper/means_by_date_no_period_line.png'
        period_effects_vector = None
    compare_to_seasonal_cycles.plot_means_by_date(results, 
        [opposite_pair], 
        data_to_use='means_by_date_no_individual_mean', 
                           min_date='2015-01-01', 
                           max_date='2018-01-01',
                          substratification=substratification, 
                          substratification_level=substratification_level, 
                          outliers_to_highlight=['2015-12-25', '2016-12-25', '2016-11-09'],
                          min_obs=1000,
                        ylimit=8, # 12
                        period_effects_vector=period_effects_vector, 
                        figname=figname)

def make_cycle_amplitudes_bar_plot_for_figure_1(results):
    """
    Make bar graph of all the cycle amplitudes. 
    """

    fig = plt.figure(figsize=[24, 5])
    substratification = 'by_very_active_northern_hemisphere_loggers'
    substratification_level = True
    data_source_to_use = 'linear_regression'
    metric_to_use = 'max_minus_min'
    hourly_period_to_exclude = None

    behavior_symptoms = ['sex*had_sex_versus_sex*did_not_have_sex',
                        'exercise*exercised_versus_exercise*did_not_exercise', 
                         'sleep*slept_6_hours_or_more_versus_sleep*slept_6_hours_or_less']

    mood_symptoms = ['emotion*happy_versus_emotion*sad',
                    'emotion*happy_versus_emotion*sensitive_emotion',
                     'energy*energized_versus_energy*exhausted', 
                     'motivation*motivated_versus_motivation*unmotivated',
                     'social*supportive_social_versus_social*conflict_social',
                      'motivation*productive_versus_motivation*unproductive',
                     'mental*calm_versus_mental*stressed',
                     'mental*focused_versus_mental*distracted',
                     'social*sociable_versus_social*withdrawn_social']

    vital_sign_symptoms = ['continuous_features*heart_rate_versus_continuous_features*null', 
                           'continuous_features*bbt_versus_continuous_features*null', 
                           'continuous_features*weight_versus_continuous_features*null']

    assert sorted(mood_symptoms + behavior_symptoms + vital_sign_symptoms) == sorted(results.keys())
    assert sorted(mood_symptoms + behavior_symptoms + vital_sign_symptoms) == sorted(ORDERED_SYMPTOM_NAMES)

    # set up plot as a grid. 
    gridsize = tuple([1, 19])
    gs1 = gridspec.GridSpec(*gridsize, width_ratios=[1] * 16 + [10] * 3)
    
    # mood symptoms
    plt.subplot2grid(gridsize, (0, 0), colspan=9)
    make_bar_graph_of_total_cycle_variation(results, 
                                            opposite_pairs=mood_symptoms,
                                            metric_to_use=metric_to_use,
                                            title_string='Mood\n',
                                            substratification=substratification, 
                                            substratification_level=substratification_level, 
                                            yticks=[0, 5, 10], 
                                            ytick_labels=['0%', '5%', '10%'],
                                            ymax=11, 
                                            yaxis_label=True, 
                                            hourly_period_to_exclude=hourly_period_to_exclude, 
                                            data_source_to_use=data_source_to_use)

    plt.legend(prop={'size':15}, # 'weight':'bold'
     bbox_transform=fig.transFigure, bbox_to_anchor=(.92, .73))

    # behavior symptoms
    plt.subplot2grid(gridsize, (0, 10), colspan=3)
    make_bar_graph_of_total_cycle_variation(results, 
                                            opposite_pairs=behavior_symptoms,
                                            cycles_to_plot=['date_relative_to_period', 'weekday', 'month'],
                                            title_string='Behavior\n',
                                            metric_to_use=metric_to_use,
                                            substratification=substratification, 
                                            substratification_level=substratification_level, 
                                            legend=False,
                                            yticks=[0, 10, 20], 
                                            ytick_labels=['0%', '10%', '20%'],
                                            ymax=20, 
                                            yaxis_label=False, 
                                            hourly_period_to_exclude=None, 
                                            data_source_to_use=data_source_to_use)
    
    # vital signs: BBT
    plt.subplot2grid(gridsize, (0, 14))
    make_bar_graph_of_total_cycle_variation(results, 
                                            opposite_pairs=['continuous_features*bbt_versus_continuous_features*null'],
                                            cycles_to_plot=['date_relative_to_period', 'weekday', 'month'],
                                            title_string='',
                                            metric_to_use=metric_to_use,
                                            substratification=substratification, 
                                            substratification_level=substratification_level, 
                                            legend=False,
                                            draw_xticks=True,
                                            yticks=[0, .4, .8], 
                                            ytick_labels=[0, .4, .8],
                                            ymax=.8, 
                                            yaxis_label=False,
                                            hourly_period_to_exclude=None, 
                                            data_source_to_use=data_source_to_use)

    # vital signs: Weight
    plt.subplot2grid(gridsize, (0, 15))
    make_bar_graph_of_total_cycle_variation(results, 
                                            opposite_pairs=['continuous_features*weight_versus_continuous_features*null'],
                                            cycles_to_plot=['date_relative_to_period', 'weekday', 'month'],
                                            title_string='',
                                            metric_to_use=metric_to_use,
                                            substratification=substratification, 
                                            substratification_level=substratification_level, 
                                            legend=False,
                                            draw_xticks=True,
                                            yticks=[0, .3, .6], 
                                            ytick_labels=[0, .3, .6],
                                            ymax=.6, 
                                            yaxis_label=False,
                                            hourly_period_to_exclude=None, 
                                            data_source_to_use=data_source_to_use)

    # vital signs: Heart Rate
    plt.subplot2grid(gridsize, (0, 16))
    make_bar_graph_of_total_cycle_variation(results, 
                                            opposite_pairs=['continuous_features*heart_rate_versus_continuous_features*null'],
                                            cycles_to_plot=['date_relative_to_period', 'weekday'],
                                            title_string='',
                                            metric_to_use=metric_to_use,
                                            substratification=substratification, 
                                            substratification_level=substratification_level, 
                                            legend=False,
                                            draw_xticks=True,
                                            yticks=[0, 1, 2, 3], 
                                            ytick_labels=[0, 1, 2, 3],
                                            ymax=3, 
                                            yaxis_label=False,
                                            hourly_period_to_exclude=None, 
                                            data_source_to_use=data_source_to_use)



    # Messily adjust spacing. I don't think I got this entirely right but the plot looks ok, so...
    gs1.set_width_ratios([1] * 18 + [2])
    plt.subplots_adjust(wspace=1.3, top=.8, bottom=.4)
    plt.text(.73, .82, "Vital signs\n", fontsize=20, transform=fig.transFigure) # title for last plot.
    plt.savefig('figures_for_paper/amplitude_plot.png', dpi=300)
    plt.show()

def make_bar_graph_of_total_cycle_variation(results, 
                                            opposite_pairs,
                                            metric_to_use,
                                            title_string,
                                            cycles_to_plot=None,
                                            substratification=None, 
                                            substratification_level=None, 
                                            legend=True,
                                            draw_xticks=True,
                                            yticks=None,
                                            ytick_labels=None,
                                           ymax=10, 
                                           yaxis_label=False,
                                            fig_filename=None, 
                                            hourly_period_to_exclude=None, 
                                            data_source_to_use=None):
    """
    Helper method used by make_cycle_amplitudes_bar_plot_for_figure_1. 
    """
    assert metric_to_use == 'max_minus_min'
    all_cycles = ['date_relative_to_period', 'local_hour', 'weekday', 'month']
    if cycles_to_plot is None:
        cycles_to_plot = all_cycles

    bar_width = .3
    space_between_groups = 1
    
    tick_locations = []
    tick_names = []
    
    cycles_to_colors = {'date_relative_to_period':PERIOD_CYCLE_COLOR, 
                        'local_hour':HOUR_CYCLE_COLOR, 
                        'weekday':WEEKDAY_CYCLE_COLOR, 
                        'month':SEASONAL_CYCLE_COLOR}
    
    current_bar_location = 0
    all_cycle_amplitudes = {} # keep track of average cycle amplitudes 
    for opposite_pair in opposite_pairs:
        
        
        assert data_source_to_use == 'linear_regression'
        if data_source_to_use == 'linear_regression':
            regression_data = results[opposite_pair][substratification][substratification_level][data_source_to_use]
            data = compare_to_seasonal_cycles.convert_regression_format_to_simple_mean_format(regression_data, 'linear_regression')
        else:
            data = deepcopy(results[opposite_pair][substratification][substratification_level][data_source_to_use])
        tick_locations.append(current_bar_location + bar_width * (len(all_cycles) - 1) / 2.)
        tick_names.append(CANONICAL_PRETTY_SYMPTOM_NAMES[opposite_pair])
        ratios = {} # compute errorbars on ratio of amplitudes. 
        
        for cycle in sorted(all_cycles, key=lambda x:x not in cycles_to_plot):
            # we always loop over all four cycles so the spacing is consistent; 
            # we just set the cycles we don't want to plot for 0. 
            if cycle not in cycles_to_plot:
                metric_val = 0
            else:
                metric_val = compare_to_seasonal_cycles.get_cycle_amplitude(data, cycle, metric_to_use, hourly_period_to_exclude)

                
                # get errorbars by bootstrapping
                all_bootstrapped_amps = []
                iterate_data = deepcopy(results[opposite_pair][substratification][substratification_level]['linear_regression_bootstrapped_iterates'])
                for iterate in range(len(iterate_data)):
                    bootstrapped_linear_regression_format = compare_to_seasonal_cycles.convert_regression_format_to_simple_mean_format(
                        iterate_data[iterate],
                        'linear_regression')
                    bootstrapped_amp = compare_to_seasonal_cycles.get_cycle_amplitude(bootstrapped_linear_regression_format, 
                                                cycle, 
                                                metric_to_use='max_minus_min', 
                                                hourly_period_to_exclude=None)
                    all_bootstrapped_amps.append(bootstrapped_amp)
                
                bootstrapped_CI = bootstrap_CI(all_bootstrapped_amps, metric_val)
                lower_err = metric_val - bootstrapped_CI['lower_95_CI']
                upper_err = bootstrapped_CI['upper_95_CI'] - metric_val
                ratios[cycle] = {'val':metric_val, 'bootstrapped_iters':all_bootstrapped_amps, 'lower_95_CI':bootstrapped_CI['lower_95_CI'], 'upper_95_CI':bootstrapped_CI['upper_95_CI']}
                assert lower_err > 0
                assert upper_err > 0
                
                #lower_err, upper_err = compare_to_seasonal_cycles.get_amplitude_standard_error_from_regression_data(regression_data, cycle)
            if 'continuous' not in opposite_pair:
                metric_val = metric_val * 100
                lower_err = lower_err * 100
                upper_err = upper_err * 100

            assert metric_val < ymax
            label_to_use = PRETTY_CYCLE_NAMES[cycle] if opposite_pair == opposite_pairs[0] else None
            plt.bar([current_bar_location], 
                [metric_val], 
                width=bar_width, 
                color=[cycles_to_colors[cycle]], 
                label=label_to_use)
            if cycle in cycles_to_plot:
                plt.errorbar([current_bar_location], [metric_val], yerr=[[lower_err], [upper_err]], color='black', capsize=2)
            if cycle not in all_cycle_amplitudes:
                all_cycle_amplitudes[cycle] = []
            all_cycle_amplitudes[cycle].append(metric_val)
            current_bar_location += bar_width
        print('Period cycle amplitude for %s: %2.5f (95%% CI: %2.5f-%2.5f)' % 
            (opposite_pair,
            ratios['date_relative_to_period']['val'], 
             ratios['date_relative_to_period']['lower_95_CI'], 
             ratios['date_relative_to_period']['upper_95_CI']))
        for k in ratios.keys():
            if k != 'date_relative_to_period':
                bootstrapped_ratios = np.array(ratios['date_relative_to_period']['bootstrapped_iters'])/np.array(ratios[k]['bootstrapped_iters'])
                original_ratio = ratios['date_relative_to_period']['val']/ratios[k]['val']
                # We use the percentiles on the ratios to compute CIs, for consistency with the age trends (where we also compute ratios, and because the distribution is quite skewed, we use percentiles). 
                # Confirmed that this yields essentially identical CIs to using 1.96 * std of bootstrapped iterates. 
                print("Ratio to amplitude of %s cycle: %2.5f (95%% CI: %2.5f-%2.5f); fraction of ratios greater than 1, %2.5f;  amplitude of %s cycle, %2.5f (95%% CI: %2.5f-%2.5f)" % (k, 
                        original_ratio, 
                        scoreatpercentile(bootstrapped_ratios, 2.5), 
                        scoreatpercentile(bootstrapped_ratios, 97.5), 
                        np.mean(bootstrapped_ratios > 1), 
                        k,
                        ratios[k]['val'], 
                        ratios[k]['lower_95_CI'], 
                        ratios[k]['upper_95_CI']))
                

        current_bar_location += space_between_groups
    if draw_xticks:
        plt.xticks(tick_locations, tick_names, fontsize=15, rotation=90)
    else:
        plt.xticks([])
    
    plt.yticks(yticks,
        ytick_labels,
        fontsize=15)

    plt.ylim([0, ymax])
    assert max(yticks) <= ymax

    if yaxis_label:
        if metric_to_use == 'max_minus_min':
            plt.ylabel("Cycle amplitude\n(max - min)", fontsize=20)
        else:
            plt.ylabel("Average difference from cycle mean", fontsize=20)
    plt.title(title_string, fontsize=20)
    print("Average amplitudes of all bars for symptoms")
    print(opposite_pairs)
    for cycle in all_cycle_amplitudes:
        print 'Average for all symptoms for %s: %2.3f' % (cycle, np.mean(all_cycle_amplitudes[cycle]))

def make_maps_for_figure_2(results):
    """
    Make a map for each symptom pair. 
    These results have already been filtered for 
    min_users = MIN_USERS_FOR_SUBGROUP
    min_obs = MIN_OBS_FOR_SUBGROUP
    Confirmed this looks ok using test country values. Test case: 
    extreme_bin_edge = 10
    bin_edges = np.arange(-extreme_bin_edge, extreme_bin_edge + .0001, extreme_bin_edge * 2./ 7)
    bin_edge_labels=['-%2.3f' % extreme_bin_edge] + ['' for a in range(len(bin_edges) - 2)] + ['%2.3f' % extreme_bin_edge]
        
    print bin_edges
    compare_to_seasonal_cycles.make_basemap(country_vals={'Sweden':8,
                                                         'China':5, 
                                                          'United States':3, 
                                                          'Australia':0, 
                                                         'Brazil':-3, 
                                                         'Mexico':-5, 
                                                         'Argentina':-8}, 
                             bin_edges=bin_edges, 
                             bin_edge_labels=bin_edge_labels, 
                             title_string='test')
    """
    for k in ORDERED_SYMPTOM_NAMES:
        binary_effects = deepcopy(results[k]
         ['no_substratification']
         ['period_effects_with_covariates']
         ['near_period*C(country)']
         ['predicted_effects_by_country'])


        print("Number of binary effects examined for %s:\n %i, range %2.3f to %2.3f" % (k,
                                                                               len(binary_effects), 
                                                                               min(binary_effects.values()), 
                                                                               max(binary_effects.values())))

        percent_formatter = '{:.0%}' # percent formatter
        float_formatter = "{0:.1f}" # float formatter
        if k == 'emotion*happy_versus_emotion*sad':
            # main plot for paper
            formatter = percent_formatter
            extreme_bin_edge = .15 # the extreme end of the color bin. 
            bad_edge = '\nmore sad'
            good_edge = '\nmore happy'
            filename = 'figures_for_paper/fig2.png'
        else:
            filename = 'figures_for_paper/supplementary_map_%s.png' % k

            if k == 'sex*had_sex_versus_sex*did_not_have_sex':
                extreme_bin_edge = .3
                formatter = percent_formatter
                bad_edge = '\nless sex'
                good_edge = '\nmore sex'
            elif 'exercise' in k:
                extreme_bin_edge = .15
                formatter = percent_formatter
                bad_edge = '\nless exercise'
                good_edge = '\nmore exercise'
            elif 'sleep' in k:
                extreme_bin_edge = .15
                formatter = percent_formatter
                bad_edge = '\nslept <6 hrs'
                good_edge = '\nslept >6 hrs'
            elif WEIGHT_SUBSTRING in k:
                extreme_bin_edge = 1.0
                formatter = float_formatter
                bad_edge = '\nlbs lighter'
                good_edge = '\nlbs heavier'
            elif HEART_SUBSTRING in k:
                extreme_bin_edge = 2.5
                formatter = float_formatter
                bad_edge = '\nBPM slower'
                good_edge = '\nBPM faster'
            elif BBT_SUBSTRING in k:
                extreme_bin_edge = .5
                formatter = float_formatter
                bad_edge = ' deg F\nlower BBT'
                good_edge = ' deg F\nhigher BBT'
            elif determine_mood_behavior_or_vital_sign(k) == 'Mood':
                canonical_pretty_name = CANONICAL_PRETTY_SYMPTOM_NAMES[k].replace('\n', ' ')
                good_symptom, bad_symptom = canonical_pretty_name.split( ' versus ')
                extreme_bin_edge = .15
                bad_edge = '\nmore ' + bad_symptom
                good_edge = '\nmore ' + good_symptom
                formatter = percent_formatter
            else:
                raise Exception("%s is not a valid symptom" % k)
        assert extreme_bin_edge > 0
        bin_edges = np.arange(-extreme_bin_edge, extreme_bin_edge + .0001, extreme_bin_edge * 2./ 7)
        bin_edge_labels=['%s%s' % (formatter.format(extreme_bin_edge), bad_edge)] + ['' for a in range(len(bin_edges) - 2)] + ['%s%s' % (formatter.format(extreme_bin_edge), good_edge)]
        title_string = '%s' % (CANONICAL_PRETTY_SYMPTOM_NAMES[k].replace('\n', ' '))


        compare_to_seasonal_cycles.make_basemap(country_vals=binary_effects, 
                         bin_edges=bin_edges, 
                         bin_edge_labels=bin_edge_labels, 
                         title_string=title_string,
                         filename=filename)


def make_age_trend_plot(results,
    opposite_pairs_to_plot,
    specifications_to_plot,
    figname,
    plot_curves_for_two_age_groups,
    n_subplot_rows,
    n_subplot_columns, 
    figsize, 
    subplot_kwargs, 
    age_ticks_only_at_bottom=False, 
    label_kwargs=None, 
    linewidth=2, 
    plot_legend=False, 
    include_ylabel=True, 
    plot_yerr=False):
    """
    Shows that age trends are robust to inclusion of other covariates. 
    This is actually used to make two plots in the paper: the main figure 3, and also a supplementary 
    robustness check. 
    """
    
    fig = plt.figure(figsize=figsize)
    if label_kwargs is None:
        label_kwargs = {'fontsize':13}
    for subplot_idx, opposite_pair in enumerate(opposite_pairs_to_plot):
        print(opposite_pair)

        plt.subplot(n_subplot_rows, n_subplot_columns, subplot_idx + 1)
        period_effects_with_covariates = deepcopy(results[opposite_pair]
         ['no_substratification']
         ['period_effects_with_covariates'])


        grouping = 'predicted_effects_by_categorical_age'
        pretty_age_names = {'[10, 15)':'10-15', 
                            '[15, 20)':'15-20', 
                            '[20, 25)':'20-25', 
                            '[25, 30)':'25-30', 
                            '[30, 35)':'30-35', 
                            '[35, inf)':'35+'}
        specifications_to_colors = {'age':'#1f77b4', 
        'country+age':'#ff7f0e', 
        'country+age+behavior':'#2ca02c', 
        'country+age+behavior+app usage':'#d62728'}
        for specification in specifications_to_plot:
            

            age_cats = ['[10, 15)', '[15, 20)', '[20, 25)', '[25, 30)', '[30, 35)', '[35, inf)']
            if opposite_pair == 'continuous_features*bbt_versus_continuous_features*null':
                age_cats.remove('[10, 15)')
            # extract the age effects for the relevant specification. 
            ys = [period_effects_with_covariates[SHORT_NAMES_TO_REGRESSION_COVS[specification]][grouping][a] for a in age_cats]
            
            if plot_yerr:
                assert specification == 'age'
                yerrs = []
                ratios = {}
                for i, age_cat in enumerate(age_cats):
                    bootstrapped_vals = [period_effects_with_covariates['bootstrapped_iterates'][iterate]['near_period*C(categorical_age)']['predicted_effects_by_categorical_age'][age_cat] for iterate in range(len(period_effects_with_covariates['bootstrapped_iterates']))]
                    # make sure this is consistent with how we're computing bootstrapped CIs in the bootstrap CI method. 
                    assert len(set(bootstrapped_vals)) == N_BOOTSTRAP_ITERATES
                    bootstrapped_CIs = bootstrap_CI(bootstrap_iterates=bootstrapped_vals, original_estimate=ys[i])
                    yerrs.append([ys[i] - bootstrapped_CIs['lower_95_CI'], 
                                  bootstrapped_CIs['upper_95_CI'] - ys[i]])
                    if age_cat in ['[15, 20)', '[30, 35)']:
                        print('%s %2.5f (95%% CI %2.5f-%2.5f)' % (age_cat, ys[i], bootstrapped_CIs['lower_95_CI'], bootstrapped_CIs['upper_95_CI']))
                        ratios[age_cat] = {'val':ys[i], 'bootstrapped_iters':bootstrapped_vals}
                original_ratio = ratios['[30, 35)']['val']/ratios['[15, 20)']['val']
                bootstrapped_ratios = np.array(ratios['[30, 35)']['bootstrapped_iters'])/np.array(ratios['[15, 20)']['bootstrapped_iters'])
                print('Ratio: %2.5f (%2.5f-%2.5f)' % (original_ratio, scoreatpercentile(bootstrapped_ratios, 2.5), 
                    scoreatpercentile(bootstrapped_ratios, 97.5)))
                print("Fraction of bootstrapped ratios that are greater than 1: %2.3f" % (bootstrapped_ratios > 1).mean())
                yerrs = np.array(yerrs).transpose()


            else:
                yerrs = None
                for i in range(len(ys)):
                    print '%-65s %-20s %-9s %2.5f' % (opposite_pair, specification, age_cats[i], ys[i])


            
            plt.errorbar(range(len(ys)), 
                ys, yerr=yerrs, label=specification, color=specifications_to_colors[specification], linewidth=linewidth, elinewidth=1, capsize=2)
            if age_ticks_only_at_bottom and (subplot_idx < n_subplot_columns * (n_subplot_rows - 1)):
                plt.xticks([])
            else:
                plt.xticks(range(len(ys)), 
                           [pretty_age_names[a] for a in age_cats], 
                           rotation=90, 
                           **label_kwargs)
                plt.xlabel("Age", **label_kwargs)
        titlestring = CANONICAL_PRETTY_SYMPTOM_NAMES[opposite_pair]
        if opposite_pair == 'continuous_features*weight_versus_continuous_features*null':
            ylimits = [-.6, .6]
            plt.yticks([-.6, 0, .6],
                ['-0.6\nlbs\n', '', '+0.6\nlbs'], 
                **label_kwargs)
        elif opposite_pair == 'continuous_features*bbt_versus_continuous_features*null':
            ylimits = [-.5, .5]
            plt.yticks([-.5, 0, .5], 
                ['-0.5\ndeg F\n', '', '+0.5\ndeg F'],
                **label_kwargs)
        elif opposite_pair == 'continuous_features*heart_rate_versus_continuous_features*null':
            ylimits = [-3, 3]
            plt.yticks([-3, 0, 3], 
                ['-3\nBPM\n', '', '+3\nBPM'],
                **label_kwargs)
        elif determine_mood_behavior_or_vital_sign(opposite_pair) == 'Behavior':
            ylimits = [-.15, .15]
            if 'exercise*' in opposite_pair:
                plt.yticks([-.15, 0, .15], ['15%\nless exercise\n', 
                'no\neffect', 
                '15%\nmore exercise'], 
                **label_kwargs)
            elif 'sex*' in opposite_pair:
                plt.yticks([-.15, 0, .15], ['15%\nless sex\n', 
                'no\neffect', 
                '15%\nmore sex'], 
                **label_kwargs)
            elif 'sleep'  in opposite_pair:
                plt.yticks([-.15, 0, .15], ['15% more\nslept <6 hrs\n', 
                'no\neffect', 
                '15% more\nslept >6 hrs'], 
                **label_kwargs)
        elif determine_mood_behavior_or_vital_sign(opposite_pair) == 'Mood':
            good_symptom, bad_symptom = titlestring.split('\nversus\n')
            if 'sensitive' in opposite_pair:
                ylimits = [-.1, .1]
                plt.yticks([-.1, 0, .1], ['10%% more\n%s\n' % bad_symptom,
                    'no\neffect', 
                    '10%% more\n%s' % good_symptom], 
                    **label_kwargs)
            else:
                ylimits = [-.06, .06]
                plt.yticks([-.06, 0, .06], ['6%% more\n%s\n' % bad_symptom,
                    'no\neffect', 
                    '6%% more\n%s' % good_symptom], 
                    **label_kwargs)
        else:
            raise Exception("Invalid symptom")
        
        plt.yticks(**label_kwargs)
        if (subplot_idx == 0) and include_ylabel:
            plt.ylabel("Premenstrual effect", **label_kwargs)

        plt.xlim([-.1, len(ys) - 1 + .1])
        plt.ylim(ylimits)
        plt.plot([-.1, len(ys) - 1 + .1], [0, 0], color='black')
        plt.title(titlestring, **label_kwargs)
        if plot_legend and subplot_idx == n_subplot_columns - 1:
            plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

        # This is a little confusing, but if plot_curves_for_two_age_groups is True
        # we make a second set of subplots which show the menstrual cycles for younger women and older group. 
        if plot_curves_for_two_age_groups:
            assert n_subplot_rows == 2 # make sure all values are what we expect or things will break. 
            assert n_subplot_columns == 4
            assert subplot_idx <= 3
            plt.subplot(n_subplot_rows, n_subplot_columns, subplot_idx + 5)
            ages_to_colors = {'[15, 20)':'#2ca02c', 
                              '[30, 35)':'#d62728'}
            for age_group in ['[15, 20)', '[30, 35)']:
                curve_to_plot = compare_to_seasonal_cycles.convert_regression_format_to_simple_mean_format(
                    results[opposite_pair]['by_categorical_age'][age_group]['linear_regression'],
                    type_of_regression='linear_regression')
                curve_to_plot = curve_to_plot['date_relative_to_period']
                xs = curve_to_plot.index
                assert list(xs) == range(-14, 15)
                ys = curve_to_plot['mean'].values - curve_to_plot['mean'].values.mean()
                plt.errorbar(xs, ys, 
                             label='Age ' + pretty_age_names[age_group], 
                             color=ages_to_colors[age_group], linewidth=linewidth)
            if opposite_pair == 'emotion*happy_versus_emotion*sad':
                plt.yticks([-.06, 0, .06], 
                           ['6% more\nsad\n', 'baseline', '6% more\nhappy'])
            if opposite_pair == 'continuous_features*weight_versus_continuous_features*null':
                plt.yticks([-.6, 0, .6],
                ['-0.6\nlbs\n', '', '+0.6\nlbs'], 
                **label_kwargs)
            elif opposite_pair == 'continuous_features*bbt_versus_continuous_features*null':
                plt.yticks([-.5, 0, .5], 
                    ['-0.5\ndeg F\n', '', '+0.5\ndeg F'],
                    **label_kwargs)
            elif opposite_pair == 'continuous_features*heart_rate_versus_continuous_features*null':
                plt.yticks([-3, 0, 3], 
                    ['-3\nBPM\n', '', '+3\nBPM'],
                    **label_kwargs)
            plt.plot([-14, 14], [0, 0], color='black')
            plt.xticks([-14, -7, 0, 7, 14], **label_kwargs)
            plt.xlim([-14, 14])
            plt.ylim(ylimits)
            plt.yticks(**label_kwargs)
            plt.title(titlestring, **label_kwargs)

            plt.xlabel("Day relative to period", **label_kwargs)
            if subplot_idx == 0:
                plt.legend(**label_kwargs)
                plt.ylabel("Effect relative to baseline", **label_kwargs)
    plt.subplots_adjust(**subplot_kwargs)
    
    plt.savefig(figname)

def make_clock_plot(results, just_plot_single_cycle):
    """
    Makes a circular plot that shows all four cycles as concentric circles. 
    if just_plot_single_cycle is True, plots a single cycle in frames that can be turned into a movie. 
    This is not tremendously useful, but it is sort of fun :)
    """
    substratification = 'by_very_active_northern_hemisphere_loggers'
    substratification_level = True

    if just_plot_single_cycle:
        plt.figure(figsize=[5, 5])
        opposite_pairs_to_plot = ['emotion*happy_versus_emotion*sensitive_emotion']
    else:
        plt.figure(figsize=[12, 20])
        opposite_pairs_to_plot = ORDERED_SYMPTOM_NAMES

    for opposite_idx, opposite_pair in enumerate(opposite_pairs_to_plot):
        print("Making clock plot for %s" % opposite_pair)
        data = deepcopy(results[opposite_pair][substratification][substratification_level]['linear_regression'])
        data = compare_to_seasonal_cycles.convert_regression_format_to_simple_mean_format(data, 'linear_regression')
        for cycle in data:
            data[cycle]['mean'] = data[cycle]['mean'] - data[cycle]['mean'].mean()
            assert np.allclose(data[cycle]['mean'].mean(), 0)
        if 'continuous' in opposite_pair:
            max_abs_val = np.abs(data['date_relative_to_period']['mean']).max()
            min_val = -max_abs_val * 1.2
            max_val = max_abs_val * 1.2
        elif 'sex' in opposite_pair:
            min_val = -.15
            max_val = .15
        else:
            min_val = -.1
            max_val = .1

        if 'continuous' in opposite_pair or 'sex*' in opposite_pair or 'sleep*' in opposite_pair or 'exercise*' in opposite_pair:
            # no hourly info for these symptoms
            assert 'local_hour' in data
            data['local_hour']['mean'] = 0
        if HEART_SUBSTRING in opposite_pair:
            assert 'month' in data
            data['month']['mean'] = 0

        def convert_to_x_and_y(r, theta):
            # small helper method that converts r, theta to x and y. 
            r = np.array(r)
            theta = np.array(theta)
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            return x, y

        if not just_plot_single_cycle:
            plt.subplot(5, 3, opposite_idx + 1)
            plt.subplots_adjust(bottom=.02, top=.98, left=.02, right=.98, wspace=.05, hspace=.1)
        else:
            plt.subplot(1, 1, 1)
            frame_idx = 0
            plt.subplots_adjust(bottom=.1, top=.9, left=.1, right=.9)

        # set up basic plot circles etc. 
        cycles_to_plot = ['month','weekday', 'local_hour', 'date_relative_to_period']
        plt.xlim([-5.1, 5.1])
        plt.ylim([-5.1, 5.1])
        # plot the circles. They will have radius 1, 2, 3, 4, 5. 
        for circle_idx in range(len(cycles_to_plot) + 1):
            circle_edge_x, circle_edge_y = convert_to_x_and_y(r=circle_idx + 1, theta=np.arange(0, 2 * np.pi, .01))
            plt.plot(circle_edge_x, circle_edge_y, color='black', linewidth=3)

        # plot the cycles names. 
        for cycle_idx, cycle in enumerate(cycles_to_plot):
            plt.text(0, 
                     -1.35 - cycle_idx, 
                     PRETTY_CYCLE_NAMES[cycle], 
                     fontsize=14,
                     horizontalalignment='center', 
                     verticalalignment='center', 
                     fontweight='bold')

        plt.plot([0, 0], [1, 5], color='black', linewidth=3) # the "hand" of the clock, which points at 12. 
        plt.title(CANONICAL_PRETTY_SYMPTOM_NAMES[opposite_pair].replace('\n', ' ').replace('versus', 'vs'), 
                  fontsize=14, 
                  fontweight='bold')

        plt.xticks([])
        plt.yticks([])

        # there is probably a better way of filling the segments of the pie. Sigh. Instead we use LOTS OF DOTS. 
        n_total_samples = 30000
        # offset for each cycle. 
        starting_val_for_cycle = {'date_relative_to_period':np.pi * 3 / 2, # day -14 is at the bottom of the clock.
                                 'local_hour':1.*np.pi / 2, # hour 0 (midnight) is top of clock. 
                                 'weekday':np.pi / 2 - 2.*np.pi / 7, # saturday + sunday are top of clock. (This is Monday's position). 
                                 'month':np.pi/2} # January is top of clock.



        for cycle_idx, cycle in enumerate(cycles_to_plot):
            all_xs = []
            all_ys = []
            all_colors = []
            grouped_d = deepcopy(data[cycle]['mean'])
            if cycle == 'weekday':
                grouped_d = grouped_d.loc[['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']]
            else:
                assert list(grouped_d.index) == sorted(list(grouped_d.index))
            for i in range(len(grouped_d)):
                p = grouped_d.iloc[i] # value we want to plot in the segment. 
                assert p > min_val
                assert p < max_val
                p = (p - min_val) / (max_val - min_val) # scale the value to the appropriate color range. 
                n_samples = int(1.*n_total_samples / len(grouped_d))
                r = np.random.random(n_samples) + 1 + cycle_idx # randomly generate r in appropriate range for circle. 
                dtheta = 2. * np.pi / len(grouped_d) # segment width is 2 pi/n_groups
                starting_val = starting_val_for_cycle[cycle]

                # to compute theta, we start at the starting_val
                # go counterclockwise by the group index (-dtheta * i) 
                # and add random noise to fill the segment. 
                theta = starting_val - dtheta * i - np.random.random(n_samples) * dtheta 
                x, y = convert_to_x_and_y(r, theta)
                all_xs += list(x)
                all_ys += list(y)
                colors_for_cmap = ['#023eff', '#ffffff', '#e8000b']
                cm_fxn = LinearSegmentedColormap.from_list(
                    'mycmap', colors_for_cmap, N=256)
                colors = [cm_fxn(p)] * n_samples
                all_colors += colors
                if just_plot_single_cycle:
                    plt.scatter(x, y, c=np.array(colors).squeeze(), alpha=.3, s=4)
                    plt.savefig('frames_for_movie/frame_%i.png' % frame_idx, dpi=100)
                    frame_idx += 1
                    print 'frame %i' % frame_idx
            all_colors = np.array(all_colors).squeeze()
            idxs = range(len(all_colors))
            random.shuffle(idxs)
            all_xs = np.array(all_xs)
            all_ys = np.array(all_ys)
            all_xs = all_xs[idxs]
            all_ys = all_ys[idxs]
            all_colors=all_colors[idxs]
            plt.scatter(all_xs, all_ys, c=all_colors, alpha=.3, s=4)

    if not just_plot_single_cycle:
        plt.savefig('figures_for_paper/clock_plot.png', dpi=300)

def make_name_pretty_for_table(x):
        """
        This gets used for both category and type, which is a little weird
        not sure it will generalize, so be careful using this method (safest just to use for plotting stuff). 
        """
        specific_mapping = {'continuous_features':'vital signs', 
                            '0-3':'0-3 hrs', 
                            '3-6':'3-6 hrs', 
                            '6-9':'6-9 hrs', 
                            '>9':'9 hrs', 
                            'protected_sex':'protected', 
                            'unprotected_sex':'unprotected', 
                            'withdrawal_sex':'withdrawal',
                            'sensitive_emotion':'sensitive', 
                            'cold_flu_medication':'cold/flu medication', 
                            'cold_flu_ailment':'cold/flu ailment', 
                            'bbt':'BBT', 
                            'pms':'PMS'}

        if x in specific_mapping:
            return specific_mapping[x]
        x = x.replace('_craving', '')
        x = x.replace('_social', '')
        x = x.replace('_hair', '')
        x = x.replace('_skin', '')
        x = x.replace('_poop', '')
        x = x.replace('_collection_method', '')
        x = x.replace('_medication', '')
        x = x.replace('_ailment', '')
        x = x.replace('_appointment', '')
        x = x.replace('_test_neg', '_negative')
        x = x.replace('_test_pos', '_positive')
        x = x.replace('_hbc', ' birth control')
        x = x.replace('_', ' ')

        return x


def make_table_of_all_symptoms_not_just_those_in_analysis():
    """
    Makes a table of all symptoms, not just those included in analysis. 
    Uses symptom counts in processed data.
    """ 
    

    symptom_counts = cPickle.load(open(os.path.join(base_results_dir, 'symptom_counts_in_processed_data.pkl'), 'rb'))
    print("Total symptom logs read in: %i of %i types" % (sum(symptom_counts.values()), len(symptom_counts.keys())))
    symptom_table = {'Category':[], 'Symptoms':[], 'Count':[]}
    for k in symptom_counts:
        assert len(k.split('*')) == 2
        category = make_name_pretty_for_table(k.split('*')[0])
        symptom = make_name_pretty_for_table(k.split('*')[1])
        if category not in symptom_table['Category']:
            symptom_table['Category'].append(category)
            symptom_table['Symptoms'].append([])
            symptom_table['Count'].append(0)

        idx = symptom_table['Category'].index(category)
        symptom_table['Symptoms'][idx].append(symptom)
        symptom_table['Count'][idx] += symptom_counts[k]
    pd.set_option('max_colwidth', 500)
    symptom_table = pd.DataFrame(symptom_table)
    symptom_table['Symptoms'] = symptom_table['Symptoms'].map(lambda x:', '.join(sorted(x)))
    symptom_table = symptom_table.sort_values(by='Count')[::-1]
    symptom_table['Percent of logs'] = 100.*symptom_table['Count'] / symptom_table['Count'].sum()
    symptom_table['Percent of logs'] = symptom_table['Percent of logs'].map(lambda x:'%2.0f%%' % x if x > 1 else '<1%')
    symptom_table = symptom_table[['Category', 'Symptoms', 'Percent of logs']]
    print symptom_table.to_latex(index=False).replace('\\\\', '\\\\ \\hline').replace('<1', '$<1$')

def make_table_of_user_statistics(n_chunks_to_use):
    """
    Make a table of mean / median / 90CI age, symptoms logged, etc. 
    Does this for all Clue users, not only for users who logged one of the 15 symptoms. 
    Computes symptom counts + app usage time using the filtered symptom dataset. 
    """
    all_features = {}
    for user_group in ['all_users', 'very_active_loggers']:
        all_features[user_group] = {'age':[], 
                    'exact_n_days_using_app':[], 
                    'exact_total_symptoms_logged':[]}

    for i in range(n_chunks_to_use):
        print("Loading user dataframe %i" % i)
        user_d = cPickle.load(open(os.path.join(processed_chunked_tracking_dir, 
                                                'user_features',
                                                'user_features_chunk_%i.pkl' % i)))
        for user_group in all_features:
            df = pd.DataFrame(user_d.values())
            if user_group == 'very_active_loggers':
                df = df.loc[df['very_active_loggers'] == True]
            all_features[user_group]['age'] += list(df['age'].dropna())
            all_features[user_group]['exact_n_days_using_app'] += list(df['exact_n_days_using_app'])
            all_features[user_group]['exact_total_symptoms_logged'] += list(df['exact_total_symptoms_logged'])

    for user_group in all_features:
        print(user_group)
        summary_stats = []
        for k in all_features[user_group].keys():
            summary_stats.append({'feature':k
                                  .replace('exact_n_days_using_app', 'Days using Clue')
                                  .replace('exact_total_symptoms_logged', 'Total symptoms logged').
                                  replace('age', 'Age'),
                                 'Mean':'%2.1f' % np.mean(all_features[user_group][k]), 
                                  'Median':'%2.1f' % np.median(all_features[user_group][k]), 
                                  '5th percentile':'%2.1f' % scoreatpercentile(all_features[user_group][k], 5), 
                                  '95th percentile':'%2.1f' % scoreatpercentile(all_features[user_group][k], 95)})
        print (pd.DataFrame(summary_stats)[['feature', 'Mean', 'Median', '5th percentile', '95th percentile']]
               .to_latex(index=False).replace('\\\\', '\\\\ \\hline'))


def make_user_count_table(results):
    """
    Counts of very active loggers + total loggers logging each symptom. 
    """
    symptom_table = []
    
    # keep counts of how many users and symptoms there are across ALL symptoms. 
    total_very_active_n_obs = 0
    total_very_active_ids = set()
    
    total_overall_n_obs = 0
    total_overall_user_ids = set()
    
    for k in results.keys():
        very_active_logger_data = results[k]['no_substratification']
        non_very_active_logger_data = results[k]['ALL_OMITTED_USERS_ROBUSTNESS_CHECK_ONLY'][True]

        pretty_feature_name = CANONICAL_PRETTY_SYMPTOM_NAMES[k].replace('\n', ' ')
        feature_type = determine_mood_behavior_or_vital_sign(k)
        
        very_active_n_obs = 1.0*very_active_logger_data['overall_n_obs']
        non_active_n_obs = 1.0*non_very_active_logger_data['overall_n_obs']
        overall_n_obs = very_active_n_obs + non_active_n_obs
        
        very_active_user_ids = very_active_logger_data['unique_user_ids']
        non_active_user_ids = non_very_active_logger_data['unique_user_ids']
        assert len(very_active_user_ids.intersection(non_active_user_ids)) == 0
        overall_user_ids = very_active_user_ids.union(non_active_user_ids)
        
        very_active_mu = very_active_logger_data['overall_positive_frac']
        overall_mu = (very_active_logger_data['overall_positive_frac'] * very_active_n_obs + 
                      non_very_active_logger_data['overall_positive_frac'] * non_active_n_obs) / overall_n_obs
        
        if feature_type == 'Vital sign':
            very_active_mu = '%2.1f' % very_active_mu
            overall_mu = '%2.1f' % overall_mu
        else:
            very_active_mu = '%2.1f%%' % (very_active_mu * 100.)
            overall_mu = '%2.1f%%' % (overall_mu * 100.)
        
        total_very_active_n_obs += very_active_n_obs
        total_overall_n_obs += overall_n_obs
        
        total_very_active_ids = total_very_active_ids.union(very_active_user_ids)
        total_overall_user_ids = total_overall_user_ids.union(overall_user_ids)
        
        symptom_table.append({'Dimension':pretty_feature_name, 
                              'Category':feature_type,
                              'N Obs (LTLs)':"{:,.0f}".format(very_active_n_obs), 
                              'unformatted_n_obs':very_active_n_obs,
                              'N Users (LTLs)':"{:,.0f}".format(len(very_active_user_ids)), 
                              'Mean value (LTLs)':very_active_mu, 
                              'N Obs (overall)':"{:,.0f}".format(overall_n_obs), 
                              'N Users (overall)':"{:,.0f}".format(len(overall_user_ids)), 
                              'Mean value (overall)':overall_mu})

    symptom_table.append({'Dimension':'-', 
                          'Category':'All combined',
                          'N Obs (LTLs)':"{:,.0f}".format(total_very_active_n_obs), 
                          'unformatted_n_obs':total_very_active_n_obs,
                          'N Users (LTLs)':"{:,.0f}".format(len(total_very_active_ids)), 
                          'Mean value (LTLs)':'-',
                          'N Obs (overall)':"{:,.0f}".format(total_overall_n_obs), 
                          'N Users (overall)':"{:,.0f}".format(len(total_overall_user_ids)), 
                          'Mean value (overall)':'-'
                          })

    symptom_table = pd.DataFrame(symptom_table)
    symptom_table = (symptom_table.sort_values(by=['Category', 'unformatted_n_obs'])[::-1]
                     [['Category', 
                       'Dimension', 
                       'Mean value (LTLs)', 
                       'Mean value (overall)',
                       'N Obs (LTLs)', 
                       'N Obs (overall)',
                       'N Users (LTLs)',
                       'N Users (overall)']])
    print symptom_table.to_latex(index=False).replace('\\\\', '\\\\ \\hline')

def make_near_period_start_table(results):
    """
    What is the near-period interval for each symptom? 
    """
    symptom_table = []
    for opposite_pair in ORDERED_SYMPTOM_NAMES:
        start_day = results[opposite_pair]['no_substratification']['start_day_for_period_individual_effect']
        end_day = results[opposite_pair]['no_substratification']['end_day_for_period_individual_effect']
        symptom_table.append({'Category':determine_mood_behavior_or_vital_sign(opposite_pair),
                              'Dimension':CANONICAL_PRETTY_SYMPTOM_NAMES[opposite_pair].replace('\n', ' '), 
                              'Start day':start_day, 
                              'End day':end_day})
    symptom_table = pd.DataFrame(symptom_table)
    print (symptom_table[['Category', 'Dimension', 'Start day', 'End day']]
           .sort_values(by='Category').to_latex(index=False)
           .replace('\\\\', '\\\\ \\hline'))


def make_substratification_robustness_plot(results, categories_to_substratify_by, top_n_countries, min_users=MIN_USERS_FOR_SUBGROUP, min_obs=MIN_OBS_FOR_SUBGROUP, savefig=True, plot_birth_control_cats_only=False):
    """
    Show that amplitudes don't vary too much depending on what substratification you use. 
    """
        
    def no_hourly_info(opposite_pair):
        assert type(opposite_pair) == str
        for k in ['continuous', 'exercise*', 'sleep*', 'sex*']:
            if k in opposite_pair:
                return True
        return False

    for k in results.keys():
        # small hack -- create a substratification label for included vs removed users. 
        results[k]['by_any_filtering'] = {'removed_users':{}, 'included_users':{}}
        for data_to_copy in ['overall_n_users', 
                             'overall_n_obs', 
                             'linear_regression']:
            results[k]['by_any_filtering']['removed_users'][data_to_copy] = deepcopy(results[k]
                                                                                ['ALL_OMITTED_USERS_ROBUSTNESS_CHECK_ONLY']
                                                                                [True]
                                                                                [data_to_copy])
            results[k]['by_any_filtering']['included_users'][data_to_copy] = deepcopy(results[k]
                                                                             ['no_substratification']
                                                                             [data_to_copy])
    metric_to_use = 'max_minus_min'
    cycles_to_colors = {'date_relative_to_period':PERIOD_CYCLE_COLOR, 
                            'local_hour':HOUR_CYCLE_COLOR, 
                            'weekday':WEEKDAY_CYCLE_COLOR, 
                            'month':SEASONAL_CYCLE_COLOR}
    cycles = ['date_relative_to_period', 'month', 'weekday', 'local_hour']
    all_results = []
    fontsize = 17

    for opposite_pair in ORDERED_SYMPTOM_NAMES:
        fig = plt.figure(figsize=[16, 4])
        for substrat_idx, substratification in enumerate(categories_to_substratify_by):
            df = [] # this keeps track of results for each subcategory (eg, country or age). 
            levels = results[opposite_pair][substratification].keys()
            if substratification != 'by_country':
                levels = compare_to_seasonal_cycles.order_subcategories(levels, 
                                                                        substratification)
            else:
                levels = top_n_countries
            for level in levels:
                assert level in results[opposite_pair][substratification]
                data = results[opposite_pair][substratification][level]['linear_regression']
                data = compare_to_seasonal_cycles.convert_regression_format_to_simple_mean_format(data, 
                                                                                                 'linear_regression')
                def get_pretty_name_for_level(level):

                    if level == 'included_users':
                        return 'included'
                    elif level == 'removed_users':
                        return 'removed'
                    elif level == 'United States':
                        return 'US'
                    elif level == 'Britain (UK)':
                        return 'UK'
                    elif level in [True, False]:
                        return level
                    elif level[0] == '[' and level[-1] == ')':
                        level = level.replace('[', '').replace(')', '')
                        bottom, top = level.split(', ')
                        if bottom == '-1':
                            bottom = '0'
                        if top == 'inf':
                            return '>%s' % bottom
                        return '%s to %s' % (bottom, top)
                    else:
                        return level
                results_for_level = {'substratification':substratification, 
                           'opposite_pair':opposite_pair, 
                           'level':get_pretty_name_for_level(level), 
                           'n_users':results[opposite_pair][substratification][level]['overall_n_users'], 
                           'n_obs':results[opposite_pair][substratification][level]['overall_n_obs']}
                print '%-50s %-25s %6i %6i' % (opposite_pair, 
                                         substratification + ':' + str(level), 
                                         results_for_level['n_users'], 
                                         results_for_level['n_obs'])
                if ((results_for_level['n_users'] < min_users) 
                    or (results_for_level['n_obs'] < min_obs)):
                    continue
                for cycle in cycles:
                    if cycle == 'local_hour' and no_hourly_info(opposite_pair):
                        amplitude = 0
                    elif cycle == 'month' and HEART_SUBSTRING in opposite_pair:
                        amplitude = 0
                    else:
                        amplitude = compare_to_seasonal_cycles.get_cycle_amplitude(data, 
                                                                                   cycle, 
                                                                                   metric_to_use, 
                                                                                   hourly_period_to_exclude=None)


                    if 'continuous' not in opposite_pair:
                        amplitude = amplitude * 100.
                    results_for_level[cycle] = amplitude
                df.append(results_for_level)
            df = pd.DataFrame(df)
            
            if not plot_birth_control_cats_only:
                assert len(categories_to_substratify_by) == 6
            if len(df) == 1:
                assert plot_birth_control_cats_only # for IUD sometimes we don't have enough data to make enough to make a plot. 
                continue

            # now we have the amplitude data, it's time to make a plot. 
            plt.subplot(1, len(categories_to_substratify_by), substrat_idx + 1)
            for cycle in cycles:
                if sum(df[cycle]) == 0: 
                    # eg, we're missing hourly data. 
                    continue
                plt.plot(df[cycle], color=cycles_to_colors[cycle], label=PRETTY_CYCLE_NAMES[cycle])
                if HEART_SUBSTRING in opposite_pair:
                    plt.ylim([0,4])
                    yticks = [0, 2, 4]
                    ytick_labels = [0, 2, 4]
                elif 'sex' in opposite_pair:
                    plt.ylim([0, 50])
                    yticks = [0, 25, 50]
                    ytick_labels = ['0%', '25%', '50%']
                elif WEIGHT_SUBSTRING in opposite_pair:
                    plt.ylim([0, 3])
                    yticks = [0, 1.5, 3]
                    ytick_labels = [0, 1.5, 3]
                elif BBT_SUBSTRING in opposite_pair:
                    plt.ylim([0, 1])
                    yticks = [0, .5, 1]
                    ytick_labels = [0, .5, 1]
                else:
                    plt.ylim([0, 30])
                    yticks = [0, 15, 30]
                    ytick_labels = ['0%', '15%', '30%']

            plt.xticks(range(len(df)), df['level'], rotation=90, fontsize=fontsize)
            pretty_substratification_names = {'by_n_symptom_categories_used':'categories\nlogged', 
                                              'by_total_symptoms_logged':'symptoms\nlogged', 
                                              'by_categorical_age':'age group', 
                                              'by_country':'country', 
                                              'by_categorical_latitude':'latitude', 
                                              'by_any_filtering':'included in\nmain analysis', 
                                              'by_logged_hormonal_birth_control':'Logged hormonal BC', 
                                              'by_logged_birth_control_pill':'Logged BC pill', 
                                              'by_logged_iud':'Logged IUD'}

            plt.title(pretty_substratification_names[substratification], fontsize=fontsize)
            if substrat_idx == 0:
                plt.ylabel("Cycle amplitude\nfor subgroup", fontsize=fontsize)
                plt.yticks(yticks, ytick_labels, fontsize=fontsize)
            else:
                plt.yticks([])

        plt.legend(fontsize=fontsize, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        big_fontsize = fontsize + 2
        plt.suptitle(CANONICAL_PRETTY_SYMPTOM_NAMES[opposite_pair].replace('\n', ' '), fontsize=big_fontsize, fontweight='bold')

        plt.subplots_adjust(top=.8, bottom=.35, left=.1, right=.85)
        if savefig:
            if plot_birth_control_cats_only:
                plt.savefig('figures_for_paper/birth_control_only_substratification_robustness_plot_%s.png' % opposite_pair, dpi=300)
            else:
                plt.savefig('figures_for_paper/substratification_robustness_plot_%s.png' % opposite_pair, dpi=300)
        plt.show()

def make_period_regression_specifications_robustness_check(results):
    """
    Try different period parameterizations to see if it changes the amplitudes. 
    """

    all_amplitudes = []
    specifications_to_pretty_names = {"days_after_last_cycle_start":'Days after last cycle start', 
                                      "days_before_next_cycle_start":'Days before next cycle start', 
                                      'frac_through_cycle':"Normalized for user's cycle length"}

    for opposite_pair in results:
        plt.figure()
        original_regression_results = compare_to_seasonal_cycles.convert_regression_format_to_simple_mean_format(
        results[opposite_pair]['no_substratification']['linear_regression'], 
        'linear_regression')
        ys = deepcopy(original_regression_results['date_relative_to_period']['mean'].values)
        ys = ys - ys.mean()
        plt.plot(original_regression_results['date_relative_to_period'].index, 
                    ys, 
                    label='Original')
        plt.xlabel("Cycle day")
        plt.ylabel("Change relative to baseline")
        original_period_amp = compare_to_seasonal_cycles.get_cycle_amplitude(original_regression_results, 
                                            'date_relative_to_period', 
                                            metric_to_use='max_minus_min', 
                                            hourly_period_to_exclude=None)
        
        all_amplitudes.append({'amp':original_period_amp, 'specification':'original', 'opposite_pair':opposite_pair})
        data = deepcopy(results[opposite_pair]['no_substratification']['linear_regression_with_alternate_period_specifications'])
        for k in data:
            covariate = k.replace('with_period_parameterization_', '')
            period_beta = data[k]['params'].loc[data[k]['params'].index.map(lambda x:covariate in x)]
            period_beta.index = period_beta.index.map(lambda x:float(x.split('))[T.')[1].replace(']', '')))
            
            period_beta = period_beta.sort_index()
            if covariate == 'days_before_next_cycle_start':
                assert list(period_beta.index) == range(-39, 1)
                period_beta = period_beta.append(pd.Series([0], index=[-40]))
                period_beta = period_beta.loc[period_beta.index.map(lambda x:x >= -28)]
            elif covariate == 'days_after_last_cycle_start':
                assert list(period_beta.index) == range(1, 41)
                period_beta = period_beta.append(pd.Series([0], index=[0]))
                period_beta = period_beta.loc[period_beta.index.map(lambda x:x <= 28)]
            elif covariate == 'frac_through_cycle':
                period_beta = period_beta.append(pd.Series([0], index=[-.5]))
            else:
                raise Exception("Invalid covariate")
            assert len(period_beta) == 29
            period_beta = period_beta.sort_index()
            assert len(set(period_beta.index)) == len(period_beta)
            if covariate == 'frac_through_cycle':
                xs = period_beta.index * 28
            else:
                xs = period_beta.index
            ys = deepcopy(period_beta.values)
            ys = ys - ys.mean()
            plt.plot(xs, ys, label=specifications_to_pretty_names[covariate])
            amp = period_beta.values.max() - period_beta.values.min()
            all_amplitudes.append({'amp':amp, 
                                   'specification':covariate, 
                                   'opposite_pair':opposite_pair})
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        plt.title(opposite_pair)

    all_amplitudes = pd.DataFrame(all_amplitudes)[['opposite_pair', 
                                  'specification', 
                                  'amp']]
    print(all_amplitudes.to_string())
    
    # Make bar graph showing all amplitudes. Probably not included in paper? 
    plt.figure(figsize=[10, 5])
    x_loc = 0
    bar_width = .1
    space_between_bars = .3

    current_palette = sns.color_palette()

    specifications = ['original', 
    'days_after_last_cycle_start',
    'days_before_next_cycle_start',
    'frac_through_cycle']
                                      
    specifications_to_colors = dict(zip(specifications, 
                                        current_palette.as_hex()[1:5]))
                                                 
    specifications_to_bars = {}     
    pairs_to_plot = ORDERED_SYMPTOM_NAMES
    max_diff_from_100 = 0
    all_diffs_from_100 = []
    for opposite_pair in pairs_to_plot:
        original_val = None
        for specification in specifications:
            if specification not in specifications_to_bars and specification != 'original':
                specifications_to_bars[specification] = {'xs': [], 'ys':[]}
            
            val = all_amplitudes.loc[(all_amplitudes['opposite_pair'] == opposite_pair) & 
                                    (all_amplitudes['specification'] == specification)]
            assert len(val) == 1
            val = float(val['amp'])
            if specification == 'original':
                original_val = val
            else:
                val = 100 * val / original_val
                diff_from_100 = abs(val - 100)
                all_diffs_from_100.append(diff_from_100)
                if diff_from_100 > max_diff_from_100:
                    max_diff_from_100 = diff_from_100
                    max_diff_from_100_specification = specification
                    max_diff_from_100_opposite_pair = opposite_pair
            
                specifications_to_bars[specification]['xs'].append(x_loc)
                x_loc += bar_width
                specifications_to_bars[specification]['ys'].append(val)


        x_loc += space_between_bars
    xlim = [-bar_width, x_loc ]
    plt.plot(xlim, [100, 100], color='black', linestyle='--')
    plt.xlim(xlim)
    for specification in specifications_to_bars.keys():
        plt.bar(specifications_to_bars[specification]['xs'], 
                specifications_to_bars[specification]['ys'], 
                width=bar_width, 
                color=specifications_to_colors[specification], 
                label=specifications_to_pretty_names[specification])
    n_pairs = len(pairs_to_plot)
    plt.xticks((space_between_bars + 3*bar_width) * np.arange(n_pairs) + 1*bar_width, 
               [CANONICAL_PRETTY_SYMPTOM_NAMES[a] for a in pairs_to_plot])
    plt.legend()
    plt.ylim([0, 150])
    plt.ylabel("Amplitude relative to original estimate")
    plt.yticks([0, 25, 50, 75, 100, 125, 150], ['0%', '25%', '50%', '75%', '100%', '125%', '150%'])
    plt.xlabel("Dimension")
    print("Average difference from the original estimate is %2.1f%%; Largest difference from the original estimates is %2.1f%%\nSpecification: %s\nDimension: %s" % (
        np.mean(all_diffs_from_100),
        max_diff_from_100, 
        specifications_to_pretty_names[max_diff_from_100_specification], 
        CANONICAL_PRETTY_SYMPTOM_NAMES[max_diff_from_100_opposite_pair].replace('\n', ' ')))


def make_seasonal_regression_specifications_robustness_check(results):
    """
    Show that results don't vary too much under different parameterization of seasonal cycles. 
    """
    for opposite_pair in results:
        if HEART_SUBSTRING in opposite_pair:
            # no data for heart data. 
            continue
        data = deepcopy(results[opposite_pair]['by_very_active_northern_hemisphere_loggers'][True]['linear_regression_with_alternate_seasonal_specifications'])
        fake_results = {}
        fake_results[opposite_pair] = {'all_regression_specifications':{}}
        # give the regression specifications interpretable names. 
        pretty_spec_names = {'swap_out_both':'Both', 
                             'with_week_of_year_instead_of_month':'Week of year', 
                             'with_days_since_start_instead_of_year':'Linear time trend'}
        for specification in data:
            data_for_spec = data[specification]
            if specification == 'swap_out_both' or specification == 'with_week_of_year_instead_of_month':
                seasonal_variable = 'week_of_year'
            else:
                seasonal_variable = 'month'
            standard_format = compare_to_seasonal_cycles.convert_regression_format_to_simple_mean_format(
                data_for_spec,
                'linear_regression', 
                seasonal_variable=seasonal_variable)
            fake_results[opposite_pair]['all_regression_specifications'][pretty_spec_names[specification]] = {
                'take_simple_means_by_group_no_individual_mean':standard_format}

        # also extract original data. 
        original_data = compare_to_seasonal_cycles.convert_regression_format_to_simple_mean_format(
            results[opposite_pair]['by_very_active_northern_hemisphere_loggers'][True]['linear_regression'], 
            'linear_regression')

        (fake_results[opposite_pair]
         ['all_regression_specifications']
         ['Original']) = {'take_simple_means_by_group_no_individual_mean': original_data}
        period_amp = compare_to_seasonal_cycles.get_cycle_amplitude(
                fake_results[opposite_pair]['all_regression_specifications']['Original']['take_simple_means_by_group_no_individual_mean'],
                'date_relative_to_period',
                'max_minus_min',
                hourly_period_to_exclude=None)
        print '\n\n****Analyzing data for %s; star indicates that seasonal amp > period amp' % opposite_pair
        print("Period amp is %2.3f" % period_amp)
        for k in ['Week of year', 'Original', 'Linear time trend', 'Both']:
            if k in ['Original', 'Linear time trend']:
                cycle_to_test = 'month'
            else:
                cycle_to_test = 'week_of_year'
            amp = compare_to_seasonal_cycles.get_cycle_amplitude(
                fake_results[opposite_pair]['all_regression_specifications'][k]['take_simple_means_by_group_no_individual_mean'],
                cycle_to_test,
                'max_minus_min',
                hourly_period_to_exclude=None)
            print 'Amplitude for %s is %2.3f %s' % (k, amp, '*' if amp > period_amp else '')
        compare_to_seasonal_cycles.make_four_cycle_plots(fake_results, 
                                                         ['all_regression_specifications'], 
                                                         emotion_pairs=[opposite_pair], 
                                                         data_to_use='take_simple_means_by_group_no_individual_mean', 
                                                         suptitle=False, 
                                                         colors_for_lines={'Original':'magenta', 
                                                                 'Both':'green',
                                                                 'Week of year':'cyan', 
                                                                 'Linear time trend':'blue'},
                                                         use_expanded_seasonal_x_axis=True, 
                                                         show_errorbars=False)

def make_regression_specifications_robustness_plot(results):
    """
    show that results don't vary too much under mixed models or taking simple mean. 
    """
    def create_fake_results_for_specification_plot(results, symptom_pair):
        """
        Helper method. Loop over linear_regression, mixed_model_regression, and take_simple_means_by_group_no_individual_mean
        and get inferred trajectories + amplitudes. 
        """
        fake_results = {}
        fake_results[symptom_pair] = {'all_regression_specifications':{}}
        desired_regressions_to_plot = ['linear_regression', 
                  'mixed_model_regression', 
                  'take_simple_means_by_group_no_individual_mean']
        all_regressions_to_plot = [a for a in desired_regressions_to_plot if a in results.keys()]
        all_amplitudes = []
        for k in all_regressions_to_plot:
            if k == 'linear_regression' or k == 'mixed_model_regression':
                data = compare_to_seasonal_cycles.convert_regression_format_to_simple_mean_format(results[k], 
                                                                                                  k)
            else:
                data = results[k]

            pretty_reg_names = {'linear_regression':'Lin reg', 
                                'take_simple_means_by_group_no_individual_mean':'Means by group', 
                                'mixed_model_regression':'Mixed model'}
            fake_results[symptom_pair]['all_regression_specifications'][pretty_reg_names[k]] = {'take_simple_means_by_group_no_individual_mean':data}
            amplitudes_for_regression = {'regression':k, 'symptom':symptom_pair}
            for cycle in ['date_relative_to_period', 'local_hour', 'month', 'weekday']:
                if HEART_SUBSTRING in symptom_pair and cycle == 'month':
                    amplitudes_for_regression['%s_amp' % cycle] = 0
                else:
                    amplitudes_for_regression['%s_amp' % cycle] = compare_to_seasonal_cycles.get_cycle_amplitude(data,
                                                                   cycle, 
                                                                   metric_to_use='max_minus_min', 
                                                                   hourly_period_to_exclude=None)
            all_amplitudes.append(amplitudes_for_regression)

        if len(all_amplitudes) != len(desired_regressions_to_plot):
            all_amplitudes = []
        return fake_results, all_amplitudes

    amplitude_df = []
    for symptom_pair in results.keys():
        fake_results, amplitudes = create_fake_results_for_specification_plot(results[symptom_pair]['by_very_active_northern_hemisphere_loggers'][True],
                                               symptom_pair)
        amplitude_df += amplitudes

        if WEIGHT_SUBSTRING in symptom_pair:
            date_ylimit = .5
        else:
            date_ylimit = 10
        # make plots of date trajectory. 
        compare_to_seasonal_cycles.plot_means_by_date(results, 
                                                     [symptom_pair], 
                                                      data_to_use='means_by_date_no_individual_mean', 
                                                      min_date='2015-01-01', 
                                                      max_date='2018-01-01', 
                                                      ylimit=date_ylimit, 
                                                      min_obs=100, 
                                                      print_outlier_dates=False)
        compare_to_seasonal_cycles.plot_means_by_date(results, 
                                                     [symptom_pair], 
                                                      data_to_use='means_by_date', 
                                                      min_date='2015-01-01', 
                                                      max_date='2018-01-01', 
                                                      ylimit=date_ylimit, 
                                                      min_obs=100, 
                                                      print_outlier_dates=False)
        if symptom_pair == 'emotion*happy_versus_emotion*sad':
            figname = 'figures_for_paper/regression_specifications_happiness.png'
        elif WEIGHT_SUBSTRING in symptom_pair:
            figname = 'figures_for_paper/regression_specifications_weight.png'
        else:
            figname = None
        compare_to_seasonal_cycles.make_four_cycle_plots(fake_results, 
                                                    ['all_regression_specifications'], 
                                                     [symptom_pair], 
                                                     data_to_use='take_simple_means_by_group_no_individual_mean', 
                                                         colors_for_lines={'Lin reg':'magenta', 
                                                                 'Mixed model':'green',
                                                                 'Means by group':'cyan'}, 
                                                         suptitle=False, 
                                                         figname=figname)

    amplitude_df = pd.DataFrame(amplitude_df)

    print('printing out relative amplitudes under different specifications.')
    mixed_model_amps = (amplitude_df.loc[amplitude_df['regression'] == 'mixed_model_regression', 
                 ['symptom', 'date_relative_to_period_amp']]
                 .sort_values(by='symptom'))
    linear_regression_amps = (amplitude_df.loc[amplitude_df['regression'] == 'linear_regression', 
                 ['symptom', 'date_relative_to_period_amp']]
                    .sort_values(by='symptom'))
    ratios = []
    for i in range(len(linear_regression_amps)):
        assert mixed_model_amps['symptom'].iloc[i] == linear_regression_amps['symptom'].iloc[i]
        ratio = mixed_model_amps['date_relative_to_period_amp'].values[i] / linear_regression_amps['date_relative_to_period_amp'].values[i]
        print '%s %2.3f' % (mixed_model_amps['symptom'].values[i], ratio)


    print('plots of how much the amplitudes vary using the different regression specifications.')

    # I'm not a huge fan of these plots at present because the mixed model amplitudes are super-noisy 
    # (as you can see from the plots above) so it's not clear to me that we can trust them. 
    # Actually, the errorbars are likely unreliable for mixed models. 

    cycles = ['date_relative_to_period', 'local_hour', 'month', 'weekday']
    regression_types = list(set(amplitude_df['regression']))
    cycles_to_colors = {'date_relative_to_period':PERIOD_CYCLE_COLOR, 
                            'local_hour':HOUR_CYCLE_COLOR, 
                            'weekday':WEEKDAY_CYCLE_COLOR, 
                            'month':SEASONAL_CYCLE_COLOR}
    for symptom in list(set(amplitude_df['symptom'])):
        plt.figure()
        idxs = amplitude_df['symptom'] == symptom
        small_df = deepcopy(amplitude_df.loc[idxs])
        pretty_reg_names = {'linear_regression':'OLS', 'mixed_model_regression':'Mixed', 
                           'take_simple_means_by_group_no_individual_mean':'simple mean'}
        small_df['regression'] = small_df['regression'].map(lambda x:pretty_reg_names[x])
        for cycle in cycles:
            plt.plot(small_df['%s_amp' % cycle].values, color=cycles_to_colors[cycle], label=cycle)
        if 'sex*' in symptom:
            plt.ylim([0, .2])
        elif WEIGHT_SUBSTRING in symptom:
            plt.ylim([0, 1])
        elif HEART_SUBSTRING in symptom:
            plt.ylim([0, 3])
        elif BBT_SUBSTRING in symptom:
            plt.ylim([0, 1])
        else:
            plt.ylim([0, .2])
        plt.xticks(range(len(small_df)), small_df['regression'])
        plt.legend()
        plt.title(symptom)
        plt.xlabel("Regression specification")
        plt.ylabel("Amplitude")

    # Confirm that mood symptoms are still largest for period symptoms under mixed model specifications
    for specification in ['linear_regression', 'mixed_model_regression']:
        print 'Average amplitude of mood symptoms under specification %s' % specification
        idxs = ((amplitude_df['regression'] == specification) & 
         (amplitude_df['symptom'].map(lambda x:'sex*' not in x 
                                      and 'exercise*' not in x 
                                      and 'sleep*' not in x 
                                      and 'continuous' not in x)))
        assert idxs.sum() == 9
        print amplitude_df.loc[idxs, ['%s_amp' % cycle for cycle in cycles]].mean()

    return amplitude_df
            

def make_mood_symptoms_robustness_plot_without_opposite_symptoms():
    """
    rather than expressing each mood symptom relative to its opposite, 
    look at p(mood_symptom | at_least_one_symptom_on_screen). 
    This method is a bit confusing because it actually does two things. 
    First, it looks at mood symptoms: what happens if we plot them using the non-opposite parameterization, is the period cycle still largest on average?
    Second, it looks at all the symptoms we don't include in the main analysis, makes sure the period cycle is still large. 
    """
    cycles = ['date_relative_to_period', 'local_hour', 'month', 'weekday']
    individual_symptom_files = sorted([a for a in os.listdir(base_results_dir) if 'fluctuations' in a])
    all_fluctuation_amplitudes = []
    for f in sorted(individual_symptom_files):
        if 'continuous' in f:
            # doesn't make sense for these symptoms, since it's probability of emitting
            # a binary symptom. 
            continue
        opposite_pair = f.replace('fluctuations_in_symptom_logged_', '').replace('.pkl', '')
        is_mood_symptom = any([cat in opposite_pair and 
                               ('pms' not in opposite_pair) for cat in MOOD_SYMPTOM_CATEGORIES])

        data_by_hour_and_not_by_hour = cPickle.load(open(os.path.join(base_results_dir, f), 'rb'))
        
        included_in_main_analysis = (is_mood_symptom 
                                     or ('sex*' in opposite_pair and 'drive' not in opposite_pair)
                                     or 'exercise*' in opposite_pair
                                     or 'sleep*' in opposite_pair 
                                     or 'continuous*' in opposite_pair)

        results_for_opposite_pair = {'symptom':opposite_pair.replace('symptom_', '').split('_versus')[0][:50], 
                                     'is_mood_symptom':is_mood_symptom, 
                                     'included_in_main_analysis':included_in_main_analysis}

        for use_hour in ['by_hour', 'not_by_hour']:
            print(f, data_by_hour_and_not_by_hour[use_hour].keys())
            if 'normalizer: all_mood_symptoms' in data_by_hour_and_not_by_hour[use_hour]:
                normalizer = 'normalizer: symptoms_on_screen'
            else:
                normalizer = 'normalizer: none'

            individual_symptom_data = data_by_hour_and_not_by_hour[use_hour][normalizer]
            results_to_plot = {}
            results_to_plot[opposite_pair] = {'no_substratification':{}}
            results_to_plot[opposite_pair]['no_substratification']['linear_regression'] = individual_symptom_data
            individual_symptom_data = compare_to_seasonal_cycles.convert_regression_format_to_simple_mean_format(individual_symptom_data, 
                                                                                                                 'linear_regression')
            for cycle in cycles:
                results_for_opposite_pair[cycle + '_' + use_hour] = compare_to_seasonal_cycles.get_cycle_amplitude(individual_symptom_data, 
                                                               cycle, 
                                                               metric_to_use='max_minus_min', 
                                                               hourly_period_to_exclude=None)
            largest_cycle_not_including_daily = sorted([a for a in cycles if a != 'local_hour'], 
                                                       key=lambda cycle:results_for_opposite_pair[cycle + '_' + use_hour])[::-1][0]
            results_for_opposite_pair['largest_cycle_not_including_daily_%s' % use_hour] = largest_cycle_not_including_daily
            if is_mood_symptom and (use_hour == 'by_hour'):

                compare_to_seasonal_cycles.make_four_cycle_plots(results_to_plot, 
                                                                    ['no_substratification'], 
                                                                    [opposite_pair], 
                                                                    data_to_use='linear_regression', 
                                                                 suptitle=False)
        all_fluctuation_amplitudes.append(results_for_opposite_pair)

    all_fluctuation_amplitudes = pd.DataFrame(all_fluctuation_amplitudes)[['symptom'] + 
                                                                          ['is_mood_symptom', 'included_in_main_analysis'] + 
                                                                          [a + '_not_by_hour' for a in cycles] + 
                                                                          [a + '_by_hour' for a in cycles] + 
                                                                          ['largest_cycle_not_including_daily_not_by_hour']]
    print("All mood symptoms are")
    print all_fluctuation_amplitudes.loc[all_fluctuation_amplitudes['is_mood_symptom'] == True, 'symptom'].values
    print 'Averaged across all mood symptoms, the average fluctuation amplitude is'
    print all_fluctuation_amplitudes.loc[all_fluctuation_amplitudes['is_mood_symptom'] == True, 
                                   ['symptom'] + [cycle + '_by_hour' for cycle in cycles]].mean()


    ### Show that when we look at all symptoms (not just the ones included) menstrual cycle > weekly + seasonal cycles 
    pd.set_option('display.width', 500)
    print("Printing out which cycle is largest, not including local hour...")
    not_included_in_analysis = all_fluctuation_amplitudes.loc[all_fluctuation_amplitudes['included_in_main_analysis'] == False]
    for cycle in sorted(list(set(not_included_in_analysis['largest_cycle_not_including_daily_not_by_hour']))):
        
        idxs = not_included_in_analysis['largest_cycle_not_including_daily_not_by_hour'] == cycle
        print '%s: largest for %i/%i symptoms' % (cycle, idxs.sum(), len(idxs))
        print (not_included_in_analysis.loc[idxs, ['symptom', cycle + '_not_by_hour']]
               .sort_values(by=cycle + '_not_by_hour')[::-1]).to_string()


def make_country_robustness_plot(results):
    """
    Scatter plot showing country effects don't vary too much by specification. 
    """
    plt.figure(figsize=[13.5, 15])
    fontsize = 13
    for opposite_idx, opposite_pair in enumerate(ORDERED_SYMPTOM_NAMES):
        period_effects_with_covariates = (results[opposite_pair]
         ['no_substratification']
         ['period_effects_with_covariates'])

        specifications_to_use = ['country', 'country+age', 'country+age+behavior', 'country+age+behavior+app usage']
        
        wide_df = {}
        countries = None
        for specification in specifications_to_use:
            if countries is not None:
                assert countries == sorted(period_effects_with_covariates[SHORT_NAMES_TO_REGRESSION_COVS[specification]]['predicted_effects_by_country'].keys())
            else:
                countries = sorted(period_effects_with_covariates[SHORT_NAMES_TO_REGRESSION_COVS[specification]]['predicted_effects_by_country'].keys())
            vals = [period_effects_with_covariates[SHORT_NAMES_TO_REGRESSION_COVS[specification]]['predicted_effects_by_country'][c] for c in countries]
            wide_df[specification] = vals
        wide_df = pd.DataFrame(wide_df)
        plt.subplot(5, 3, opposite_idx + 1)
        lim = np.abs(wide_df.values).max() * 1.1
        plt.plot([-lim, lim], [0, 0], color='grey', alpha=.5, linestyle='--')
        plt.plot([0, 0], [-lim, lim], color='grey', alpha=.5, linestyle='--')
        min_correlation = 1
        # loop over alternate specifications and make sure they'll well-correlated. 
        for spec2 in wide_df.columns:
            if spec2 == 'country':
                continue
            plt.scatter(wide_df['country'], 
                        wide_df[spec2], 
                        alpha=1, 
                        s=3, 
                        label=spec2)
            r, p = pearsonr(wide_df['country'], wide_df[spec2])
            if r < min_correlation:
                min_correlation = r
        plt.title(CANONICAL_PRETTY_SYMPTOM_NAMES[opposite_pair].replace('\n', ' ') + '\n(min $r=%2.2f$)' % min_correlation, fontsize=fontsize)
        
        plt.ylim([-lim, lim])
        plt.xlim([-lim, lim])
        plt.plot([-lim, lim], [-lim, lim], color='black')
        
        if opposite_idx % 3 == 0:
            plt.ylabel('Alternate specification', fontsize=fontsize)
        if opposite_idx >= 12:
            plt.xlabel('Country-only specification', fontsize=fontsize)
        if opposite_idx == 2:
            plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize=fontsize)
    plt.subplots_adjust(left=.08, right=.65, top=.95, bottom=.05, wspace=.3, hspace=.4)
    plt.savefig('figures_for_paper/country_robustness.png', dpi=300)
    plt.show()

def deprecated_country_specific_effects_plot(results):
    # country-specific effects. THIS IS DEPRECATED, WE ARE NOT USING IT AT PRESENT. 
    raise Exception("THIS IS DEPRECATED, JUST POKING AROUND. DON'T BOTHER CHECKING.")

    for opposite_pair in results.keys():
        period_effects_with_covariates = (results[opposite_pair]
         ['no_substratification']
         ['period_effects_with_covariates'])
        baseline_effect = (period_effects_with_covariates['near_period']['predicted_effects_by_country'])
        assert len(set(baseline_effect.values())) == 1
        baseline_effect = baseline_effect.values()[0]
        specifications_to_use = ['country', 'country+age', 'country+age+behavior', 'country+age+behavior+app usage']
        
        df_to_plot = {'specification':[], 
                     'Period effect by country':[], 
                      'country':[]}
        wide_df = {}
        countries = None
        plt.figure(figsize=[10, 5])
        plt.subplot(131)
        for specification in specifications_to_use:
            if countries is not None:
                assert countries == sorted(period_effects_with_covariates[SHORT_NAMES_TO_REGRESSION_COVS[specification]]['predicted_effects_by_country'].keys())
            else:
                countries = sorted(period_effects_with_covariates[SHORT_NAMES_TO_REGRESSION_COVS[specification]]['predicted_effects_by_country'].keys())
            vals = [period_effects_with_covariates[SHORT_NAMES_TO_REGRESSION_COVS[specification]]['predicted_effects_by_country'][c] for c in countries]
            df_to_plot['specification'] += [specification.replace('+', '+\n') for a in range(len(vals))]
            df_to_plot['Period effect by country'] += vals
            df_to_plot['country'] += countries
            wide_df[specification] = vals
            plt.plot(vals, label=specification)
        plt.xlabel("Country (same order, unlabeled)")
        plt.legend()
        plt.ylabel("Period effect by country")
        
        df_to_plot = pd.DataFrame(df_to_plot)
        # botplot of country effects, showing consistency. 
        plt.subplot(132)
        sns.boxplot(x='specification', 
                       y='Period effect by country', 
                       data=df_to_plot, 
                    color='steelblue')
        plt.plot([-.5, len(specifications_to_use) - .5], 
                 [baseline_effect, baseline_effect],
                 color='black', 
                 linestyle='--')
        plt.title(opposite_pair + '\n%i countries' % len(countries))

        # print out some correlations
        wide_df = pd.DataFrame(wide_df)
        
        correlations_across_specifications = []
        for i in range(len(specifications_to_use)):
            for j in range(i):
                slope, intercept, r_value, p_value, std_err = linregress(wide_df[specifications_to_use[i]], 
                                                                         wide_df[specifications_to_use[j]])
                correlations_across_specifications.append({'spec 1':specifications_to_use[i], 
                                                          'spec 2':specifications_to_use[j], 
                                                          'slope':slope, 
                                                          'r_value':r_value})
        
        
        plt.subplot(133)
        plt.scatter(wide_df['country'], 
                    wide_df['country+age+behavior+app usage'])
        
        
        correlations_across_specifications = pd.DataFrame(correlations_across_specifications)
        
        print opposite_pair + ': %i countries' % len(countries)
        print 'Worst correlations between specifications (correlations are taken across countries)'
        print correlations_across_specifications.sort_values(by='r_value').head(n=1)
        print 'Smallest slopes between specifications (again, across country)'
        print correlations_across_specifications.sort_values(by='slope').head(n=1)
        
        
        #sns.pairplot(wide_df)

def make_period_lengths_plot(n_chunks_to_use):
    """
    # we compare to Chiazze 1968: "The length and variability of the human menstrual cycle"
    # Their numbers are reproduced in "Menstruation and Menstrual Disorders: The Epidemiology of Menstruation and Menstrual Dysfunction"
    # We compare to the numbers filtering between 15 - 45 day cycles but the numbers are also similar if 
    # you apply looser filtering to our dataset (just to avoid users who forget a cycle) 
    # and use their unfiltered numbers. 
    """
    chunks_to_use = range(n_chunks_to_use)
    ages_to_all_period_lengths = {}
    chiazze_numbers = {'15-19':{'their_mean':29.3, 'their_std':4.74},
                       '20-24':{'their_mean':29.1, 'their_std':4.55}, 
                       '25-29':{'their_mean':28.5, 'their_std':3.61},
                       '30-34':{'their_mean':28.0, 'their_std':3.45},
                       '35-39':{'their_mean':27.3, 'their_std':3.39},
                       '40-44':{'their_mean':26.9, 'their_std':3.70}}
    users_examined = set()
    for chunk in chunks_to_use:
        print("Loading period length chunk %i" % chunk)
        path = os.path.join(processed_chunked_tracking_dir, 'users_to_period_starts_chunk_%i.pkl' % chunk)
        period_starts_for_chunk = cPickle.load(open(path, 'rb'))
        path = os.path.join(processed_chunked_tracking_dir, 
                        'user_features', 
                        'user_features_chunk_%i.pkl' % chunk)
        user_features_for_chunk = cPickle.load(open(path, 'rb'))

        # categorical_age
        def do_categorical_age_the_way_Chiazze_does(age):
            if np.isnan(age):
                return None
            if age >= 15 and age < 20:
                return '15-19'
            if age >= 20 and age < 25:
                return '20-24'
            if age >= 25 and age < 30:
                return '25-29'
            if age >= 30 and age < 35:
                return '30-34'
            if age >= 35 and age < 40:
                return '35-39'
            if age >= 40 and age < 45:
                return '40-44'
            return None

        for user in period_starts_for_chunk:
            if user not in user_features_for_chunk:
                continue # some users have period starts but aren't included in analysis because no time info. 
            chiazze_user_age = do_categorical_age_the_way_Chiazze_does(user_features_for_chunk[user]['age'])

            if chiazze_user_age not in ages_to_all_period_lengths:
                ages_to_all_period_lengths[chiazze_user_age] = []
            assert user not in users_examined
            users_examined.add(user)
            for i in range(len(period_starts_for_chunk[user]) - 1):
                length = (period_starts_for_chunk[user][i + 1] - period_starts_for_chunk[user][i]).days
                assert length >= 0
                if length > 0: # sometimes lengths can be zero because users log eg light + medium bleeding on same day. 
                    ages_to_all_period_lengths[chiazze_user_age].append(length)


    for categorical_age in sorted(ages_to_all_period_lengths.keys()):
        period_lengths = np.array(ages_to_all_period_lengths[categorical_age])
        plt.figure()
        plt.hist(period_lengths,
                 bins=range(100))
        plt.xlabel("Menstrual cycle length")
        plt.ylabel("Total number of cycles")
        plt.title('Age group %s' % categorical_age)
        plt.show()
        period_length_counts = Counter(period_lengths)
        all_period_lengths = np.array(period_lengths)
        print("Categorical age %s" % categorical_age)
        print("Total period count: %i" % len(period_lengths))
        min_outlier_cutoff = 15
        max_outlier_cutoff = 45
        non_outliers = (period_lengths <= max_outlier_cutoff) & (period_lengths >= min_outlier_cutoff)
        if categorical_age in chiazze_numbers:
            chiazze_numbers[categorical_age]['our_mean'] = np.mean(period_lengths[non_outliers])
            chiazze_numbers[categorical_age]['our_std'] = np.std(period_lengths[non_outliers])
            chiazze_numbers[categorical_age]['our_mean_higher_cutoff'] = np.mean(period_lengths[period_lengths < 60])
            chiazze_numbers[categorical_age]['our_std_higher_cutoff'] = np.std(period_lengths[period_lengths < 60])
        print("Mean length: %2.1f\nmedian length %2.1f\nstd in lengths %2.1f\n5th pctile %2.1f\n95th pctile %2.1f\nmean discarding outliers outside of %i-%i, %2.1f\nmedian %2.1f\nstd discarding outliers >=%2.2f\nmin length %i\nmax length %i" % (np.mean(period_lengths),
                                                                             np.median(period_lengths), 
                                                                             np.std(period_lengths),
                                                                             scoreatpercentile(period_lengths, 5), 
                                                                             scoreatpercentile(period_lengths, 95),
                                                                             min_outlier_cutoff,    
                                                                             max_outlier_cutoff,
                                                                             np.mean(period_lengths[non_outliers]), 
                                                                             np.median(period_lengths[non_outliers]), 
                                                                             np.std(period_lengths[non_outliers]), 
                                                                             np.min(period_lengths),
                                                                             np.max(period_lengths)))


    # actual figure for paper
    plt.figure(figsize=[5, 4])
    chiazze_ages = ['15-19', '20-24', '25-29', '30-34', '35-39', '40-44']

    plt.errorbar(x=np.arange(len(chiazze_ages)) + .03, # horizontal offset so errorbars don't overlap, 
                 y=[chiazze_numbers[age]['our_mean'] for age in chiazze_ages], 
                 yerr=[chiazze_numbers[age]['our_std'] for age in chiazze_ages], 
                 label='Clue data', 
                 capsize=5)
    plt.errorbar(x=np.arange(len(chiazze_ages)), 
                 y=[chiazze_numbers[age]['their_mean'] for age in chiazze_ages], 
                 yerr=[chiazze_numbers[age]['their_std'] for age in chiazze_ages], 
                 label='Chiazze et al, 1968', 
                 capsize=5)
    plt.xticks(range(len(chiazze_ages)), chiazze_ages, fontsize=14)
    plt.ylim([0, 35])
    plt.ylabel("Menstrual cycle length (days)", fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlabel("Age", fontsize=14)
    plt.legend(loc=4, fontsize=14)
    plt.subplots_adjust(bottom=.15, left=.2)
    plt.savefig('figures_for_paper/validation_menstrual_cycle_lengths.png', dpi=300)
    print pd.DataFrame(chiazze_numbers)

def recapitulate_country_specific_happiness_trends(results, min_obs, min_users):
    """
    Show that we can recapitulate country-specific happiness variation. 
    Uses Gallup data. Source: https://www.gallup.com/analytics/213617/gallup-analytics.asp
    """

    files = [a for a in os.listdir('external_happiness_datasets') if 'GallupAnalytics_Export_20180828' in a]
    combined_gallup_d = None
    for f in files:
        question = f.split('-')[1].replace('.csv', '').strip()
        print question
        gallup_d = pd.read_csv(os.path.join('external_happiness_datasets', f), skiprows=7)
        print gallup_d.head()
        if question in ['Life Today', 'Negative Experience Index', 'Positive Experience Index']:
            gallup_d[question] = gallup_d['Value'].map(float)
        else:
            raise Exception("This is deprecated, we should not be using these questions")
            gallup_d[question] = gallup_d['Yes'].map(lambda x:float(x.replace('%', '')))
        gallup_d = gallup_d.loc[gallup_d['Demographic'] == 'Gender']
        assert (gallup_d['Time'] == 2017).all()
        gallup_d = gallup_d.loc[gallup_d['Demographic Value'] == 'Female', ['Geography', question]]
        if combined_gallup_d is None:
            combined_gallup_d = gallup_d
        else:
            combined_gallup_d = pd.merge(combined_gallup_d, gallup_d, on='Geography', how='outer')

    print 'Number of countries', len(combined_gallup_d)
    print 'Number of countries with full data', len(combined_gallup_d.dropna())
    combined_gallup_d = combined_gallup_d.dropna()

    # correlate with our happy/sad levels. 
    for symptom_pair in ['emotion*happy_versus_emotion*sad']:
        print("******\n%s" % symptom_pair)
        d = results[symptom_pair]

        country_stratifications = d['by_country']
        country_names = sorted(country_stratifications.keys())
        our_data = []

        for country in country_names:
            #print results[symptom_pair]['by_country'][country].keys()
            if ((results[symptom_pair]['by_country'][country]['overall_n_obs'] >= min_obs) and 
               (results[symptom_pair]['by_country'][country]['overall_n_users'] >= min_users)):
                our_data.append({'country':country, 
                                 'country_mean':results[symptom_pair]['by_country'][country]['overall_positive_frac']})
        our_data = pd.DataFrame(our_data)
        for c in combined_gallup_d.columns:
            if c != 'Geography':
                column_d = dict(zip(combined_gallup_d['Geography'],
                                combined_gallup_d[c]))

                our_data[c] = our_data['country'].map(lambda x:column_d[x] if x in column_d else None)
                df_to_regress = our_data[[c, 'country_mean']].dropna()
                slope, intercept, r_value, p_value, std_err = linregress(df_to_regress[c], 
                                                             df_to_regress['country_mean'])

                plt.figure(figsize=[4, 4])
                plt.title("r=%2.2f, p=%2.1e, n countries=%i" % (r_value, p_value, len(df_to_regress)))
                plt.scatter(df_to_regress[c].values, df_to_regress['country_mean'].values)
                plt.xlabel('Gallup data: %s' % c)
                plt.ylabel("Our data:\n%s" % symptom_pair)
                plt.show()
                        
def recapitulate_country_specific_weight_trends(results, min_obs, min_users):
    for filename in ['NCD_BMI_25A.csv',
                'NCD_BMI_30A.csv',
                 'weight_by_country_both_men_and_women.csv']:
    
        if filename == 'weight_by_country_both_men_and_women.csv':
            # data source: https://en.wikipedia.org/wiki/Human_body_weight#Global_statistics
            # which gets it from a paper. Note -- this is not gender-specific, and it would be good to get gender-specific data. 
            weight_file = open(os.path.join('external_happiness_datasets/%s' % filename))
            weight_file.next()
            external_weight_data_by_country = {}

            for line in weight_file:
                l = line.split()
                if len(l) == 0:
                    continue
                country_name = ' '.join(l[1:-2])
                weight = float(l[-1])
                external_weight_data_by_country[country_name] = weight
            external_weight_data_by_country['Korea (South)'] = external_weight_data_by_country['South Korea']
            external_weight_data_by_country['Britain (UK)'] = external_weight_data_by_country['United Kingdom']
            external_weight_data_by_country['Dominican Republic'] = external_weight_data_by_country['Dominican Rep.']
            external_weight_data_by_country['United Arab Emirates'] = external_weight_data_by_country['UAE']
        else: 
            # WHO data
            # Sources are http://apps.who.int/gho/data/view.main.GDO2105v and
            # http://apps.who.int/gho/data/view.main.GDO2106v
            # see also https://ourworldindata.org/obesity which I think has the same data. 
            who_weight_data_by_country = pd.read_csv('external_happiness_datasets/%s' % filename)
            desired_col = '2016.2'
            n_header_rows = 3
            header = list(who_weight_data_by_country[desired_col].iloc[:n_header_rows])
            if filename == 'NCD_BMI_25A.csv':
                assert list(header)[0] == ' Prevalence of overweight among adults, BMI &amp;GreaterEqual, 25 (age-standardized estimate) (%)'
            else:
                assert list(header)[0] == ' Prevalence of obesity among adults, BMI &amp;GreaterEqual, 30 (age-standardized estimate) (%)'
            print list(header)[0]
            assert list(header)[1] == ' 18+  years' 
            assert list(header)[2] == ' Female'
            
            countries = list(who_weight_data_by_country['Unnamed: 0'].iloc[n_header_rows:])
            
            def convert_who_data_to_float(a):
                if a == "No data":
                    return np.nan
                else:
                    return float(a.split()[0])

            vals = [convert_who_data_to_float(a) for a in list(who_weight_data_by_country[desired_col].iloc[n_header_rows:])]
            
            external_weight_data_by_country = dict(zip(countries, vals))
            for k in countries:
                if np.isnan(external_weight_data_by_country[k]):
                    del external_weight_data_by_country[k]
            external_weight_data_by_country['Vietnam'] = external_weight_data_by_country['Viet Nam']
            external_weight_data_by_country['Korea (South)'] = external_weight_data_by_country['Republic of Korea']
            external_weight_data_by_country['Britain (UK)'] = external_weight_data_by_country['United Kingdom of Great Britain and Northern Ireland']
            external_weight_data_by_country['Venezuela'] = external_weight_data_by_country['Venezuela (Bolivarian Republic of)'] 
            external_weight_data_by_country['United States'] = external_weight_data_by_country['United States of America'] 
            external_weight_data_by_country['Russia'] = external_weight_data_by_country['Russian Federation'] 

        our_weight_data_by_country = []
        for country in results['continuous_features*weight_versus_continuous_features*null']['by_country'].keys():
            if country == 'Taiwan':
                continue
            if ((results['continuous_features*weight_versus_continuous_features*null']['by_country'][country]['overall_n_obs'] >= min_obs) and 
               (results['continuous_features*weight_versus_continuous_features*null']['by_country'][country]['overall_n_users'] >= min_users)):
                our_weight_data_by_country.append({'country':country, 
                                                   'our_val':results['continuous_features*weight_versus_continuous_features*null']['by_country'][country]['overall_positive_frac'], 
                                                   'external_val':external_weight_data_by_country[country]})
            
            
        
        our_weight_data_by_country = pd.DataFrame(our_weight_data_by_country)
        plt.scatter(our_weight_data_by_country['our_val'], 
                    our_weight_data_by_country['external_val'])
        slope, intercept, r_value, p_value, std_err = linregress(our_weight_data_by_country['our_val'],
                                                                 our_weight_data_by_country['external_val'])
        plt.xlabel("Average weight of Clue users", fontsize=16)
        plt.ylabel("%s" % filename.replace('-', ' ').replace('.csv', ''), fontsize=16)
        plt.title("r: %2.2f; p: %2.3e; countries: %i" % (r_value, p_value, len(our_weight_data_by_country)))
        plt.show()



def make_previously_known_cycles_plot():
    """
    Plot previously known menstrual, weekly, and seasonal cycles. 
    """
    symptoms = {'month':['ailment*allergy_ailment', 
                             'appointment*vacation_appointment', 
                         'ailment*cold_flu_ailment'],
                'date_relative_to_period':['pain*cramps', 
                                           'pain*tender_breasts',
                         'pain*headache',
                         'pain*ovulation_pain'], 
                'weekday':['party*drinks_party','party*hangover']}
    pretty_symptom_names = {'ailment*allergy_ailment':'Allergies', 
                    'ailment*cold_flu_ailment':'Cold/flu', 
                    'appointment*vacation_appointment':'Vacation', 
                            'pain*cramps':'Cramps', 
                            'pain*headache':'Headache', 
                            'pain*ovulation_pain':'Ovulation pain', 
                            'pain*tender_breasts':'Tender breasts', 
                            'party*drinks_party':'Party with alcohol', 
                            'party*hangover':'Hangover'}

    plt.figure(figsize=[12, 5])
    fontsize=14
    for subplot_idx, cycle in enumerate(['date_relative_to_period', 'month', 'weekday']):
        plt.subplot(1, 3, subplot_idx + 1)
        for symptom in symptoms[cycle]:

            filename = 'fluctuations_in_symptom_logged_symptom_%s_versus_didnt*didnt_log_symptom.pkl' % symptom
            data = cPickle.load(open(os.path.join(base_results_dir, filename), 'rb'))['not_by_hour']['normalizer: none']
            data = compare_to_seasonal_cycles.convert_regression_format_to_simple_mean_format(data, 'linear_regression')
            xtick_kwargs = {'rotation':90, 'fontsize':fontsize}#, 'fontweight':'bold'}

            if cycle == 'month':
                xs = data[cycle].index
                assert list(xs) == range(1, 13)
                plt.xticks([1, 4, 7, 10], ['Jan', 'Apr', 'Jul', 'Oct'], **xtick_kwargs)
                ylimit = .08
                middle_tick = ''
            elif cycle == 'date_relative_to_period':
                xs = data[cycle].index
                assert list(xs) == range(-14, 15)
                ylimit = .3
                plt.xticks([-14, -7, 0, 7, 14], 
                           ['Day -14', 'Day -7', 'Day 0:\nperiod start', 'Day 7', 'Day 14'], 
                           **xtick_kwargs)
                middle_tick = 'baseline'
            elif cycle == 'weekday':
                xs = range(7)
                data[cycle] = data[cycle].loc[['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']]
                plt.xticks(range(7),
                           ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'],
                           **xtick_kwargs)
                middle_tick = ''
                ylimit = .05
            plt.ylim([-ylimit, ylimit])

            plt.yticks([-ylimit, 0, ylimit], 
                       ['-%.0f%%' % (ylimit * 100), middle_tick, '+%.0f%%' % (ylimit * 100)], 
                       fontsize=fontsize), 
                      # fontweight='bold')

            ys = data[cycle]['mean'].values
            yerr = data[cycle]['err'].values
            plt.plot([min(xs), max(xs)], [0, 0], color='black')
            ys = ys - ys.mean()
            plt.errorbar(xs, 
                     ys, 
                     yerr, 
                     label=pretty_symptom_names[symptom])
            plt.title(PRETTY_CYCLE_NAMES[cycle], fontsize=fontsize + 2)#, fontweight='bold')

            plt.xlim([min(xs), max(xs)])
            plt.legend(loc=3, fontsize=fontsize - 3, handlelength=1)
    plt.subplots_adjust(left=.1, bottom=.30, wspace=.25, right=.95)
    plt.savefig('figures_for_paper/previously_known_cycles.png', dpi=300)
    plt.show()
