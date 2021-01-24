This repo contains code to reproduce results in the paper "Daily, weekly, seasonal and menstrual cycles in women’s mood, behaviour and vital signs". Data that support the findings of this paper are available with appropriate permission from Clue. The data are not publicly available to preserve the privacy of Clue users. Please contact emmap1@cs.stanford.edu with any questions about data or code. 

# Citation

"Daily, weekly, seasonal and menstrual cycles in women’s mood, behaviour and vital signs". Emma Pierson, Tim Althoff, Daniel Thomas, Paula Hillard, and Jure Leskovec. *Nature Human Behaviour*, 2021. 

# Files

**requirements.txt**: Package versions for pip virtualenv. 

**constants_and_util.py**: Constants, paths, and utility methods used throughout the analysis. 

**dataprocessor.py**	: Helper methods for processing the original data from Clue. 		
	
**compare_to_seasonal_cycles.py**: Primary code used for heavy lifting in the analysis; saves cached results which are used to generate figures in paper. 

**generate_results_for_paper.py**: Creates figures used in the actual paper. 

**how_many_users_are_there.py**: Counts people and observations in the original dataset.

**make_plots_for_each_feature.py**: Decompose each feature logged in the Clue data into four cycles (even for features not analyzed in the main analysis). 

**opposite_emotional_features.csv**: helper file: pairs of emotions treated as opposite pairs in the mood analysis (eg, happy/sad).

**results_for_paper.ipynb**: Generates the main figures in the paper. 

**robustness_checks.ipynb**: Generates supplementary figures/tables and robustness checks. 

# Reproducing results

Analysis was run using a Python 2.7.12 virtualenv with package versions specified in `requirements.txt` included in this repo. As a warning, the original files are large, and the analysis starts multiple threads in parallel; the original analysis was performed on a cluster with hundreds of cores and terabytes of RAM. 
 
#### Data processing

1. Break data into more manageable pieces: run
```
nohup python -u dataprocessor.py split_original_tracking_data_into_manageable_pieces > split_original_tracking_data_into_manageable_pieces.out & 
```

in a screen session or similar. 

2. Add cycle annotations (date relative to period, etc): run 
```
nohup python -u dataprocessor.py annotate_with_cycle_days > annotate_with_cycle_days.out &. 
```
Warning: this will start 64 threads (one for each chunk). Both steps 1 + 2 will take ~8 hours to run.  

3. Add user feature annotations: run 
```
nohup python -u dataprocessor.py compute_user_features_and_individual_symptoms > compute_user_features_and_individual_symptoms.out &
```
This should be relatively fast, but it will also use 64 threads.

#### Analysis

To actually analyze the data, run 
```
nohup python -u compare_to_seasonal_cycles.py > compare_to_seasonal_cycles.py &
```

this will perform analysis for each of the 15 mood, behavior, and vital sign dimensions discussed in the paper and cache the results in .pkl files. Currently, this will take a very long time to run because `N_BOOTSTRAP_ITERATES` in `constants_and_util.py` is set to 1000. While this is used to generate precise confidence intervals, you can greatly decrease the runtime of the code (and still get point estimates) by reducing this constant. 

The plots and figures in the paper are generated using the ipython notebooks once `compare_to_seasonal_cycles` has finished running and the results have been cached. 



 
