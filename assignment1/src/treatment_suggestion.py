# Serena Zafiris, Donovan Moini, Lucille Njoo
# Treatment Suggestion

import pandas as pd
from pomegranate import *
import itertools

COLUMN_NAMES = ['X', 'Z', 'Y', 'M', 'W']
LABEL_INDEX = {l:i for i, l in enumerate(COLUMN_NAMES)}

def treatment_suggestion():
    obs_data = pd.read_csv('../dat/medical_studies/med_ex_obs.csv')
    exp_data = pd.read_csv('../dat/medical_studies/med_ex_exp.csv')

    obs_model = BayesianNetwork('Observational Model').from_samples(
        obs_data, state_names=COLUMN_NAMES)
    exp_model = BayesianNetwork('Experimental Model').from_samples(
        exp_data, state_names=COLUMN_NAMES)
    
    # P(Y=1 | W=0, M=0, X=x) on experimental model
    # Save best x as best_treatment
    best_treatment = (-1, -1)
    for x in (0, 1):
        exp_data = exp_model.predict_proba({'W': 0, 'M': 0, 'X': x})[LABEL_INDEX['Y']].probability(1)
        print(f'{x}: {exp_data}')
        if exp_data > best_treatment[1]:
            best_treatment = (x, exp_data)
    print(f'Best treatment: {best_treatment}\n')


    # P(X=x | W=0, M=0) for x=best_treatment[0] (in this case, x=0) on observational data
    # P(X=0 | W=0, M=0)
    prescribing_optimal_treatment = obs_model.predict_proba({'W': 0, 'M': 0})[LABEL_INDEX['X']].probability(best_treatment[0])
    prescribing_not_optimal_treatment = 1 - prescribing_optimal_treatment
    print(f'Prescribing optimal treatment:\t{prescribing_optimal_treatment}')
    print(f'Not prescribing best treatment:\t{prescribing_not_optimal_treatment}')

    return True

treatment_suggestion()