# Serena Zafiris, Donovan Moini, Lucille Njoo
# Model Validator

import pandas as pd
from pomegranate import *
import itertools

COLUMN_NAMES = ['X', 'Z', 'Y', 'M', 'W']
LABEL_INDEX = {l:i for i, l in enumerate(COLUMN_NAMES)}


def model_validator():
    obs_data = pd.read_csv('../dat/medical_studies/med_ex_obs.csv')
    exp_data = pd.read_csv('../dat/medical_studies/med_ex_exp.csv')

    obs_model = BayesianNetwork('Observational Model').from_samples(
        obs_data, state_names=COLUMN_NAMES)
    exp_model = BayesianNetwork('Experimental Model').from_samples(
        exp_data, state_names=COLUMN_NAMES)
    
    # P(Y | X=x)
    obs_prob = {}
    for x, w in itertools.product((0, 1), repeat=2):
        obs_prob[x] = obs_prob.get(x, 0) + obs_model.predict_proba({'X': x, 'W': w})[LABEL_INDEX['Y']].probability(1) * obs_model.predict_proba({})[LABEL_INDEX['W']].probability(w)

    # P(Y=1 | do(X=x)) where Z = {W}
    exp_prob = {}
    for x in (0, 1):
        exp_prob[x] = exp_model.predict_proba({'X': x})[LABEL_INDEX['Y']].probability(1)    

    return {'obs_prob': obs_prob, 'exp_prob': exp_prob}


print(model_validator())
