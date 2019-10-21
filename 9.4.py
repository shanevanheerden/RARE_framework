"""Interpret Model



Inputs
----------
in_data : [Orange.data.Table, Orange.data.Table]

Outputs
-------
out_data: Orange.data.Table
"""

######## PACKAGES ########

from __future__ import division
import operator
import Orange
import shap
import time
import skimage.color
from sklearn import model_selection
from sklearn.linear_model import ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from dragonfly import Window
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
from scipy.stats import gaussian_kde
try:
    import matplotlib.pyplot as pl
except ImportError as e:
    warnings.warn("matplotlib could not be loaded!", e)
    pass
from shap.plots import labels
from shap.plots import colors
from matplotlib import rcParams
from matplotlib.colors import LinearSegmentedColormap
rcParams['font.family'] = 'cmr10'

####### PARAMETERS #######

script_name = 'seg-severity-mlp'
execute_in_orange = False
args = {'num_models': 100,
        'k': 5,
        'scoring': ['r2', 'neg_mean_squared_error'],
        'num_clust': 10}

######### SCRIPT #########

def script(num_models, k, scoring, num_clust):
    X = in_data.X
    y = in_data.Y
    best_params = {'hidden_layer_sizes': (28, 28), 'solver': 'adam'}

    print(in_data.X.shape)
    print(best_params)

    results = {}
    scores = {}
    models = {}
    for m in range(num_models):
        model = MLPRegressor(**best_params)
        kfold = model_selection.KFold(n_splits=k)
        cv_results = model_selection.cross_validate(model, X, y, cv=kfold, scoring=scoring)
        print(cv_results)
        score = sum(cv_results['test_r2'])/len(cv_results['test_r2'])
        results[m] = cv_results
        scores[m] = score
        models[m] = model
        print(m, max(scores.values()), score)
    
    print(max(scores, key=scores.get), scores, results[max(scores, key=scores.get)])
    best_model = models[max(scores, key=scores.get)]
 
    feat_names = [x.name for x in in_data.domain.attributes]
    #new_feat_names = [x.split('=')[-1] for x in feat_names]
    print(feat_names)

    shap.initjs()

    best_model.fit(X, y)
    Xsum = shap.kmeans(X, num_clust) if num_clust is not None else X
    explainer = shap.KernelExplainer(best_model.predict, Xsum.data)

    shap_values = explainer.shap_values(X)

    #feat_names = ['{0}: {1}'.format(x[0], x[1]) for x in list(zip(feat_names, np.around(np.mean(np.absolute(shap_values), axis=0), 3)))]
    #id = len(Window.get_all_windows())
    #summary_plot(shap_values, features=X, feature_names=feat_names, class_names=class_names, plot_type='bar', color='black', id=id)
    #summary_plot(shap_values, features=X, feature_names=feat_names, class_names=class_names, plot_type='dot', title=title, id=id)

    d = Orange.data.Domain([Orange.data.ContinuousVariable(f) for f in feat_names] + [Orange.data.ContinuousVariable('S_' + f) for f in feat_names], class_vars=in_data.domain.class_vars, metas=in_data.domain.metas)

    xs = np.concatenate((in_data.X, shap_values), axis=1)
    ys = in_data.Y
    metas = in_data.metas

    out_data = Orange.data.Table.from_numpy(d, xs, Y=ys, metas=metas)
    
    return out_data, None, None, shap_values


##### INPUTS/OUTPUTS #####

from orangecontrib.rare.handlers import IOHandler

iohandler = IOHandler(script=script_name, execute_in_orange=execute_in_orange)

if not iohandler.ide_is_orange:
    in_data, in_datas, in_learner, in_learners, in_classifier, in_classifiers, in_object, in_objects = iohandler.load_inputs()
elif iohandler.ide_is_orange and not iohandler.execute_in_orange:
    iohandler.save_inputs(in_data=in_data, in_datas=in_datas, in_learner=in_learner, in_learners=in_learners, in_classifier=in_classifier, in_classifiers=in_classifiers, in_object=in_object, in_objects=in_objects)

if (iohandler.execute_in_orange and iohandler.ide_is_orange) or (not iohandler.ide_is_orange):
    out_data, out_learner, out_classifier, out_object = script(**args)

if not iohandler.ide_is_orange:
    iohandler.save_outputs(out_data=out_data, out_learner=out_learner, out_classifier=out_classifier, out_object=out_object)
elif iohandler.ide_is_orange and not iohandler.execute_in_orange:
    out_data, out_learner, out_classifier, out_object = iohandler.load_outputs()