"""KDE



Inputs
----------
in_data : [Orange.data.Table, Orange.data.Table]

Outputs
-------
out_data: Orange.data.Table
"""

######## PACKAGES ########

import numpy as np
import Orange
import sklearn
from sklearn import model_selection
from sklearn.linear_model import ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
import random
import itertools
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
import warnings
from sklearn.metrics import r2_score, mean_squared_error
warnings.filterwarnings("ignore")

#[print(x) for x in sklearn.metrics.SCORERS.keys()]

####### PARAMETERS #######

script_name = 'train_test'
execute_in_orange = False
args = {'k': 5,
        'scoring': 'r2',
        'test_size': 0.3}

######### SCRIPT #########
'''
SEGMENT - RATE
Searching LR
LR best score:  0.18872575599790764
LR best params:  {'alpha': 0.0005, 'l1_ratio': 0.04}
LR Train R2: 0.11983554145997588
LR Train MSE: 479.949536616025
LR Test R2: 0.21180522553921144
LR Test MSE: 259.1354193335823
Searching DT
DT best score:  0.28058699458559727
DT best params:  {'criterion': 'mae', 'min_samples_leaf': 37}
DT Train R2: 0.2019035724228464
DT Train MSE: 435.1982255974511
DT Test R2: 0.29354104006036363
DT Test MSE: 232.26307095370302
Searching RF
RF best score:  0.42460368882062643
RF best params:  {'max_features': 1, 'n_estimators': 150}
RF Train R2: 0.8603415994786685
RF Train MSE: 76.1550684811028
RF Test R2: 0.49431724465180027
RF Test MSE: 166.25371938879314
Searching KNN
KNN best score:  0.3786887839668241
KNN best params:  {'metric': 'manhattan', 'n_neighbors': 7, 'weights': 'distance'}
KNN Train R2: 0.9732992893108116
KNN Train MSE: 14.559771867920526
KNN Test R2: 0.27580329859157693
KNN Test MSE: 238.09472224406994
Searching SVM
SVM best score:  0.18788934768257332
SVM best params:  {'C': 10.0, 'kernel': 'rbf'}
SVM Train R2: 0.11218908315169596
SVM Train MSE: 484.1191143423287
SVM Test R2: 0.19744321931907693
SVM Test MSE: 263.8572274765913
Searching MLP
MLP best score:  0.3076959492033575
MLP best params:  {'layers': (28, 28)}
MLP Train R2: 0.2720514588167814
MLP Train MSE: 396.946913308372
MLP Test R2: 0.3002110993331122
MLP Test MSE: 230.07015029166746

SEGMENT - SEVERITY
Searching LR
LR best score:  0.022398286626325165
LR best params:  {'alpha': 0.0002, 'l1_ratio': 0.17}
LR Train R2: 0.025681344538878048
LR Train MSE: 52079167170.25851
LR Test R2: 0.02187158222777552
LR Test MSE: 53711027339.56914
Searching DT
DT best score:  0.024877647332365624
DT best params:  {'criterion': 'mse', 'min_samples_leaf': 96}
DT Train R2: 0.046977629210337546
DT Train MSE: 50940840645.05671
DT Test R2: 0.023652551692957213
DT Test MSE: 53613230672.079216
Searching RF
RF best score:  0.028223896522571103
RF best params:  {'max_features': 2, 'n_estimators': 190}
RF Train R2: 0.3276017797285037
RF Train MSE: 35940951271.15307
RF Test R2: 0.02980832976630099
RF Test MSE: 53275204336.90051
Searching KNN
KNN best score:  0.03711592243946239
KNN best params:  {'metric': 'chebyshev', 'n_neighbors': 97, 'weights': 'distance'}
KNN Train R2: 0.3703780484730651
KNN Train MSE: 33654479141.751446
KNN Test R2: 0.03256015529016054
KNN Test MSE: 53124095982.148506
Searching SVM
SVM best score:  -0.06137592793035086
SVM best params:  {'C': 9.0, 'kernel': 'linear'}
SVM Train R2: -0.01005073893336315
SVM Train MSE: 56712719501.026146
SVM Test R2: -0.06150090760744309
SVM Test MSE: 58289180882.1341

JUNCTION - RATE
Searching LR
LR best score:  0.23536816397261315
LR best params:  {'alpha': 0.02, 'l1_ratio': 0.98}
LR Train R2: 0.2414785506237913
LR Train MSE: 4825.019976588924
LR Test R2: 0.2092678806683611
LR Test MSE: 6846.456329381202
Searching DT
DT best score:  0.3723671401838422
DT best params:  {'criterion': 'mse', 'min_samples_leaf': 9}
DT Train R2: 0.5822884780509159
DT Train MSE: 2657.0988065178235
DT Test R2: 0.25194821943568324
DT Test MSE: 6476.913890987964
Searching RF
RF best score:  0.5061809659083162
RF best params:  {'max_features': 15, 'n_estimators': 150}
RF Train R2: 0.8378569036408364
RF Train MSE: 1031.4061384056172
RF Test R2: 0.37636466743691155
RF Test MSE: 5399.669452483145
Searching KNN
KNN best score:  0.3552291577706224
KNN best params:  {'metric': 'manhattan', 'n_neighbors': 16, 'weights': 'distance'}
KNN Train R2: 0.8884240806910945
KNN Train MSE: 709.7439894606476
KNN Test R2: 0.25573777036357814
KNN Test MSE: 6444.102532625882
Searching SVM
SVM best score:  0.179840617713035
SVM best params:  {'C': 10.0, 'kernel': 'linear'}
SVM Train R2: 0.17862044678108036
SVM Train MSE: 5224.865764708696
SVM Test R2: 0.14506012644423127
SVM Test MSE: 7402.391233953829
Searching MLP
MLP best score:  0.2604563864277273
MLP best params:  {'hidden_layer_sizes': (30, 33), 'solver': 'adam'}
MLP Train R2: 0.29272528382149765
MLP Train MSE: 4499.035112723568
MLP Test R2: 0.22300705969580903
MLP Test MSE: 6727.497345784482

JUNCTION - SEVERITY
Searching LR
LR best score:  -0.0004262717482408808
LR best params:  {'alpha': 0.2, 'l1_ratio': 0.49}
LR Train R2: 0.008676322611887666
LR Train MSE: 29711583200.178547
LR Test R2: 0.0012025759876052033
LR Test MSE: 28681783018.511635
Searching DT
DT best score:  -0.0038316768317637755
DT best params:  {'criterion': 'friedman_mse', 'min_samples_leaf': 648}
DT Train R2: 0.004645384174902012
DT Train MSE: 29832396982.273262
DT Test R2: -0.0002590252747343147
DT Test MSE: 28723754823.061947
Searching RF
RF best score:  -0.1682813944157875
RF best params:  {'max_features': 1, 'n_estimators': 80}
RF Train R2: 0.32995208521024066
RF Train MSE: 20082425974.970276
RF Test R2: -0.11871931225559318
RF Test MSE: 32125497925.128105
Searching KNN
KNN best score:  -0.011737987028221885
KNN best params:  {'metric': 'chebyshev', 'n_neighbors': 85, 'weights': 'uniform'}
KNN Train R2: 0.00656816067979793
KNN Train MSE: 29774768242.635887
KNN Test R2: -0.0002024293236073671
KNN Test MSE: 28722129595.81273
Searching SVM
SVM best score:  -0.06789701617763581
SVM best params:  {'C': 10.0, 'kernel': 'linear'}
SVM Train R2: -0.05777306847577934
SVM Train MSE: 31703179544.47676
SVM Test R2: -0.051275504031711217
SVM Test MSE: 30188760177.398903
Searching MLP
MLP best score:  -0.0027597008538640336
MLP best params:  {'hidden_layer_sizes': (19, 28), 'solver': 'adam'}
MLP Train R2: -2.2278730341218633e-05
MLP Train MSE: 29972294432.443035
MLP Test R2: -0.0010144856594584972
MLP Test MSE: 28745448862.62674
'''

def script(k, scoring, test_size):
    X_train, X_test, y_train, y_test = train_test_split(in_data.X, in_data.Y, test_size=test_size, random_state=1)

    run_lr = False
    run_dt = False
    run_rf = False
    run_knn = False
    run_svm = False
    run_mlp = False
    run_gb = False

    if run_lr is True:
        lr_parameters = {'alpha': [0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007, 0.0008, 0.0009, 0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
                         'l1_ratio': [0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.11, 0.12, 0.13, 0.14, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.2, 0.21, 0.22, 0.23, 0.24, 0.25, 0.26, 0.27, 0.28, 0.29, 0.3, 0.31, 0.32, 0.33, 0.34, 0.35, 0.36, 0.37, 0.38, 0.39, 0.4, 0.41, 0.42, 0.43, 0.44, 0.45, 0.46, 0.47, 0.48, 0.49, 0.5, 0.51, 0.52, 0.53, 0.54, 0.55, 0.56, 0.57, 0.58, 0.59, 0.6, 0.61, 0.62, 0.63, 0.64, 0.65, 0.66, 0.67, 0.68, 0.69, 0.7, 0.71, 0.72, 0.73, 0.74, 0.75, 0.76, 0.77, 0.78, 0.79, 0.8, 0.81, 0.82, 0.83, 0.84, 0.85, 0.86, 0.87, 0.88, 0.89, 0.9, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99, 1.0]}
        lr = ElasticNet()
        lr_results = GridSearchCV(lr, lr_parameters, scoring=scoring, cv=k, verbose=0, n_jobs=-1)
        print('Searching LR')
        best_lr = lr_results.fit(X_train, y_train)
        print('LR best score: ', best_lr.best_score_)
        print('LR best params: ', best_lr.best_params_)
        y_pred_train = best_lr.predict(X_train)
        print('LR Train R2:', r2_score(y_train, y_pred_train))
        print('LR Train MSE:', mean_squared_error(y_train, y_pred_train))
        y_pred_test = best_lr.predict(X_test)
        print('LR Test R2:', r2_score(y_test, y_pred_test))
        print('LR Test MSE:', mean_squared_error(y_test, y_pred_test))

    if run_dt is True:
        dt_parameters = {'criterion': ['mse', 'friedman_mse', 'mae'],
                         'min_samples_leaf': list(range(1, 1000, 1))}
        dt = DecisionTreeRegressor()
        dt_results = GridSearchCV(dt, dt_parameters, scoring=scoring, cv=k, verbose=0, n_jobs=-1)
        print('Searching DT')
        best_dt = dt_results.fit(X_train, y_train)
        print('DT best score: ', best_dt.best_score_)
        print('DT best params: ', best_dt.best_params_)
        y_pred_train = best_dt.predict(X_train)
        print('DT Train R2:', r2_score(y_train, y_pred_train))
        print('DT Train MSE:', mean_squared_error(y_train, y_pred_train))
        y_pred_test = best_dt.predict(X_test)
        print('DT Test R2:', r2_score(y_test, y_pred_test))
        print('DT Test MSE:', mean_squared_error(y_test, y_pred_test))

    if run_rf is True:
        rf_parameters = {'n_estimators': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200],
                         'max_features': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28]}
        rf = RandomForestRegressor()
        rf_results = GridSearchCV(rf, rf_parameters, scoring=scoring, cv=k, verbose=0, n_jobs=-1)
        print('Searching RF')
        best_rf = rf_results.fit(X_train, y_train)
        print('RF best score: ', best_rf.best_score_)
        print('RF best params: ', best_rf.best_params_)
        y_pred_train = best_rf.predict(X_train)
        print('RF Train R2:', r2_score(y_train, y_pred_train))
        print('RF Train MSE:', mean_squared_error(y_train, y_pred_train))
        y_pred_test = best_rf.predict(X_test)
        print('RF Test R2:', r2_score(y_test, y_pred_test))
        print('RF Test MSE:', mean_squared_error(y_test, y_pred_test))

    if run_knn is True:
        knn_parameters = {'n_neighbors': list(range(1, 100, 1)),
                          'weights': ['uniform', 'distance'],
                          'metric': ['euclidean', 'manhattan', 'chebyshev']}
        knn = KNeighborsRegressor()
        knn_results = GridSearchCV(knn, knn_parameters, scoring=scoring, cv=k, verbose=0, n_jobs=-1)
        print('Searching KNN')
        best_knn = knn_results.fit(X_train, y_train)
        print('KNN best score: ', best_knn.best_score_)
        print('KNN best params: ', best_knn.best_params_)
        y_pred_train = best_knn.predict(X_train)
        print('KNN Train R2:', r2_score(y_train, y_pred_train))
        print('KNN Train MSE:', mean_squared_error(y_train, y_pred_train))
        y_pred_test = best_knn.predict(X_test)
        print('KNN Test R2:', r2_score(y_test, y_pred_test))
        print('KNN Test MSE:', mean_squared_error(y_test, y_pred_test))

    if run_svm is True:
        svm_parameters = {'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
                          'C': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3.0, 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9, 4.0, 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 4.8, 4.9, 5.0, 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7, 5.8, 5.9, 6.0, 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7, 6.8, 6.9, 7.0, 7.1, 7.2, 7.3, 7.4, 7.5, 7.6, 7.7, 7.8, 7.9, 8.0, 8.1, 8.2, 8.3, 8.4, 8.5, 8.6, 8.7, 8.8, 8.9, 9.0, 9.1, 9.2, 9.3, 9.4, 9.5, 9.6, 9.7, 9.8, 9.9, 10.0]}
        svm = SVR()
        svm_results = GridSearchCV(svm, svm_parameters, scoring=scoring, cv=k, verbose=0, n_jobs=-1)
        print('Searching SVM')
        best_svm = svm_results.fit(X_train, y_train)
        print('SVM best score: ', best_svm.best_score_)
        print('SVM best params: ', best_svm.best_params_)
        y_pred_train = best_svm.predict(X_train)
        print('SVM Train R2:', r2_score(y_train, y_pred_train))
        print('SVM Train MSE:', mean_squared_error(y_train, y_pred_train))
        y_pred_test = best_svm.predict(X_test)
        print('SVM Test R2:', r2_score(y_test, y_pred_test))
        print('SVM Test MSE:', mean_squared_error(y_test, y_pred_test))

    if run_mlp is True:
        mlp_parameters = {'hidden_layer_sizes': list(itertools.combinations_with_replacement(list(range(1, 29 * 2 + 1)), 1)) + list(itertools.combinations_with_replacement(list(range(1, 17 * 2 + 1)), 2)),
                         'solver': ['adam'] * 5}
        mlp = MLPRegressor()
        mlp_results = GridSearchCV(mlp, mlp_parameters, scoring=scoring, cv=k, verbose=0, n_jobs=-1)
        print('Searching MLP')
        best_mlp = mlp_results.fit(X_train, y_train)
        print('MLP best score: ', best_mlp.best_score_)
        print('MLP best params: ', best_mlp.best_params_)
        y_pred_train = best_mlp.predict(X_train)
        print('MLP Train R2:', r2_score(y_train, y_pred_train))
        print('MLP Train MSE:', mean_squared_error(y_train, y_pred_train))
        y_pred_test = best_mlp.predict(X_test)
        print('MLP Test R2:', r2_score(y_test, y_pred_test))
        print('MLP Test MSE:', mean_squared_error(y_test, y_pred_test))

    if run_gb is True:
        gb_parameters = {'loss': ['ls', 'lad', 'huber', 'quantile'],
                         'learning_rate': [0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                         'n_estimators': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200]}
        mlp = GradientBoostingRegressor()
        gb_results = GridSearchCV(mlp, gb_parameters, scoring=scoring, cv=k, verbose=0, n_jobs=-1)
        print('Searching GB')
        best_gb = gb_results.fit(X_train, y_train)
        print('GB best score: ', best_gb.best_score_)
        print('GB best params: ', best_gb.best_params_)
        y_pred_train = best_gb.predict(X_train)
        print('GB Train R2:', r2_score(y_train, y_pred_train))
        print('GB Train MSE:', mean_squared_error(y_train, y_pred_train))
        y_pred_test = best_gb.predict(X_test)
        print('GB Test R2:', r2_score(y_test, y_pred_test))
        print('GB Test MSE:', mean_squared_error(y_test, y_pred_test))


    d = Orange.data.Domain(in_data.domain.attributes, class_vars=in_data.domain.class_vars)
    out_data = Orange.data.Table.from_numpy(d, X=np.array(X_train), Y=np.array(y_train))

    return out_data, None, None, None#[best_lr.best_params_, best_dt.best_params_, best_rf.best_params_, best_knn.best_params_, best_svm.best_params_]


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