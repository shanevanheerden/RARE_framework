"""

Inputs
----------
in_datas : [Orange.data.Table, Orange.data.Table, Orange.data.Table]

Outputs
-------
out_data: Orange.data.Table
out_object : [Orange.data.Table, Orange.data.Table, Orange.data.Table]
"""

######## PACKAGES ########

import numpy as np
import pandas as pd
import Orange
from orangecontrib.rare.utils import orange_table_to_pandas_dataframe, pandas_dataframe_to_orange_table

####### PARAMETERS #######

script_name = 'reconcile_rows'
execute_in_orange = True
args = {'reconcile_casualties': True}

######### SCRIPT #########

def script(reconcile_casualties):
    accidents = in_datas[0]
    vehicles = in_datas[1]
    try:
        casualties = in_datas[2]
        reconcile_casualties = True
    except:
        reconcile_casualties = False
        
    acc_df = orange_table_to_pandas_dataframe(accidents)
    acc_ind = acc_df.AccidentIndex.values
    veh_df = orange_table_to_pandas_dataframe(vehicles)
    veh_ind = veh_df.AccidentIndex.values
    if reconcile_casualties:
        cas_df = orange_table_to_pandas_dataframe(casualties)
        cas_ind = cas_df.AccidentIndex.values

    #if accident row has been deleted, delete corresponding vehicle and casualty row(s)
    veh_to_del = list(set(veh_ind)-set(acc_ind))
    veh_df = veh_df[~veh_df.AccidentIndex.isin(veh_to_del)]
    if reconcile_casualties:
        cas_to_del = list(set(cas_ind)-set(acc_ind))
        cas_df = cas_df[~cas_df.AccidentIndex.isin(cas_to_del)]

    #if one or more vehicle row(s) has been deleted, delete corresponding vehicle and accident rows
    num_veh_before = dict(zip(acc_df.AccidentIndex.values, acc_df.NumberOfVehicles.values))
    num_veh_after = veh_df.AccidentIndex.value_counts().to_dict()
    rows_to_del = []
    for k, v in num_veh_before.items():
        try:
            if num_veh_after[k]!=v:
                rows_to_del.append(k)
        except KeyError:
            rows_to_del.append(k)
    acc_df = acc_df[~acc_df.AccidentIndex.isin(rows_to_del)]
    veh_df = veh_df[~veh_df.AccidentIndex.isin(rows_to_del)]
    
    #if one or more casulaty row(s) has been deleted, delete corresponding casulaty and accident rows
    if reconcile_casualties:
        num_cas_before = dict(zip(acc_df.AccidentIndex.values, acc_df.NumberOfCasualties.values))
        num_cas_after = cas_df.AccidentIndex.value_counts().to_dict()
        rows_to_del = []
        for k, v in num_cas_before.items():
            try:
                if num_cas_after[k]!=v:
                    rows_to_del.append(k)
            except KeyError:
                rows_to_del.append(k)
        acc_df = acc_df[~acc_df.AccidentIndex.isin(rows_to_del)]
        cas_df = cas_df[~cas_df.AccidentIndex.isin(rows_to_del)]
    
    #if accident row has been deleted, delete corresponding vehicle and casualty row(s)
    veh_to_del = list(set(veh_ind)-set(acc_ind))
    veh_df = veh_df[~veh_df.AccidentIndex.isin(veh_to_del)]
    if reconcile_casualties:
        cas_to_del = list(set(cas_ind)-set(acc_ind))
        cas_df = cas_df[~cas_df.AccidentIndex.isin(cas_to_del)]
    
    if reconcile_casualties:
        print('Acc: {},'.format(len(acc_df)), 'Veh: {},'.format(len(veh_df)), 'Cas: {}'.format(len(cas_df)))
    else:
        print('Acc: {},'.format(len(acc_df)), 'Veh: {}'.format(len(veh_df)))
    
    acc_table = pandas_dataframe_to_orange_table(acc_df, accidents.domain, 'accidents')
    veh_table = pandas_dataframe_to_orange_table(veh_df, vehicles.domain, 'vehicles')
    if reconcile_casualties:
        cas_table = pandas_dataframe_to_orange_table(cas_df, casualties.domain, 'casualties')
    
    if reconcile_casualties:
        out_object = [acc_table, veh_table, cas_table]
    else:
        out_object = [acc_table, veh_table]
    
    return None, None, None, out_object

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