"""Categorise Accidents

Categorise accident features into road and environment features.

Inputs
----------
in_object : accidents, vehicles
           [(Orange.data.Table), (Orange.data.Table)]

Outputs
-------
out_object : road, environment, driver, vehicle, crash
             [[(Orange.data.Table), (Orange.data.Table)], [(Orange.data.Table), (Orange.data.Table), (Orange.data.Table)]]
"""

######## PACKAGES ########

import numpy as np
import pandas as pd
import Orange
from orangecontrib.rare.utils import get_feature_names, orange_table_to_pandas_dataframe, pandas_dataframe_to_orange_table, get_domain_subset

####### PARAMETERS #######

script_name = 'categorise_accidents'
execute_in_orange = True
args = {'table': 0}

######### SCRIPT #########

def script(table):
    current_input = in_object[0].name
    required_input = 'accidents'
    if current_input != required_input:
        raise ValueError('Current inputs {0} do not match required inputs {1}'.format(current_input, required_input))

    accidents = in_object[0]

    acc_attr_cols, acc_class_cols, acc_meta_cols = get_feature_names(accidents.domain)
    acc_cols = acc_attr_cols + acc_class_cols + acc_meta_cols

    acc_df = orange_table_to_pandas_dataframe(accidents)

    road_cols = ['FirstRoadClass', 'RoadType', 'SpeedLimit', 'JunctionDetail', 'JunctionControl', 'PedestrianCrossingPhysicalFacilities', 'UrbanOrRuralArea']
    environment_cols = ['DayOfWeek', 'TimeBand', 'PedestrianCrossingHumanControl', 'LightConditions', 'WeatherConditions', 'RoadSurfaceConditions', 'SpecialConditionsAtSite', 'CarriagewayHazards']

    road_df = acc_df[acc_meta_cols + road_cols]
    environment_df = acc_df[acc_meta_cols + environment_cols]

    road_table = pandas_dataframe_to_orange_table(road_df, get_domain_subset(accidents.domain, acc_meta_cols + road_cols), 'road')
    environment_table = pandas_dataframe_to_orange_table(environment_df, get_domain_subset(accidents.domain, acc_meta_cols + environment_cols), 'environment')

    out_object = [road_table, environment_table]
    out_data = out_object[table]
    
    return out_data, None, None, out_object

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