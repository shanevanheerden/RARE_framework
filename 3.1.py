"""3.1

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
import time
import psutil
from orangecontrib.rare.utils import orange_table_to_pandas_dataframe, pandas_dataframe_to_orange_table

####### PARAMETERS #######

script_name = '3.1'
execute_in_orange = False
args = {'accident_index': 'AccidentIndex',
        'vehicle_index': 'VehicleReference',
        'press': ['VehicleType', 'TowingAndArticulation', 'JunctionLocation', 'VehicleLocationRestrictedLane', 'VehicleManoeuvre', 'FirstPointOfImpact', 'HitObjectInCarriageway', 'SkiddingAndOverturning', 'VehicleLeavingCarriageway', 'HitObjectOffCarriageway']}

######### SCRIPT #########

def OrderVehicles(Vu, p, s):
    k = len(Vu)
    nV = len(p)
    if s > nV - 1:
        Vo = Vu
        return Vo
    elif len(set(Vu[p[s]].tolist())) == k:
        Vo = Vu.sort_values(p[s])
        return Vo
    else:
        Vu = Vu.sort_values(p[s])
        j = 1
        for j in Vu[p[s]].unique():
            Vt = Vu[Vu[p[s]] == j]
            Vt = OrderVehicles(Vt, p, s + 1)
            Vu[Vu[p[s]] == j] = Vt
        Vo = Vu
        return Vo

def script(accident_index, vehicle_index, press):
    vehicles = orange_table_to_pandas_dataframe(in_data)
    new_vehicles = pd.DataFrame(columns=vehicles.columns)

    i = 0
    start_time = time.time()
    accidents = vehicles.AccidentIndex.unique()
    for k in accidents:
        veh_unordered = vehicles[vehicles.AccidentIndex == k]
        veh_ordered = OrderVehicles(veh_unordered, press, 0)
        veh_ordered[vehicle_index] = list(range(1, len(veh_ordered) + 1))
        new_vehicles = new_vehicles.append(veh_ordered, ignore_index=True)
        dt = time.time() - start_time
        print(i, round(i / len(accidents) * 100, 3), '\t', time.strftime("%H:%M:%S", time.gmtime(dt)), '\t', round((dt / ((i + 1) / len(accidents)) - dt) / (60 ** 2), 3), '\t', time.ctime((time.time() + (dt / ((i + 1) / len(accidents)) - dt))), '\t', psutil.virtual_memory())
        i += 1

    attr_vals = {}
    for i in in_data.domain.attributes + in_data.domain.class_vars + in_data.domain.metas:
        if isinstance(i, Orange.data.variable.DiscreteVariable):
            attr_vals[i.name] = i.values
        else:
            attr_vals[i.name] = None
        
    cols = list(new_vehicles.columns)
    cols.remove(vehicle_index)

    new_new_vehicles = pd.DataFrame(columns=cols)
    for k in new_vehicles.AccidentIndex.unique():
        veh_unmerged = new_vehicles[new_vehicles.AccidentIndex == k].copy()
        veh_unmerged = veh_unmerged.drop(vehicle_index, axis=1)
        row = []
        for j in veh_unmerged.columns:
            val = ''
            col = veh_unmerged[j].tolist()
            if j == accident_index:
                val = col[0]
            else:
                for k in range(len(veh_unmerged)):
                    if attr_vals[j] is not None:
                        val += attr_vals[j][int(col[k])]
                    else:
                        val += col[k]
                    if k != len(veh_unmerged) - 1:
                        val +=','
            row.append(val)
        new_row = pd.DataFrame(np.array([row]), columns=cols)
        new_new_vehicles = new_new_vehicles.append(new_row, ignore_index=True)

    attr = []
    for i in in_data.domain.attributes:
        if i.name != vehicle_index:
            new_new_vehicles[i.name] = pd.Categorical(new_new_vehicles[i.name])
            attr.append(Orange.data.DiscreteVariable(i.name, values=new_new_vehicles[i.name].cat.categories))
            new_new_vehicles[i.name] = new_new_vehicles[i.name].cat.codes
    class_vars = []
    for i in in_data.domain.class_vars:
        if i.name != vehicle_index:
            new_new_vehicles[i.name] = pd.Categorical(new_new_vehicles[i.name])
            class_vars.append(Orange.data.DiscreteVariable(i.name, values=new_new_vehicles[i.name].cat.categories))
            new_new_vehicles[i.name] = new_new_vehicles[i.name].cat.codes
    metas = []
    for i in in_data.domain.metas:
        if i.name != vehicle_index:
            if isinstance(i, Orange.data.variable.DiscreteVariable):
                new_new_vehicles[i.name] = pd.Categorical(new_new_vehicles[i.name])
                metas.append(Orange.data.DiscreteVariable(i.name, values=new_new_vehicles[i.name].cat.categories))
                new_new_vehicles[i.name] = new_new_vehicles[i.name].cat.codes
            else:
                metas.append(Orange.data.StringVariable(i.name))
    
    d = Orange.data.Domain(attr, class_vars, metas)
    out_data = pandas_dataframe_to_orange_table(new_new_vehicles, d)
    
    return out_data, None, None, None

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