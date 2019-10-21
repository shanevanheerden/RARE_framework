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

import Orange
import psutil
import time
import math
import numpy as np
from orangecontrib.rare.utils import orange_table_to_pandas_dataframe, pandas_dataframe_to_orange_table

####### PARAMETERS #######

script_name = '5.1'
execute_in_orange = False
args = {'accident_index': 'AccidentIndex',
        'severity_index': 'CasualtySeverity',
        'class_index': 'CasualtyClass',
        'scaling_function': lambda x: x**math.log(0.1, 2/3)}

######### SCRIPT #########

def script(accident_index, severity_index, class_index, scaling_function):
    accidents = orange_table_to_pandas_dataframe(in_datas[0])
    victims = orange_table_to_pandas_dataframe(in_datas[1])

    severity = []
    pedestrians = []
    dangerous = []

    i = 0
    start_time = time.time()
    acc_indexes = accidents[accident_index].tolist()
    for a in acc_indexes:
        tot_sev = 0
        ped_acc = False
        dangerous_acc = False
        sev = victims[victims[accident_index] == a][severity_index].tolist()
        ped = victims[victims[accident_index] == a][class_index].tolist()
        for s, p in zip(sev, ped):
            if s == 0:
                tot_sev += 1897129
                dangerous_acc = True
            elif s == 1:
                tot_sev += 213184
                dangerous_acc = True
            else:
                tot_sev += 16434
            if p == 2:
                ped_acc = True
        severity.append(tot_sev)
        dangerous.append(1.0 if dangerous_acc else 0.0)
        pedestrians.append(1.0 if ped_acc else 0.0)
        dt = time.time() - start_time
        print(round(i / len(acc_indexes) * 100, 3), '\t', time.strftime("%H:%M:%S", time.gmtime(dt)), '\t', round((dt / ((i + 1) / len(acc_indexes)) - dt) / (60 ** 2), 3), '\t', time.ctime((time.time() + (dt / ((i + 1) / len(acc_indexes)) - dt))), '\t', psutil.virtual_memory())
        i += 1

    accidents['Severity'] = severity
    accidents['PedestrianAccident'] = pedestrians
    accidents['DangerousAccident'] = dangerous

    attr = in_datas[0].domain.attributes + (Orange.data.ContinuousVariable('Severity'), Orange.data.DiscreteVariable('PedestrianAccident', values=['No', 'Yes']), Orange.data.DiscreteVariable('DangerousAccident', values=['No', 'Yes']))
    class_vars = in_datas[0].domain.class_vars
    metas = in_datas[0].domain.metas

    d = Orange.data.Domain(attr, class_vars, metas)
    out_data = pandas_dataframe_to_orange_table(accidents, d)
        
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