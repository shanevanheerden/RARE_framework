"""Compute Centroids



Inputs
----------
in_data : [Orange.data.Table, Orange.data.Table]

Outputs
-------
out_data: Orange.data.Table
"""

######## PACKAGES ########

import numpy as np
import pandas as pd
from shapely.geometry import MultiPoint
import shapely.wkt
import Orange
import time
import psutil
from collections import Counter
import operator
from orangecontrib.rare.utils import orange_table_to_pandas_dataframe, pandas_dataframe_to_orange_table
pd.set_option('display.max_rows', 6000)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

####### PARAMETERS #######

script_name = 'compute_centroids'
execute_in_orange = False
args = {'xycols': ['Longitude', 'Latitude']}

######### SCRIPT #########

def script(xycols):
    accidents = orange_table_to_pandas_dataframe(in_data)



    xs = []
    metas = []
    i = 0
    print(accidents.head())
    #c_names = accidents['Cluster'].unique().astype(int)
    clusters = accidents['Junction'].to_numpy().astype(int)
    #geom = accidents['geometry'].astype(str).apply(shapely.wkt.loads).to_numpy()
    lats = accidents['Latitude'].to_numpy().astype(float)
    longs = accidents['Longitude'].to_numpy().astype(float)
    #aadt = accidents['AADF'].to_numpy().astype(int)
    #longs = accidents[xycols[0]].to_numpy()
    #lats = accidents[xycols[1]].to_numpy()
    #col = np.random.shuffle(np.array([x for x in in_data.domain.metas if x.name=='Cluster'][0].colors.copy()))

    continuizer = Orange.preprocess.Continuize()
    new_data = continuizer(in_data)

    feat_names = [i.name for i in new_data.domain.attributes] + ['Junction']

    #print(feat_names)

    df = orange_table_to_pandas_dataframe(new_data)


    #print(len(in_data.domain.metas[11].values))
    print(len(df['Junction'].unique()))

    df_new = df[feat_names].groupby(['Junction']).sum().astype(int).div(df.groupby('Junction').size(), axis=0)

    #print(df.groupby('Cluster').size().divide())

    #print(pd.DataFrame(df.groupby('Cluster').size()).astype(float).to_numpy()*1000000/pd.DataFrame(df[['AADF', 'Cluster']].astype(float).groupby(['Cluster']).mean()).to_numpy())
    #print()
    #print(len(sorted(pd.DataFrame(df.groupby('Cluster').size()).astype(float).to_numpy()*1000000000/pd.DataFrame(df[['AADF', 'Cluster']].astype(float).groupby(['Cluster']).mean()).to_numpy())))
    #print(df[['AADF', 'Cluster']].astype(int).groupby(['Cluster']).mean())
    df_new['Rate'] = pd.DataFrame(pd.DataFrame(df.groupby('Junction').size()).astype(float).to_numpy().astype(float)*1000000000/pd.DataFrame(df[['AADF', 'Junction']].astype(float).groupby(['Junction']).mean()).to_numpy().astype(float))
    df_new['Severity'] = df[['Severity', 'Junction']].astype(int).groupby(['Junction']).sum().div(df.groupby('Junction').size(), axis=0)
    df_new['Frequency'] = pd.DataFrame(df.groupby('Junction').size()).astype(float).to_numpy().astype(float)
    print(df_new)

    #print(pd.DataFrame(df[['AADF', 'Cluster']].astype(float).groupby(['Cluster']).mean()))
    #print(len(pd.DataFrame(df.groupby('Cluster').size()).astype(float).to_numpy().astype(float)))
    #print(len(df_new['Severity']))
    #print(len(df_new))

    clust_list = []
    lat_list = []
    long_list = []
    numacc_list = []

    #xs.append(row)
    for clust in range(len(df['Junction'].unique())):
        lat = [l for l, c in zip(lats, clusters) if c == clust]
        long = [l for l, c in zip(longs, clusters) if c == clust]
        num_acc = len(list(lat))
        lat = sum(lat)/len(lat)
        long = sum(long) / len(long)
        clust_list.append(float(clust))
        lat_list.append(lat)
        long_list.append(long)
        numacc_list.append(num_acc)
        #metas.append([float(clust), centroid[1], centroid[0], len(points)])

    df_new['Junction'] = clust_list
    print(len(lat_list), lat_list)
    df_new['Latitude'] = lat_list
    df_new['Longitude'] = long_list
    df_new['NumberOfAccidents'] = numacc_list

    #dt = time.time() - start_time
    #print(i, round(i / len(c_names) * 100, 3), '\t', time.strftime("%H:%M:%S", time.gmtime(dt)), '\t', round((dt / ((i + 1) / len(c_names)) - dt) / (60 ** 2), 3), '\t', time.ctime((time.time() + (dt / ((i + 1) / len(c_names)) - dt))), '\t', psutil.virtual_memory())
    #i += 1

    d = Orange.data.Domain([Orange.data.ContinuousVariable(i.name) for i in new_data.domain.attributes], class_vars=[Orange.data.ContinuousVariable('Rate'), Orange.data.ContinuousVariable('Severity'), Orange.data.ContinuousVariable('Frequency')], metas=[in_data.domain.metas[13], Orange.data.ContinuousVariable('Latitude'), Orange.data.ContinuousVariable('Longitude'), Orange.data.ContinuousVariable('NumberOfAccidents')])#, metas=[Orange.data.DiscreteVariable('Cluster', values = ['J{0}'.format(int(i)+1) for i in range(len(df_new))])])

    out_data = pandas_dataframe_to_orange_table(df_new, d)

    #temp = Orange.data.Table.from_numpy(d, X=np.array(xs), Y=np.array(ys), metas=np.array(metas))
    #ind = [i.name for i in in_data.domain.metas].index('Cluster')
    #[temp.domain.metas[ind].set_color(i, c) for i, c in zip(range(len(temp.domain.metas[ind].colors)), col)]
    #out_data = temp

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