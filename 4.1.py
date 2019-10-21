"""4.1

Inputs
----------
in_data : [Orange.data.Table, Orange.data.Table]

Outputs
-------
out_data: Orange.data.Table
"""

######## PACKAGES ########

import numpy as np
import math
import geopandas as gpd
import shapely
import pandas as pd
import time
import psutil
import Orange
from shapely.geometry import Point
from orangecontrib.rare.utils import orange_table_to_pandas_dataframe, pandas_dataframe_to_orange_table

####### PARAMETERS ####### 23:53:44

script_name = '4.1'
execute_in_orange = False
args = {}

######### SCRIPT #########

def line_gradient(p1, p2):
    try:
        m = (p2[1]-p1[1])/(p2[0]-p1[0])
    except ZeroDivisionError:
        m = np.nan_to_num(np.inf)
    return m

def in_line_region(point, line):
    x = point.x
    y = point.y

    x1 = line.coords[0][0]
    y1 = line.coords[0][1]
    x2 = line.coords[1][0]
    y2 = line.coords[1][1]
    m = line_gradient(line.coords[0], line.coords[1])

    if x1 == x2 and y1 > y2:  # Case 1
        if y2 < y < y1:
            return True
    elif x1 > x2 and y1 > y2:  # Case 2
        if -m*(y - y2) + x2 < x < -m*(y - y1) + x1 and -1/m*(x - x2) + y2 < y < -1/m*(x - x1) + y1:
            return True
    elif x1 > x2 and y1 == y2:  # Case 3
        if x2 < x < x1:
            return True
    elif x1 > x2 and y1 < y2:  # Case 4
        if -m*(y - y2) + x2 < x < -m*(y - y1) + x1 and -1/m*(x - x1) + y1 < y < -1/m*(x - x2) + y2:
            return True
    elif x1 == x2 and y1 < y2:  # Case 5
        if y1 <= y <= y2:
            return True
    elif x1 < x2 and y1 < y2:  # Case 6
        if -m*(y - y1) + x1 < x < -m*(y - y2) + x2 and -1/m*(x - x1) + y1 < y < -1/m*(x - x2) + y2:
            return True
    elif x1 < x2 and y1 == y2:  # Case 7
        if x1 <= x <= x2:
            return True
    elif x1 < x2 and y1 > y2:  # Case 8
        if -m*(y - y1) + x1 < x < -m*(y - y2) + x2 and -1/m*(x - x2) + y2 < y < -1/m*(x - x1) + y1:
            return True
    return False

def find_closest_line(point, lines, nodes_only=False):
    candidates = {}
    edge_points = {}
    perp_dist = lambda p, l: math.fabs(
        (l[1][1] - l[0][1]) * p[0] - (l[1][0] - l[0][0]) * p[1] + l[1][0] * l[0][1] - l[1][1] * l[0][0]) / math.sqrt(
        (l[1][1] - l[0][1]) ** 2 + (l[1][0] - l[0][0]) ** 2)
    perp_point = lambda p, l: ((-(l[1][0]-l[0][0])*(-(l[1][0]-l[0][0])*p[0]-(l[1][1]-l[0][1])*p[1])-((l[1][1]-l[0][1])*(l[1][0]*l[0][1]-l[1][1]*l[0][0])))/((l[1][1]-l[0][1])**2+(l[1][0]-l[0][0])**2),
                               ((l[1][1] - l[0][1])*((l[1][0]-l[0][0])*p[0]+(l[1][1] - l[0][1])*p[1])-(-(l[1][0]-l[0][0])*(l[1][0]*l[0][1]-l[1][1]*l[0][0])))/((l[1][1]-l[0][1])**2+(l[1][0]-l[0][0])**2))
    euclid_dist = lambda p1, p2: math.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)
    for k in range(len(lines)):
        if in_line_region(point, lines[k]) and nodes_only is False:
            candidates[k] = perp_dist(point.coords[0], lines[k].coords)
        else:
            dist1 = euclid_dist(point.coords[0], lines[k].coords[0])
            dist2 = euclid_dist(point.coords[0], lines[k].coords[1])
            if dist1 < dist2:
                candidates[k] = dist1
                edge_points[k] = Point(lines[k].coords[0])
            else:
                candidates[k] = dist2
                edge_points[k] = Point(lines[k].coords[1])
    J = [k for k, v in candidates.items() if v == min(candidates.values())]
    if nodes_only:
        return J
    if len(J) == 1 or not all([x in edge_points.keys() for x in J]):
        k_star = J[0]
        proj_point = Point(perp_point(point.coords[0], lines[k_star].coords))
    else:
        theta_star = np.nan_to_num(np.inf)
        for k in J:
            m1 = line_gradient(point.coords[0], edge_points[k].coords[0])
            m2 = -1/line_gradient(lines[k].coords[0], lines[k].coords[1])
            theta = math.fabs(math.atan((m1-m2)/(1+m1*m2))*180/math.pi)
            if theta < theta_star:
                theta_star = theta
                k_star = k
                proj_point = edge_points[k]
    return k_star, proj_point

def script():
    accidents = orange_table_to_pandas_dataframe(in_datas[0])
    roads = orange_table_to_pandas_dataframe(in_datas[1])

    points = accidents['geometry'].to_numpy()
    lines = roads['geometry'].to_numpy()

    i = 0
    start_time = time.time()
    road_ids = []
    for p in points:
        k_line, new_point = find_closest_line(p, lines)
        road_ids.append(roads.iloc[k_line]['RoadID'])
        accidents.at[i, 'geometry'] = new_point
        dt = time.time() - start_time
        print(round(i / len(points) * 100, 3), '\t', time.strftime("%H:%M:%S", time.gmtime(dt)), '\t', round((dt / ((i + 1) / len(points)) - dt) / (60 ** 2), 3), '\t', time.ctime((time.time() + (dt / ((i + 1) / len(points)) - dt))), '\t', psutil.virtual_memory())
        i += 1
    accidents['RoadID'] = road_ids
    d = Orange.data.Domain(in_data[0].domain.attributes, class_vars=in_data[0].domain.class_vars, metas=in_data[0].domain.metas + (Orange.data.StringVariable('RoadID'),))
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