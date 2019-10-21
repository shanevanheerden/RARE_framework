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
import math
import geopandas as gpd
import matplotlib.pyplot as plt
import shapely
import pandas as pd
import statistics
import Orange
import time
import psutil
import fiona
import math
from shapely.geometry import Point, LineString
from orangecontrib.rare.utils import orange_table_to_pandas_dataframe, pandas_dataframe_to_orange_table
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

####### PARAMETERS #######

script_name = 'kde'
execute_in_orange = False
args = {'max_bin_size': 5,
        'kernel_function': lambda x, w: (3 / 4 * (1 - (x / w)**2)) / w,
        'bandwidth': 50,
        'max_ro_length': 20,
        'severity_index': 'Severity',
        'make_shapefile': True}

######### SCRIPT #########

def line_gradient(p1, p2):
    try:
        m = (p2[1]-p1[1])/(p2[0]-p1[0])
    except ZeroDivisionError:
        m = np.nan_to_num(np.inf)
    return m

def script(max_bin_size, kernel_function, bandwidth, max_ro_length, severity_index, make_shapefile):
    accidents = orange_table_to_pandas_dataframe(in_datas[0])
    roads = orange_table_to_pandas_dataframe(in_datas[1])

    print('Data loaded.')

    points = accidents.geometry.to_dict()
    lines = roads[roads.Year == max(roads.Year.unique())].reset_index().geometry.to_dict()

    temp = {v: k for k, v in roads[roads.Year == max(roads.Year.unique())].reset_index().RoadID.to_dict().items()}

    accident_road_dict = {k: temp[v] for k, v in accidents.RoadID.to_dict().items()}
    temp2 = {v: k for k, v in temp.items()}

    line_network = {}
    traffic_flows = {}
    for i in range(len(lines)):
        aadf_df = roads[roads.RoadID == temp2[i]][['Year', 'AADF']].copy()
        aadf_dict = dict(zip(aadf_df.Year.astype(int), aadf_df.AADF))
        traffic_flows[i] = list(aadf_dict.values())[0]
        line_network[i] = [[], []]
        centroid = lines[i].centroid
        l1 = LineString((Point(lines[i].coords[0]), centroid))
        l2 = LineString((Point(lines[i].coords[1]), centroid))
        for j in range(len(lines)):
            if i != j:
                if l1.intersects(lines[j]):
                    line_network[i][0].append(j)
                elif l2.intersects(lines[j]):
                    line_network[i][1].append(j)

    print('Line network built.')

    #bins = np.array(range(100, 0, -1)).astype(float)
    #for bin in bins:
    chunk_network = {}
    to_do = []
    line_chunks_dict = {}
    file_line = []
    file_chunk = []
    file_geom = []
    for i in range(len(lines)):
        num_bins = int(lines[i].length // max_bin_size + 1)
        chunk_length = lines[i].length / num_bins
        m = line_gradient(lines[i].coords[0], lines[i].coords[1])
        theta = math.atan(m)
        x_move = (2*(lines[i].coords[0][0] < lines[i].coords[1][0]) - 1) * chunk_length * math.fabs(math.cos(theta))
        y_move = (2*(lines[i].coords[0][1] < lines[i].coords[1][1]) - 1) * chunk_length * math.fabs(math.sin(theta))
        temp = []
        for b in range(num_bins):
            chunk_network[(i, b)] = [[], []]
            temp.append(LineString((Point((lines[i].coords[0][0] + b*x_move, lines[i].coords[0][1] + b*y_move)), Point((lines[i].coords[0][0] + (b + 1)*x_move, lines[i].coords[0][1] + (b + 1)*y_move)))))
            if b == 0:
                if num_bins > 2:
                    chunk_network[(i, b)][1].append((i, b + 1))
                to_do.append((i, b))
            elif b == (num_bins - 1):
                if num_bins > 2:
                    chunk_network[(i, b)][0].append((i, b - 1))
                to_do.append((i, b))
            else:
                chunk_network[(i, b)][0].append((i, b - 1))
                chunk_network[(i, b)][1].append((i, b + 1))
        line_chunks_dict[i] = dict(zip(range(num_bins), temp))
        file_line.extend([i] * num_bins)
        file_chunk.extend(list(range(num_bins)))
        file_geom.extend(temp)
        #x = [l.length for l in file_geom]

    print('Chunked lines.')

    for li, ci in to_do:
        centroid = line_chunks_dict[li][ci].centroid
        l1 = LineString((Point(line_chunks_dict[li][ci].coords[0]), centroid))
        l2 = LineString((Point(line_chunks_dict[li][ci].coords[1]), centroid))
        for lj, cj in to_do:
            if (li, ci) != (lj, cj):
                if l1.intersects(line_chunks_dict[lj][cj]):
                    chunk_network[(li, ci)][0].append((lj, cj))
                elif l2.intersects(line_chunks_dict[lj][cj]):
                    chunk_network[(li, ci)][1].append((lj, cj))

    print('Chunk network built.')

    #i = 0
    #start_time = time.time()
    accident_distrib = {key: 0 for key in chunk_network.keys()}
    severity_distrib = {key: 0 for key in chunk_network.keys()}
    cols = [i.name for i in in_datas[0].domain.attributes]
    vals = {}
    for c in cols:
        vals[c] = [k.values for k in in_datas[0].domain.attributes if k.name == c][0]
    lc_keys = chunk_network.keys()
    attribute_distrib = {}
    for key in lc_keys:
        attribute_distrib[key] = {}
        for k in cols:
            attribute_distrib[key][k] = {}
            #for kk in vals[k]:
                #attribute_distrib[key][k][kk] = 0
        #dt = time.time() - start_time
        #print(i, round(i / len(lc_keys) * 100, 3), '\t', time.strftime("%H:%M:%S", time.gmtime(dt)), '\t', round((dt / ((i + 1) / len(lc_keys)) - dt) / (60 ** 2), 3), '\t', time.ctime((time.time() + (dt / ((i + 1) / len(lc_keys)) - dt))), '\t', psutil.virtual_memory())
        #i += 1

    print('Attribute dict initialised.')

    def recursion(i, l, c, kernel, w, banned_chunks, old_l, old_c, prev_dist, orig_line, crossed, deg_crossed, dist_pe_start, dist_pe_end, sign):
        if i == 0:
            dist_pe_start = line_chunks_dict[l][c].centroid.distance(Point(lines[orig_line].coords[0]))
            dist_pe_end = -line_chunks_dict[l][c].centroid.distance(Point(lines[orig_line].coords[1]))
            if dist_pe_start > w and abs(dist_pe_end) > w:
                #print('CASE 1:', w, dist_pe_start, dist_pe_end)
                pass
            elif dist_pe_start > w and abs(dist_pe_end) < w:
                #print('CASE 2:', w, dist_pe_start, dist_pe_end)
                pass
            elif dist_pe_start < w and abs(dist_pe_end) > w:
                #print('CASE 3:', w, dist_pe_start, dist_pe_end)
                pass
            elif dist_pe_start < w and abs(dist_pe_end) < w:
                #print('CASE 4:', w, dist_pe_start, dist_pe_end)
                pass
            else:
                #print('ERROR', w, dist_pe_start, dist_pe_end)
                pass
            dist_pe = min(dist_pe_start, dist_pe_end)
        elif i == 1:
            if old_l == l:
                if old_c < c:
                    sign = -1
                else:
                    sign = 1
            else:
                if old_c == 0:
                    sign = 1
                else:
                    sign = -1

        abs_dist_pq = 0 if i == 0 else prev_dist + (line_chunks_dict[old_l][old_c].length / 2) + (line_chunks_dict[l][c].length / 2)
        if abs_dist_pq < w:
            ne_start = len(line_network[orig_line][0]) + 1
            ne_end = len(line_network[orig_line][1]) + 1

            if l != old_l and i != 0:
                crossed += 1
                if crossed > 1:
                    abc = line_network[old_l][[z for z in range(2) if l in line_network[old_l][z]][0]]
                    deg_crossed *= len(abc)

            dist_pq = sign * abs_dist_pq

            if 2 * dist_pe_end + w < dist_pq < 2 * dist_pe_start - w:#-w <= dist_pq < 2 * dist_pe - w:
                val_to_add = deg_crossed * line_chunks_dict[l][c].length * kernel(dist_pq, w)
                accident_distrib[(l, c)] += val_to_add
                severity_distrib[(l, c)] += sev*val_to_add
                for k, v in feat_cat.items():
                    if v not in attribute_distrib[(l, c)][k].keys():
                        attribute_distrib[(l, c)][k][v] = 0
                    attribute_distrib[(l, c)][k][v] += val_to_add
                #print('A: {}, S:{}, C:{}, D:{}, N:{}, PE:{}, QE:{}, 2PE-w:{}, PQ:{}, Val:{}'.format((l, c), sign, crossed, (round(dist_qe_start, 2), round(dist_qe_end, 2)), (ne_start, ne_end), (round(dist_pe_start, 2), round(dist_pe_end, 2)), (round(dist_qe_start, 2), round(dist_qe_end, 2)),  (round(2 * dist_pe_start - w, 2), round(2 * dist_pe_end + w, 2)), round(dist_pq, 2), val_to_add))
            elif 2 * dist_pe_start - w <= dist_pq < dist_pe_start and dist_pe_end < dist_pq <= 2 * dist_pe_end + w:#2 * dist_pe - w <= dist_pq < dist_pe:
                val_to_add = deg_crossed * line_chunks_dict[l][c].length * kernel(dist_pq, w) - (ne_start - 2) / ne_start * kernel(2 * dist_pe_start - dist_pq, w) - (ne_end - 2) / ne_end * kernel(2 * dist_pe_end - dist_pq, w)
                accident_distrib[(l, c)] += val_to_add
                severity_distrib[(l, c)] += sev * val_to_add
                for k, v in feat_cat.items():
                    if v not in attribute_distrib[(l, c)][k].keys():
                        attribute_distrib[(l, c)][k][v] = 0
                    attribute_distrib[(l, c)][k][v] += val_to_add
                #print('B0: {}, S:{}, C:{}, D:{}, N:{}, PE:{}, QE:{}, 2PE-w:{}, PQ:{}, Val:{}'.format((l, c), sign, crossed, (round(dist_qe_start, 2), round(dist_qe_end, 2)), (ne_start, ne_end), (round(dist_pe_start, 2), round(dist_pe_end, 2)), (round(dist_qe_start, 2), round(dist_qe_end, 2)),  (round(2 * dist_pe_start - w, 2), round(2 * dist_pe_end + w, 2)), round(dist_pq, 2), val_to_add))
            elif dist_pe_start <= dist_pq < w and dist_pe_end < dist_pq <= 2 * dist_pe_end + w: #dist_pe <= dist_pq < w:
                val_to_add = deg_crossed * line_chunks_dict[l][c].length * 2 / ne_start * kernel(dist_pq, w) - 1 / (ne_start - 1) * (ne_end - 2) / ne_end * kernel(2 * dist_pe_end - dist_pq, w)
                accident_distrib[(l, c)] += val_to_add
                severity_distrib[(l, c)] += sev * val_to_add
                for k, v in feat_cat.items():
                    if v not in attribute_distrib[(l, c)][k].keys():
                        attribute_distrib[(l, c)][k][v] = 0
                    attribute_distrib[(l, c)][k][v] += val_to_add
                #print('C01: {}, S:{}, C:{}, D:{}, N:{}, PE:{}, QE:{}, 2PE-w:{}, PQ:{}, Val:{}'.format((l, c), sign, crossed, (round(dist_qe_start, 2), round(dist_qe_end, 2)), (ne_start, ne_end), (round(dist_pe_start, 2), round(dist_pe_end, 2)), (round(dist_qe_start, 2), round(dist_qe_end, 2)),  (round(2 * dist_pe_start - w, 2), round(2 * dist_pe_end + w, 2)), round(dist_pq, 2), val_to_add))
            elif -w < dist_pq <= dist_pe_end and 2 * dist_pe_start - w <= dist_pq < dist_pe_start: #dist_pe <= dist_pq < w:
                val_to_add = deg_crossed * line_chunks_dict[l][c].length * 2 / ne_end * kernel(dist_pq, w) - 1 / (ne_end - 1) * (ne_start - 2) / ne_start * kernel(2 * dist_pe_start - dist_pq, w)
                accident_distrib[(l, c)] += val_to_add
                severity_distrib[(l, c)] += sev * val_to_add
                for k, v in feat_cat.items():
                    if v not in attribute_distrib[(l, c)][k].keys():
                        attribute_distrib[(l, c)][k][v] = 0
                    attribute_distrib[(l, c)][k][v] += val_to_add
                #print('C02: {}, S:{}, C:{}, D:{}, N:{}, PE:{}, QE:{}, 2PE-w:{}, PQ:{}, Val:{}'.format((l, c), sign, crossed, (round(dist_qe_start, 2), round(dist_qe_end, 2)), (ne_start, ne_end), (round(dist_pe_start, 2), round(dist_pe_end, 2)), (round(dist_qe_start, 2), round(dist_qe_end, 2)),  (round(2 * dist_pe_start - w, 2), round(2 * dist_pe_end + w, 2)), round(dist_pq, 2), val_to_add))
            elif 2 * dist_pe_start - w <= dist_pq < dist_pe_start:#2 * dist_pe - w <= dist_pq < dist_pe:
                val_to_add = deg_crossed * line_chunks_dict[l][c].length * kernel(dist_pq, w) - (ne_start - 2) / ne_start * kernel(2 * dist_pe_start - dist_pq, w)
                accident_distrib[(l, c)] += val_to_add
                severity_distrib[(l, c)] += sev * val_to_add
                for k, v in feat_cat.items():
                    if v not in attribute_distrib[(l, c)][k].keys():
                        attribute_distrib[(l, c)][k][v] = 0
                    attribute_distrib[(l, c)][k][v] += val_to_add
                #print('B1: {}, S:{}, C:{}, D:{}, N:{}, PE:{}, QE:{}, 2PE-w:{}, PQ:{}, Val:{}'.format((l, c), sign, crossed, (round(dist_qe_start, 2), round(dist_qe_end, 2)), (ne_start, ne_end), (round(dist_pe_start, 2), round(dist_pe_end, 2)), (round(dist_qe_start, 2), round(dist_qe_end, 2)),  (round(2 * dist_pe_start - w, 2), round(2 * dist_pe_end + w, 2)), round(dist_pq, 2), val_to_add))
            elif dist_pe_end < dist_pq <= 2 * dist_pe_end + w:
                val_to_add = deg_crossed * line_chunks_dict[l][c].length * kernel(dist_pq, w) - (ne_end - 2) / ne_end * kernel(2 * dist_pe_end - dist_pq, w)
                accident_distrib[(l, c)] += val_to_add
                severity_distrib[(l, c)] += sev * val_to_add
                for k, v in feat_cat.items():
                    if v not in attribute_distrib[(l, c)][k].keys():
                        attribute_distrib[(l, c)][k][v] = 0
                    attribute_distrib[(l, c)][k][v] += val_to_add
                #print('B2: {}, S:{}, C:{}, D:{}, N:{}, PE:{}, QE:{}, 2PE-w:{}, PQ:{}, Val:{}'.format((l, c), sign, crossed, (round(dist_qe_start, 2), round(dist_qe_end, 2)), (ne_start, ne_end), (round(dist_pe_start, 2), round(dist_pe_end, 2)), (round(dist_qe_start, 2), round(dist_qe_end, 2)),  (round(2 * dist_pe_start - w, 2), round(2 * dist_pe_end + w, 2)), round(dist_pq, 2), val_to_add))
            elif dist_pe_start <= dist_pq < w: #dist_pe <= dist_pq < w:
                val_to_add = deg_crossed * line_chunks_dict[l][c].length * 2 / ne_start * kernel(dist_pq, w)
                accident_distrib[(l, c)] += val_to_add
                severity_distrib[(l, c)] += sev * val_to_add
                for k, v in feat_cat.items():
                    if v not in attribute_distrib[(l, c)][k].keys():
                        attribute_distrib[(l, c)][k][v] = 0
                    attribute_distrib[(l, c)][k][v] += val_to_add
                #print('C1: {}, S:{}, C:{}, D:{}, N:{}, PE:{}, QE:{}, 2PE-w:{}, PQ:{}, Val:{}'.format((l, c), sign, crossed, (round(dist_qe_start, 2), round(dist_qe_end, 2)), (ne_start, ne_end), (round(dist_pe_start, 2), round(dist_pe_end, 2)), (round(dist_qe_start, 2), round(dist_qe_end, 2)),  (round(2 * dist_pe_start - w, 2), round(2 * dist_pe_end + w, 2)), round(dist_pq, 2), val_to_add))
            elif -w < dist_pq <= dist_pe_end:
                val_to_add = deg_crossed * line_chunks_dict[l][c].length * 2 / ne_end * kernel(dist_pq, w)
                accident_distrib[(l, c)] += val_to_add
                severity_distrib[(l, c)] += sev * val_to_add
                for k, v in feat_cat.items():
                    if v not in attribute_distrib[(l, c)][k].keys():
                        attribute_distrib[(l, c)][k][v] = 0
                    attribute_distrib[(l, c)][k][v] += val_to_add
                #print('C2: {}, S:{}, C:{}, D:{}, N:{}, PE:{}, QE:{}, 2PE-w:{}, PQ:{}, Val:{}'.format((l, c), sign, crossed, (round(dist_qe_start, 2), round(dist_qe_end, 2)), (ne_start, ne_end), (round(dist_pe_start, 2), round(dist_pe_end, 2)), (round(dist_qe_start, 2), round(dist_qe_end, 2)),  (round(2 * dist_pe_start - w, 2), round(2 * dist_pe_end + w, 2)), round(dist_pq, 2), val_to_add))
            else:
                #print('NOTHING ADDED!!!: {}, S:{}, C:{}, D:{}, N:{}, PE:{}, QE:{}, 2PE-w:{}, PQ:{}'.format((l, c), sign, crossed, (round(dist_qe_start, 2), round(dist_qe_end, 2)), (ne_start, ne_end), (round(dist_pe_start, 2), round(dist_pe_end, 2)), (round(dist_qe_start, 2), round(dist_qe_end, 2)),  (round(2 * dist_pe_start - w, 2), round(2 * dist_pe_end + w, 2)), round(dist_pq, 2)))
                pass

            step = i + 1
            for chunk in chunk_network[(l, c)]:
                if chunk is not []:
                    for new_l, new_c in chunk:
                        #print(i, (l, c), chunk_network[(l, c)], (new_l, new_c), banned_chunks, (new_l, new_c) not in banned_chunks)
                        if (new_l, new_c) not in banned_chunks:
                            banned = chunk + [(l, c)]
                            recursion(step, new_l, new_c, kernel, w, banned, l, c, abs_dist_pq, orig_line, crossed, deg_crossed, dist_pe_start, dist_pe_end, sign)
        else:
            #print('END', (l, c))
            pass

    i = 0
    start_time = time.time()
    for p in points.keys():
        sev = accidents[i:i+1][severity_index].to_list()[0]
        feat_cat = dict([(k.name, vals[k.name][int(l)]) for k, l in zip(in_datas[0].domain.attributes, list(in_datas[0].X[i]))])
        l = int(accident_road_dict[p])
        line = line_chunks_dict[l]
        min_dist = np.inf
        c = None
        for k, v in line.items():
            new_dist = v.distance(points[p])
            if new_dist < min_dist:
                min_dist = new_dist
                c = k

        #print('XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX', p, (l, c), len(chunk_network[(l, c)][0]) + len(chunk_network[(l, c)][1]), chunk_network[(l, c)])
        recursion(0, l, c, kernel_function, bandwidth, [], None, None, 0, l, 0, 1, None, None, 1)
        #area = sum([v * line_chunks_dict[k[0]][k[1]].length for k, v in chunk_values.items()])
        dt = time.time() - start_time
        print(i, round(i / len(points) * 100, 3), '\t', time.strftime("%H:%M:%S", time.gmtime(dt)), '\t', round((dt / ((i + 1) / len(points)) - dt) / (60 ** 2), 3), '\t', time.ctime((time.time() + (dt / ((i + 1) / len(points)) - dt))), '\t', psutil.virtual_memory())
        i += 1

    #print(bin, sum(x) / len(x), len(x), (bin - sum(x) / len(x)) / bin * 100, statistics.stdev(x), statistics.stdev(x) / bin, len(points.keys()), sum(accident_distrib.values()), sum(accident_distrib.values())/len(points.keys()))
    print('Accident distribution constructed.')

    comb_accident_distrib = {}
    comb_severity_distrib = {}
    comb_attribute_distrib = {}
    for l, line in lines.items():
        num_bins = int(line.length // max_ro_length + 1)
        ro_length = line.length / num_bins
        r = 0
        cum_len = 0
        for c in range(len(line_chunks_dict[l])):
            chunk = line_chunks_dict[l][c]
            if cum_len + chunk.length > ro_length:
                prop = (ro_length - cum_len) / chunk.length
                if (l, r) not in list(comb_accident_distrib.keys()):
                    comb_accident_distrib[(l, r)] = 0.0
                    comb_severity_distrib[(l, r)] = 0.0
                if (l, r + 1) not in list(comb_accident_distrib.keys()):
                    comb_accident_distrib[(l, r + 1)] = 0.0
                    comb_severity_distrib[(l, r + 1)] = 0.0
                comb_accident_distrib[(l, r)] += accident_distrib[(l, c)] * prop
                #comb_accident_distrib[(l, r)] = comb_accident_distrib[(l, r)] * (1 + (max_ro_length - ro_length) / max_ro_length)
                comb_accident_distrib[(l, r + 1)] += accident_distrib[(l, c)] * (1 - prop)
                comb_severity_distrib[(l, r)] += severity_distrib[(l, c)] * prop
                #comb_severity_distrib[(l, r)] = comb_severity_distrib[(l, r)] * (1 + (max_ro_length - ro_length) / max_ro_length)
                comb_severity_distrib[(l, r + 1)] += severity_distrib[(l, c)] * (1 - prop)
                if (l, r) not in list(comb_attribute_distrib.keys()):
                    comb_attribute_distrib[(l, r)] = {}
                if (l, r + 1) not in list(comb_attribute_distrib.keys()):
                    comb_attribute_distrib[(l, r + 1)] = {}
                for k in attribute_distrib[(l, c)].keys():
                    if k not in list(comb_attribute_distrib[(l, r)].keys()):
                        comb_attribute_distrib[(l, r)][k] = {}
                    if k not in list(comb_attribute_distrib[(l, r + 1)].keys()):
                        comb_attribute_distrib[(l, r + 1)][k] = {}
                    for v in attribute_distrib[(l, c)][k].keys():
                        if v not in comb_attribute_distrib[(l, r)][k].keys():
                            comb_attribute_distrib[(l, r)][k][v] = attribute_distrib[(l, c)][k][v] * prop
                        if v not in comb_attribute_distrib[(l, r + 1)][k].keys():
                            comb_attribute_distrib[(l, r + 1)][k][v] = attribute_distrib[(l, c)][k][v] * (1 - prop)
                        else:
                            comb_attribute_distrib[(l, r)][k][v] += attribute_distrib[(l, c)][k][v] * prop
                            #comb_attribute_distrib[(l, r)][k][v] = comb_attribute_distrib[(l, r)][k][v] * (1 + (max_ro_length - ro_length) / max_ro_length)
                            comb_attribute_distrib[(l, r + 1)][k][v] += attribute_distrib[(l, c)][k][v] * (1 - prop)
                cum_len = cum_len + chunk.length - ro_length
                r += 1
            else:
                if (l, r) not in list(comb_accident_distrib.keys()):
                    comb_accident_distrib[(l, r)] = 0.0
                    comb_severity_distrib[(l, r)] = 0.0
                comb_accident_distrib[(l, r)] += accident_distrib[(l, c)]
                comb_severity_distrib[(l, r)] += severity_distrib[(l, c)]
                if (l, r) not in list(comb_attribute_distrib.keys()):
                    comb_attribute_distrib[(l, r)] = {}
                for k in attribute_distrib[(l, c)].keys():
                    if k not in list(comb_attribute_distrib[(l, r)].keys()):
                        comb_attribute_distrib[(l, r)][k] = {}
                    for v in attribute_distrib[(l, c)][k].keys():
                        if v not in comb_attribute_distrib[(l, r)][k].keys():
                            comb_attribute_distrib[(l, r)][k][v] = attribute_distrib[(l, c)][k][v]
                        else:
                            comb_attribute_distrib[(l, r)][k][v] += attribute_distrib[(l, c)][k][v]
                cum_len += chunk.length

    print('Constructed road observations.')

    new_attribute_distrib = {}
    cats = []
    for key in accident_distrib.keys():
        new_attribute_distrib[key] = {}
        for k in cols:
            for kk in vals[k]:
                cats.append(kk)
                try:
                    new_attribute_distrib[key][k + '=' + kk] = 0.0 if accident_distrib[key] == 0.0 else attribute_distrib[key][k][kk]/sum(attribute_distrib[key][k].values())
                except KeyError:
                    new_attribute_distrib[key][k + '=' + kk] = 0.0

    print('Attribute distribution constructed.')

    #[print(k, v) for k, v in chunk_values.items()]
    comb_keys = new_attribute_distrib[(0, 0)].keys()
    attribute_names = [i.name for i in in_datas[0].domain.attributes]
    feats = [[k for k in comb_keys if a in k] for a in attribute_names]
    print(attribute_names)
    print(feats)
    if make_shapefile:
        prop_types = {'line': 'int', 'chunk': 'int', 'accidents': 'float', 'traffic': 'int', 'rate': 'float', 'severity': 'float'}
        for a in attribute_names:
            prop_types[a] = 'int'
        schema = {'geometry': 'LineString', 'properties': prop_types,}
        with fiona.open(r'C:\Users\17683068\Desktop\GM\shapefiles\CHUNKS.shp', 'w', 'ESRI Shapefile', schema) as file:
            for l, c, g in zip(file_line, file_chunk, file_geom):
                prop_vals = {'line': l, 'chunk': c, 'accidents': np.nan if accident_distrib[(l, c)] == 0.0 else accident_distrib[(l, c)], 'traffic': traffic_flows[l], 'rate': np.nan if accident_distrib[(l, c)] == 0.0 else accident_distrib[(l, c)]/traffic_flows[l]*1000000000, 'severity': np.nan if accident_distrib[(l, c)] == 0.0 else severity_distrib[(l, c)] / accident_distrib[(l, c)]}
                for feat, att in zip(feats, attribute_names):
                    temppp = [new_attribute_distrib[(l, c)][f] for f in feat]
                    if sum(temppp) != 0:
                        prop_vals[att] = int(np.argmax(np.array(temppp))) + 1
                    else:
                        prop_vals[att] = 0
                file.write({'geometry': shapely.geometry.mapping(g), 'properties': prop_vals,})
        print('Shapefile saved.')

    xs = []
    ys = []
    metas = []
    for l, c in comb_accident_distrib.keys():
        xs.append([new_attribute_distrib[(l, c)][k] for k in new_attribute_distrib[(l, c)].keys()])
        #xs.append([max(attribute_distrib[(l, c)][k], key=attribute_distrib[(l, c)][k].get) for k in attribute_distrib[(l, c)].keys()])
        ys.append([0.0 if comb_accident_distrib[(l, c)] == 0.0 else comb_accident_distrib[(l, c)], 0.0 if comb_accident_distrib[(l, c)] == 0.0 else comb_accident_distrib[(l, c)]/traffic_flows[l]*1000000000, 0.0 if comb_accident_distrib[(l, c)] == 0.0 else comb_severity_distrib[(l, c)] / comb_accident_distrib[(l, c)]])
        metas.append([str(l), str(c)])#, line_chunks_dict[l][c].length, np.nan if comb_accident_distrib[(l, c)] == 0.0 else 1.0, str(line_chunks_dict[l][c])])
    #print(np.nanmean(np.array(ys), axis=0))
    attr = [Orange.data.ContinuousVariable(k) for k in comb_keys]
    d = Orange.data.Domain(attr, class_vars=[Orange.data.ContinuousVariable('Frequency'), Orange.data.ContinuousVariable('Rate'), Orange.data.ContinuousVariable('Severity')], metas=[Orange.data.StringVariable('RoadID'), Orange.data.StringVariable('ChunkID')])#, Orange.data.ContinuousVariable('ChunkLength'), Orange.data.ContinuousVariable('AccidentsOnChunk'), Orange.data.StringVariable('geometry')])
    out_data = Orange.data.Table.from_numpy(d, X=np.array(xs), Y=np.array(ys), metas=np.array(metas))

    print('Saving Orange file.')

    #return out_data, None, None, None


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