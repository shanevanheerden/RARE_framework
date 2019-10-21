from __future__ import absolute_import, division, print_function
import os
#import arcpy
import numpy as np
import pandas as pd
import scipy
from scipy import misc, special
import itertools
import csv
import time
import pywinauto
import pyautogui
import glob
import random
import math
import warnings
import Orange
import data_configuration as dc
import pickle

warnings.filterwarnings('ignore')
#pd.set_option('display.height', 10000)
pd.set_option('display.width', 10000)
pd.set_option('display.max_rows', 20)
pd.set_option('display.max_columns', 20)

STATS = {'rows': {}, 'filter_time': {}, 'nnd_time': {}, 'score_time': {}, 'acc_time': {}}
SCORES_LOOKUP = {}
START_TIME = None


class FeatureFilter:  # class for feature filtering

    def __init__(self, extension, features, points):  # receive features dictionary and points shapefile
        self.extension = extension  # data extension
        self.features = features
        self.points = points  # accidents shapefile
        self.filter_folder = r'\filtered'  # filter folder
        self.time_round = 2

    @staticmethod
    def check_exists(folder, c_name, file_types):  # check if files already exist

        if not os.path.exists(folder):  # if k-folder does not exist
            os.makedirs(folder)  # make a k-folder
            return False  # file doesn't exist

        check = [os.path.isfile(folder + '\\' + c_name + x) for x in file_types]  # check if each file is there
        remove = [x for x, y in zip(file_types, check) if y is True]  # identify all the files that should be removed if not all file types are in the folder

        if sum(check) != len(file_types):  # check if all file types are not there
            [os.remove(folder + '\\' + c_name + x) for x in remove]  # delete the types that are there
            return False  # file doesn't exist
        else:
            return True  # file exists

    def cat_code(self, feats, cat):  # get unique category code identifier
        code = []
        for f, c in zip(feats, cat):  # for each feature-category
            code.append(self.features[f]['f_code'] + self.features[f]['c_name'][c]['c_code'])  # append all f_code and c_code pairs

        code = ''.join(sorted([list(x) for x in itertools.permutations(code)])[0])  # sort all permutations and take first as code identifier

        return code  # return category code identifier

    def cat_name(self, feats, cat):  # get category name
        name = ''  # initialise blank name
        name += self.cat_code(feats, cat) + '_' + str(cat[0])  # append first code identifier and category name(s)

        for k in range(1, len(cat)):  # for any other categories
            name += '_' + str(cat[k])  # append the category name
        return name  # return category name

    def build_query(self, feats, cat):
        where = ''  # initialise where
        c = 0
        for k in cat:  # for each category
            gis_name = self.features[feats[c]]['gis_name']
            c_num = self.features[feats[c]]['c_name'][k]['c_num']
            if where != '':  # if not the first iteration
                where += ' AND '  # join with AND
            if feats[c] == "RoadClass":  # if dealing with RoadClass
                where += '(\"1st_' + gis_name + '\" = ' + str(c_num) + ' OR \"2nd_' + gis_name + '\" = ' + str(c_num) + ')'  # where clause combining 1st and 2nd road classes
            else:  # if anything else
                where += '(\"' + gis_name + '\" = ' + str(c_num) + ')'  # where clause
            c += 1
        return where

    def run_ff(self, feats, cat):  # create filtered shapefile
        c_name = self.cat_name(feats, cat)  # get category name
        out_folder = self.extension + self.filter_folder + '\k' + str(len(feats))  # define folder for filtered files
        file_types = ['.cpg', '.dbf', '.prj', '.sbn', '.sbx', '.shp', '.shp.xml', '.shx']  # file types
        if self.check_exists(out_folder, c_name, file_types) is True:  # if the files already exist
            STATS['rows'][c_name] = int((os.path.getsize(out_folder + '\\' + c_name + '.dbf') - 1058) / 1313)
            STATS['filter_time'][c_name] = ''
            return True  # don't filter

        where = self.build_query(feats, cat)

        start_time = time.time()
        try:
            layer = arcpy.MakeFeatureLayer_management(self.points, c_name, where)  # query ArcGIS layer
            rows = int(arcpy.GetCount_management(layer).getOutput(0))

            if rows >= 2:
                arcpy.FeatureClassToShapefile_conversion(layer, out_folder)  # convert ArcGIS layer to shapefile
                print('FILTER', feats, c_name, time.time() - start_time)
                STATS['filter_time'][c_name] = round(time.time() - start_time, self.time_round)
                return True
            else:
                STATS['filter_time'][c_name] = round(time.time() - start_time, self.time_round)
                return False
        except:
            #print('ARCPY ERROR')
            return False


class NearestNeighbour(FeatureFilter):

    def __init__(self, extension, features, points, network):
        FeatureFilter.__init__(self, extension, features, points)
        self.network = network  # roads shapefile
        self.nnd_folder = r'\nnd'  # nnd folder

    def run_nn(self, feats, cat):  # run nearest neighbour distance algorithm in SANET toolbox
        c_name = self.cat_name(feats, cat)  # get category name

        out_folder = self.extension + self.nnd_folder + '\k' + str(len(feats))  # define folder for nnd files
        file_types = ['_expected.csv', '_observed.csv', '_graphics.R']  # file types
        if self.check_exists(out_folder, c_name, file_types) is True:  # if the files already exist
            return True  # don't run SANET

        in_folder = self.extension + self.filter_folder + '\k' + str(len(feats))  # define folder containing filtered shapefiles
        if not os.path.isfile(in_folder + '\\' + c_name + '.shp'):  # if the shapefile does not exist
            filtered = self.run_ff(feats, cat)  # run the filter to create the shapefile
            if filtered is False:  # if there are no points in the shapefile
                return False  # don't run SANET

        points_file = in_folder + '\\' + c_name + '.shp'

        out = self.extension + self.nnd_folder + '\k' + str(len(feats))
        observed_file = out + '\\' + c_name + '_observed.csv'
        expected_file = out + '\\' + c_name + '_expected.csv'
        graphics_file = out + '\\' + c_name + '_graphics.R'

        sanet = pywinauto.application.Application().start(r'C:\Program Files\PASCO\SANET_WIN\SANET_WIN.exe')

        press = [['tab'] * 9, 'enter', ['tab'] * 7, ['tab'] * 2, ['tab'] * 2, ['tab'] * 2, ['tab'] * 3, 'enter', 'enter']
        typewrite = [self.network, points_file, observed_file, expected_file, graphics_file]
        sequence = [0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0]
        counter = [0, 0]

        for k in sequence:
            sanet.top_window().set_focus()
            if k == 0:
                pyautogui.press(press[counter[0]])
                counter[0] += 1
            elif k == 1:
                pyautogui.typewrite(typewrite[counter[1]])
                counter[1] += 1

        start_time = time.time()

        while list(pywinauto.findwindows.find_windows(title='Grobal auto nearest neighbor distance method')) != []:
            pass

        print('NND', feats, c_name, time.time() - start_time)
        STATS['nnd_time'][c_name] = round(time.time() - start_time, self.time_round)
        sanet.kill()
        return True


class SpatialScore(NearestNeighbour):

    def __init__(self, extension, features, points, network):
        NearestNeighbour.__init__(self, extension, features, points, network)
        self.out_lines = [0, 1, 4, 5, 8, 9, 12, 13]
        self.curve = 2  # 1 = mean, 2 = upper, 3 = lower

    def order_features(self, feats):  # order features by f_code
        sort = []  # initialise sort list
        [sort.append([f, self.features[f]['f_code']]) for f in list(feats)]  # append all features and f_codes to list
        feats = list(zip(*sorted(sort, key=lambda x: x[1])))[0]  # sort list by f_code and redefine features

        return feats  # return ordered features

    def feat_code(self, feats):  # get unique feature code identifier
        code = []
        for f in feats:  # for each feature
            code.append(self.features[f]['f_code'])  # append all f_codes

        code = ''.join(sorted([list(x) for x in itertools.permutations(code)])[0])  # sort all permutations and take first as code identifier

        return code  # return feature code identifier

    def decode(self, code):
        code = [code[i:i+2] for i in range(0, len(code), 2)]

        f_name = list(self.features.keys())
        f_code = [self.features[x]['f_code'] for x in f_name]

        feats = dict(zip(f_code, f_name))
        cats = {}

        for k in f_code:
            c_name = list(self.features[feats[k]]['c_name'].keys())
            c_code = [self.features[feats[k]]['c_name'][x]['c_code'] for x in c_name]
            cats[k] = dict(zip(c_code, c_name))

        decode = []
        [decode.append(cats[k[0]][k[1]]) for k in code]

        return decode

    def run_ss(self, feats, throttle=1.0):
        throttle = round(throttle, 2)
        feats = self.order_features(feats)
        f_code = self.feat_code(feats)
        num_rows = {}

        try:
            spatial_score = SCORES_LOOKUP[f_code]
            print('LOOKUP', spatial_score, feats)
            return spatial_score
        except:
            pass

        cats = (list(self.features[x]['c_name'].keys()) for x in feats)

        if throttle != 1:
            for cat in itertools.product(*cats):
                c_name = self.cat_name(feats, cat)
                in_folder = self.extension + self.filter_folder + '\k' + str(len(feats))
                if not os.path.isfile(in_folder + '\\' + c_name + '.shp'):
                    self.run_ff(feats, cat)
                try:
                    num_rows[c_name.split('_')[0]] = int((os.path.getsize(in_folder + '\\' + c_name + '.dbf') - 1058) / 1313)
                except:
                    pass
            sorted_cats = sorted(num_rows.items(), key=lambda x: x[1], reverse=True)
            tot_rows = 0
            categs = []
            c = 0

            while tot_rows / round(sum(num_rows.values()) * throttle) < 1:
                tot_rows += sorted_cats[c][1]
                categs.append(sorted_cats[c][0])
                c += 1

            cats = [self.decode(x) for x in categs]
            cats = map(list, zip(*cats))
            cats = [list(set(x)) for x in cats]

        counts = []
        scores = []
        categos = []

        for cat in itertools.product(*cats):
            categos.append(cat[0])
            c_name = self.cat_name(feats, cat)
            c_code = c_name.split('_')[0]

            try:
                in_folder = glob.glob(self.extension + self.nnd_folder + '\k' + str(len(feats)) + '\\' + c_code + '*.R')[0]
            except:
                self.run_nn(feats, cat)
                try:
                    in_folder = glob.glob(self.extension + self.nnd_folder + '\k' + str(len(feats)) + '\\' + c_code + '*.R')[0]
                except:
                    counts.append(0)
                    scores.append(0)
                    continue

            start_time = time.time()

            r_file = open(in_folder, 'r')

            lines = []

            for line in r_file:
                lines.append(line.split())

            out = []
            count = 0
            for i in self.out_lines:
                temp = lines[i][3].split(',')
                temp.insert(0, lines[i][2][2:-1])
                temp[-1] = temp[-1][:-1]
                temp = list(map(float, temp))
                if count % 2 == 0:
                    out.append([])
                out[int(count / 2)].append(temp)
                count += 1

            area_tot1 = 0
            area_tot2 = 0
            h = out[0][0][1] - out[0][0][0]

            for i in range(len(out[self.curve][0]) - 1):
                if i < (len(out[0][0]) - 1):
                    a1 = out[0][1][i] / max(out[0][1]) - out[self.curve][1][i] / max(out[self.curve][1])
                    b1 = out[0][1][i + 1] / max(out[0][1]) - out[self.curve][1][i + 1] / max(out[self.curve][1])
                else:
                    a1 = 1 - out[self.curve][1][i] / max(out[self.curve][1])
                    b1 = 1 - out[self.curve][1][i + 1] / max(out[self.curve][1])

                a2 = 1 - out[self.curve][1][i] / max(out[self.curve][1])
                b2 = 1 - out[self.curve][1][i + 1] / max(out[self.curve][1])

                area1 = ((a1 + b1) / 2) * h
                area2 = ((a2 + b2) / 2) * h

                if area1 > 0:
                    area_tot1 += area1
                area_tot2 += area2

            rows = int(max(out[0][1]))
            counts.append(rows)
            scores.append(area_tot1 / area_tot2)

            STATS['rows'][c_name] = rows
            STATS['score_time'][c_name] = round(time.time() - start_time, self.time_round)
            STATS['acc_time'][c_name] = round(time.time() - START_TIME, self.time_round)

            r_file.close()

        total = sum(counts)
        norm_counts = [k / total for k in counts]
        print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@', feats[0])
        [print(c, n, s) for c, n, s in list(zip(categos, counts, scores))]

        spatial_score = sum([x * y for x, y in zip(norm_counts, scores)])
        SCORES_LOOKUP[f_code] = spatial_score
        #print('CALCULATED', float(spatial_score), feats)

        return spatial_score


class OldCode:

    def delete_files(self, folder, name):  # check if files already exist
        types = ['.cpg', '.dbf', '.prj', '.sbn', '.sbx', '.shp', '.shp.xml', '.shx']  # different file types

        check = [os.path.isfile(folder + '\\' + name + x) for x in types]  # check if each file is there
        remove = [x for x, y in zip(types, check) if y is True]  # identify all the files that should be removed if not all file types are in the folder
        [os.remove(folder + '\\' + name + x) for x in remove]  # delete the types that are there

        return


    def clean_points(self, points, features, feats):  # remove all accidents that have missing values
        delete = {'AccidentSeverity': [],
                  'DayOfWeek': [],
                  'RoadClass': [1, 2, 3],  # NOT
                  'RoadType': [-1],
                  'SpeedLimit': [],
                  'JunctionDetail': [-1],
                  'JunctionControl': [-1],
                  'PedestrianCrossingHuman': [-1],
                  'PedestrianCrossingPhysical': [-1],
                  'LightConditions': [-1],
                  'WeatherConditions': [9, -1],
                  'RoadSurfaceConditions': [-1],
                  'SpecialConditionsAtSite': [-1],
                  'CarriagewayHazards': [-1],
                  'UrbanOrRuralArea': [3],
                  'DidPoliceOfficerAttendSceneOfAccident': []}

        name = 'manchester_accidents_cleaned'  # new file name
        folder = points[:points.rfind('\\')]  # take the previous accidents shapefile folder

        delete_files(folder, name)  # delete any previously made file

        where = ''  # initialise where
        for f in range(len(feats)):  # for all the features
            for k in range(len(delete[feats[f]])):  # for all the categories to be deleted
                if where != '' and feats[f] != 'RoadClass':  # if not the first iteration and not dealing with RoadClass
                    where += ' OR '  # join with OR
                if feats[f] == 'RoadClass':  # if dealing with RoadClass
                    if k != 0:  # if not the first iteration
                        where += ' AND '  # join with AND
                    where += '(NOT (\"1st_' + features[feats[f]]['gis_name'] + '\" = ' + str(
                        delete[feats[f]][k]) + ' OR \"2nd_' + features[feats[f]]['gis_name'] + '\" = ' + str(delete[feats[f]][k]) + '))'  # where clause combining 1st and 2nd road classes
                else:  # if anything else
                    where += '(\"' + features[feats[f]]['gis_name'] + '\" = ' + str(delete[feats[f]][k]) + ')'  # where clause
        where = 'NOT (' + where + ')'  # invert final where clause

        layer = arcpy.MakeFeatureLayer_management(points, name, where)  # query ArcGIS layer
        arcpy.FeatureClassToShapefile_conversion(layer, folder)  # convert ArcGIS layer to shapefile

        return folder + '\\' + name + '.shp'


    def get_features(self):  # get features dictionary from Excel file
        filename = r'C:\Users\17683068\Desktop\manchester\data\features.csv'  # csv file
        with open(filename, "r") as f:  # open csv file to read
            reader = csv.reader(f)  # csv reader
            raw = list(reader)  # read rows to list
        f.close()  # close csv file

        header = raw[0:3]  # define header as first 3 rows
        data = raw[3:]  # define data as the remaining rows

        features = {}  # initialise features dictionary

        for row in data:  # for every row in data
            e = 0  # initialise element counter to 1
            for element in row:  # for every remaining element in the row
                element = element.replace(' ', '')  # remove all the spaces from strings
                if e == 0:
                    f_key = str(element)  # define f_name as key
                    features[f_key] = {}  # initialise dictionary for key value
                elif e == 1:
                    c_keys = element.split(',')
                    features[f_key]['c_name'] = {}
                elif header[1][e] == 'scalar':
                    if header[2][e] == 'integer' and header[1][e] == 'scalar':  # if integer and scalar
                        features[f_key][header[0][e]] = int(element)  # convert string to integer
                    elif header[2][e] == 'string' and header[1][e] == 'scalar':  # if string and scalar
                        features[f_key][header[0][e]] = element  # keep as is
                    elif header[2][e] == 'integer' and header[1][e] == 'vector':  # if string and vector
                        features[f_key][header[0][e]] = element.split(',')  # make string into list
                    else:
                        print('ERROR')  # somethings wrong
                elif header[1][e] == 'vector' and header[0][e] != 'exclude':
                    for c in c_keys:
                        if c not in list(features[f_key]['c_name'].keys()):
                            features[f_key]['c_name'][c] = {}
                        if header[2][e] == 'integer' and header[1][e] == 'vector':  # if integer and vector
                            features[f_key]['c_name'][c][header[0][e]] = element.split(',')[c_keys.index(c)]
                        elif header[2][e] == 'string' and header[1][e] == 'vector':  # if string and vector
                            features[f_key]['c_name'][c][header[0][e]] = element.split(',')[c_keys.index(c)]
                        else:
                            print('ERROR')  # somethings wrong
                e += 1  # increment element counter

        keys = [raw[x][header[0].index('f_name')] for x in range(3, len(raw))]
        excl = [raw[x][header[0].index('exclude')].replace(' ', '').split(',') for x in range(3, len(raw))]

        exclude = dict(zip(keys, excl))

        for i in list(features.keys()):
            if exclude[i] != '':
                for j in list(features[i]['c_name'].keys()):
                    if features[i]['c_name'][j]['c_num'] in exclude[i]:
                        del features[i]['c_name'][j]

        for f in list(features.keys()):
            print(f, features[f])

        return features  # return features dictionary


    def order_features(self, features, feats):  # order features by f_code
        sort = []  # initialise sort list
        [sort.append([f, features[f]['f_code']]) for f in list(feats)]  # append all features and f_codes to list
        print(sort)
        feats = list(zip(*sorted(sort, key=lambda x: x[1])))[0]  # sort list by f_code and redefine features

        return feats  # return ordered features


    def import_scores(self, extension, k, throttle):
        global SCORES_LOOKUP
        try:
            print(extension + '\lookups\lookup' + str(k) + '-' + str(int(round(throttle * 100))) + '.csv')
            reader = csv.reader(open(extension + '\lookups\lookup' + str(k) + '-' + str(int(round(throttle * 100))) + '.csv', 'r'))
            for row in reader:
                key, value = row
                SCORES_LOOKUP[key] = float(value)
        except:
            pass

        return

    def old_code(self):
        global START_TIME
        global SCORES_LOOKUP
        global STATS
        features = self.get_features()
        extension = r'C:\Users\17683068\Desktop\manchester\data'
        network = extension + r'\raw\manchester_roads.shp'
        points = extension + r'\raw\cleaned_manchester_accidents.shp'
        START_TIME = time.time()
        stats_id = int(round(time.time()))
        feats = self.order_features(features, list(features.keys()))
        print(feats)
        #points = clean_points(points, features, feats)

        k = 1
        throttle = 1
        self.import_scores(extension, k, throttle)

        for x in itertools.combinations(feats, k):
            SpatialScore(extension, features, points, network).run_ss(x, throttle)
            try:
                pd.DataFrame(STATS)[['rows', 'acc_time', 'score_time', 'nnd_time', 'filter_time']].to_csv(r'C:\Users\17683068\Desktop\manchester\data\stats\stats' + str(k) + '-' + str(int(round(throttle * 100))) + '-' + str(stats_id) + '.csv')
            except MemoryError:
                STATS = {'rows': {}, 'filter_time': {}, 'nnd_time': {}, 'score_time': {}, 'acc_time': {}}
                stats_id = int(round(time.time()))
            pd.DataFrame(SCORES_LOOKUP, columns=SCORES_LOOKUP.keys(), index=[0]).transpose().to_csv(r'C:\Users\17683068\Desktop\manchester\data\lookups\lookup' + str(k) + '-' + str(int(round(throttle * 100))) + '.csv', header=False)

        print(time.time() - START_TIME)


class AccidentDomains:

    def __init__(self):
        self.code_value_dict = {}
        self.value_code_dict = {}

    def get_locations(self, accidents, feats, alpha):
        accidents.data['Count'] = [1] * len(accidents.data.index)
        table = pd.DataFrame(pd.pivot_table(accidents.data, index=feats, values=['Count'], aggfunc=np.sum).to_records()).sort_values('Count', ascending=False).reset_index()
        table['Prob'] = table.apply(lambda row: row['Count'] / table['Count'].sum(), axis=1).astype(object)
        accidents.data.drop('Count', axis=1, inplace=True)
        table['CumProb'] = table['Prob'].cumsum(axis=0)
        if alpha < 1:
            first = table[table.CumProb < alpha]
            last = table[table.CumProb > alpha].reset_index().loc[[0]]
            table = first.append(last.drop(last.columns[0], axis=1), ignore_index=True).drop(first.columns[0], axis=1)
        else:
            table.drop('index', axis=1, inplace=True)
        print(table)
        locations_temp = table.drop(table.columns[[-3, -2, -1]], axis=1).T.to_dict('dict')
        locations = {}
        i_count = 0
        for i in locations_temp.keys():
            locations[i_count] = locations_temp[i]
            i_count += 1
        return locations

    def build_query(self, index, locations, feats):
        query = ''
        for k in feats:
            query += '{0}==\'{1}\''.format(k, locations[index][k])
            if k != feats[-1]:
                query += ' and '
        return query

    def rename_categories(self, data):
        for k in list(data.columns)[list(data.columns).index('VehicleType'):]:
            sorted_list = sorted(sorted(data[k].unique()), key=len)
            front_list = [x for x in sorted_list if "_" not in x]
            back_list = [x for x in sorted_list if "_" in x]
            reordered_list = front_list + back_list
            self.code_value_dict[k] = dict(zip(range(len(reordered_list)), reordered_list))
            self.value_code_dict[k] = dict(zip(reordered_list, range(len(reordered_list))))
            data[k] = data[k].apply(lambda row: self.value_code_dict[k][row])

    def calculate_score_matrix(self, accidents, spatial_feats, locations, method, fold=1):
        data_instance = dc.ConfigData()
        remove_cols = [i for i in accidents.data.columns if data_instance.metadata[accidents.country][i].info == 'm' or data_instance.metadata[accidents.country][i].info == 'i']
        accidents.data.drop(remove_cols, axis=1, inplace=True)
        score_matrix_list = []
        logreg = Orange.classification.LogisticRegressionLearner()
        if method == 'AUC':
            fold = 1
        for f in range(fold):
            score_matrix_list.append([])
            score_matrix_dict = {}
            i_count = 0
            for i in sorted(list(locations.keys())):
                j_count = 0
                score_matrix_dict[i] = {}
                score_matrix_list[f].append([])
                i_query = self.build_query(i, locations, spatial_feats)
                i_data = accidents.data.query(i_query).copy()
                i_data.drop(spatial_feats, axis=1, inplace=True)
                for j in sorted(list(locations.keys())):
                    print(i_count, j_count, (i_count*len(locations) + j_count)/(len(locations)**2))
                    if i_count == j_count:
                        score_matrix_dict[i][j] = 0.0
                        score_matrix_list[f][i_count].append(0.0)
                    else:
                        try:
                            score_matrix_dict[i][j] = score_matrix_dict[j][i]
                            score_matrix_list[f][i_count].append(score_matrix_dict[j][i])
                        except (KeyError, IndexError):
                            j_query = self.build_query(j, locations, spatial_feats)
                            j_data = accidents.data.query(j_query).copy()
                            j_data.drop(spatial_feats, axis=1, inplace=True)
                            if method == 'CA':
                                if len(i_data.index) > len(j_data.index):
                                    i_data = i_data.sample(len(j_data.index), replace=True)
                                else:
                                    j_data = j_data.sample(len(i_data.index), replace=True)
                            i_data['Target'] = [1] * len(i_data.index)
                            j_data['Target'] = [0] * len(j_data.index)
                            ij_data = pd.concat([i_data, j_data])
                            data_instance.metadata[accidents.country]['Target'] = data_instance.Feature('Target', 'd', 'c', None, None, {0: data_instance.Category('Top', None), 1: data_instance.Category('Bottom', None)})
                            accidents_instance = data_instance.Data(accidents.country, accidents.stream, accidents.table, accidents.format, accidents.year, ij_data, accidents.filter)
                            orange_table = data_instance._convert_csv_to_orangetable(accidents_instance, False, False)[0]
                            res = Orange.evaluation.CrossValidation(orange_table.data, [logreg], k=5)
                            if method == 'AUC':
                                score = max(2 * Orange.evaluation.scoring.AUC(res)[0] - 1, 0)
                            elif method == 'CA':
                                score = max(2 * Orange.evaluation.scoring.CA(res)[0] - 1, 0)
                            score_matrix_dict[i][j] = score
                            score_matrix_list[f][i_count].append(score)
                    j_count += 1
                [print(['%.2f' % elem for elem in k]) for k in score_matrix_list[f]]
                print(f, fold*len(locations.keys()), len([item for sublist in score_matrix_list for item in sublist]), len([item for sublist in score_matrix_list for item in sublist])/(len(locations.keys())*fold), max([max(i) for i in score_matrix_list[f]]), '\n')
                i_count += 1
        avg_score_matrix_list = [[sum([score_matrix_list[f][i][j] for f in range(fold)])/float(fold) for j in range(len(locations.keys()))] for i in range(len(locations.keys()))]
        [print(['%.2f' % elem for elem in k]) for k in avg_score_matrix_list]
        return avg_score_matrix_list

    def define_regions(self, accidents, spatial_feats, locations, score_matrix_list):
        accidents = dc.ConfigData()._convert_csv_to_orangetable(accidents, False, False)[0]
        attr = [k for k in accidents.data.domain if k.name in spatial_feats]
        for k, v in locations.items():
            string_temp = []
            for kk, vv in locations[k].items():
                string_temp.append(vv)
            locations[k]['label'] = '|'.join(string_temp)
        #[print(locations[w]) for w in sorted(locations.keys())]
        xs = []
        i_count = 0
        for i in sorted(list(locations.keys())):
            xs.append([])
            for j in spatial_feats:
                xs[i_count].append([k.values.index(locations[i][j]) for k in attr if k.name == j][0])
            i_count += 1
        metas = [[i] for i in [locations[j]['label'] for j in sorted(list(locations.keys()))]]
        d = Orange.data.Domain(attr, metas=[Orange.data.StringVariable('label')])
        rows = Orange.data.Table.from_numpy(domain=d, X=xs, metas=metas)
        dist_mat = Orange.misc.distmatrix.DistMatrix(np.array(score_matrix_list), row_items=rows)
        file_object = open(r'C:\Users\17683068\Desktop\masters\data\dist_matXXX.pickle', 'wb')
        pickle.dump(dist_mat, file_object)
        file_object.close()
        #dist_mat.save(r'C:\Users\17683068\Desktop\masters\data\dist_mat.dst')

    def run_ad(self):
        country = 'great_britain'
        stream = 'gbr_road_accidents'
        table = 'gbr_accidents_roads_vehicles'
        format = 'csv_label'
        years = list(range(2016, 2017, 1))
        concat = True
        query = 'FirstRoadClass==\'Motorway\' or FirstRoadClass==\'AM\' or FirstRoadClass==\'A\''
        base = None

        accidents = dc.ConfigData()._get_data(country, stream, table, format, years, concat, query, base)
        print(accidents.data.head(10))
        spatial_feats = ['FirstRoadClass', 'RoadType', 'SpeedLimit', 'JunctionDetail', 'JunctionControl', 'PedestrianCrossingPhysicalFacilities', 'UrbanOrRuralArea']
        locations = self.get_locations(accidents, spatial_feats, 1)
        score_matrix_list = self.calculate_score_matrix(accidents, spatial_feats, locations, 'AUC')
        self.define_regions(accidents, spatial_feats, locations, score_matrix_list)


def main():
    #AccidentDomains().run_ad()
    OldCode().old_code()


if __name__ == '__main__':
    main()
