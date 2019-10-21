"""Junction Similarity

Compute similarity matrix of junction locations.

Inputs
----------
in_data : Orange.data.Table

Outputs
-------
out_object : Orange.misc.distmatrix.DistMatrix

"""

######## PACKAGES ########

import numpy as np
import math
import Orange
import time
from scipy import sparse
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix, coo_matrix
import itertools
import pickle
from orangecontrib.rare.utils import orange_table_to_pandas_dataframe
import psutil
from matplotlib import rcParams
rcParams['font.family'] = 'cmr10'

####### PARAMETERS #######

script_name = 'junction_similarity32'
execute_in_orange = False
args = {'measure': 'of',  # of, lin, goodall3
        'radius': 100,
        'threshold': 0.7}

######### SCRIPT #########

class Similarity:
    def __init__(self, data):
        self.D = data.X
        self.D_inv = np.array(self.D).transpose().tolist()
        self.N = len(data.X)
        self.d = len(data.X[0])
        self.A = list(range(self.d))
        self.A_set = {}
        for k in range(self.d):
            self.A_set[k] = list(range(len(data.domain[k].values)))

    def n(self, k):
        return len(self.A_set[k])

    def f(self, k, x):
        return self.D_inv[k].count(x)

    def p(self, k, x):
        return self.f(k, x)/self.N

    def p2(self, k, x):
        return (self.f(k, x)*(self.f(k, x)-1))/(self.N*(self.N-1))

    def sim(self, X, Y, measure='overlap'):
        sim = []

        if measure == 'anderberg':
            num_den = sum([(1 / self.p(k, X[k])) ** 2 * (2 / (self.n(k) * (self.n(k) + 1))) if X[k] == Y[k] else 0 for k in range(self.d)])
            den = sum([(1 / (2 * self.p(k, X[k]) * self.p(k, Y[k]))) * (2 / (self.n(k) * (self.n(k) + 1))) if X[k] != Y[k] else 0 for k in range(self.d)])
            return num_den / (num_den + den)
        else:
            for k in range(self.d):
                if measure == 'overlap':
                    if X[k] == Y[k]:
                        s = 1
                    else:
                        s = 0
                    w = 1/self.d
                elif measure == 'eskin':
                    if X[k] == Y[k]:
                        s = 1
                    else:
                        s = self.n(k)**2 / (self.n(k)**2 + 2)
                    w = 1/self.d
                elif measure == 'iof':
                    if X[k] == Y[k]:
                        s = 1
                    else:
                        s = 1 / (1 + math.log(self.f(k, X[k]) * math.log(self.f(k, Y[k]))))
                    w = 1/self.d
                elif measure == 'of':
                    if X[k] == Y[k]:
                        s = 1
                    else:
                        s = 1 / (1 + math.log(self.N / self.f(k, X[k])) * math.log(self.N / self.f(k, Y[k])))
                    w = 1/self.d
                elif measure == 'lin':
                    if X[k] == Y[k]:
                        s = 2 * math.log(self.p(k, X[k]))
                    else:
                        s = 2 * math.log(self.p(k, X[k]) + self.p(k, Y[k]))
                    w = 1 / sum([math.log(self.p(i, X[i])) + math.log(self.p(i, Y[i])) for i in range(self.d)])
                elif measure == 'lin1':
                    if X[k] == Y[k]:
                        s = sum([math.log(self.p(k, q)) if self.p(k, X[k]) <= self.p(k, q) <= self.p(k, Y[k]) else 0 for q in self.A_set[k]])
                    else:
                        s = 2 * math.log(sum([self.p(k, q) if self.p(k, X[k]) <= self.p(k, q) <= self.p(k, Y[k]) else 0 for q in self.A_set[k]]))
                    w = 1 / (sum([sum([math.log(self.p(k, q)) if self.p(k, X[k]) <= self.p(k, q) <= self.p(k, Y[k]) else 0 for q in self.A_set[k]]) for i in range(self.d)]))
                elif measure == 'goodall1':
                    pass
                elif measure == 'goodall2':
                    pass
                elif measure == 'goodall3':
                    if X[k] == Y[k]:
                        s = 1-self.p2(k, X[k])
                    else:
                        s = 0
                    w = 1/self.d
                elif measure == 'goodall4':
                    if X[k] == Y[k]:
                        s = self.p2(k, X[k])
                    else:
                        s = 0
                    w = 1/self.d
                elif measure == 'smirnov':
                    if X[k] == Y[k]:
                        s = (self.N - self.f(k, X[k])) / self.f(k, X[k]) + sum([self.f(k, q) / (self.N - self.f(k, q)) for q in self.A if q != X[k]])
                    else:
                        s = -2 + sum([self.f(k, q) / (self.N - self.f(k, q)) for q in self.A if q != X[k] and q != Y[k]])
                    w = 1/sum([self.n(i) for i in range(self.d)])
                elif measure == 'gambaryan':
                    if X[k] == Y[k]:
                        s = -(self.p(k, X[k]) * math.log(self.p(k, X[k]), 2) + (1-self.p(k, X[k])) * math.log(1-self.p(k, X[k]), 2))
                    else:
                        s = 0
                    w = 1/sum([self.n(i) for i in range(self.d)])
                elif measure == 'burnaby':
                    if X[k] == Y[k]:
                        s = 1
                    else:
                        s = sum([2 * math.log(1 - self.p(k, q)) for q in self.A]) / (math.log((self.p(k, X[k]) * self.p(k, Y[k])) / ((1 - self.p(k, X[k])) * (1 - self.p(k, Y[k]))))+sum([2 * math.log(1 - self.p(k, q)) for q in self.A]))
                    w = 1/self.d
                sim.append(w * s)
            return sum(sim)

def script(measure, radius, threshold):
    rows = []
    cols = []
    data = []
    bignum = math.exp(8) #13.021084882936476, 10.882761655985368, 9.826672220046262
    save_path = r'C:\Users\17683068\Desktop\GM\Junctions'
    run1 = True

    if run1:
        sim_data = Similarity(in_data)
        xy_coords = orange_table_to_pandas_dataframe(in_data)['geometry'].to_numpy()
        start_time = time.time()
        for i in range(len(in_data)):
            for j in range(len(in_data)):
                if j > i:
                    x1 = xy_coords[i].x
                    y1 = xy_coords[i].y
                    x2 = xy_coords[j].x
                    y2 = xy_coords[j].y
                    distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                    if distance <= radius:
                        similarity = sim_data.sim(in_data.X[i], in_data.X[j], measure)
                        print(similarity)
                        try:
                            score = similarity / ((distance) ** 2)
                        except ZeroDivisionError:
                            score = bignum
                        rows.append(i)
                        cols.append(j)
                        data.append(score)
                        rows.append(j)
                        cols.append(i)
                        data.append(score)
                elif j == i:
                    pass
                    rows.append(i)
                    cols.append(j)
                    data.append(bignum)
            dt = time.time() - start_time
            print(i, round(i / len(in_data) * 100, 3), '\t', time.strftime("%H:%M:%S", time.gmtime(dt)), '\t', round((dt / ((i + 1) / len(in_data)) - dt) / (60 ** 2), 3), '\t', time.ctime((time.time() + (dt / ((i + 1) / len(in_data)) - dt))), '\t', len(data) / (len(in_data) * (i + 1)) * 100, '\t', psutil.virtual_memory())

        #file_object = open(r'{0}\rows{1}{2}.pickle'.format(save_path, str(radius), measure), 'wb')
        #pickle.dump(rows, file_object, protocol=4)
        #file_object.close()

        #file_object = open(r'{0}\cols{1}{2}.pickle'.format(save_path, str(radius), measure), 'wb')
        #pickle.dump(cols, file_object, protocol=4)
        #file_object.close()

        #file_object = open(r'{0}\data{1}{2}.pickle'.format(save_path, str(radius), measure), 'wb')
        #pickle.dump(data, file_object, protocol=4)
        #file_object.close()

    if True:
        #file_object = open(r'{0}\rows{1}{2}.pickle'.format(save_path, str(radius), measure), 'rb')
        #rows = pickle.load(file_object)
        #file_object = open(r'{0}\cols{1}{2}.pickle'.format(save_path, str(radius), measure), 'rb')
        #cols = pickle.load(file_object)
        #file_object = open(r'{0}\data{1}{2}.pickle'.format(save_path, str(radius), measure), 'rb')
        #data = pickle.load(file_object)
        #test_data = np.array(pickle.load(file_object))

        #test_data = 1 / np.array(test_data)
        #test_data[test_data == np.inf] = bignum
        #test_data = np.log(test_data + 1)

        #print(np.amin(test_data), np.amax(test_data))
        #min_val = 0 #np.amin(data)
        #max_val = 14.61840433881251 #np.amax(data)
        #data = (data.astype(float) - min_val) / (max_val - min_val)

        '''
        fig, ax = plt.subplots(figsize=(10, 8))
        data = 1 / np.array(data)
        data[data == np.inf] = bignum
        data = np.log(data + 1)
        min_val = np.amin(data)
        max_val = np.amax(data)
        print(max_val)
        data = (data.astype(float) - min_val) / (max_val - min_val)
        ax.xaxis.grid(color='lightgray', linestyle='dashed')
        ax.yaxis.grid(color='lightgray', linestyle='dashed')
        plt.xlabel('Similarity score', fontsize=16)
        plt.ylabel('Count', fontsize=16)
        plt.hist(data, bins='auto')
        plt.show()
        '''

    sparce_mat = sparse.coo_matrix((np.array(data), (np.array(rows), np.array(cols)))).toarray()
    mat_coo = np.reciprocal(sparce_mat)
    mat_coo[mat_coo == np.inf] = bignum
    mat_coo = np.log(mat_coo + 1)

    #min_val = np.amin(mat_coo)
    #max_val = np.amax(mat_coo)

    print(np.amin(mat_coo), np.amax(mat_coo))

    #mat_coo = (mat_coo.astype(float) - min_val) / (max_val - min_val)
    out_object = Orange.misc.distmatrix.DistMatrix(mat_coo, row_items=in_data)

    file_object = open(r'{0}\data{1}{2}.pickle'.format(save_path, str(radius), measure), 'wb')
    pickle.dump(data, file_object, protocol=4)
    file_object.close()

    return None, None, None, out_object

    clusters = {}
    n_clust = 0
    count = 0
    prev_r = 0
    flag1 = False
    #[print(i, i[2] < threshold) for i in sorted(zip(rows, cols, data), key=lambda x: x[0])]


    dist_mat = {}
    for r, c, d in zip(rows, cols, data):
        dist_mat[(r, c)] = d
        dist_mat[(c, r)] = d

    loop = sorted(list(set([(r, c, round(d, 3)) if r < c else (c, r, round(d, 3)) for r, c, d in zip(rows, cols, data)])), key=lambda x: x[0])

    prev_r = -1
    clust = -1
    for r, c, d in loop:
        if r == prev_r:
            clusters[clust].append(c)
        else:
            clust += 1
            clusters[clust] = [r]
        prev_r = r

    [print(k, v) for k, v in clusters.items() if k < 100]
    print(len(clusters))

    #mini_clusters = {}
    #for k, v in clusters.items():
        #print(list(clusters.keys()))
        #print(list(zip(*list(clusters.keys()))))
        #temp_dict = {}


        #sparce_mat = sparse.coo_matrix([])
        #for itertools.combinations(v):


    '''
    for r, c, d in loop:
        count += 1
        #print(count, r, c, d, flag1, prev_r)
        if d < threshold and r != c:
            flag1 = True
            if n_clust == 0:
                clusters[n_clust] = [r, c]
                n_clust += 1
                continue
            else:
                flag2 = False
                for k, v in clusters.items():
                    if r in v and c in v:
                        flag2 = True
                        break
                    elif r in v and c not in v:
                        clusters[k].append(c)
                        flag2 = True
                        break
                    elif c in v and r not in v:
                        #print(c, 'INNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNN')
                        clusters[k].append(r)
                        flag2 = True
                        break
                if not flag2:
                    clusters[n_clust] = [r, c]
                    n_clust += 1
            print(count / len(loop))
            #print(clusters)
        elif r != c:
            clusters[n_clust] = [c]
            n_clust += 1
        if prev_r != r:
            #print('INNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNN')
            if not flag1 and r != c:
                #print(prev_r, 'NEWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWW')
                clusters[n_clust] = [prev_r]
                n_clust += 1
            flag1 = False
        #print(clusters)
        prev_r = r
    print(n_clust, len(clusters.keys()))
    '''
    cluster_data = []

    for i in range(len(in_data)):
        flag3 = False
        for k, v in clusters.items():
            if i in v:
                flag3 = True
                cluster_data.append(k)
                break
        if not flag3:
            cluster_data.append(n_clust)
            n_clust += 1

    print(len(set(cluster_data)), n_clust)

    metas = np.c_[in_data.metas, cluster_data]

    d = Orange.data.Domain(in_data.domain.attributes, class_vars=in_data.domain.class_vars, metas=in_data.domain.metas + (Orange.data.DiscreteVariable('Cluster', values=['J{0}'.format(int(i)+1) for i in range(n_clust)]),))
    out_data = Orange.data.Table.from_numpy(d, X=in_data.X, Y=in_data.Y, metas=metas)

    ind = [i.name for i in out_data.domain.metas].index('Cluster')
    col = out_data.domain.metas[ind].colors.copy()
    np.random.shuffle(col)
    [out_data.domain.metas[ind].set_color(i, c) for i, c in zip(range(len(out_data.domain.metas[ind].colors)), col)]

    return out_data, None, None, None

##### INPUTS/OUTPUTS #####

from orangecontrib.rare.handlers import IOHandler

iohandler = IOHandler(script=script_name, execute_in_orange=execute_in_orange)

if not iohandler.ide_is_orange:
    in_data, in_datas, in_learner, in_learners, in_classifier, in_classifiers, in_object, in_objects = iohandler.load_inputs()
elif iohandler.ide_is_orange and not iohandler.execute_in_orange:
    iohandler.save_inputs(in_data=in_data, in_datas=in_datas, in_learner=in_learner, in_learners=in_learners,
                          in_classifier=in_classifier, in_classifiers=in_classifiers, in_object=in_object,
                          in_objects=in_objects)

if (iohandler.execute_in_orange and iohandler.ide_is_orange) or (not iohandler.ide_is_orange):
    out_data, out_learner, out_classifier, out_object = script(**args)

if not iohandler.ide_is_orange:
    iohandler.save_outputs(out_data=out_data, out_learner=out_learner, out_classifier=out_classifier,
                           out_object=out_object)
elif iohandler.ide_is_orange and not iohandler.execute_in_orange:
    out_data, out_learner, out_classifier, out_object = iohandler.load_outputs()