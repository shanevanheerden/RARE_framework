import numpy as np
import matplotlib.pyplot as plt
from Orange.widgets.utils import colorpalette
from matplotlib import rcParams

feat_names = [in_data.domain.attributes[i].name for i in range(len(in_data.domain.attributes))]

feat_cat_dict = {}
col = 0
for f in feat_names:
    feat_cat = f.split('=')
    if feat_cat[0] not in feat_cat_dict.keys():
        feat_cat_dict[feat_cat[0]] = {}
    feat_cat_dict[feat_cat[0]][feat_cat[1]] = {'f': in_data.Y[:,0].dot(in_data.X[:, col]), 'r': in_data.Y[:,1].dot(in_data.X[:, col])/sum(in_data.X[:,col]), 's': in_data.Y[:,2].dot(in_data.X[:, col])/sum(in_data.X[:,col])}
    col += 1
    
f_scatter = 'RoadType' # 'FirstRoadClass' 'RoadType' 'SpeedLimit' 'JunctionDetail' 'JunctionControl' 'PedestrianCrossingPhysicalFacilities' 'UrbanOrRuralArea'
f_filter = 'SpeedLimit'

n_scatter = len(feat_cat_dict[f_scatter].keys())
n_filter = len(feat_cat_dict[f_filter].keys())

c_scatter = list(feat_cat_dict[f_scatter].keys())
c_filter = list(feat_cat_dict[f_filter].keys())

freq = [v['f'] for k, v in feat_cat_dict[f_scatter].items()]
rate = [v['r'] for k, v in feat_cat_dict[f_scatter].items()]
severity = [v['s'] for k, v in feat_cat_dict[f_scatter].items()]

rs = []
i = 0
for cs in c_scatter:
    rs.append([])
    for cf in c_filter:
        s = feat_names.index(f_scatter + '=' + cs)
        f = feat_names.index(f_filter + '=' + cf)
        count = in_data.X[:, s].dot(in_data.X[:, f])
        rs[i].append(count)
    i += 1

rs_new = [[r[k]/sum(r) for k in range(n_filter)] for r in rs]

sizes = np.array([10000*f/sum(freq) for f in freq])

palette = colorpalette.ColorPaletteGenerator(n_filter)
colours = []
for i in range(n_filter):
    colours.append([x/255 for x in list(palette[i].getRgb()[:-1])])

fig, ax = plt.subplots(figsize=(10, 8))
for j in range(n_scatter):
    xy = []
    ses = []
    for k in range(n_filter):
        x = [0] + np.cos(np.linspace(2 * np.pi * sum(rs_new[j][:k]), 2 * np.pi * sum(rs_new[j][:k+1]), 100)).tolist()
        y = [0] + np.sin(np.linspace(2 * np.pi * sum(rs_new[j][:k]), 2 * np.pi * sum(rs_new[j][:k+1]), 100)).tolist()
        xy.append(np.column_stack([x, y]))
        ses.append(np.abs(xy).max())
        
    if j == 0:
        [ax.scatter(99999999, 99999999, marker='s', s=100, facecolor=colours[k], label=c_filter[k]) for k in range(n_filter)]
    
    [ax.scatter(severity[j], rate[j], marker=(xy[k], 0), s=ses[k] ** 2 * sizes[j], facecolor=colours[k]) for k in range(n_filter)]

rcParams['font.family'] = 'cmr10'
shift_scale = 0.002
n_pad = 0.15
e_pad = 0.15
s_pad = 0.1
w_pad = 0.1
title_fontsize = 16
axis_fontsize = 16
label_fontsize = 16
legend_fontsize = 16
legendtitle_fontsize = 16
leg_loc = 'upper left'

[ax.annotate(txt, (severity[i]+shift_scale*((sizes[i]/np.pi)**0.5*(max(severity)-min(severity))), rate[i]+shift_scale*((sizes[i]/np.pi)**0.5*(max(rate)-min(rate)))), fontsize=label_fontsize) for i, txt in enumerate(c_scatter)]
plt.ylim(min(rate)-s_pad*(max(rate)-min(rate)),max(rate)+n_pad*(max(rate)-min(rate)))
plt.xlim(min(severity)-w_pad*(max(severity)-min(severity)),max(severity)+e_pad*(max(severity)-min(severity)))
plt.title(f_scatter, fontsize=title_fontsize)
plt.xlabel('Average Accident Severity', fontsize=axis_fontsize)
plt.ylabel('Average Accident rate', fontsize=axis_fontsize)
leg = plt.legend(loc=leg_loc, title=f_filter, fontsize=legend_fontsize)
leg._legend_box.align = 'left'
leg.get_title().set_fontsize(legendtitle_fontsize)
plt.grid()
ax.set_axisbelow(True)
ax.xaxis.grid(color='lightgray', linestyle='dashed')
ax.yaxis.grid(color='lightgray', linestyle='dashed')
plt.show()