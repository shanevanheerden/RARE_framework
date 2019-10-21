"""View Shapefile

View shapefile layer(s).

Inputs
----------
in_datas : Orange.data.Table

Outputs
-------
None

"""

######## PACKAGES ########

import geopandas as gpd
import matplotlib.pyplot as plt
import contextily as ctx
import time

####### PARAMETERS #######

script_name = 'reconcile_rows'
execute_in_orange = True
args = {'zoom': 10}

######### SCRIPT #########

def add_basemap(ax, zoom, url='http://tile.stamen.com/terrain/tileZ/tileX/tileY.png'):
    xmin, xmax, ymin, ymax = ax.axis()
    basemap, extent = ctx.bounds2img(xmin, ymin, xmax, ymax, zoom=zoom, url=url)
    ax.imshow(basemap, extent=extent, interpolation='bilinear')
    ax.axis((xmin, xmax, ymin, ymax))

def zoom(event):
    cur_xlim = ax.get_xlim()
    cur_ylim = ax.get_ylim()
    cur_xrange = (cur_xlim[1] - cur_xlim[0])*.5
    cur_yrange = (cur_ylim[1] - cur_ylim[0])*.5
    xdata = event.xdata
    ydata = event.ydata
    if event.button == 'up':
        scale_factor = 1/2
    elif event.button == 'down':
        scale_factor = 2
    else:
        scale_factor = 1
    ax.set_xlim([xdata - cur_xrange*scale_factor, xdata + cur_xrange*scale_factor])
    ax.set_ylim([ydata - cur_yrange*scale_factor, ydata + cur_yrange*scale_factor])
    add_basemap(ax, zoom=10, url=ctx.sources.ST_TONER_LITE)
    plt.draw()

def script(zoom):
    shapefiles = [i.name for i in in_datas]
    script_args = [{'color':'red', 'markersize':3}, {'color':'black'}]
    fig, ax = plt.subplots()
    [gpd.read_file(s).to_crs(epsg=3857).plot(ax=ax, **a) for s, a in zip(shapefiles, script_args)]
    add_basemap(ax, zoom=zoom, url=ctx.sources.ST_TONER_LITE)
    fig.canvas.mpl_connect('scroll_event', zoom)
    plt.tight_layout()
    plt.show()
    return None, None, None, None

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
