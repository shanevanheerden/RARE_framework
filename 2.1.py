import Orange
from orangecontrib.rare.utils import orange_table_to_pandas_dataframe, pandas_dataframe_to_orange_table

roads = orange_table_to_pandas_dataframe(in_data)

temp = {v: k for k, v in roads[roads.Year == max(roads.Year.unique())].reset_index().RoadID.to_dict().items()}
temp2 = {v: k for k, v in temp.items()}

traffic_flows = []
for i in temp2.keys():
    aadf_df = roads[roads.RoadID == temp2[i]][['Year', 'AADF']].copy()
    aadf_dict = dict(zip(aadf_df.Year.astype(int), aadf_df.AADF))
    traffic_flows.append(sum([v * 366 if k % 4 == 0 else v * 365 for k, v in aadf_dict.items()]))

new_roads = roads[roads.Year == max(roads.Year.unique())].reset_index()
new_roads = new_roads.drop(['index', 'AADF'], axis=1)
new_roads['AADF'] = traffic_flows

d = Orange.data.Domain(in_data.domain.attributes, class_vars=in_data.domain.class_vars, metas=in_data.domain.metas)

out_data = pandas_dataframe_to_orange_table(new_roads, d)
