import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
import glob
import os
import seaborn as sns
from functools import reduce

plt.rcParams.update(
    {'font.size': 13, 'font.family': "serif", 'mathtext.fontset': 'dejavuserif', 'xtick.direction': 'in',
     'xtick.major.size': 0.5, 'grid.linestyle': "--", 'axes.grid': False, "grid.alpha": 1, "grid.color": "#cccccc",
     'xtick.minor.size': 1.5, 'xtick.minor.width': 0.5, 'xtick.minor.visible': True, 'xtick.top': True,
     'ytick.direction': 'in', 'ytick.major.size': 0.5, 'ytick.minor.size': 1.5, 'ytick.minor.width': 0.5,
     'ytick.minor.visible': True, 'ytick.right': True, 'axes.linewidth': 0.5, 'grid.linewidth': 0.5,
     'lines.linewidth': 1.5, 'legend.frameon': False, 'savefig.bbox': 'tight', 'savefig.pad_inches': 0.05})
pd.options.mode.chained_assignment = None

# Read MSA Geo data
MSA_geo = gpd.GeoDataFrame.from_file(r'D:\Google_Review\Parking\tl_2019_us_cbsa\tl_2019_us_cbsa.shp')
MSA_geo = MSA_geo.to_crs(epsg=5070)
# print(len(set(MSA_geo['CSAFP'])))  # CSAFP means CSA

# Read CBG Geo data
'''
# poly = gpd.GeoDataFrame.from_file(
#     r'F:\Data\Dewey\SAFEGRAPH\Open Census Data\Census Website\2019\nhgis0011_shape\\US_blck_grp_2019_84.shp')
# poly['BGFIPS'] = poly['GISJOIN'].str[1:3] + poly['GISJOIN'].str[4:7] + \
#                  poly['GISJOIN'].str[8:14] + poly['GISJOIN'].str[14:15]
# poly = poly[~poly['GISJOIN'].str[1:3].isin(['02', '15', '60', '66', '69', '72', '78'])].reset_index(drop=True)
# poly = poly.to_crs('EPSG:5070')  # 5070
# poly.to_pickle(r'D:\Google_Review\Parking\temp\poly_5070.pkl')
'''
poly = pd.read_pickle(r'D:\Google_Review\Parking\temp\poly_5070.pkl')

# Read other features
CT_Features = pd.read_csv(r'F:\Research_Old\COVID19-Socio\Data\CBG_COVID_19.csv', index_col=0)
CT_Features['BGFIPS'] = CT_Features['BGFIPS'].astype(str).apply(lambda x: x.zfill(12))
smart_loc = pd.read_pickle(r'D:\Incentrip_research\data\SmartLocationDatabaseV3\SmartLocationDatabase.pkl')
msa_population = smart_loc[['CBSA', 'CBSA_POP', 'CBSA_Name']].drop_duplicates(subset=['CBSA']).reset_index(drop=True)
poi_ty = pd.read_excel(r'D:\Google_Review\Parking\temp\places_summary_hsh.xlsx')
poi_ty = poi_ty[['naics_code', 'Categories']]

# Read POI info and sjoin with CBG
'''
filens = glob.glob(r'D:\Google_Review\Parking\parking-places-metrics\*classification*.csv')
for kk in tqdm(filens):
    temp = pd.read_csv(kk)
    g_pois = gpd.GeoDataFrame(temp, geometry=gpd.points_from_xy(temp['longitude'], temp['latitude']))
    g_pois = g_pois.set_crs('EPSG:4326')
    g_pois = g_pois.to_crs('EPSG:5070')
    SInUS = gpd.sjoin(g_pois, poly, how='inner', op='within').reset_index(drop=True)
    SInUS = SInUS[['gmap_id', 'BGFIPS']]
    g_pois = g_pois.merge(SInUS, on='gmap_id')
    g_pois.to_csv(r'D:\Google_Review\Parking\parking-sjoin-new\%s' % kk.split('\\')[-1], index=False)

# Read all csvs
g_pois = pd.concat(map(pd.read_csv, glob.glob(os.path.join(r'D:\Google_Review\Parking\parking-sjoin-new\\', "*.csv"))))
g_pois.to_pickle(r'D:\Google_Review\Parking\temp\park_all_new.pkl')
'''

# Read sentiment data
g_pois = pd.read_pickle(r'D:\Google_Review\Parking\temp\park_all_new.pkl')
g_pois['BGFIPS'] = g_pois['BGFIPS'].astype(str).apply(lambda x: x.zfill(12))
g_pois = g_pois.merge(poi_ty, on='naics_code')
g_pois['Categories'] = g_pois['Categories'].fillna('Others')
g_pois.loc[(g_pois['category'].str.contains('|'.join(['parking lot', 'parking garage']), case=False)) & (
        g_pois['Categories'] == 'Personal Service'), 'Categories'] = 'Parking'
g_pois['Categories'].value_counts()
g_pois['total_parking_reviews'] = g_pois['total_parking_reviews'].astype(int)
g_pois['avg_parking_sentiment'] = g_pois['avg_parking_sentiment'].astype(float)
g_pois['sum_comment'] = g_pois.groupby(['BGFIPS', 'Categories'])['total_parking_reviews'].transform("sum")
g_pois['weight_st'] = (g_pois['total_parking_reviews'] / g_pois['sum_comment']) * g_pois['avg_parking_sentiment']

# Average sentiment to CBG
bg_pois = g_pois.groupby(['BGFIPS', 'Categories'])['weight_st'].sum().reset_index()
bg_count = g_pois.groupby(['BGFIPS', 'Categories'])['total_parking_reviews'].sum().reset_index()
bg_pois = bg_pois.merge(bg_count, on=['BGFIPS', 'Categories'])
bg_pois['Categories'].value_counts()
# Six type: 'Restaurant', 'Retail Trade', 'Recreation', 'Hotel', 'Personal Service', 'Apartment'
# type_list = list(g_pois['Categories'].value_counts().head(6).index) + ['Parking']
# bg_pois = bg_pois[bg_pois['Categories'].isin(type_list)]
bg_pois.groupby(['Categories'])['BGFIPS'].count().reset_index()
# Only contiguous US
bg_pois = bg_pois[bg_pois['BGFIPS'].isin(poly['BGFIPS'])].reset_index(drop=True)

# Build some new CBG features based on GMAP POIs
gall_pois = pd.read_pickle(r'D:\Google_Review\\Urban_sen\data\\us_gm_poi.pkl')
gall_pois['BGFIPS'] = gall_pois['BGFIPS'].astype(str).apply(lambda x: x.zfill(12))
# Total number of reviews: all pois
g_fe1 = gall_pois.groupby(['BGFIPS'])['num_of_reviews'].sum().reset_index()
# Total number of all POIs
g_fe2 = g_pois.groupby(['BGFIPS'])['gmap_id'].count().reset_index()
g_fe2.columns = ['BGFIPS', 'total_poi_count']
# Avg rating score of all POIs
gall_pois['sum_comment_all'] = gall_pois.groupby(['BGFIPS'])['num_of_reviews'].transform("sum")
gall_pois['weight_st_all'] = (gall_pois['num_of_reviews'] / gall_pois['sum_comment_all']) * gall_pois['avg_rating']
g_fe3 = gall_pois.groupby(['BGFIPS'])['weight_st_all'].sum().reset_index()
g_fe3.columns = ['BGFIPS', 'avg_poi_score']

# Parking related POIs
parkings = gall_pois[
    gall_pois['category'].astype(str).str.contains('|'.join(['parking lot', 'parking garage']), case=False)]
g_fe4 = parkings.groupby(['BGFIPS'])['num_of_reviews'].sum().reset_index()
g_fe4.columns = ['BGFIPS', 'num_of_parking_review']
g_fe5 = parkings.groupby(['BGFIPS'])['gmap_id'].count().reset_index()
g_fe5.columns = ['BGFIPS', 'total_parking_count']
# Avg parking score
parkings['sum_comment_all'] = parkings.groupby(['BGFIPS'])['num_of_reviews'].transform("sum")
parkings['weight_st_all'] = (parkings['num_of_reviews'] / parkings['sum_comment_all']) * parkings['avg_rating']
g_fe6 = parkings.groupby(['BGFIPS'])['weight_st_all'].sum().reset_index()
g_fe6.columns = ['BGFIPS', 'avg_parking_score']
# Avg parking sentiment
g_fe7 = bg_pois.loc[
    (bg_pois['Categories'] == 'Parking') & (bg_pois['total_parking_reviews'] > 10), ['BGFIPS', 'weight_st']]
g_fe7.columns = ['BGFIPS', 'avg_parking_st']
g_fe = reduce(lambda left, right: pd.merge(left, right, on='BGFIPS', how='outer'),
              [g_fe1, g_fe2, g_fe3, g_fe4, g_fe5, g_fe6, g_fe7])
g_fe = g_fe.fillna(0)

# Merge
bg_pois = bg_pois.merge(g_fe, on='BGFIPS').reset_index(drop=True)
# Merge with CBG features
bg_pois = bg_pois.merge(CT_Features, on='BGFIPS').reset_index(drop=True)
# Merge with smart location features
bg_pois = bg_pois.merge(smart_loc[['BGFIPS', 'D3B', 'Pct_AO0', 'Pct_AO1', 'Pct_AO2p', 'NatWalkInd', 'D3A', 'D1C', 'D4C',
                                   'CBSA', 'CBSA_POP', 'Ac_Total', 'CSA', 'D4A', 'D4D', 'D4E']], on='BGFIPS')
# Merge with MSA features
bg_pois = bg_pois.merge(MSA_geo, left_on='CBSA', right_on='CBSAFP', how='left')
# Only greater than 10
bg_pois = bg_pois[(bg_pois['total_parking_reviews'] > 10)].reset_index(drop=True)
sns.displot(bg_pois['weight_st'])  # .loc[bg_pois['Categories'] == 'Hotel', 'weight_st']

# Rename:
bg_pois = bg_pois.rename(columns={'D3B': 'Intersection_Density', 'Pct_AO0': 'Zero_car_R', 'Pct_AO1': 'One_car_R',
                                  'Pct_AO2p': 'Two_plus_car_R', 'NatWalkInd': 'Walkability', 'D3A': 'Road_Density',
                                  'D1C': 'Employment_Density', 'D4C': 'Transit_Freq', 'D4A': 'Distance_Transit',
                                  'D4D': 'Transit_Freq_Area', 'D4E': 'Transit_Freq_Pop'})

# Polish data
bg_pois.loc[bg_pois['Transit_Freq'] < 0, 'Transit_Freq'] = 0
bg_pois.loc[bg_pois['Transit_Freq_Area'] < 0, 'Transit_Freq_Area'] = 0
bg_pois.loc[bg_pois['Transit_Freq_Pop'] < 0, 'Transit_Freq_Pop'] = 0
bg_pois.loc[bg_pois['Distance_Transit'] < 0, 'Distance_Transit'] = 1207.008
bg_pois['Parking_review_density'] = (bg_pois['total_parking_reviews'] + bg_pois['num_of_parking_review']) / bg_pois[
    'ALAND_x']
bg_pois['Total_review_density'] = bg_pois['num_of_reviews'] / bg_pois['ALAND_x']
bg_pois['Parking_poi_density'] = bg_pois['total_parking_count'] / bg_pois['ALAND_x']
bg_pois['Total_poi_density'] = bg_pois['total_poi_count'] / bg_pois['ALAND_x']

# Output data: To R for modelling
bg_pois = bg_pois.dropna(subset=['LSAD']).reset_index(drop=True)
need_scio = ['Population_Density', 'Bt_18_44_R', 'Asian_R', 'Over_65_R', 'Public_Transit_R', 'Republican_R', 'GINI',
             'Intersection_Density', 'Bt_45_64_R', 'Urbanized_Areas_Population_R', 'Indian_R', 'Rural_Population_R',
             'Household_Below_Poverty_R', 'Employment_Density', 'Worked_at_home_R', 'Two_plus_car_R', 'Ac_Total',
             'HISPANIC_LATINO_R', 'Zero_car_R', 'Black_Non_Hispanic_R', 'Drive_alone_R', 'Bicycle_R', 'CBSA_POP',
             'White_Non_Hispanic_R', 'Carpool_R', 'Male_R', 'Urban_Clusters_Population_R', 'Road_Density', 'Walk_R',
             'Walkability', 'One_car_R', 'Transit_Freq', 'Education_Degree_R', 'Democrat_R', 'Median_income',
             'Distance_Transit', 'Transit_Freq_Area', 'Transit_Freq_Pop', 'total_parking_reviews', 'num_of_reviews',
             'total_poi_count', 'avg_poi_score', 'num_of_parking_review', 'total_parking_count', 'avg_parking_score',
             'Total_Population', "Parking_review_density", "Total_review_density", "Parking_poi_density",
             "Total_poi_density"]
fbg_pois = bg_pois[
    ['BGFIPS', 'Categories', 'weight_st', 'avg_parking_st', 'ALAND_x', 'Lng', 'Lat', 'CBSA', 'LSAD'] + need_scio]
fbg_pois.to_csv(r'D:\Google_Review\Parking\temp\bg_poitype_parking.csv')
fbg_pois.isnull().sum()
fbg_pois.corr(numeric_only=True).to_csv(r'D:\Google_Review\Parking\temp\bg_poitype_parking_corr.csv')
fbg_pois.describe().T.to_csv(r'D:\Google_Review\Parking\temp\bg_poitype_parking_des.csv')

## Discription analysis
# How many CBG in each MSA
# bg_msa = bg_pois[bg_pois['LSAD'] == 'M1'].reset_index(drop=True)
bg_msa = bg_pois.copy()
ct_cbg = bg_msa.groupby("NAMELSAD").count()[['BGFIPS']].sort_values(by='BGFIPS', ascending=False).reset_index()
bg_msa = bg_msa[bg_msa['NAMELSAD'].isin(ct_cbg.loc[ct_cbg['BGFIPS'] > 10, 'NAMELSAD'])].reset_index(drop=True)
print('No of MSA: %s' % len(ct_cbg.loc[ct_cbg['BGFIPS'] > 10, 'NAMELSAD']))

# Groupby MSA
bg_msa_avg = bg_msa.groupby("NAMELSAD")[['weight_st'] + need_scio].mean()
temp = bg_msa_avg.corr()
plt.plot(bg_msa_avg['Black_Non_Hispanic_R'], bg_msa_avg['weight_st'], 'o', alpha=0.5)

# # Plot boxplot: Avg sentiment by MSA and POI type
# for ep in set(bg_pois['Categories']):
#     poi_msa = bg_msa[bg_msa['Categories'] == ep].reset_index(drop=True)
#     ranks = poi_msa.groupby("NAMELSAD")["weight_st"].mean().fillna(0).sort_values()[::-1].index
#     ranks01 = list(ranks[0:10]) + list(ranks[-10:])
#     fig, ax = plt.subplots(figsize=(8, 6))
#     sns.boxplot(data=poi_msa[poi_msa['NAMELSAD'].isin(ranks01)], y='NAMELSAD', x='weight_st', order=ranks01, ax=ax)
#     ax.xaxis.grid(True)
#     plt.title(ep)
#     plt.tight_layout()
#     plt.savefig(r'D:\Google_Review\Parking\results\sentiment_poi_%s.png' % ep, dpi=1000)
#     plt.close()

# Correlation by MSA and POI type
all_poi_corr = pd.DataFrame()
for ep in set(bg_pois['Categories']):
    poi_msa = bg_msa[bg_msa['Categories'] == ep].reset_index(drop=True)
    all_corr = pd.DataFrame()
    for csa in set(bg_msa['NAMELSAD'].dropna()):
        temp = poi_msa[(poi_msa['NAMELSAD'] == csa)]
        temp['weight_st'] = temp['weight_st'].astype(float)
        corr_t = temp.corr(numeric_only=True)[['weight_st']]
        corr_t.columns = [csa]
        corr_t = corr_t.reset_index()
        if len(all_corr) == 0:
            all_corr = corr_t
        else:
            all_corr = corr_t.merge(all_corr, on='index', how='outer')
    all_corr = all_corr.set_index('index')

    # Get all corr and plot
    des_cor = all_corr.T.describe()
    des_mean = des_cor.loc['mean',].sort_values(ascending=False)
    des_mean = des_mean[
        des_mean.index.isin(need_scio)]
    des_mean_tt = pd.concat([des_mean.head(10), des_mean.tail(10)]).reset_index()

    des_mean = des_mean.reset_index()
    des_mean.columns = ['index', ep]
    if len(all_poi_corr) == 0:
        all_poi_corr = des_mean
    else:
        all_poi_corr = des_mean.merge(all_poi_corr, on='index', how='outer')

    # Plot correlation
    pty_corr = all_corr[all_corr.index.isin(des_mean_tt['index'])].T.reset_index(drop=True)
    pty_corr = pd.melt(pty_corr, value_vars=pty_corr.columns)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.boxplot(data=pty_corr, x='value', y='index', order=list(des_mean_tt['index']), ax=ax)
    ax.xaxis.grid(True)
    plt.title(ep)
    plt.xlabel('Correlation')
    plt.tight_layout()
    # plt.savefig(r'D:\Google_Review\Parking\results\corr_poi_%s_new.png' % ep, dpi=1000)
    # plt.close()

all_poi_corr = pd.melt(all_poi_corr, id_vars='index', value_vars=all_poi_corr.columns[1:])
# all_poi_corr = all_poi_corr[all_poi_corr['variable'].isin(bg_pois['Categories'].value_counts().head(6).index)]
all_poi_corr = all_poi_corr[all_poi_corr['index'].isin(all_poi_corr[all_poi_corr['value'].abs() > 0.1]['index'])]
fig, ax = plt.subplots(figsize=(10, 10))
sns.barplot(data=all_poi_corr, x='value', y='index', hue='variable', ax=ax)
ax.xaxis.grid(True)
ax.yaxis.grid(True)
plt.legend(loc=2)
plt.xlabel('Correlation')
plt.tight_layout()
plt.savefig(r'D:\Google_Review\Parking\results\corr_poi_each_new.png', dpi=1000)
