import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import geopandas as gpd
import calendar
import glob
import os
import seaborn as sns

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
# print(len(set(MSA_geo['CSAFP'])))  # CSAFP means CSA
# MSA_geo = MSA_geo.to_crs('EPSG:4326')

# Read CBG Geo data
poly = gpd.GeoDataFrame.from_file(
    r'F:\\Data\\SafeGraph\\Open Census Data\\Census Website\\2019\\nhgis0011_shape\\US_blck_grp_2019_84.shp')
poly['BGFIPS'] = poly['GISJOIN'].str[1:3] + poly['GISJOIN'].str[4:7] + \
                 poly['GISJOIN'].str[8:14] + poly['GISJOIN'].str[14:15]
poly = poly[~poly['GISJOIN'].str[1:3].isin(['02', '15', '60', '66', '69', '72', '78'])].reset_index(drop=True)
# poly = poly.to_crs('EPSG:4326')  # 5070

'''
# Read POI info and sjoin with CBG
filens = glob.glob(r'D:\Google_Review\Parking\parking-places-metrics\*classification*.csv')
for kk in tqdm(filens):
    temp = pd.read_csv(kk)
    g_pois = gpd.GeoDataFrame(temp, geometry=gpd.points_from_xy(temp['longitude'], temp['latitude']))
    g_pois = g_pois.set_crs('EPSG:4326')
    SInUS = gpd.sjoin(g_pois, poly, how='inner', op='within').reset_index(drop=True)
    SInUS = SInUS[['gmap_id', 'BGFIPS']]
    g_pois = g_pois.merge(SInUS, on='gmap_id')
    g_pois.to_csv(r'D:\Google_Review\Parking\parking-sjoin-new\%s' % kk.split('\\')[-1], index=False)

# Read all csvs
g_pois = pd.concat(map(pd.read_csv, glob.glob(os.path.join(r'D:\Google_Review\Parking\parking-sjoin-new\\', "*.csv"))))
g_pois.to_pickle(r'D:\Google_Review\Parking\temp\park_all_new.pkl')
'''

# Read POI type
poi_ty = pd.read_excel(r'D:\Google_Review\Parking\temp\places_summary_hsh.xlsx')
poi_ty = poi_ty[['naics_code', 'Categories']]

# Read sentiment data
g_pois = pd.read_pickle(r'D:\Google_Review\Parking\temp\park_all_new.pkl')
g_pois = g_pois.merge(poi_ty, on='naics_code')
g_pois['Categories'] = g_pois['Categories'].fillna('Others')
g_pois['Categories'].value_counts()

# Average sentiment to CBG
g_pois['sum_comment'] = g_pois.groupby('BGFIPS')['total_parking_reviews'].transform("sum")
g_pois['weight'] = g_pois['total_parking_reviews'] / g_pois['sum_comment']
g_pois['weight_st'] = g_pois['weight'] * g_pois['avg_parking_sentiment']
bg_pois = g_pois.groupby(['BGFIPS', 'Categories'])['weight_st'].sum().reset_index()
bg_count = g_pois.groupby(['BGFIPS', 'Categories'])['total_parking_reviews'].sum().reset_index()
bg_pois = bg_pois.merge(bg_count, on=['BGFIPS', 'Categories'])
bg_pois = bg_pois[(bg_pois['total_parking_reviews'] > 10)].reset_index(drop=True)
bg_pois['BGFIPS'] = bg_pois['BGFIPS'].astype(str).apply(lambda x: x.zfill(12))
# bg_pois = bg_pois.fillna(0)
bg_pois = bg_pois[bg_pois['Categories'].isin(bg_pois['Categories'].value_counts().head(6).index)].reset_index(
    drop=True)

# Merge with CBG features
CT_Features = pd.read_csv(r'F:\Research\COVID19-Socio\Data\CBG_COVID_19.csv', index_col=0)
CT_Features['BGFIPS'] = CT_Features['BGFIPS'].astype(str).apply(lambda x: x.zfill(12))
bg_pois = bg_pois.merge(CT_Features, on='BGFIPS', how='left')
# Merge with smart location features
smart_loc = pd.read_pickle(r'D:\Incentrip_research\data\SmartLocationDatabaseV3\SmartLocationDatabase.pkl')
bg_pois = bg_pois.merge(smart_loc[['BGFIPS', 'D3B', 'Pct_AO0', 'Pct_AO1', 'Pct_AO2p', 'NatWalkInd', 'D3A', 'D1C', 'D4C',
                                   'CBSA']], on='BGFIPS', how='left')
# Merge with MSA features
bg_pois = bg_pois.merge(MSA_geo, right_on='CBSAFP', left_on='CBSA', how='left')
# Only contiguous US
bg_pois = bg_pois[bg_pois['BGFIPS'].isin(poly['BGFIPS'])].reset_index(drop=True)
# sns.displot(bg_pois['weight_st'])

# Rename:
bg_pois = bg_pois.rename(columns={'D3B': 'Intersection_Density', 'Pct_AO0': 'Zero_car_R', 'Pct_AO1': 'One_car_R',
                                  'Pct_AO2p': 'Two_plus_car_R', 'NatWalkInd': 'Walkability', 'D3A': 'Road_Density',
                                  'D1C': 'Employment_Density', 'D4C': 'Transit_Freq'})

# Output data
need_scio = ['Population_Density', 'Bt_18_44_R', 'Asian_R', 'Over_65_R', 'Public_Transit_R', 'Republican_R', 'GINI',
             'Intersection_Density', 'Bt_45_64_R', 'Urbanized_Areas_Population_R', 'Indian_R', 'Rural_Population_R',
             'Household_Below_Poverty_R', 'Employment_Density', 'Worked_at_home_R', 'Two_plus_car_R',
             'HISPANIC_LATINO_R', 'Zero_car_R', 'Black_Non_Hispanic_R', 'Drive_alone_R', 'Bicycle_R',
             'White_Non_Hispanic_R', 'Carpool_R', 'Male_R', 'Urban_Clusters_Population_R', 'Road_Density', 'Walk_R',
             'Walkability', 'One_car_R', 'Transit_Freq', 'Education_Degree_R', 'Democrat_R', 'Median_income']
bg_pois[['BGFIPS', 'Categories', 'weight_st', 'total_parking_reviews', 'Total_Population', 'ALAND_x', 'Lng', 'Lat',
         'CBSA', 'CSAFP', 'CBSAFP', 'LSAD', 'geometry'] + need_scio].to_pickle(
    r'D:\Google_Review\Parking\temp\bg_pois_parking_poi.pkl')

# How many CBG in each MSA
bg_msa = bg_pois[bg_pois['LSAD'] == 'M1'].reset_index(drop=True)
ct_cbg = bg_msa.groupby("NAMELSAD").count()[['BGFIPS']].sort_values(by='BGFIPS', ascending=False).reset_index()
bg_msa = bg_msa[bg_msa['NAMELSAD'].isin(ct_cbg.loc[ct_cbg['BGFIPS'] > 40, 'NAMELSAD'])].reset_index(drop=True)

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
    plt.savefig(r'D:\Google_Review\Parking\results\corr_poi_%s_new.png' % ep, dpi=1000)
    plt.close()

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
