import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import geopandas as gpd
import calendar
import glob
import os
import seaborn as sns

pd.options.mode.chained_assignment = None
plt.rcParams.update(
    {'font.size': 13, 'font.family': "serif", 'mathtext.fontset': 'dejavuserif', 'xtick.direction': 'in',
     'xtick.major.size': 0.5, 'grid.linestyle': "--", 'axes.grid': False, "grid.alpha": 1, "grid.color": "#cccccc",
     'xtick.minor.size': 1.5, 'xtick.minor.width': 0.5, 'xtick.minor.visible': True, 'xtick.top': True,
     'ytick.direction': 'in', 'ytick.major.size': 0.5, 'ytick.minor.size': 1.5, 'ytick.minor.width': 0.5,
     'ytick.minor.visible': True, 'ytick.right': True, 'axes.linewidth': 0.5, 'grid.linewidth': 0.5,
     'lines.linewidth': 1.5, 'legend.frameon': False, 'savefig.bbox': 'tight', 'savefig.pad_inches': 0.05})

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

# Read sentiment data
g_pois = pd.read_pickle(r'D:\Google_Review\Parking\temp\park_all_new.pkl')

# Average sentiment to CBG
# g_pois['naics_code'].value_counts()
g_pois['sum_comment'] = g_pois.groupby('BGFIPS')['total_parking_reviews'].transform("sum")
g_pois['weight'] = g_pois['total_parking_reviews'] / g_pois['sum_comment']
g_pois['weight_st'] = g_pois['weight'] * g_pois['avg_parking_sentiment']
bg_pois = g_pois.groupby('BGFIPS')['weight_st'].sum().reset_index()
bg_count = g_pois.groupby('BGFIPS')['total_parking_reviews'].sum().reset_index()
bg_pois = bg_pois.merge(bg_count, on='BGFIPS')
bg_pois = bg_pois[(bg_pois['total_parking_reviews'] > 10)].reset_index(drop=True)
bg_pois['BGFIPS'] = bg_pois['BGFIPS'].astype(str).apply(lambda x: x.zfill(12))
# bg_pois = bg_pois.fillna(0)

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
bg_pois[['BGFIPS', 'weight_st', 'total_parking_reviews', 'Total_Population', 'ALAND_x', 'Lng', 'Lat', 'CBSA', 'CSAFP',
         'CBSAFP', 'LSAD', 'geometry'] + need_scio].to_pickle(r'D:\Google_Review\Parking\temp\bg_pois_parking.pkl')

# How many CBG in each MSA
bg_msa = bg_pois[bg_pois['LSAD'] == 'M1'].reset_index(drop=True)
ct_cbg = bg_msa.groupby("NAMELSAD").count()[['BGFIPS']].sort_values(by='BGFIPS', ascending=False)

# Plot spatial map: total us sentiment
bg_pois_geo = poly.merge(bg_pois[['BGFIPS', 'weight_st', 'CBSA']], on='BGFIPS')
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 8))
bg_pois_geo.plot(column='weight_st', ax=ax, legend=True, scheme='natural_breaks', cmap='coolwarm', k=6,
                 legend_kwds=dict(frameon=False, ncol=3, loc='lower center', bbox_to_anchor=(0.5, 0)), linewidth=0,
                 edgecolor='white', alpha=0.8)
(MSA_geo[(MSA_geo['LSAD'] == 'M1') & (MSA_geo['CBSAFP'].isin(bg_pois['CBSA']))]
 .geometry.boundary.plot(color=None, edgecolor='k', linewidth=0.5, alpha=0.8, ax=ax))
ax.tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)
ax.axis('off')
plt.tight_layout()
plt.savefig(r'D:\Google_Review\Parking\results\spatial_sentiment_us_new.png', dpi=1000)

# Plot spatial map: the top 10 MSA
for kk in range(0, 10):
    msa_t = bg_pois[(bg_pois['NAMELSAD'] == ct_cbg.index[kk])]
    msa_t_geo = poly.merge(msa_t[['BGFIPS', 'weight_st', 'CBSA']], on='BGFIPS')
    # msa_t_geo.to_crs('EPSG:5070')
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(9, 6))
    msa_t_geo.plot(column='weight_st', ax=ax, legend=True, scheme='natural_breaks', cmap='coolwarm', k=6,
                   legend_kwds=dict(frameon=False, ncol=3, loc='lower center', bbox_to_anchor=(0.5, 0)), linewidth=0,
                   edgecolor='white', alpha=1)
    ax.tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(r'D:\Google_Review\Parking\results\spatial_sentiment_%s_new.png' % ct_cbg.index[kk], dpi=1000)
    plt.close()

# Plot boxplot: Avg sentiment by MSA
ranks = bg_msa.groupby("NAMELSAD")["weight_st"].mean().fillna(0).sort_values()[::-1].index
ranks01 = list(ranks[0:10]) + list(ranks[-10:])
fig, ax = plt.subplots(figsize=(8, 6))
sns.boxplot(data=bg_pois[bg_pois['NAMELSAD'].isin(ranks01)], y='NAMELSAD', x='weight_st', order=ranks01, ax=ax)
ax.xaxis.grid(True)
plt.title('All POIs')
plt.tight_layout()
plt.savefig(r'D:\Google_Review\Parking\results\sentiment_msa_new.png', dpi=1000)

'''
# Correlation by MSA: change the min count
for min_c in tqdm(range(0, 50, 2)):
    bg_pois = bg_pois[(bg_pois['total_parking_reviews'] > min_c)].reset_index(drop=True)
    all_corr = pd.DataFrame()
    for csa in set(bg_msa['NAMELSAD'].dropna()):
        temp = bg_pois[(bg_pois['NAMELSAD'] == csa)]
        temp['weight_st'] = temp['weight_st'].astype(float)
        corr_t = temp.corr(numeric_only=True)[['weight_st']]
        corr_t.columns = [csa]
        corr_t = corr_t.reset_index()
        if len(all_corr) == 0:
            all_corr = corr_t
        else:
            all_corr = corr_t.merge(all_corr, on='index', how='outer')
    all_corr = all_corr.set_index('index')

    des_cor = all_corr.T.describe()
    des_mean = des_cor.loc['mean',].sort_values(ascending=False).reset_index()
    des_mean.columns = ['index', min_c]
    if min_c == 0:
        des_all = des_mean
    else:
        des_all = des_mean.merge(des_all, on='index', how='outer')

des_all = pd.concat([des_all.head(5), des_all.tail(5)])
fig, ax = plt.subplots(figsize=(9, 6))
des_all.set_index('index').T.sort_index().plot(ax=ax, linewidth=2)
ax.plot([5, 5], [-0.35, 0.25], color='k', linestyle='--')
plt.legend(ncol=2, borderaxespad=0.)
plt.ylabel('Correlation')
plt.xlabel('Minimum # of reviews')
plt.tight_layout()
plt.savefig(r'D:\Google_Review\Parking\results\Threshold.png', dpi=1000)
'''

# Correlation by MSA
all_corr = pd.DataFrame()
for csa in set(bg_msa['NAMELSAD'].dropna()):
    temp = bg_pois[(bg_pois['NAMELSAD'] == csa)]
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
des_mean = des_mean[des_mean.index.isin(need_scio)]
des_mean_tt = pd.concat([des_mean.head(10), des_mean.tail(10)]).reset_index()
pty_corr = all_corr[all_corr.index.isin(des_mean_tt['index'])].T.reset_index(drop=True)
pty_corr = pd.melt(pty_corr, value_vars=pty_corr.columns)
fig, ax = plt.subplots(figsize=(8, 6))
sns.boxplot(data=pty_corr, x='value', y='index', order=list(des_mean_tt['index']), ax=ax)
ax.xaxis.grid(True)
plt.title('All POIs')
plt.xlabel('Correlation')
plt.tight_layout()
plt.savefig(r'D:\Google_Review\Parking\results\corr_poi_all_new.png', dpi=1000)

# # Plot correlation by MSA
# for kk in ['Rural_Population_R', 'Population_Density', 'White_Non_Hispanic_R', "Median_income"]:
#     ranks = all_corr.loc[kk,].fillna(0).sort_values(ascending=False)
#     ranks = pd.concat([ranks.head(10), ranks.tail(10)]).reset_index()
#     fig, ax = plt.subplots(figsize=(8, 6))
#     sns.barplot(data=ranks, x=kk, y='index', ax=ax)
#     ax.xaxis.grid(True)
#     plt.title('All POIs')
#     plt.tight_layout()
#     plt.savefig(r'D:\Google_Review\Parking\results\corr_msa_%s.png' % kk, dpi=1000)
#     plt.close()
