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
# CSAFP: 176; CBSAFP: 938; MSA: 392, M2SA: 546
# print(len(set(MSA_geo.loc[MSA_geo['LSAD']=='M2','CBSAFP'])))

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
smart_loc['BGFIPS'] = smart_loc['BGFIPS'].astype(str).apply(lambda x: x.zfill(12))
msa_population = smart_loc[['CBSA', 'CBSA_POP', 'CBSA_Name']].drop_duplicates(subset=['CBSA']).reset_index(drop=True)
poi_ty = pd.read_excel(r'D:\Google_Review\Parking\temp\places_summary_hsh.xlsx')
poi_ty = poi_ty[['naics_code', 'Categories']]

# Read sentiment data
g_pois = pd.read_pickle(r'D:\Google_Review\Parking\temp\park_all_new.pkl')
g_pois['BGFIPS'] = g_pois['BGFIPS'].astype(str).apply(lambda x: x.zfill(12))
# Merge with CSA code
g_pois = g_pois.merge(smart_loc[['BGFIPS', 'CBSA']], on='BGFIPS')
# print(len(set(g_pois['CBSA'])))
# Only contiguous US
g_pois = g_pois[g_pois['BGFIPS'].isin(poly['BGFIPS'])].reset_index(drop=True)

# Merge with POI type
g_pois = g_pois.merge(poi_ty, on='naics_code')
g_pois['Categories'] = g_pois['Categories'].fillna('Others')
g_pois.loc[(g_pois['category'].str.contains('|'.join(['parking lot', 'parking garage']), case=False)) & (
        g_pois['Categories'] == 'Personal Service'), 'Categories'] = 'Parking'
g_pois['Categories'].value_counts()
g_pois['total_parking_reviews'] = g_pois['total_parking_reviews'].astype(int)
g_pois['avg_parking_sentiment'] = g_pois['avg_parking_sentiment'].astype(float)
g_pois['sum_comment'] = g_pois.groupby(['CBSA', 'Categories'])['total_parking_reviews'].transform("sum")
g_pois['weight_st'] = (g_pois['total_parking_reviews'] / g_pois['sum_comment']) * g_pois['avg_parking_sentiment']

# Average sentiment to MSA
bg_pois = g_pois.groupby(['CBSA', 'Categories'])['weight_st'].sum().reset_index()
bg_count = g_pois.groupby(['CBSA', 'Categories'])['total_parking_reviews'].sum().reset_index()
bg_pois = bg_pois.merge(bg_count, on=['CBSA', 'Categories'])

# Build some new MSA features based on GMAP POIs
gall_pois = pd.read_pickle(r'D:\Google_Review\\Urban_sen\data\\us_gm_poi.pkl')
gall_pois['BGFIPS'] = gall_pois['BGFIPS'].astype(str).apply(lambda x: x.zfill(12))
gall_pois = gall_pois.merge(smart_loc[['BGFIPS', 'CBSA']], on='BGFIPS')

# Total number of reviews: all pois
g_fe1 = gall_pois.groupby(['CBSA'])['num_of_reviews'].sum().reset_index()
# Total number of all POIs
g_fe2 = g_pois.groupby(['CBSA'])['gmap_id'].count().reset_index()
g_fe2.columns = ['CBSA', 'total_poi_count']
# Avg rating score of all POIs
gall_pois['sum_comment_all'] = gall_pois.groupby(['CBSA'])['num_of_reviews'].transform("sum")
gall_pois['weight_st_all'] = (gall_pois['num_of_reviews'] / gall_pois['sum_comment_all']) * gall_pois['avg_rating']
g_fe3 = gall_pois.groupby(['CBSA'])['weight_st_all'].sum().reset_index()
g_fe3.columns = ['CBSA', 'avg_poi_score']

# Parking related POIs
parkings = gall_pois[
    gall_pois['category'].astype(str).str.contains('|'.join(['parking lot', 'parking garage']), case=False)]
g_fe4 = parkings.groupby(['CBSA'])['num_of_reviews'].sum().reset_index()
g_fe4.columns = ['CBSA', 'num_of_parking_review']
g_fe5 = parkings.groupby(['CBSA'])['gmap_id'].count().reset_index()
g_fe5.columns = ['CBSA', 'total_parking_count']
# Avg parking score
parkings['sum_comment_all'] = parkings.groupby(['CBSA'])['num_of_reviews'].transform("sum")
parkings['weight_st_all'] = (parkings['num_of_reviews'] / parkings['sum_comment_all']) * parkings['avg_rating']
g_fe6 = parkings.groupby(['CBSA'])['weight_st_all'].sum().reset_index()
g_fe6.columns = ['CBSA', 'avg_parking_score']
# Avg parking sentiment
g_fe7 = bg_pois.loc[
    (bg_pois['Categories'] == 'Parking') & (bg_pois['total_parking_reviews'] > 10), ['CBSA', 'weight_st']]
g_fe7.columns = ['CBSA', 'avg_parking_st']
g_fe = reduce(lambda left, right: pd.merge(left, right, on='CBSA', how='outer'),
              [g_fe1, g_fe2, g_fe3, g_fe4, g_fe5, g_fe6, g_fe7])
g_fe = g_fe.fillna(0)

# CBG SOCIO-DEMO features
CT_Features = CT_Features.merge(
    smart_loc[['BGFIPS', 'D3B', 'Pct_AO0', 'Pct_AO1', 'Pct_AO2p', 'NatWalkInd', 'D3A', 'D1C', 'D4C', 'CBSA', 'CBSA_POP',
               'Ac_Total', 'CSA', 'D4A', 'D4D', 'D4E']], on='BGFIPS')
# Rename
CT_Features = CT_Features.rename(
    columns={'D3B': 'Intersection_Density', 'Pct_AO0': 'Zero_car_R', 'Pct_AO1': 'One_car_R',
             'Pct_AO2p': 'Two_plus_car_R', 'NatWalkInd': 'Walkability', 'D3A': 'Road_Density',
             'D1C': 'Employment_Density', 'D4C': 'Transit_Freq', 'D4A': 'Distance_Transit', 'D4D': 'Transit_Freq_Area',
             'D4E': 'Transit_Freq_Pop'})
# Remove outliers
CT_Features.loc[CT_Features['Transit_Freq'] < 0, 'Transit_Freq'] = 0
CT_Features.loc[CT_Features['Transit_Freq_Area'] < 0, 'Transit_Freq_Area'] = 0
CT_Features.loc[CT_Features['Transit_Freq_Pop'] < 0, 'Transit_Freq_Pop'] = 0
CT_Features.loc[CT_Features['Distance_Transit'] < 0, 'Distance_Transit'] = 1207.008

# Read household count
B01 = pd.read_csv(
    r'G:\Data\SafeGraph\Open Census Data\Census Website\2019\nhgis0010_csv\nhgis0010_ds244_20195_2019_blck_grp.csv',
    encoding='latin-1', skiprows=[1])
B01['BGFIPS'] = B01['GISJOIN'].str[1:3] + B01['GISJOIN'].str[4:7] + B01['GISJOIN'].str[8:14] + B01['GISJOIN'].str[14:15]
B01 = B01[['BGFIPS', 'ALW0E001']]
B01.columns = ['BGFIPS', 'Total_Household']
CT_Features = CT_Features.merge(B01, on='BGFIPS')

# Change CBG to CSA-level
# Area-based: First return back to area and then recalculate
cct = 0
for kk in ['Population_Density', 'Intersection_Density', 'Employment_Density', 'Road_Density']:
    CT_Features['total'] = CT_Features[kk] * CT_Features['ALAND']
    CSA_Feature = CT_Features.groupby('CBSA')[['ALAND', 'total']].sum().reset_index()
    CSA_Feature[kk] = CSA_Feature['total'] / CSA_Feature['ALAND']
    CSA_Feature = CSA_Feature[['CBSA', kk]]
    if cct == 0:
        CSA_Features = CSA_Feature
    else:
        CSA_Features = CSA_Features.merge(CSA_Feature, on='CBSA', how='outer')
    cct += 1

# Population-based
for kk in ['Bt_18_44_R', 'Asian_R', 'Over_65_R', 'Republican_R', 'Bt_45_64_R', 'Urbanized_Areas_Population_R',
           'Indian_R', 'Rural_Population_R', 'HISPANIC_LATINO_R', 'Black_Non_Hispanic_R', 'White_Non_Hispanic_R',
           'Male_R', 'Urban_Clusters_Population_R', 'Education_Degree_R', 'Democrat_R']:
    CT_Features['total'] = CT_Features[kk] * CT_Features['Total_Population']
    CSA_Feature = CT_Features.groupby('CBSA')[['Total_Population', 'total']].sum().reset_index()
    CSA_Feature[kk] = CSA_Feature['total'] / CSA_Feature['Total_Population']
    CSA_Feature = CSA_Feature[['CBSA', kk]]
    CSA_Features = CSA_Features.merge(CSA_Feature, on='CBSA', how='outer')

# Household-based
for kk in ['Household_Below_Poverty_R', 'Two_plus_car_R', 'Zero_car_R', 'One_car_R', 'Median_income']:
    CT_Features['total'] = CT_Features[kk] * CT_Features['Total_Household']
    CSA_Feature = CT_Features.groupby('CBSA')[['Total_Household', 'total']].sum().reset_index()
    CSA_Feature[kk] = CSA_Feature['total'] / CSA_Feature['Total_Household']
    CSA_Feature = CSA_Feature[['CBSA', kk]]
    CSA_Features = CSA_Features.merge(CSA_Feature, on='CBSA', how='outer')

# Sum-based
CSA_Feature = CT_Features.groupby('CBSA')[['Total_Household', 'Total_Population', 'ALAND']].sum().reset_index()
CSA_Features = CSA_Features.merge(CSA_Feature, on='CBSA', how='outer')
# Mean-based
CSA_Feature = CT_Features.groupby('CBSA')[['Walkability', 'Transit_Freq']].mean().reset_index()
CSA_Features = CSA_Features.merge(CSA_Feature, on='CBSA', how='outer')

# Merge with review features
bg_pois = bg_pois.merge(g_fe, on='CBSA').reset_index(drop=True)
# Merge with CBSA features
bg_pois = bg_pois.merge(CSA_Features, on='CBSA').reset_index(drop=True)
MSA_geo.rename({'CBSAFP': 'CBSA'}, axis=1, inplace=True)
bg_pois = bg_pois.merge(MSA_geo, on='CBSA')
# Only greater than 10
bg_pois = bg_pois[(bg_pois['total_parking_reviews'] > 10)].reset_index(drop=True)
sns.displot(bg_pois['weight_st'])  # .loc[bg_pois['Categories'] == 'Hotel', 'weight_st']

# Polish data
bg_pois['Parking_review_density'] = (bg_pois['total_parking_reviews'] + bg_pois['num_of_parking_review']) / bg_pois[
    'ALAND']
bg_pois['Total_review_density'] = bg_pois['num_of_reviews'] / bg_pois['ALAND']
bg_pois['Parking_poi_density'] = bg_pois['total_parking_count'] / bg_pois['ALAND']
bg_pois['Total_poi_density'] = bg_pois['total_poi_count'] / bg_pois['ALAND']
