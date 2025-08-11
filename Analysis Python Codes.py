#!/usr/bin/env python
# coding: utf-8

# In[397]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
from shapely.geometry import Point
get_ipython().system('conda install -c conda-forge contextily -y')
import contextily as ctx
import ee
ee.Initialize()


# In[6]:


df_geo = pd.read_csv("s00_me_bfa2021.csv")


# In[7]:


df_geo


# In[8]:


# Rename geographic and metadata columns to clear English descriptors
# This improves readability and consistency across merged datasets
df_geo.rename(columns={
    'hhid': 'hhid',
    'grappe': 'cluster',
    'menage': 'household_number',
    'vague': 'wave',
    'hhweight': 'household_weight',
    's00q00': 'country',
    's00q01': 'region',
    's00q02': 'prefecture_arrondissement',
    's00q03': 'commune',
    's00q04': 'residence_type',
    's00q05': 'village_quartier',
    's00q07a': 'household_type',
    's00q07b': 'recent_mover',
    's00q07c': 'migration_reason',
    's00q07d': 'interviewed_2018_2019',
    's00q07d2': 'interviewed_2018_other',
    's00q22': 'interview_visits_required',
    's00q23a': 'interview_date_1_start',
    's00q24a': 'interview_date_2_start',
    's00q25a': 'interview_date_3_start',
    's00q23b': 'interview_date_1_end',
    's00q24b': 'interview_date_2_end',
    's00q25b': 'interview_date_3_end',
    's00q08': 'interview_result',
    's00q27': 'questionnaire_result',
    's00q28': 'interview_language',
    'GPS__Latitude': 'gps_latitude',
    'GPS__Longitude': 'gps_longitude'
}, inplace=True)


# In[9]:


df_geo


# In[10]:


# Basic cleanup: ensure identifiers are stripped and lowercase where appropriate
# Prepare GPS and string columns for merging or spatial processing
df_geo['region'] = df_geo['region'].astype(str).str.strip().str.lower()
df_geo['commune'] = df_geo['commune'].astype(str).str.strip().str.lower()
df_geo['prefecture_arrondissement'] = df_geo['prefecture_arrondissement'].astype(str).str.strip().str.lower()
df_geo['village_quartier'] = df_geo['village_quartier'].astype(str).str.strip().str.lower()
df_geo['residence_type'] = df_geo['residence_type'].astype(str).str.strip().str.lower()

# Clean GPS coordinates: coerce invalid GPS entries to NaN and round to 6 decimals
df_geo['gps_latitude'] = pd.to_numeric(df_geo['gps_latitude'], errors='coerce').round(6)
df_geo['gps_longitude'] = pd.to_numeric(df_geo['gps_longitude'], errors='coerce').round(6)


# In[11]:


# Parse interview start and end dates into standardized datetime format
# This allows for later analysis like interview timing, visit duration, etc.

date_cols = [
    'interview_date_1_start', 'interview_date_2_start', 'interview_date_3_start',
    'interview_date_1_end', 'interview_date_2_end', 'interview_date_3_end'
]

for col in date_cols:
    df_geo[col] = pd.to_datetime(df_geo[col], errors='coerce')
    
    
    
    
    
    # Compute survey duration using available interview start/end dates
# We take the earliest start and latest end to get duration per household

df_geo['survey_duration_days'] = (
    df_geo[
        ['interview_date_1_start', 'interview_date_2_start', 'interview_date_3_start',
         'interview_date_1_end', 'interview_date_2_end', 'interview_date_3_end']
    ].max(axis=1)
    - df_geo[
        ['interview_date_1_start', 'interview_date_2_start', 'interview_date_3_start',
         'interview_date_1_end', 'interview_date_2_end', 'interview_date_3_end']
    ].min(axis=1)
).dt.days



# In[12]:


# Clean and validate GPS coordinates
# Remove extreme or missing GPS values
df_geo['gps_latitude'] = pd.to_numeric(df_geo['gps_latitude'], errors='coerce')
df_geo['gps_longitude'] = pd.to_numeric(df_geo['gps_longitude'], errors='coerce')

# Flag invalid or missing GPS coordinates (for diagnostics)
df_geo['gps_valid'] = df_geo[['gps_latitude', 'gps_longitude']].notnull().all(axis=1)

# Remove entries with clearly invalid coordinates (e.g., 0,0 or outside Burkina bounds)
df_geo.loc[
    (df_geo['gps_latitude'] < 9.5) | (df_geo['gps_latitude'] > 15) |
    (df_geo['gps_longitude'] < -6) | (df_geo['gps_longitude'] > 3),
    'gps_valid'
] = False


# In[13]:


# Create geometry column using cleaned GPS coordinates
df_geo['geometry'] = df_geo.apply(
    lambda row: Point(row['gps_longitude'], row['gps_latitude']) if row['gps_valid'] else None,
    axis=1
)

# Convert to GeoDataFrame (for spatial visualization)
gdf_geo = gpd.GeoDataFrame(df_geo, geometry='geometry', crs='EPSG:4326')

# Filter only valid GPS rows for mapping
gdf_valid = gdf_geo[df_geo['gps_valid']]

# Plot the map
plt.figure(figsize=(10, 8))
gdf_valid.plot(markersize=2, color='darkblue', alpha=0.7)
plt.title("Household GPS Locations (Burkina Faso, EHCVM 2021)", fontsize=14)
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.grid(True)
plt.show()


# In[14]:


df_geo.columns


# In[16]:


df_geo.to_csv("bfa_geospatial_data_cleaned.csv")


# In[15]:


shock_data = pd.read_csv("bfa_shock_raw_renamed_cleaned.csv")
shock_data


# In[16]:


shock_data.columns


# In[17]:


import unicodedata

def normalize_string(x):
    if pd.isnull(x): return ""
    x = str(x).lower().strip()
    return unicodedata.normalize('NFKD', x).encode('ascii', 'ignore').decode('utf-8')

shock_data['shock_type_clean'] = shock_data['shock_type'].apply(normalize_string)


# In[18]:


shock_type_map = {
    # Economic shocks
    'prix eleves des produits alimentaires': 'economic',
    'prix eleves des intrants agricoles': 'economic',
    'baisse importante des prix des produits agricoles': 'economic',
    "perte d'emploi salarie d'un membre": 'economic',
    "perte importante de revenus salariaux  (autre que du fait d'un accident ou d'une maladie)": 'economic',
    "perte importante du revenu non agricole du menage  (autre que du fait d'un accident ou d'une maladie)": 'economic',
    "faillite d'une entreprise non agricole du menage": 'economic',
    "fin de transferts reguliers provenant d'autres menages": 'economic',

    # Social shocks
    "vol d'argent, de biens, de recolte ou de betail": 'social',
    "maladie grave ou accident d'un membre du menage": 'social',
    "deces d'un membre du menage": 'social',
    'divorce, separation': 'social',
    'conflit agriculteur/eleveur': 'social',
    'conflit arme/violence/insecurite': 'social',

    # Environmental shocks
    'secheresse/pluies irregulieres': 'environmental',
    'inondations': 'environmental',
    'glissement de terrain': 'environmental',
    'attaques acridiennes ou autres ravageurs de recolte': 'environmental',
    'taux eleve de maladies des cultures': 'environmental',
    'taux eleve de maladies des animaux': 'environmental',

    # Reassign 'autre' to social
    'autre (a preciser)': 'social'
}


# In[19]:


shock_data['shock_group'] = shock_data['shock_type_clean'].map(shock_type_map)


# In[20]:


# Group shocks by household and count each shock type
shock_grouped = shock_data.groupby('hhid').agg(
    shock_economic=('shock_group', lambda x: (x == 'economic').sum()),
    shock_social=('shock_group', lambda x: (x == 'social').sum()),
    shock_environmental=('shock_group', lambda x: (x == 'environmental').sum())
).reset_index()

# Compute total and mean shocks per household
shock_grouped['shock_total'] = shock_grouped[['shock_economic', 'shock_social', 'shock_environmental']].sum(axis=1)
shock_grouped['shock_mean'] = shock_grouped[['shock_economic', 'shock_social', 'shock_environmental']].mean(axis=1)


# In[21]:


# Merge household-level shock counts into the original data
shock_data = shock_data.merge(shock_grouped, on='hhid', how='left')


# In[22]:


# Convert 'oui'/'non' to binary 1/0
shock_data['shock_occurred_last3y'] = shock_data['shock_occurred_last3y'].str.lower().map({'oui': 1, 'non': 0})


# In[23]:


# Identify all columns
cols = list(shock_data.columns)

# Define target shock columns
shock_summary_cols = [
    'shock_economic', 'shock_social', 'shock_environmental',
    'shock_mean', 'shock_total'
]

# Remove those from current list to reposition
for col in shock_summary_cols:
    cols.remove(col)

# Insert after 'shock_occurred_last3y'
insert_pos = cols.index('shock_occurred_last3y') + 1
for i, col in enumerate(shock_summary_cols):
    cols.insert(insert_pos + i, col)

# Reorder the DataFrame
shock_data = shock_data[cols]


# In[24]:


# Build dictionary for aggregation
agg_dict = {}

for col in shock_data.columns:
    if col == 'hhid':
        continue
    elif col == 'shock_mean':
        agg_dict[col] = 'mean'  # compute average shock per hh
    elif col in ['shock_total', 'shock_economic', 'shock_social', 'shock_environmental', 'shock_occurred_last3y']:
        agg_dict[col] = 'max'  # take maximum (e.g., if any member reported shock)
    elif col.startswith('coping_'):
        agg_dict[col] = 'max'  # 1 if any household member used that strategy
    else:
        agg_dict[col] = 'first'  # keep the first valid value (e.g., for identifiers or descriptions)


# In[25]:


# Identify numeric and non-numeric columns
numeric_cols = shock_data.select_dtypes(include=['number']).columns.tolist()
non_numeric_cols = shock_data.select_dtypes(exclude=['number']).columns.tolist()

# Initialize aggregation dictionary
agg_dict = {}

for col in shock_data.columns:
    if col == 'hhid':
        continue
    elif col in numeric_cols:
        if col.startswith('coping_') or col in ['shock_mean', 'shock_total', 'shock_economic', 'shock_social', 'shock_environmental', 'shock_occurred_last3y']:
            agg_dict[col] = 'max'  # numeric indicators: keep 1 if any member reported
        else:
            agg_dict[col] = 'first'  # keep first numeric value if not a shock indicator
    else:
        agg_dict[col] = 'first'  # keep string values like 'shock_type', 'shock_group', etc.


# In[26]:


# Aggregate to household level with corrected aggregation
shock_hh_all_cols = shock_data.groupby('hhid').agg(agg_dict).reset_index()


# In[27]:


# Extract the fixed total shock columns
shock_totals = ['shock_economic', 'shock_social', 'shock_environmental', 'shock_mean', 'shock_total']


# In[28]:


shock_totals 


# In[29]:


# List of columns to take first (excluding totals and hhid)
cols_to_first = [col for col in shock_data.columns if col not in shock_totals]
shock_base = shock_data[cols_to_first].drop_duplicates('hhid').copy()


# In[30]:


shock_counts = shock_data[['hhid'] + shock_totals].drop_duplicates('hhid').copy()
shock_counts


# In[31]:


shock_base


# In[32]:


shock_hh_all_cols


# In[33]:


shock_data[['shock_economic', 'shock_social', 'shock_environmental', 'shock_total', 'shock_mean']].describe()


# In[ ]:





# In[ ]:





# In[34]:


shock_data.columns


# In[35]:


# List values that were not successfully mapped
unmatched_shocks = shock_data.loc[shock_data['shock_group'].isna(), 'shock_type_clean'].unique()
print(unmatched_shocks)


# In[36]:


# Check value counts in the 'shock_group' column
shock_data['shock_group'].value_counts(dropna=False)


# In[44]:


# Ensure dummy-like cols are numeric 0/1
for col in shock_data.columns:
    if col.startswith('coping_') or col.startswith('shock_type_'):
        shock_data[col] = pd.to_numeric(shock_data[col], errors='coerce').fillna(0).astype(int)

# Aggregation dictionary
agg_dict = {
    'shock_respondent': 'first',
    'shock_occurred_last3y': 'sum',   # total number of shocks occurred
    'shock_environmental': 'sum',
    'shock_social': 'sum',
    'shock_economic': 'sum',
    'shock_year': 'first',
    'shock_effect_income': ['sum', 'mean'],
    'shock_effect_assets': ['sum', 'mean'],
    'shock_effect_agri_prod': ['sum', 'mean'],
    'shock_effect_livestock': ['sum', 'mean'],
    'shock_effect_food_stock': ['sum', 'mean'],
    'shock_effect_food_purchase': ['sum', 'mean'],
}

# Include all coping strategies as totals
for col in shock_data.columns:
    if col.startswith('coping_') and '_specify' not in col:
        agg_dict[col] = 'sum'

# Include each shock type dummy column as totals
for col in shock_data.columns:
    if col.startswith('shock_type_'):
        agg_dict[col] = 'sum'

# Aggregate individual-level data to household level
df_agg = shock_data.groupby(group_keys).agg(agg_dict)
df_agg.columns = ['_'.join(col) if isinstance(col, tuple) else col for col in df_agg.columns]
df_agg.reset_index(inplace=True)


# In[42]:


shock_data.columns


# In[47]:


# Collapse shock months and coping strategy details
shock_months = shock_data.groupby(group_keys)['shock_month'].apply(
    lambda x: ', '.join(x.dropna().astype(str).unique())
).reset_index()

coping_other = shock_data.groupby(group_keys)['coping_other_specifies'].apply(
    lambda x: ', '.join(x.dropna().astype(str).unique())
).reset_index()

df_agg = df_agg.merge(shock_months, on=group_keys, how='left')
df_agg = df_agg.merge(coping_other, on=group_keys, how='left')


# In[27]:


# Add country column with a fixed value
df_household['country'] = 'burkina faso'

# Define new column order
first_cols = ['year', 'country']
id_cols = ['hhid', 'cluster', 'household_code']
remaining_cols = [col for col in df_household.columns if col not in first_cols + id_cols]

# Reorder columns: year, country, IDs, then all others
df_household = df_household[first_cols + id_cols + remaining_cols]


# In[35]:


df_household.to_csv("bfa_shock_household_level_cleaned_data_with_experienced_composite_shocks.csv")


# In[30]:


df_household


# In[33]:


df_household.rename(columns={'household_code': 'household_number'}, inplace=True)
df_household


# In[34]:


df_household.columns


# In[36]:


# Merge the cleaned shock data with the geospatial data using household identifiers
merge_keys = ['hhid', 'cluster', 'household_number']

df_merged = df_geo.merge(df_household, on=merge_keys, how='left')

# Optional: Confirm merge success
print("Final merged dataset shape:", df_merged.shape)
print("Missing shock values after merge (should be 0):")
print(df_merged[['shock_occurred_last3y_max']].isnull().sum())


# In[37]:


df_merged


# In[40]:


# Rename 'country_x' to 'country'
if 'country_x' in df_merged.columns:
    df_merged.rename(columns={'country_x': 'country'}, inplace=True)

df_merged['year'] = 2021

# Reorder columns: year, country, identifiers, then the rest
first_cols = ['year', 'country']
id_cols = ['hhid', 'cluster', 'household_number']
other_cols = [col for col in df_merged.columns if col not in first_cols + id_cols]

df_merged = df_merged[first_cols + id_cols + other_cols]


# In[42]:


df_merged.to_csv("bfa_merged_geospatial and shock cleand_data.csv")


# In[54]:


import geopandas as gpd
import matplotlib.pyplot as plt
import contextily as ctx

# convert the merged DataFrame into a GeoDataFrame for spatial plotting
gdf = gpd.GeoDataFrame(df_merged, geometry='geometry')


gdf = gdf.set_crs(epsg=4326)

# reproject the data to Web Mercator (EPSG:3857) so I can add a basemap using contextily
gdf = gdf.to_crs(epsg=3857)

#  plot composite shock intensity at the household level on a map
fig, ax = plt.subplots(figsize=(12, 10))
gdf.plot(
    ax=ax,
    column='total_experienced_composite_shocks',
    cmap='Reds',
    legend=True,
    markersize=25,
    alpha=0.8
)

# add the OpenStreetMap basemap to provide geographic context
ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik)
ax.set_title("Hotspots of Composite Shocks in Burkina Faso (Household Level)", fontsize=14)
ax.set_axis_off()


plt.show()


# In[55]:


# visualize the spatial distribution of total experienced composite shocks.
# first convert the merged dataset to a GeoDataFrame, ensure it uses the correct CRS (WGS84),
# then reproject it to Web Mercator to enable adding a basemap. Finally, I plot the household-level
# shock intensities using a red gradient and overlay an OpenStreetMap basemap for context.


gdf = gpd.GeoDataFrame(df_merged, geometry='geometry')
gdf = gdf.set_crs(epsg=4326)
gdf = gdf.to_crs(epsg=3857)

fig, ax = plt.subplots(figsize=(12, 10))
gdf.plot(
    ax=ax,
    column='total_experienced_composite_shocks',
    cmap='Reds',
    legend=True,
    markersize=25,
    alpha=0.8
)

ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik)
ax.set_title("Hotspots of Composite Shocks in Burkina Faso (Household Level)", fontsize=14)
ax.set_axis_off()
plt.show()


# In[56]:


# generates an enhanced household-level map of composite shocks,
# Convert to GeoDataFrame and project for tile compatibility
gdf = gpd.GeoDataFrame(df_merged, geometry='geometry')
gdf = gdf.set_crs(epsg=4326).to_crs(epsg=3857)

# Create quantile bins for clearer legend mapping
gdf['shock_bin'] = pd.qcut(gdf['total_experienced_composite_shocks'], q=4, labels=['Low', 'Moderate', 'High', 'Extreme'])

# Map color palette
color_map = {
    'Low': '#ffffb2',
    'Moderate': '#fecc5c',
    'High': '#fd8d3c',
    'Extreme': '#e31a1c'
}

gdf['color'] = gdf['shock_bin'].map(color_map)

# Plot
fig, ax = plt.subplots(figsize=(13, 11))
gdf.plot(ax=ax, color=gdf['color'], alpha=0.8, markersize=20)

# Add basemap
ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik)

legend_patches = [mpatches.Patch(color=color_map[b], label=b) for b in color_map]
ax.legend(handles=legend_patches, title='Shock Intensity')


ax.set_title('Burkina Faso: Composite Shock Hotspots (Household Level)', fontsize=15)
ax.set_axis_off()

plt.tight_layout()
plt.show()


# In[57]:


# visualizes composite household shocks across Burkina Faso with full country extent.
# I project coordinates, bin shock intensity, and add a contextual basemap, ensuring full spatial coverage.


# Project to Web Mercator
gdf = gpd.GeoDataFrame(df_merged, geometry='geometry')
gdf = gdf.set_crs(epsg=4326).to_crs(epsg=3857)

# Create bins for shock levels
gdf['shock_bin'] = pd.qcut(gdf['total_experienced_composite_shocks'], q=4, labels=['Low', 'Moderate', 'High', 'Extreme'])

# Color mapping
color_map = {
    'Low': '#ffffb2',
    'Moderate': '#fecc5c',
    'High': '#fd8d3c',
    'Extreme': '#e31a1c'
}
gdf['color'] = gdf['shock_bin'].map(color_map)

# Plot with full extent
fig, ax = plt.subplots(figsize=(13, 11))
gdf.plot(ax=ax, color=gdf['color'], alpha=0.8, markersize=20)

# Set full spatial extent
xmin, ymin, xmax, ymax = gdf.total_bounds
ax.set_xlim(xmin - 10000, xmax + 10000)
ax.set_ylim(ymin - 10000, ymax + 10000)

# Add base map
ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik)

# Add legend
legend_patches = [mpatches.Patch(color=color_map[b], label=b) for b in color_map]
ax.legend(handles=legend_patches, title='Shock Intensity')

# Title and styling
ax.set_title('Burkina Faso: Composite Shock Hotspots (Household Level)', fontsize=15)
ax.set_axis_off()
plt.tight_layout()
plt.show()


# In[59]:


conda install -c conda-forge geodatasets


# In[ ]:


import geodatasets
import geopandas as gpd

# Load world boundaries from geodatasets
world = gpd.read_file(geodatasets.get_path('naturalearth.land'))
bfa = world[world['NAME'] == 'Burkina Faso'].to_crs(epsg=3857)


# In[ ]:





# In[55]:


# visualize household-level composite shocks across Burkina Faso using the official GADM boundary overlay and a contextual basemap.
# Load the Burkina Faso admin1 shapefile (assuming .shp, .shx, and .dbf are all present in the folder)

shapefile_path = "/Users/user/Desktop/OSAKA UNIVERSITY/FINDEX_DATA/LSMS/WHOLE_LSMS_CLEANING/UNICEF/gadm41_BFA_shp/gadm41_BFA_1.shp"

bfa_admin1 = gpd.read_file(shapefile_path).to_crs(epsg=3857)


gdf = gpd.GeoDataFrame(df_merged, geometry='geometry')
gdf = gdf.set_crs(epsg=4326).to_crs(epsg=3857)

fig, ax = plt.subplots(figsize=(14, 12))
bfa_admin1.boundary.plot(ax=ax, edgecolor='black', linewidth=1)
gdf.plot(ax=ax, column='total_experienced_composite_shocks', cmap='Reds', legend=True, markersize=25, alpha=0.7)
ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik)
ax.set_title("Hotspots of Composite Shocks in Burkina Faso (Household Level)", fontsize=14)
ax.set_axis_off()
plt.tight_layout()
plt.show()


# In[65]:


# Create bins for shock intensity

gdf['shock_category'] = pd.cut(
    gdf['total_experienced_composite_shocks'],
    bins=[-1, 2, 5, 9, gdf['total_experienced_composite_shocks'].max()],
    labels=["Low", "Moderate", "High", "Extreme"]
)

# Define color mapping
category_colors = {
    "Low": "#ffeda0",
    "Moderate": "#feb24c",
    "High": "#f03b20",
    "Extreme": "#bd0026"
}
gdf['color'] = gdf['shock_category'].map(category_colors)

# Plot setup
fig, ax = plt.subplots(figsize=(12, 10))
bfa_admin1.boundary.plot(ax=ax, color="black", linewidth=1)
gdf.plot(ax=ax, color=gdf['color'], markersize=30, alpha=0.85)

ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik)
ax.set_title("Burkina Faso: Composite Shock Hotspots (Household Level)", fontsize=15)
ax.set_axis_off()

# Custom Legend
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker='o', color='w', label='Low', markerfacecolor=category_colors["Low"], markersize=10),
    Line2D([0], [0], marker='o', color='w', label='Moderate', markerfacecolor=category_colors["Moderate"], markersize=10),
    Line2D([0], [0], marker='o', color='w', label='High', markerfacecolor=category_colors["High"], markersize=10),
    Line2D([0], [0], marker='o', color='w', label='Extreme', markerfacecolor=category_colors["Extreme"], markersize=10)
]
ax.legend(handles=legend_elements, loc='upper right', title='Shock Intensity')

plt.tight_layout()
plt.show()


# In[66]:


# Visualize household-level shock intensity with full Burkina Faso map extent


shapefile_path = "/Users/user/Desktop/OSAKA UNIVERSITY/FINDEX_DATA/LSMS/WHOLE_LSMS_CLEANING/UNICEF/gadm41_BFA_shp/gadm41_BFA_1.shp"
bfa_admin1 = gpd.read_file(shapefile_path).to_crs(epsg=3857)

gdf = gpd.GeoDataFrame(df_merged, geometry='geometry')
gdf = gdf.set_crs(epsg=4326).to_crs(epsg=3857)

bins = [0, 2, 5, 8, 100]
labels = ['Low', 'Moderate', 'High', 'Extreme']
colors = ['#fff5c3', '#fdbf6f', '#ef6548', '#990000']
color_map = dict(zip(labels, colors))

gdf['shock_bin'] = pd.cut(gdf['total_experienced_composite_shocks'], bins=bins, labels=labels, include_lowest=True)
gdf['color'] = gdf['shock_bin'].map(color_map)

fig, ax = plt.subplots(figsize=(14, 11))
bfa_admin1.boundary.plot(ax=ax, edgecolor='black', linewidth=1)
gdf.plot(ax=ax, color=gdf['color'], markersize=20, alpha=0.85)

legend_handles = [plt.Line2D([0], [0], marker='o', color='w', label=label,
                    markerfacecolor=color_map[label], markersize=8)
                    for label in labels]
ax.legend(handles=legend_handles, title="Shock Intensity", loc='upper right')

ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik)

# Manually set full country bounds
xmin, ymin, xmax, ymax = bfa_admin1.total_bounds
ax.set_xlim(xmin, xmax)
ax.set_ylim(ymin, ymax)

ax.set_title("Burkina Faso: Composite Shock Hotspots (Household Level)", fontsize=15)
ax.axis('off')
plt.tight_layout()
plt.show()


# In[67]:


# Plot household-level composite shock hotspots across Burkina Faso using administrative boundaries and buffer-adjusted basemap extent

import geopandas as gpd
import matplotlib.pyplot as plt
import contextily as ctx

# Load shapefile for Burkina Faso's first-level administrative boundaries (GADM)
shapefile_path = "/Users/user/Desktop/OSAKA UNIVERSITY/FINDEX_DATA/LSMS/WHOLE_LSMS_CLEANING/UNICEF/gadm41_BFA_shp/gadm41_BFA_1.shp"
bfa_admin1 = gpd.read_file(shapefile_path).to_crs(epsg=3857)

# Prepare household-level merged data with shocks (assumed already merged and cleaned)
gdf = gpd.GeoDataFrame(df_merged, geometry='geometry')
gdf = gdf.set_crs(epsg=4326).to_crs(epsg=3857)

# Define shock intensity bins and assign color labels
bins = [0, 2, 5, 9, gdf['total_experienced_composite_shocks'].max()]
labels = ['Low', 'Moderate', 'High', 'Extreme']
colors = ['#ffeeba', '#f0ad4e', '#d9534f', '#8B0000']
gdf['shock_bin'] = pd.cut(gdf['total_experienced_composite_shocks'], bins=bins, labels=labels, include_lowest=True)
gdf['color'] = gdf['shock_bin'].map(dict(zip(labels, colors)))

# Create plot
fig, ax = plt.subplots(figsize=(14, 10))

# Plot base administrative boundaries
bfa_admin1.boundary.plot(ax=ax, linewidth=1, edgecolor='black')

# Plot households colored by shock intensity
for label in labels:
    subset = gdf[gdf['shock_bin'] == label]
    subset.plot(ax=ax, markersize=25, color=dict(zip(labels, colors))[label], label=label, alpha=0.8)

# Add basemap
ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik)

# Add legend and title
ax.legend(title='Shock Intensity', loc='upper right')
ax.set_title("Burkina Faso: Composite Shock Hotspots (Household Level)", fontsize=14)

# Buffer the map bounds slightly to avoid clipping regions
buffer_ratio = 0.03
xmin, ymin, xmax, ymax = bfa_admin1.total_bounds
x_buffer = (xmax - xmin) * buffer_ratio
y_buffer = (ymax - ymin) * buffer_ratio
ax.set_xlim(xmin - x_buffer, xmax + x_buffer)
ax.set_ylim(ymin - y_buffer, ymax + y_buffer)

# Hide axes
ax.set_axis_off()

plt.tight_layout()
plt.show()


# In[68]:


import geopandas as gpd
import matplotlib.pyplot as plt
import contextily as ctx

# visualizes household-level composite shock intensity across Burkina Faso using administrative boundaries and a cleaner color scheme.

# Load household geodata
gdf = gpd.GeoDataFrame(df_merged, geometry='geometry')
gdf = gdf.set_crs(epsg=4326).to_crs(epsg=3857)

# Load Burkina Faso admin1 shapefile
shapefile_path = "/Users/user/Desktop/OSAKA UNIVERSITY/FINDEX_DATA/LSMS/WHOLE_LSMS_CLEANING/UNICEF/gadm41_BFA_shp/gadm41_BFA_1.shp"
bfa_admin1 = gpd.read_file(shapefile_path).to_crs(epsg=3857)

# Bin composite shock intensity
gdf['shock_bin'] = pd.cut(
    gdf['total_experienced_composite_shocks'],
    bins=[-1, 2, 4, 7, 99],
    labels=['Low', 'Moderate', 'High', 'Extreme']
)

# Define new custom color map
color_map = {
    'Low': '#fef0d9',
    'Moderate': '#fdcc8a',
    'High': '#fc8d59',
    'Extreme': '#d7301f'
}
gdf['color'] = gdf['shock_bin'].map(color_map)

# Plot the figure
fig, ax = plt.subplots(figsize=(14, 10))
bfa_admin1.boundary.plot(ax=ax, linewidth=1.5, edgecolor='black')
gdf.plot(ax=ax, color=gdf['color'], markersize=20, alpha=0.8)

# Add basemap
ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik)

# Set title
ax.set_title("Burkina Faso: Composite Shock Hotspots (Household Level)", fontsize=14)

# Remove axis ticks
ax.set_axis_off()

# Create custom legend
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker='o', color='w', label='Low', markerfacecolor='#fef0d9', markersize=10),
    Line2D([0], [0], marker='o', color='w', label='Moderate', markerfacecolor='#fdcc8a', markersize=10),
    Line2D([0], [0], marker='o', color='w', label='High', markerfacecolor='#fc8d59', markersize=10),
    Line2D([0], [0], marker='o', color='w', label='Extreme', markerfacecolor='#d7301f', markersize=10)
]
ax.legend(handles=legend_elements, title='Shock Intensity', loc='lower right', frameon=True)

plt.tight_layout()
plt.show()


# In[69]:


import matplotlib.pyplot as plt
import contextily as ctx
import pandas as pd
from matplotlib.lines import Line2D

# visualizes household-level composite shock intensity using a polished color palette and enhanced layout styling

gdf = gpd.GeoDataFrame(df_merged, geometry='geometry')
gdf = gdf.set_crs(epsg=4326).to_crs(epsg=3857)

gdf['shock_bin'] = pd.cut(
    gdf['total_experienced_composite_shocks'],
    bins=[-1, 2, 4, 7, 99],
    labels=['Low', 'Moderate', 'High', 'Extreme']
)

color_map = {
    'Low': '#e0f3f8',
    'Moderate': '#abd9e9',
    'High': '#74add1',
    'Extreme': '#4575b4'
}
gdf['color'] = gdf['shock_bin'].map(color_map)

fig, ax = plt.subplots(figsize=(14, 10))
bfa_admin1.boundary.plot(ax=ax, linewidth=1.2, edgecolor='black')
gdf.plot(ax=ax, color=gdf['color'], markersize=24, alpha=0.9)
ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron)

ax.set_title("Burkina Faso: Composite Shock Hotspots (Household Level)", fontsize=16, pad=15)
ax.set_axis_off()

legend_elements = [
    Line2D([0], [0], marker='o', color='w', label='Low', markerfacecolor=color_map['Low'], markersize=10),
    Line2D([0], [0], marker='o', color='w', label='Moderate', markerfacecolor=color_map['Moderate'], markersize=10),
    Line2D([0], [0], marker='o', color='w', label='High', markerfacecolor=color_map['High'], markersize=10),
    Line2D([0], [0], marker='o', color='w', label='Extreme', markerfacecolor=color_map['Extreme'], markersize=10)
]
ax.legend(handles=legend_elements, title='Shock Intensity', loc='upper right', frameon=True)

plt.tight_layout()
plt.show()


# In[70]:


import matplotlib.pyplot as plt
import contextily as ctx
import pandas as pd
from matplotlib.lines import Line2D

# This code visualizes household-level composite shock risk using a clean red-gradient scale reflecting severity

gdf = gpd.GeoDataFrame(df_merged, geometry='geometry')
gdf = gdf.set_crs(epsg=4326).to_crs(epsg=3857)

gdf['shock_bin'] = pd.cut(
    gdf['total_experienced_composite_shocks'],
    bins=[-1, 1, 3, 5, 99],
    labels=['Low', 'Medium', 'High', 'Extreme']
)

color_map = {
    'Low': '#fcaeae',       # Light red
    'Medium': '#f76868',    # Medium red
    'High': '#e31a1c',      # Strong red
    'Extreme': '#99000d'    # Deep burgundy red
}
gdf['color'] = gdf['shock_bin'].map(color_map)

fig, ax = plt.subplots(figsize=(14, 10))
bfa_admin1.boundary.plot(ax=ax, linewidth=1.2, edgecolor='black')
gdf.plot(ax=ax, color=gdf['color'], markersize=25, alpha=0.9)
ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron)

ax.set_title("Burkina Faso: Composite Shock Risk Zones (Household Level)", fontsize=16, pad=15)
ax.set_axis_off()

legend_elements = [
    Line2D([0], [0], marker='o', color='w', label='Low', markerfacecolor=color_map['Low'], markersize=10),
    Line2D([0], [0], marker='o', color='w', label='Medium', markerfacecolor=color_map['Medium'], markersize=10),
    Line2D([0], [0], marker='o', color='w', label='High', markerfacecolor=color_map['High'], markersize=10),
    Line2D([0], [0], marker='o', color='w', label='Extreme', markerfacecolor=color_map['Extreme'], markersize=10)
]
ax.legend(handles=legend_elements, title='Shock Risk Level', loc='upper right', frameon=True)

plt.tight_layout()
plt.show()


# In[75]:


import matplotlib.pyplot as plt
import contextily as ctx

# Plotting household-level intensity of each shock type with improved color schemes
fig, axes = plt.subplots(1, 3, figsize=(22, 7), constrained_layout=True)

# Define each shock type column and corresponding display properties
shock_types = {
    'num_experienced_economic_shocks': ('Economic Shocks', 'Reds'),
    'num_experienced_social_shocks': ('Social Shocks', 'Blues'),
    'num_experienced_environmental_shocks': ('Environmental Shocks', 'Greens')
}

for ax, (col, (title, cmap)) in zip(axes, shock_types.items()):
    bfa_admin1.boundary.plot(ax=ax, edgecolor='black', linewidth=1.2)
    gdf.plot(
        column=col,
        cmap=cmap,
        ax=ax,
        markersize=30,
        alpha=0.85,
        legend=True,
        legend_kwds={'label': title, 'shrink': 0.75}
    )
    ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron)
    ax.set_title(title, fontsize=14)
    ax.axis('off')


# In[73]:


print(gdf.columns)


# In[76]:


# Group by region and sum the shocks by type
region_shocks = df_merged.groupby('region')[[
    'num_experienced_economic_shocks',
    'num_experienced_social_shocks',
    'num_experienced_environmental_shocks',
    'total_experienced_composite_shocks'
]].sum().reset_index()


# In[77]:


region_shocks


# In[78]:


import matplotlib.pyplot as plt

# Set plot size and style
plt.figure(figsize=(14, 8))
region_shocks_sorted = region_shocks.sort_values(by='total_experienced_composite_shocks', ascending=False)

# Set region as index for cleaner plotting
plot_data = region_shocks_sorted.set_index('region')[
    ['num_experienced_economic_shocks', 'num_experienced_social_shocks', 'num_experienced_environmental_shocks']
]

# Create grouped bar chart
plot_data.plot(kind='bar', figsize=(14, 7), width=0.8,
               color=['#d62728', '#ff7f0e', '#c7c7c7'], edgecolor='black')

plt.title("Total Number of Shocks by Type and Region", fontsize=16)
plt.ylabel("Number of Households Affected")
plt.xlabel("Region")
plt.xticks(rotation=45, ha='right')
plt.legend(["Economic", "Social", "Environmental"], title="Shock Type")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()


# In[79]:


import matplotlib.pyplot as plt

# Sort by total shocks for better visual ranking
region_shocks_sorted = region_shocks.sort_values(
    by='total_experienced_composite_shocks', ascending=False
)

# Prepare data for plotting
plot_data = region_shocks_sorted.set_index('region')[
    ['num_experienced_economic_shocks', 'num_experienced_social_shocks', 'num_experienced_environmental_shocks']
]

# Create grouped bar chart with improved color palette
plot_data.plot(
    kind='bar',
    figsize=(14, 7),
    width=0.8,
    color=['#d62728', '#ff7f0e', '#2ca02c'],  # Red, Orange, Green
    edgecolor='black'
)

plt.title("Total Number of Shocks by Type and Region", fontsize=16)
plt.ylabel("Number of Households Affected")
plt.xlabel("Region")
plt.xticks(rotation=45, ha='right')
plt.legend(["Economic", "Social", "Environmental"], title="Shock Type")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()


# In[56]:


# Load the Burkina Faso admin1 shapefile (assuming .shp, .shx, and .dbf are all present in the folder)
shapefile_path = "/Users/user/Desktop/OSAKA UNIVERSITY/FINDEX_DATA/LSMS/WHOLE_LSMS_CLEANING/UNICEF/gadm41_BFA_shp/gadm41_BFA_1.shp"

bfa_admin1 = gpd.read_file(shapefile_path).to_crs(epsg=3857)


# In[ ]:





# In[80]:


df_roster = pd.read_csv("bfa_2021_roster_individuals_cleaned.csv")
df_roster


# In[81]:


df_roster.columns


# In[83]:


# Convert relevant variables to numeric (1 if yes/true, 0 if no/false or missing)
df_roster['has_mobile_phone'] = df_roster['has_mobile_phone'].astype(str).str.lower().str.strip().map({'1': 1, 'yes': 1, 'true': 1, '0': 0, 'no': 0, 'false': 0})
df_roster['internet_access_phone'] = df_roster['internet_access_phone'].astype(str).str.lower().str.strip().map({'1': 1, 'yes': 1, 'true': 1, '0': 0, 'no': 0, 'false': 0})


# In[84]:


# Summarize household roster into household-level features

# Tag household head rows
df_roster['is_head'] = df_roster['relationship_to_head'] == 1

# Household-level summary
df_roster_summary = df_roster.groupby(['wave', 'cluster', 'household_id']).agg(
    num_members=('member_id', 'count'),
    num_females=('sex', lambda x: (x == 2).sum()),
    num_children=('age_last_birthday', lambda x: (x < 15).sum()),
    any_mobile_phone=('has_mobile_phone', 'max'),
    any_internet_access=('internet_access_phone', 'max'),
    mean_prepaid_spending=('prepaid_card_expenditure_7_days', 'mean'),
    head_sex=('sex', lambda x: x[df_roster.loc[x.index, 'is_head']].values[0] if any(df_roster.loc[x.index, 'is_head']) else np.nan),
    head_age=('age_last_birthday', lambda x: x[df_roster.loc[x.index, 'is_head']].values[0] if any(df_roster.loc[x.index, 'is_head']) else np.nan),
    head_ethnicity=('ethnicity', lambda x: x[df_roster.loc[x.index, 'is_head']].values[0] if any(df_roster.loc[x.index, 'is_head']) else np.nan),
    head_education_level=('father_education_level', lambda x: x[df_roster.loc[x.index, 'is_head']].values[0] if any(df_roster.loc[x.index, 'is_head']) else np.nan),
).reset_index()


# In[85]:


df_roster_summary


# In[128]:


df_roster_summary.to_csv("roster_hh_level_cleaned_data.csv")


# In[90]:


df_roster_summary.rename(columns={'household_id': 'hhid'}, inplace=True)


# In[91]:


df_merged = df_merged.merge(
    df_roster_summary,
    how='left',
    on=['wave', 'cluster', 'hhid']
)


# In[92]:


df_merged


# In[93]:


df_merged.to_csv("bfa_cleaned_merged_geo_shocks_roster_data.csv")


# In[94]:


df_merged.columns


# In[112]:


df_welfare = pd.read_csv("welfare indicators.csv")


# In[113]:


df_welfare


# In[114]:


df_welfare = df_welfare.rename(columns={
    'grappe': 'cluster',
    'menage': 'household_number',
    'vague': 'wave',
    'hhweight': 'household_weight'
})

id_cols = ['year', 'country', 'hhid', 'cluster', 'household_number', 'wave', 'household_weight']
other_cols = [col for col in df_welfare.columns if col not in id_cols]
df_welfare = df_welfare[id_cols + other_cols]


# In[115]:


df_welfare


# In[116]:


df_welfare['country'] = df_welfare['country'].replace({'bfa': 'Burkina Faso'})


# In[117]:


df_welfare


# In[119]:


# Ensure both hhid columns are strings
df_merged['hhid'] = df_merged['hhid'].astype(str).str.strip()
df_welfare['hhid'] = df_welfare['hhid'].astype(str).str.strip()
df_welfare['country'] = df_welfare['country'].replace({'bfa': 'Burkina Faso'})

# Merge using hhid only
df_merged1 = df_merged.merge(df_welfare, how='left', on='hhid')


# In[122]:


df_merged_1 = pd.read_csv("bfa_cleaned_merged_geo_shocks_roster_data.csv")
df_merged_1


# In[ ]:





# In[ ]:





# In[123]:


df_geo


# In[125]:


df_household


# In[129]:


df_roster_summary


# In[130]:


# Ensure all ID columns are standardized

df_geo = df_geo.rename(columns={'household_weight': 'hhweight'})
df_welfare = df_welfare.rename(columns={'household_weight': 'hhweight'})

# Add missing 'year' and 'country' to all datasets except df_welfare
for df in [df_geo, df_household, df_roster_summary]:
    df['year'] = df_welfare['year'].iloc[0]
    df['country'] = df_welfare['country'].iloc[0]

# Merge all datasets into df_merged_full
df_merged_full = df_welfare.merge(
    df_geo, on=['hhid', 'cluster', 'household_number', 'wave', 'country', 'year'], how='left'
).merge(
    df_household, on=['hhid', 'cluster', 'household_number', 'country', 'year'], how='left'
).merge(
    df_roster_summary, on=['hhid', 'cluster', 'wave', 'country', 'year'], how='left'
)


# In[131]:


df_roster


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[132]:


df_shock


# In[134]:


df_shock.columns


# In[136]:


# This code aggregates LSMS shock microdata to the household level, preserving identifiers, coping strategies, shock impacts, and categorizing shocks into economic, social, and environmental groups.
coping_cols = [col for col in df_shock.columns if col.startswith('coping_') and '_used' not in col and col != 'coping_other_specify']
df_shock[coping_cols] = df_shock[coping_cols].fillna(0).astype(int)
df_aff = df_shock[df_shock['shock_occurred_last3y'].astype(str).str.lower() == 'oui'].copy()
shock_dummy_cols = [col for col in df_shock.columns if col.startswith("shock_type_")]
df_shock_types = df_aff[['hhid'] + shock_dummy_cols].groupby('hhid').max().reset_index()
impact_cols = [
    'shock_effect_income', 'shock_effect_assets', 'shock_effect_agri_prod',
    'shock_effect_livestock', 'shock_effect_food_stock', 'shock_effect_food_purchase'
]
df_agg = df_shock.groupby('hhid')[coping_cols + impact_cols].max().reset_index()
id_vars = ['hhid', 'cluster', 'household_code', 'year', 'hh_weight']
df_ids = df_shock[id_vars].drop_duplicates(subset='hhid')
df_shock_hh = df_ids.merge(df_shock_types, on='hhid', how='left') \
                    .merge(df_agg, on='hhid', how='left')
df_shock_hh[shock_dummy_cols] = df_shock_hh[shock_dummy_cols].fillna(0)

shock_groups = {
    'shock_economic': [
        'shock_type_Baisse importante des prix des produits agricoles',
        "shock_type_Perte d'emploi salarié d'un membre",
        "shock_type_Faillite d'une entreprise non agricole du ménage",
        "shock_type_Perte importante du revenu non agricole du ménage  (autre que du fait d'un accident ou d'une maladie)",
        "shock_type_Fin de transferts réguliers provenant d'autres ménages",
        "shock_type_Prix élevés des produits alimentaires",
        "shock_type_Prix élevés des intrants agricoles",
        "shock_type_Vol d'argent, de biens, de récolte ou de bétail"
    ],
    'shock_social': [
        "shock_type_Conflit armé/Violence/Insécurité",
        "shock_type_Conflit Agriculteur/Eleveur",
        "shock_type_Décès d'un membre du ménage",
        "shock_type_Divorce, séparation",
        "shock_type_Maladie grave ou accident d'un membre du ménage"
    ],
    'shock_environmental': [
        "shock_type_Sécheresse/Pluies irrégulières",
        "shock_type_Inondations",
        "shock_type_Incendies",
        "shock_type_Attaques acridiennes ou autres ravageurs de récolte",
        "shock_type_Taux élevé de maladies des cultures",
        "shock_type_Taux élevé de maladies des animaux",
        "shock_type_Glissement de terrain"
    ]
}

for group, cols in shock_groups.items():
    present_cols = [col for col in cols if col in df_shock_hh.columns]
    df_shock_hh[group] = df_shock_hh[present_cols].max(axis=1) if present_cols else 0


# In[137]:


df_shock_hh


# In[140]:


# This code aggregates LSMS shock microdata to the household level, preserving identifiers, coping strategies, shock impacts, and categorizing shocks into economic, social, and environmental groups, ensuring that grouped shock categories are not falsely triggered for households without actual shocks.

coping_cols = [col for col in df_shock.columns if col.startswith('coping_') and '_used' not in col and col != 'coping_other_specify']
df_shock[coping_cols] = df_shock[coping_cols].fillna(0).astype(int)
df_aff = df_shock[df_shock['shock_occurred_last3y'].astype(str).str.lower() == 'oui'].copy()
shock_dummy_cols = [col for col in df_shock.columns if col.startswith("shock_type_")]
df_shock_types = df_aff[['hhid'] + shock_dummy_cols].groupby('hhid').max().reset_index()
impact_cols = [
    'shock_effect_income', 'shock_effect_assets', 'shock_effect_agri_prod',
    'shock_effect_livestock', 'shock_effect_food_stock', 'shock_effect_food_purchase'
]
df_agg = df_shock.groupby('hhid')[coping_cols + impact_cols].max().reset_index()
id_vars = ['hhid', 'cluster', 'household_code', 'year', 'hh_weight']
df_ids = df_shock[id_vars].drop_duplicates(subset='hhid')
df_shock_hh = df_ids.merge(df_shock_types, on='hhid', how='left') \
                    .merge(df_agg, on='hhid', how='left')

shock_groups = {
    'shock_economic': [
        'shock_type_Baisse importante des prix des produits agricoles',
        "shock_type_Perte d'emploi salarié d'un membre",
        "shock_type_Faillite d'une entreprise non agricole du ménage",
        "shock_type_Perte importante du revenu non agricole du ménage  (autre que du fait d'un accident ou d'une maladie)",
        "shock_type_Fin de transferts réguliers provenant d'autres ménages",
        "shock_type_Prix élevés des produits alimentaires",
        "shock_type_Prix élevés des intrants agricoles",
        "shock_type_Vol d'argent, de biens, de récolte ou de bétail"
    ],
    'shock_social': [
        "shock_type_Conflit armé/Violence/Insécurité",
        "shock_type_Conflit Agriculteur/Eleveur",
        "shock_type_Décès d'un membre du ménage",
        "shock_type_Divorce, séparation",
        "shock_type_Maladie grave ou accident d'un membre du ménage"
    ],
    'shock_environmental': [
        "shock_type_Sécheresse/Pluies irrégulières",
        "shock_type_Inondations",
        "shock_type_Incendies",
        "shock_type_Attaques acridiennes ou autres ravageurs de récolte",
        "shock_type_Taux élevé de maladies des cultures",
        "shock_type_Taux élevé de maladies des animaux",
        "shock_type_Glissement de terrain"
    ]
}

for group, cols in shock_groups.items():
    present_cols = [col for col in cols if col in df_shock_hh.columns]
    df_shock_hh[group] = df_shock_hh[present_cols].max(axis=1) if present_cols else 0

df_shock_hh[shock_dummy_cols + list(shock_groups.keys())] = df_shock_hh[
    shock_dummy_cols + list(shock_groups.keys())
].fillna(0)


# In[143]:


df_household


# In[144]:


df_household["num_experienced_economic_shocks"].value_counts()


# In[ ]:





# In[ ]:





# In[57]:


master_data = pd.read_csv("bfa_merged_geo_shock_roster_welfare_housing.csv")


# In[58]:


master_data


# In[201]:


import os
os.getcwd()


# In[59]:


gdf = master_data.copy()
gdf['geometry'] = gdf.apply(
    lambda row: Point(row['gps_longitude'], row['gps_latitude'])
    if pd.notnull(row['gps_latitude']) and pd.notnull(row['gps_longitude'])
    else None,
    axis=1
)
gdf = gpd.GeoDataFrame(gdf, geometry='geometry', crs='EPSG:4326')

shapefile_path = '/Users/user/Desktop/OSAKA UNIVERSITY/FINDEX_DATA/LSMS/WHOLE_LSMS_CLEANING/UNICEF/gadm41_BFA_shp/gadm41_BFA_1.shp'
region_shp = gpd.read_file(shapefile_path).to_crs(gdf.crs)

gdf_valid = gdf[gdf['geometry'].notnull()].copy()
gdf_invalid = gdf[gdf['geometry'].isnull()].copy()

gdf_valid = gpd.sjoin(gdf_valid, region_shp[['NAME_1', 'geometry']], how='left', predicate='within')
gdf_invalid['NAME_1'] = None

gdf = pd.concat([gdf_valid, gdf_invalid], ignore_index=True)

for col in ['shock_economic', 'shock_social', 'shock_environmental']:
    gdf[col] = gdf[col].fillna(0).astype(int)

gdf['ccri'] = gdf['shock_economic'] + gdf['shock_social'] + gdf['shock_environmental']

ccri_by_region = gdf.dropna(subset=['NAME_1']).groupby('NAME_1')['ccri'].mean().reset_index()

map_df = region_shp.merge(ccri_by_region, on='NAME_1', how='left')

fig, ax = plt.subplots(figsize=(10, 7))
map_df.plot(column='ccri', cmap='YlOrRd', linewidth=0.6, ax=ax, edgecolor='0.9', legend=True)
ax.set_title('Average Child-Sensitive CCRI by Region – Burkina Faso', fontsize=14)
ax.axis('off')
plt.tight_layout()
plt.show()


# In[60]:


# Show region names with their average CCRI scores
ccri_by_region_sorted = ccri_by_region.sort_values(by='ccri', ascending=False)
print(ccri_by_region_sorted)

# Optionally, save to CSV for inclusion in reports
ccri_by_region_sorted.to_csv('average_ccri_by_region.csv', index=False)


# In[61]:


fig, ax = plt.subplots(figsize=(10, 7))
map_df.plot(column='ccri', cmap='YlOrRd', linewidth=0.6, ax=ax, edgecolor='0.9', legend=True)
ax.set_title('Average Child-Sensitive CCRI by Region – Burkina Faso', fontsize=14)
ax.axis('off')

# Add region name labels at the centroid of each polygon
for idx, row in map_df.iterrows():
    if row['geometry'].geom_type == 'MultiPolygon':
        centroid = row['geometry'].representative_point()
    else:
        centroid = row['geometry'].centroid
    ax.annotate(row['NAME_1'], xy=(centroid.x, centroid.y),
                horizontalalignment='center', fontsize=10, fontweight='bold', color='black')

plt.tight_layout()
plt.show()


# In[62]:


fig, axs = plt.subplots(1, 3, figsize=(22, 7), sharey=True)

shock_types = ['shock_economic', 'shock_social', 'shock_environmental']
titles = ['Economic Shocks', 'Social Shocks', 'Environmental Shocks']
cmaps = ['YlOrRd', 'Blues', 'Greens']

for i, shock in enumerate(shock_types):
    # Compute average shock intensity by region
    avg_shock = gdf.dropna(subset=['NAME_1']).groupby('NAME_1')[shock].mean().reset_index()
    map_shock = region_shp.merge(avg_shock, on='NAME_1', how='left')
    
    # Plot the map
    map_shock.plot(column=shock, cmap=cmaps[i], linewidth=0.6, ax=axs[i], edgecolor='0.9', legend=True)
    axs[i].set_title(titles[i], fontsize=14)
    axs[i].axis('off')

    # Add region labels
    for idx, row in map_shock.iterrows():
        if row['geometry'].geom_type == 'MultiPolygon':
            centroid = row['geometry'].representative_point()
        else:
            centroid = row['geometry'].centroid
        axs[i].annotate(row['NAME_1'], xy=(centroid.x, centroid.y), 
                        horizontalalignment='center', fontsize=9, fontweight='bold', color='black')

plt.tight_layout()
plt.show()


# In[142]:


print(gdf.columns.tolist())


# In[171]:


map_shock.columns


# In[181]:


map_shock


# In[185]:


map_df


# In[158]:


data_master['country'] = data_master['country'].str.title()
data_master


# In[159]:


# Get the column names from each dataframe as sets
master_cols = set(master_data.columns)
sat_cols = set(sat_data.columns)
climate_cols = set(climate_data.columns)

# Common columns in all three datasets
common_all = master_cols & sat_cols & climate_cols
print("Common columns in all three datasets:")
print(sorted(common_all))

# Common columns between master and satellite
common_master_sat = master_cols & sat_cols
print("Common columns between master_data and sat_data:")
print(sorted(common_master_sat))

# Common columns between master and climate
common_master_climate = master_cols & climate_cols
print("Common columns between master_data and climate_data:")
print(sorted(common_master_climate))

# Common columns between satellite and climate
common_sat_climate = sat_cols & climate_cols
print("Common columns between sat_data and climate_data:")
print(sorted(common_sat_climate))


# In[ ]:





# In[ ]:





# In[ ]:





# In[146]:


sat_data = pd.read_csv("Bfa_Climate_data_2021_01-12.csv")
sat_data


# In[147]:


climate_data = pd.read_csv("BFA_satellite_data.csv")
climate_data


# In[151]:


# Add to satellite data
sat_data['year'] = 2021
sat_data['country'] = 'Burkina Faso'

# Add to climate data
climate_data['year'] = 2021
climate_data['country'] = 'Burkina Faso'

# Reorder columns to bring year and country to the front
def reorder_cols(df):
    cols = ['year', 'country'] + [col for col in df.columns if col not in ['year', 'country']]
    return df[cols]

sat_data = reorder_cols(sat_data)
climate_data = reorder_cols(climate_data)

# Preview
print(sat_data[['year', 'country']].head(1))
print(climate_data[['year', 'country']].head(1))


# In[160]:


# Add year and country to satellite and climate data
sat_data["year"] = 2021
sat_data["country"] = "Burkina Faso"
climate_data["year"] = 2021
climate_data["country"] = "Burkina Faso"

# Standardize country names to lowercase
data_master["country"] = data_master["country"].str.lower()
sat_data["country"] = sat_data["country"].str.lower()
climate_data["country"] = climate_data["country"].str.lower()

# Reorder columns to move ID variables to the front
sat_data = sat_data[['year', 'country', 'cluster', 'hhid'] + [col for col in sat_data.columns if col not in ['year', 'country', 'cluster', 'hhid']]]
climate_data = climate_data[['year', 'country', 'cluster', 'hhid'] + [col for col in climate_data.columns if col not in ['year', 'country', 'cluster', 'hhid']]]

# Merge datasets
merged_data = data_master.merge(sat_data, on=["year", "country", "cluster", "hhid"], how="left")
merged_data = merged_data.merge(climate_data, on=["year", "country", "cluster", "hhid"], how="left")

# Reorder columns to place satellite and climate variables at the end
sat_cols = [col for col in sat_data.columns if col not in ['year', 'country', 'cluster', 'hhid', 'system:index', '.geo']]
clim_cols = [col for col in climate_data.columns if col not in ['year', 'country', 'cluster', 'hhid', 'system:index', '.geo']]
id_cols = ['year', 'country', 'cluster', 'hhid']
main_cols = [col for col in merged_data.columns if col not in id_cols + sat_cols + clim_cols]
merged_data = merged_data[id_cols + main_cols + sat_cols + clim_cols]


# In[161]:


merged_data


# In[193]:


merged_data.to_csv("MASTER_DATA_WITH_SAT_CLIMATE_DATA.csv")


# In[162]:


key_vars = ['NDBI', 'NDVI', 'Nightlight', 'rainfall', 'temperature']
missing_summary = merged_data[key_vars].isna().sum()
duplicate_rows = merged_data.duplicated(subset=['year', 'country', 'cluster', 'hhid']).sum()
dtypes_summary = merged_data[key_vars].dtypes

print("Missing values in key columns:\n", missing_summary)
print("\nNumber of duplicate rows by ID:", duplicate_rows)
print("\nData types of key columns:\n", dtypes_summary)


# In[163]:


[col for col in merged_data.columns if 'temperature' in col.lower()]


# In[ ]:





# In[195]:


avg_shock


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[231]:


pip install geopandas contextily matplotlib


# In[63]:


import matplotlib.pyplot as plt
import geopandas as gpd
import contextily as cx 
import pandas as pd 
import matplotlib.patheffects as pe 

fig, axs = plt.subplots(1, 3, figsize=(25, 9), sharey=True) # Slightly larger figure for better detail

shock_types = ['shock_economic', 'shock_social', 'shock_environmental']
titles = ['Average Economic Shock Intensity', 'Average Social Shock Intensity', 'Average Environmental Shock Intensity']

cmaps = ['viridis', 'plasma', 'cividis'] 


for i, shock in enumerate(shock_types):
    # Compute average shock intensity by region
    # Ensure 'gdf' is accessible and 'NAME_1' is clean
    avg_shock = gdf.dropna(subset=['NAME_1']).groupby('NAME_1')[shock].mean().reset_index()

    # Merge with region_shp (ensure 'NAME_1' column exists in both and has consistent values)
    map_shock = region_shp.merge(avg_shock, on='NAME_1', how='left')

   
    # Plot the map
    # Add a distinct edge color and transparency for clarity
    map_shock.plot(
        column=shock,
        cmap=cmaps[i],
        linewidth=0.8,
        ax=axs[i],
        edgecolor='grey', # Slightly softer edge
        legend=True,
        legend_kwds={
            'label': f"Average {titles[i].split(' ')[1].replace('Intensity', 'Shock')}", # Cleaner label for legend
            'orientation': "vertical", # Vertical legend is often cleaner
            'shrink': 0.7 # Adjust legend size
        },
        missing_kwds={
            "color": "lightgrey",
            "edgecolor": "red",
            "hatch": "///",
            "label": "Missing values",
        }
    )

    cx.add_basemap(axs[i], source=cx.providers.CartoDB.PositronNoLabels) 

    axs[i].set_title(titles[i], fontsize=16, pad=15) 
    axs[i].axis('off') 

    # Add region labels with corrected styling using patheffects.withStroke
    for idx, row in map_shock.iterrows():
        if row['geometry'] is not None: # Ensure geometry exists
            if row['geometry'].geom_type == 'MultiPolygon':
                centroid = row['geometry'].representative_point()
            else:
                centroid = row['geometry'].centroid

            # Use patheffects.withStroke for a clean outline effect
            axs[i].annotate(row['NAME_1'], xy=(centroid.x, centroid.y),
                             horizontalalignment='center',
                             fontsize=9,
                             fontweight='bold',
                             color='black',
                             path_effects=[pe.withStroke(linewidth=3, foreground="white")] 
                            )
plt.suptitle('Geographical Distribution of Shock Intensities in Burkina Faso', fontsize=20, y=0.98)

plt.tight_layout(rect=[0, 0, 1, 0.95]) 
plt.show()


# In[65]:


master_data


# In[69]:


master_data["asset_index"].value_counts()


# In[ ]:





# In[ ]:





# In[68]:


from sklearn.preprocessing import MinMaxScaler

try:
    master_data = pd.read_csv("bfa_merged_geo_shock_roster_welfare_housing.csv")
    print("master_data loaded successfully.")
except FileNotFoundError:
    print("Error: 'bfa_merged_geo_shock_roster_welfare_housing.csv' not found. Exiting.")
    exit()

# --- Data Preparation and Index Calculation ---

# Define pillar variable groups
exposure_vars = ['shock_economic', 'shock_social', 'shock_environmental']
vulnerability_vars = ['num_children', 'num_adolescents', 'head_gender', 'num_rooms']
capacity_vars = ['asset_index', 'connected_to_electricity', 'connected_to_water_network',
                 'main_drinking_water_source', 'toilet_type', 'treats_drinking_water']

# To Convert 'head_gender' to numerical if present (0 for male, 1 for female)
if 'head_gender' in master_data.columns:
    master_data.loc[:, 'head_gender'] = master_data['head_gender'].apply(
        lambda x: 1 if str(x).lower() == 'female' else (0 if str(x).lower() == 'male' else x)
    )
    master_data.loc[:, 'head_gender'] = pd.to_numeric(master_data['head_gender'], errors='coerce')

# Fill NA values with 0 for all variables used in index calculations
all_index_vars = exposure_vars + vulnerability_vars + capacity_vars
for col in all_index_vars:
    if col in master_data.columns:
        master_data.loc[:, col] = pd.to_numeric(master_data[col], errors='coerce').fillna(0)
    else:
        print(f"Warning: Column '{col}' not found. Initializing with zeros.")
        master_data.loc[:, col] = 0.0

scaler = MinMaxScaler()

# Calculate 'exposure_index'
master_data['exposure_index'] = 0.0
if all(col in master_data.columns for col in exposure_vars) and not master_data[exposure_vars].empty:
    master_data['exposure_index'] = scaler.fit_transform(master_data[exposure_vars]).mean(axis=1)
    print("exposure_index calculated.")

# Calculate 'vulnerability_index'
master_data['vulnerability_index'] = 0.0
if all(col in master_data.columns for col in vulnerability_vars) and not master_data[vulnerability_vars].empty:
    master_data['vulnerability_index'] = scaler.fit_transform(master_data[vulnerability_vars]).mean(axis=1)
    print("vulnerability_index calculated.")

# Calculate 'capacity_index'

master_data['capacity_index'] = 0.0
if all(col in master_data.columns for col in capacity_vars) and not master_data[capacity_vars].empty:
    master_data['capacity_index'] = 1 - scaler.fit_transform(master_data[capacity_vars]).mean(axis=1)
    print("capacity_index calculated.")

# Calculate 'ccri_index' as average of the three pillar indices
pillar_indices = ['exposure_index', 'vulnerability_index', 'capacity_index']
for p_idx_col in pillar_indices:
    master_data.loc[:, p_idx_col] = pd.to_numeric(master_data[p_idx_col], errors='coerce')

master_data['ccri_index'] = 0.0
if all(col in master_data.columns for col in pillar_indices) and not master_data[pillar_indices].empty:
    master_data['ccri_index'] = master_data[pillar_indices].mean(axis=1)
    print("ccri_index calculated.")

# --- Display Results ---
print("\n--- New Index Columns Added ---")
print("First 5 rows with new indices:")
print(master_data[['exposure_index', 'vulnerability_index', 'capacity_index', 'ccri_index']].head().to_markdown(index=False, numalign="left", stralign="left"))

print("\nDescriptive statistics for new indices:")
print(master_data[['exposure_index', 'vulnerability_index', 'capacity_index', 'ccri_index']].describe().to_markdown(numalign="left", stralign="left"))


# In[214]:


master_data['head_gender'].value_counts(dropna=False)
master_data['hgender'].value_counts(dropna=False)


# In[215]:


# Create binary indicator: 1 = female head, 0 = male
master_data['head_gender_female'] = (master_data['hgender'].str.lower() == 'féminin').astype(int)


# In[216]:


[col for col in master_data.columns if 'asset' in col.lower() or 'water' in col.lower()]


# In[ ]:





# In[ ]:





# In[113]:


ee.Initialize(project='ee-sanohyusuf2015')


# In[116]:


import ee
ee.Initialize(project='ee-sanohyusuf2015')

# Correct: Use ImageCollection, then call .first()
collection = ee.ImageCollection("LANDSAT/LC09/C02/T1_L2") \
    .filterDate("2022-01-01", "2022-12-31") \
    .filterBounds(ee.Geometry.Point([-1.6, 13.1]))  # Optional: use any point in Burkina

first_image = collection.first()
print(first_image.getInfo())


# In[131]:


# Export Earth Engine FeatureCollection to Google Drive
task = ee.batch.Export.table.toDrive(
    collection=extracted,
    description='BFA_Satellite_Extraction',
    folder='GEE_Exports',
    fileNamePrefix='BFA_satellite_data',
    fileFormat='CSV'
)

task.start()
print("Export started! Check your Google Drive > GEE_Exports folder in a few minutes.")


# In[122]:


df_sat = pd.read_csv("BFA_satellite_data.csv")
df_sat


# In[134]:


# Normalize columns to lowercase
df_sat.columns = df_sat.columns.str.lower()

# Now safely plot
df_sat[['ndvi', 'ndbi', 'nightlight']].hist(bins=30, figsize=(10, 6))


# In[140]:


# Convert qa_pixel from float to int (by scaling ×100 or ×1000)
df_sat['qa_pixel_int'] = df_sat['qa_pixel'].multiply(100).round().astype('Int64')

# Step 2: Check frequency
print(df_sat['qa_pixel_int'].value_counts(dropna=False).head(10))


# In[139]:


df_sat['qa_pixel'].value_counts(dropna=False).head(10)


# In[128]:


# Ensure consistent column casing
df_sat.columns = df_sat.columns.str.strip().str.lower()

# Fill NaNs in 'latitude' and 'longitude' with their respective means
df_sat['latitude'] = df_sat['latitude'].fillna(df_sat['latitude'].mean())
df_sat['longitude'] = df_sat['longitude'].fillna(df_sat['longitude'].mean())


# In[129]:


import ee
ee.Initialize(project='ee-sanohyusuf2015')  # Assuming project is properly authorized

# Create a list of ee.Features
features = [
    ee.Feature(
        ee.Geometry.Point([row['longitude'], row['latitude']]),
        {
            'cluster': row['cluster'],
            'hhid': row['hhid']
        }
    )
    for _, row in df_sat.iterrows()
]

# Convert to ee.FeatureCollection
fc = ee.FeatureCollection(features)


# In[130]:


import ee
import pandas as pd
import datetime

# Initialize Earth Engine
ee.Initialize(project='ee-sanohyusuf2015')

# --- Load satellite GPS points and fill NaNs ---
df_sat.columns = df_sat.columns.str.strip().str.lower()
df_sat['latitude'] = df_sat['latitude'].fillna(df_sat['latitude'].mean())
df_sat['longitude'] = df_sat['longitude'].fillna(df_sat['longitude'].mean())

# --- Convert to FeatureCollection ---
features = [
    ee.Feature(ee.Geometry.Point([row['longitude'], row['latitude']]), {
        'hhid': str(row['hhid']),
        'cluster': str(row['cluster'])
    }) for _, row in df_sat.iterrows()
]
fc_points = ee.FeatureCollection(features)

# --- Time settings ---
year = 2021
months = list(range(1, 13))

# --- Function to extract monthly mean for each layer ---
def extract_monthly_climate(month, year=2021):
    start_date = datetime.date(year, month, 1)
    end_date = datetime.date(year, month % 12 + 1, 1) if month < 12 else datetime.date(year+1, 1, 1)
    
    # CHIRPS Rainfall (mm/day)
    chirps = ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY") \
        .filterDate(str(start_date), str(end_date)) \
        .mean().rename('rainfall')

    # MODIS LST (Kelvin) — scaled by 0.02 and converted to Celsius
    lst = ee.ImageCollection('MODIS/061/MOD11A2') \
        .filterDate(str(start_date), str(end_date)) \
        .select('LST_Day_1km') \
        .mean().multiply(0.02).subtract(273.15).rename('temperature')

    # Landsat 9 NDVI (normalized vegetation)
    ndvi = ee.ImageCollection('LANDSAT/LC09/C02/T1_L2') \
        .filterDate(str(start_date), str(end_date)) \
        .map(lambda img: img.normalizedDifference(['SR_B5', 'SR_B4']).rename('NDVI')) \
        .mean()
    
    # Combine bands
    composite = chirps.addBands([lst, ndvi])

    # Reduce over GPS points
    reduced = composite.reduceRegions(
        collection=fc_points,
        reducer=ee.Reducer.mean(),
        scale=500,
    )

    # Export
    task = ee.batch.Export.table.toDrive(
        collection=reduced,
        description=f'BFA_Climate_M{month}_Y{year}',
        folder='GEE_Exports',
        fileNamePrefix=f'climate_bfa_{year}_{month:02d}',
        fileFormat='CSV'
    )
    task.start()
    print(f"Export started for Month {month:02d} ({year}) — check Google Drive > GEE_Exports.")

# --- Loop through all months in 2021 ---
for m in months:
    extract_monthly_climate(month=m, year=year)


# In[ ]:





# In[137]:


df_sat.columns


# In[186]:


merged_data


# In[188]:


gdf


# In[189]:


map_df


# In[192]:


merged_data.columns


# In[196]:


# Compute average shock exposure by region using the 'region' column
avg_shocks = (
    merged_data.groupby("region")[['shock_economic', 'shock_social', 'shock_environmental']]
    .mean()
    .reset_index()
)


# In[203]:


map_df


# In[198]:


avg_shocks


# In[200]:


# Merge 'region' from avg_shocks and 'NAME_1' from map_df
map_df = map_df.merge(avg_shocks, left_on='NAME_1', right_on='region', how='left')


# In[201]:


map_df


# In[202]:


map_df


# In[204]:


print("From map_df (shapefile):")
print(sorted(map_df['NAME_1'].dropna().unique()))

print("\nFrom avg_shocks (hh data):")
print(sorted(avg_shocks['region'].dropna().unique()))


# In[206]:


# Clean both sides
map_df['NAME_1_clean'] = map_df['NAME_1'].str.strip().str.lower()

# Manual corrections for avg_shocks regions
region_fix_map = {
    'boucle du mouhoum': 'boucle du mouhoun',
    'hauts-bassins': 'haut-bassins'
}

avg_shocks['region_clean'] = (
    avg_shocks['region']
    .str.strip()
    .str.lower()
    .replace(region_fix_map)
)

# Merge using cleaned region names
map_df = map_df.merge(
    avg_shocks,
    left_on='NAME_1_clean',
    right_on='region_clean',
    how='left'
)


# In[207]:


map_df


# In[209]:


# Create a clean version of map_df with only relevant columns
map_df = map_df[[
    'GID_1', 'GID_0', 'COUNTRY', 'NAME_1', 'VARNAME_1', 'NL_NAME_1',
    'TYPE_1', 'ENGTYPE_1', 'CC_1', 'HASC_1', 'ISO_1', 'geometry', 'ccri',
    'shock_economic_y', 'shock_social_y', 'shock_environmental_y', 'NAME_1_clean'
]]

# Rename shock columns (remove _y suffix)
map_df.rename(columns={
    'shock_economic_y': 'shock_economic',
    'shock_social_y': 'shock_social',
    'shock_environmental_y': 'shock_environmental'
}, inplace=True)


# In[210]:


print(f"Number of rows in map_df: {len(map_df)}")
print(map_df[['NAME_1', 'shock_economic', 'shock_social', 'shock_environmental']].head())


# In[211]:


import matplotlib.pyplot as plt

# Create 3 subplots for the 3 shock types
fig, axs = plt.subplots(1, 3, figsize=(22, 7), sharey=True)

shock_types = ['shock_economic', 'shock_social', 'shock_environmental']
titles = ['Economic Shocks', 'Social Shocks', 'Environmental Shocks']
cmaps = ['YlOrRd', 'Blues', 'Greens']

for i, shock in enumerate(shock_types):
    map_df.plot(
        column=shock,
        cmap=cmaps[i],
        linewidth=0.6,
        ax=axs[i],
        edgecolor='0.9',
        legend=True
    )
    axs[i].set_title(titles[i], fontsize=14)
    axs[i].axis('off')

    # Annotate regions
    for idx, row in map_df.iterrows():
        centroid = (
            row['geometry'].representative_point()
            if row['geometry'].geom_type == 'MultiPolygon'
            else row['geometry'].centroid
        )
        axs[i].annotate(
            row['NAME_1'],
            xy=(centroid.x, centroid.y),
            horizontalalignment='center',
            fontsize=9,
            fontweight='bold',
            color='black'
        )

plt.tight_layout()
plt.show()


# In[218]:


map_df_coping = map_df.merge(
    coping_by_region,
    left_on='NAME_1_clean',
    right_on='region_clean',
    how='left',
    suffixes=('', '_coping')
)

# Fill any missing coping values with 0
coping_cols = [col for col in coping_by_region.columns if col.startswith("coping_")]
map_df_coping[coping_cols] = map_df_coping[coping_cols].fillna(0)


# In[219]:


map_df_coping


# In[221]:


coping_cols = [col for col in map_df_coping.columns if col.startswith("coping_")]
titles = ['Coping: Credit', 'Coping: Child Labor', 'Coping: Sell Livestock']
cols = ['coping_credit', 'coping_children_work', 'coping_sell_livestock']


# In[222]:


import matplotlib.pyplot as plt

# List of coping strategies to map
coping_cols_to_plot = [
    "coping_credit", 
    "coping_children_work", 
    "coping_sell_livestock"
]

# Titles and colormaps for each subplot
titles = [
    "Coping Strategy: Credit",
    "Coping Strategy: Child Labor",
    "Coping Strategy: Sell Livestock"
]
cmaps = ["Purples", "Oranges", "YlGn"]

# Set up subplots
fig, axs = plt.subplots(1, 3, figsize=(24, 8))

# Plot each coping strategy
for i, col in enumerate(coping_cols_to_plot):
    map_df_coping.plot(
        column=col,
        cmap=cmaps[i],
        linewidth=0.8,
        edgecolor="white",
        ax=axs[i],
        legend=True,
        legend_kwds={"label": "Proportion", "shrink": 0.6}
    )
    axs[i].set_title(titles[i], fontsize=14)
    axs[i].axis("off")

    # Add region labels
    for idx, row in map_df_coping.iterrows():
        centroid = (
            row["geometry"].representative_point()
            if row["geometry"].geom_type == "MultiPolygon"
            else row["geometry"].centroid
        )
        axs[i].annotate(
            row["NAME_1"],
            xy=(centroid.x, centroid.y),
            ha='center',
            fontsize=8,
            color="black"
        )

plt.tight_layout()
plt.show()


# In[223]:


# Economically appropriate label mapping
coping_titles_map = {
    'coping_savings': 'Reduce Savings',
    'coping_no_strategy': 'No Coping Strategy',
    'coping_family_help': 'Rely on Informal Transfers',
    'coping_sell_livestock': 'Liquidate Productive Assets (Livestock)',
    'coping_buy_cheaper_food': 'Buy Cheaper Food',
    'coping_change_consumption': 'Reduce Current Consumption'
}

# Select top 3 strategy columns and labels
top3_coping_cols = list(coping_titles_map.keys())[:3]
top3_coping_titles = [coping_titles_map[col] for col in top3_coping_cols]

# Plotting
fig, axs = plt.subplots(1, 3, figsize=(22, 8))

for i, col in enumerate(top3_coping_cols):
    map_df_coping[[col, 'geometry']].plot(
        column=col,
        cmap='YlOrRd',
        legend=True,
        edgecolor='0.9',
        linewidth=0.6,
        ax=axs[i]
    )
    axs[i].set_title(top3_coping_titles[i], fontsize=14)
    axs[i].axis('off')

plt.suptitle("Top 3 Coping Mechanisms by Region", fontsize=18, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()


# In[224]:


# Coping strategy labels (economically appropriate)
coping_titles_map = {
    'coping_savings': 'Reduce Savings',
    'coping_no_strategy': 'No Coping Strategy',
    'coping_family_help': 'Rely on Informal Transfers',
    'coping_sell_livestock': 'Liquidate Productive Assets (Livestock)',
    'coping_buy_cheaper_food': 'Buy Cheaper Food',
    'coping_change_consumption': 'Reduce Current Consumption'
}

# Extract columns and titles
coping_cols = list(coping_titles_map.keys())
coping_titles = [coping_titles_map[col] for col in coping_cols]

# Create 2x3 subplot grid
fig, axs = plt.subplots(2, 3, figsize=(24, 12))

# Plot each coping strategy
for i, col in enumerate(coping_cols):
    row, col_idx = divmod(i, 3)
    map_df_coping.plot(
        column=col,
        cmap='YlGnBu',
        linewidth=0.6,
        edgecolor='0.9',
        legend=True,
        ax=axs[row][col_idx]
    )
    axs[row][col_idx].set_title(coping_titles[i], fontsize=14)
    axs[row][col_idx].axis('off')

plt.suptitle("Household Coping Mechanisms by Region", fontsize=20, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()


# In[280]:


# label mapping
coping_titles_map = {
    'coping_savings': "Used Savings",
    'coping_no_strategy': 'No Coping Strategy',
    'coping_family_help': 'Received Informal Transfers (from family/friends)',
    'coping_sell_livestock': 'Sold Assets (Livestock)',
    'coping_buy_cheaper_food': 'Bought Cheaper Food',
    'coping_change_consumption': 'Reduced Consumption'
}

coping_cols = list(coping_titles_map.keys())
coping_titles = list(coping_titles_map.values())

# Plotting
fig, axs = plt.subplots(2, 3, figsize=(22, 12))

for i, col in enumerate(coping_cols):
    row_idx, col_idx = divmod(i, 3)
    ax = axs[row_idx][col_idx]

    # Plot coping strategy choropleth
    map_df_coping.plot(
        column=col,
        cmap='YlGnBu',
        linewidth=0.6,
        edgecolor='0.9',
        legend=True,
        ax=ax
    )

    # Add region name labels at centroids
    for _, row in map_df_coping.iterrows():
        if row['geometry'] is not None:
            centroid = row['geometry'].centroid
            ax.text(
                centroid.x,
                centroid.y,
                row['NAME_1'],
                fontsize=8,
                ha='center',
                color='black'
            )

    ax.set_title(coping_titles[i], fontsize=14)
    ax.axis('off')

# Final layout
plt.suptitle("Household Coping Mechanisms by Region", fontsize=20, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()


# In[281]:


# Convert to long format
coping_long = map_df_coping.melt(
    id_vars='NAME_1', 
    value_vars=list(coping_titles_map.keys()), 
    var_name='Strategy', 
    value_name='Proportion'
)
coping_long['Strategy'] = coping_long['Strategy'].map(coping_titles_map)

# Barplot
plt.figure(figsize=(14, 8))
sns.barplot(data=coping_long, x='Proportion', y='NAME_1', hue='Strategy')
plt.title('Coping Strategy Prevalence by Region', fontsize=16, fontweight='bold')
plt.xlabel('Proportion of Households')
plt.ylabel('Region')
plt.legend(title='Coping Strategy')
plt.tight_layout()
plt.show()


# In[282]:


# Create heatmap to show the intensity of coping strategies
coping_heatmap_data = map_df_coping.set_index('NAME_1')[list(coping_titles_map.keys())]
coping_heatmap_data.rename(columns=coping_titles_map, inplace=True)

# Plot
plt.figure(figsize=(12, 6))
sns.heatmap(coping_heatmap_data, annot=True, cmap="YlGnBu", fmt=".3f", linewidths=0.5)
plt.title('Heatmap of Coping Mechanisms by Region', fontsize=16, fontweight='bold')
plt.xlabel('Coping Strategy')
plt.ylabel('Region')
plt.tight_layout()
plt.show()


# In[283]:


# Stacked bar chart by region
coping_long = map_df_coping.melt(
    id_vars='NAME_1',
    value_vars=list(coping_titles_map.keys()),
    var_name='Strategy',
    value_name='Proportion'
)
coping_long['Strategy'] = coping_long['Strategy'].map(coping_titles_map)

plt.figure(figsize=(14, 8))
sns.barplot(
    data=coping_long,
    x='NAME_1',
    y='Proportion',
    hue='Strategy'
)
plt.xticks(rotation=45)
plt.title('Stacked Coping Strategies by Region', fontsize=16, fontweight='bold')
plt.xlabel('Region')
plt.ylabel('Proportion of Households')
plt.legend(title='Coping Strategy', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()


# In[284]:


# Filter for columns related to coping methods, excluding the descriptive column

coping_columns = [col for col in merged_data.columns if col.startswith('coping_') and col != 'coping_other_specifies']

# Initialize a dictionary to store the frequencies
coping_frequencies_dict = {}

# Iterate over the coping columns and count occurrences where the value is greater than 0
for col in coping_columns:
    count = (merged_data[col] > 0).sum()
    coping_frequencies_dict[col] = count

# Convert the dictionary to a pandas Series and sort in descending order
coping_frequencies = pd.Series(coping_frequencies_dict).sort_values(ascending=False)

# Rename the index for better readability
coping_frequencies.index = coping_frequencies.index.str.replace('coping_', '').str.replace('_', ' ').str.title()

# Create a bar chart to visualize the frequencies
plt.figure(figsize=(12, 8))
coping_frequencies.plot(kind='bar', color='skyblue')
plt.title('Frequency of Coping Methods', fontsize=16)
plt.xlabel('Coping Mechanisms', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.xticks(rotation=45, ha='right', fontsize=10)
plt.tight_layout()
plt.show()


# In[279]:


import pandas as pd
import matplotlib.pyplot as plt


# Create a mapping for descriptive names.
coping_name_map = {
    'coping_savings': 'Used Savings',
    'coping_family_help': 'Received Family Help',
    'coping_govt_help': 'Received Government Help',
    'coping_ngo_help': 'Received NGO Help',
    'coping_marry_daughters': 'Married Daughters Off',
    'coping_change_consumption': 'Reduced Consumption',
    'coping_buy_cheaper_food': 'Bought Cheaper Food',
    'coping_extra_jobs_active': 'Took on Extra Jobs',
    'coping_unemployed_work': 'Unemployed Took on Work',
    'coping_children_work': 'Sent Children to Work',
    'coping_children_out_school': 'Took Children Out of School',
    'coping_migration': 'Migrated',
    'coping_reduce_health_edu': 'Reduced Health/Education Spending',
    'coping_credit': 'Took on Credit',
    'coping_sell_agri_assets': 'Sold Agricultural Assets',
    'coping_sell_durables': 'Sold Durable Goods',
    'coping_sell_property': 'Sold Property',
    'coping_lease_land': 'Leased Out Land',
    'coping_sell_food_stock': 'Sold Food Stock',
    'coping_fishing': 'Relied on Fishing',
    'coping_sell_livestock': 'Sold Livestock',
    'coping_send_children_away': 'Sent Children Away',
    'coping_spiritual_activities': 'Engaged in Spiritual Activities',
    'coping_offseason_farming': 'Engaged in Off-season Farming',
    'coping_other_strategy': 'Used Other Strategy',
    'coping_no_strategy': 'Used No Strategy'
}

# Identify the columns to count based on the map
coping_cols_to_count = list(coping_name_map.keys())

# Calculate the frequencies (counting all non-zero values)
frequencies = {
    coping_name_map[col]: (merged_data[col] > 0).sum()
    for col in coping_cols_to_count
}

# Create a pandas Series from the dictionary
coping_frequencies = pd.Series(frequencies).sort_values(ascending=False)

# Filter out the zero values as requested
non_zero_frequencies = coping_frequencies[coping_frequencies > 0]

# Create a bar chart for the filtered data with the new names
plt.figure(figsize=(14, 9))
non_zero_frequencies.plot(kind='bar', color='darkcyan')
plt.title('Coping Mechanisms Adopted by Households Against Shocks in Burkina Faso', fontsize=18)
plt.xlabel('Coping Mechanisms', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.xticks(rotation=45, ha='right', fontsize=12)
plt.tight_layout()
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[234]:


# Construct Food Insecurity Index (0–4)
merged_data['food_insecurity_index'] = (
    merged_data[[
        'fs_worried_not_enough_food',
        'fs_skipped_meal_no_money',
        'fs_no_food_lack_money',
        'fs_no_food_all_day'
    ]]
    .fillna(0)
    .astype(int)
    .sum(axis=1)
)

# Create binary flag: 1 = food insecure (index ≥ 2), 0 = secure
merged_data['food_insecure'] = (merged_data['food_insecurity_index'] >= 2).astype(int)


# In[233]:


[col for col in merged_data.columns if 'food' in col.lower() or 'meal' in col.lower() or 'insec' in col.lower()]


# In[256]:


# Normalize region names
map_df['NAME_1'] = map_df['NAME_1'].str.strip().str.lower()
merged_data['region_clean'] = merged_data['region_clean'].str.strip().str.lower()


# In[257]:


# Compute weighted food insecurity prevalence per region
fs_region = (
    merged_data.groupby('region_clean', as_index=False)
    .apply(lambda d: pd.Series({
        'food_insecure_prev': np.average(d['food_insecure'], weights=d['household_weight'])
    }))
    .reset_index(drop=True)
)


# In[258]:


map_df_fs = map_df.merge(fs_region, left_on='NAME_1', right_on='region_clean', how='left')
map_df_fs


# In[260]:


fig, ax = plt.subplots(figsize=(12, 8))

map_df_fs.plot(
    column='food_insecure_prev',
    cmap='OrRd',
    linewidth=0.6,
    edgecolor='0.9',
    legend=True,
    ax=ax,
    missing_kwds={
        "color": "lightgrey",
        "label": "No data"
    }
)

# Add region labels
for idx, row in map_df_fs.iterrows():
    if row['geometry'].centroid.is_empty:
        continue
    plt.text(
        row['geometry'].centroid.x,
        row['geometry'].centroid.y,
        row['NAME_1'].title(),
        fontsize=9,
        ha='center',
        va='center',
        color='black'
    )

ax.set_title("Regional Food Insecurity Prevalence", fontsize=16, fontweight='bold')
ax.axis('off')
plt.tight_layout()
plt.show()


# In[261]:


bins = [0, 0.2, 0.35, 1]
labels = ['Low', 'Moderate', 'High']
map_df_fs['severity'] = pd.cut(map_df_fs['food_insecure_prev'], bins=bins, labels=labels)


# In[262]:


import matplotlib.pyplot as plt
import geopandas as gpd

# Set up the figure and axes
fig, ax = plt.subplots(1, 1, figsize=(12, 9))

# Plot choropleth with better style
map_df_fs.plot(
    column='food_insecure_prev',
    cmap='OrRd',
    linewidth=0.8,
    edgecolor='0.8',
    legend=True,
    legend_kwds={
        'label': "Food Insecurity Prevalence",
        'orientation': "vertical",
        'shrink': 0.6,
        'pad': 0.02
    },
    ax=ax
)

# Add region labels
for idx, row in map_df_fs.iterrows():
    if row['geometry'].centroid.is_empty:
        continue
    plt.annotate(
        text=row['NAME_1'],
        xy=(row['geometry'].centroid.x, row['geometry'].centroid.y),
        horizontalalignment='center',
        fontsize=9,
        color='black'
    )

# Title and clean aesthetics
ax.set_title("📍 Regional Food Insecurity in Burkina Faso", fontsize=18, weight='bold')
ax.axis('off')

# Tight layout for better spacing
plt.tight_layout()
plt.show()


# In[263]:


import matplotlib.pyplot as plt
import geopandas as gpd
import pandas as pd
import matplotlib.colors as mcolors

# Step 1: Define severity categories based on thresholds
def categorize_severity(val):
    if val < 0.20:
        return 'Low'
    elif val < 0.30:
        return 'Moderate'
    elif val < 0.40:
        return 'High'
    else:
        return 'Severe'

# Apply to create a severity column
map_df_fs['severity'] = map_df_fs['food_insecure_prev'].apply(categorize_severity)

# Step 2: Define color map for each severity level
severity_colors = {
    'Low': '#fee5d9',
    'Moderate': '#fcae91',
    'High': '#fb6a4a',
    'Severe': '#cb181d'
}
severity_cmap = mcolors.ListedColormap(severity_colors.values())

# Step 3: Plot the map
fig, ax = plt.subplots(1, 1, figsize=(12, 9))

# Plot using severity category
map_df_fs.plot(
    column='severity',
    cmap=severity_cmap,
    linewidth=0.8,
    edgecolor='0.9',
    ax=ax,
    legend=True,
    categorical=True,
    legend_kwds={
        'title': 'Food Insecurity Severity',
        'loc': 'lower left',
        'fontsize': 10,
        'title_fontsize': 12
    }
)

# Step 4: Add region labels
for idx, row in map_df_fs.iterrows():
    if row['geometry'].centroid.is_empty:
        continue
    plt.annotate(
        text=row['NAME_1'],
        xy=(row['geometry'].centroid.x, row['geometry'].centroid.y),
        horizontalalignment='center',
        fontsize=9,
        color='black'
    )

# Step 5: Aesthetics
ax.set_title("📍 Regional Food Insecurity Severity – Burkina Faso", fontsize=18, weight='bold')
ax.axis('off')
plt.tight_layout()
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[266]:


df = merged_data

# --- Data Preparation ---
#  Fill missing GPS coordinates with 0 instead of dropping rows.
#   Filling with 0 will place these points at (0,0) on the map, which is off the coast of Africa.
df_clean = df.fillna({
    'gps_latitude': 0,
    'gps_longitude': 0
})

# 2. Create a more descriptive column for coloring the points based on 'shock_environmental'.
#    This makes the map legend clearer (e.g., "Environmental Shock" vs. "No Environmental Shock").
#    Ensure 'shock_environmental' is treated as a numerical type before mapping.
#    Also, handle potential NaN values in 'shock_environmental' by filling them before mapping.
df_clean['Environmental Shock Occurred'] = df_clean['shock_environmental'].fillna(0).map({
    0: 'No Environmental Shock',
    1: 'Environmental Shock Reported'
})

# --- Interactive Map Creation ---
# Use plotly.express to create a scatter mapbox plot.
# This provides an interactive map.
fig = px.scatter_mapbox(
    df_clean,
    lat="gps_latitude",  # Latitude column for point placement
    lon="gps_longitude", # Longitude column for point placement
    color="Environmental Shock Occurred", # Color points based on whether an environmental shock occurred
    color_discrete_map={ # Define specific colors for clarity
        'No Environmental Shock': 'lightgray',
        'Environmental Shock Reported': 'red'
    },
    size_max=10,         # Maximum size for markers to prevent overcrowding
    zoom=5,              # Initial zoom level, suitable for viewing Burkina Faso
    mapbox_style="carto-positron", # A visually appealing base map style
    hover_name="region", # Display the 'region' name when hovering over a point
    hover_data={         # Include satellite data and other relevant info on hover
        "Environmental Shock Occurred": True, # Show the mapped shock status
        "rainfall": True,    # Show rainfall data
        "temperature": True, # Show temperature data
        "NDVI": True,        # Show Normalized Difference Vegetation Index
        "NDBI": True,        # Show Normalized Difference Built-up Index
        "Nightlight": True,  # Show Nightlight data (proxy for economic activity)
        "gps_latitude": False, # Hide raw latitude from hover info
        "gps_longitude": False # Hide raw longitude from hover info
    },
    title="Household Locations: Environmental Shocks and Satellite Data" # Title of the map
)

# --- Layout Customization ---
# Adjust map margins for better display and set a clear legend title.
fig.update_layout(
    margin={"r":0,"t":50,"l":0,"b":0}, # Set margins to zero for full map display
    legend_title_text='Environmental Shock Status' # Title for the color legend
)

# Display the generated interactive map.
fig.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[285]:


# Extract and clean WASH-related indicators from merged_data
# Based on UNICEF JMP WASH definitions (Improved = 1, Unimproved = 0)

# Define improved/safe response categories
improved_toilet = [
    "w.c. int. avec chasse d'eau", "w.c. int. chasse d'eau manuelle",
    "w.c. ext. avec chasse d'eau", "w.c. ext. chasse d'eau manuelle",
    "latrines vip (dallées, ventillées)", "latrines dallées simplement",
    "latrines sanplat (dallées, non couvertes)", "latrines ecosan (dallées, couvertes)"
]

safe_garbage = ["ramassage", "enterrées par le ménage"]

safe_wastewater = ["egout", "puisard (fosse moderne)", "fosse étanche"]

safe_child_feces = [
    "il a utilisé les toilettes/ latrines",
    "les selles ont été mises ou jetées dans les toilettes/latrines",
    "les selles ont été mises/ jetées dans les égouts ou la fosse septique",
    "les selles ont été enterrées"
]

safe_excreta = ["fosse septique", "fosse étanche", "compost", "egout"]

# Recode to binary: 1 = safe/improved, 0 = unsafe/unimproved
merged_data["toilet_improved"] = merged_data["toilet_type"].isin(improved_toilet).astype(int)
merged_data["garbage_safe"] = merged_data["garbage_disposal_method"].isin(safe_garbage).astype(int)
merged_data["wastewater_safe"] = merged_data["wastewater_disposal_method"].isin(safe_wastewater).astype(int)
merged_data["child_feces_safe"] = merged_data["child_feces_disposal_last_time"].isin(safe_child_feces).astype(int)
merged_data["excreta_safe"] = merged_data["excreta_disposal_method"].isin(safe_excreta).astype(int)

# Preview value counts for verification
for col in [
    "toilet_improved", "garbage_safe", "wastewater_safe", "child_feces_safe", "excreta_safe"
]:
    print(f"{col}:\n", merged_data[col].value_counts(dropna=False), "\n")


# In[286]:


# Clean and standardize region names for grouping
merged_data["region_clean"] = merged_data["region"].str.lower().str.strip()

# Group by region and compute average of each WASH binary variable
wash_region = (
    merged_data.groupby("region_clean")[
        ["toilet_improved", "garbage_safe", "wastewater_safe", "child_feces_safe", "excreta_safe"]
    ]
    .mean()
    .reset_index()
)

# Convert to percentage for easier interpretation
wash_region = wash_region * 100
wash_region["region_clean"] = merged_data.groupby("region_clean").size().index  # restore region names

# Compute composite WASH access score (mean of 5 indicators)
wash_region["wash_index"] = wash_region[
    ["toilet_improved", "garbage_safe", "wastewater_safe", "child_feces_safe", "excreta_safe"]
].mean(axis=1)


# In[287]:


wash_region


# In[290]:


# plot the WASH index/ scores by region

map_df["region_clean"] = map_df["region_clean"].str.lower().str.strip()
wash_region["region_clean"] = wash_region["region_clean"].str.lower().str.strip()
map_wash = map_df.merge(wash_region, on="region_clean", how="left")

fig, ax = plt.subplots(figsize=(10, 6))
map_wash.plot(
    column="wash_index",
    cmap="YlGnBu",
    linewidth=0.8,
    edgecolor="0.8",
    legend=True,
    ax=ax
)

for idx, row in map_wash.iterrows():
    if row["geometry"].centroid.is_valid:
        ax.annotate(
            text=row["region_clean"].title(),
            xy=(row["geometry"].centroid.x, row["geometry"].centroid.y),
            fontsize=8,
            color="black",
            ha="center"
        )

ax.set_title("WASH Access in Burkina Faso", fontsize=14, weight="bold")
ax.axis("off")
plt.tight_layout()
plt.show()


# In[291]:


ax.annotate(
    text=row["region_clean"].replace("-", " ").title(),
    xy=(row["geometry"].centroid.x, row["geometry"].centroid.y),
    fontsize=8,
    color="black",
    ha="center",
    va="center"
)


# In[293]:


for idx, row in map_wash.iterrows():
    if row["region_clean"] != "centre":  # skip 'centre' to avoid overlap
        ax.annotate(
            text=row["region_clean"].replace("-", " ").title(),
            xy=(row["geometry"].centroid.x, row["geometry"].centroid.y),
            fontsize=8,
            color="black",
            ha="center",
            va="center"
        )


# In[294]:


import matplotlib.pyplot as plt

map_df["region_clean"] = map_df["region_clean"].str.lower().str.strip()
wash_region["region_clean"] = wash_region["region_clean"].str.lower().str.strip()
map_wash = map_df.merge(wash_region, on="region_clean", how="left")

fig, ax = plt.subplots(figsize=(10, 6))
map_wash.plot(
    column="wash_index",
    cmap="YlGnBu",
    linewidth=0.8,
    edgecolor="0.8",
    legend=True,
    ax=ax
)

for idx, row in map_wash.iterrows():
    if row["geometry"].centroid.is_valid:
        ax.annotate(
            text=row["region_clean"].title(),
            xy=(row["geometry"].centroid.x, row["geometry"].centroid.y),
            fontsize=8,
            color="black",
            ha="center"
        )

ax.set_title("WASH Access in Burkina Faso", fontsize=14, weight="bold")
ax.axis("off")
plt.tight_layout()
plt.show()


# In[295]:


fig, ax = plt.subplots(figsize=(10, 6))
map_wash.plot(
    column="wash_index",
    cmap="YlGnBu",
    linewidth=0.8,
    edgecolor="0.8",
    legend=True,
    ax=ax
)

for idx, row in map_wash.iterrows():
    if row["region_clean"] == "plateau-central":
        # Slightly adjust label position upward
        x, y = row["geometry"].centroid.x, row["geometry"].centroid.y + 0.25
        ax.annotate("Plateau-Central", xy=(x, y), fontsize=8, ha="center", va="center")
    elif row["region_clean"] != "centre":
        ax.annotate(
            text=row["region_clean"].replace("-", " ").title(),
            xy=(row["geometry"].centroid.x, row["geometry"].centroid.y),
            fontsize=8,
            ha="center",
            va="center"
        )

ax.set_title("WASH Access in Burkina Faso", fontsize=14, weight="bold")
ax.axis("off")
plt.tight_layout()
plt.show()


# In[299]:


# Work on a copy
plot_df = map_wash.copy()

# Fill missing values with median
plot_df['wash_index'] = plot_df['wash_index'].fillna(plot_df['wash_index'].median())
plot_df['shock_social'] = plot_df['shock_social'].fillna(plot_df['shock_social'].median())

# Create quantiles
plot_df['wash_quantile'] = pd.qcut(plot_df['wash_index'], 3, labels=[0, 1, 2])
plot_df['shock_quantile'] = pd.qcut(plot_df['shock_social'], 3, labels=[0, 1, 2])

# Bivariate combination
plot_df['bivariate_class'] = plot_df['wash_quantile'].astype(str) + plot_df['shock_quantile'].astype(str)

# Define bivariate colormap
bivariate_colors = {
    '00': '#e8e8e8', '01': '#ace4e4', '02': '#5ac8c8',
    '10': '#dfb0d6', '11': '#a5add3', '12': '#5698b9',
    '20': '#be64ac', '21': '#8c62aa', '22': '#3b4994'
}

# Map colors
plot_df['color'] = plot_df['bivariate_class'].map(bivariate_colors)

# Plot
fig, ax = plt.subplots(1, 1, figsize=(10, 8))
plot_df.plot(color=plot_df['color'], edgecolor='white', linewidth=0.5, ax=ax)

ax.set_title('WASH Access × Social Shock Prevalence (Burkina Faso)', fontsize=14)
ax.axis('off')

# Manual legend
for key, color in bivariate_colors.items():
    ax.plot([], [], marker='s', linestyle='none', color=color, label=key)
ax.legend(
    title='WASH (↓) × Shocks (→)',
    loc='lower left',
    fontsize=8,
    title_fontsize=9,
    frameon=False
)

plt.tight_layout()
plt.show()


# In[300]:


# Plot map first (same as before)
fig, ax = plt.subplots(1, 1, figsize=(10, 8))
plot_df.plot(color=plot_df['color'], edgecolor='white', linewidth=0.5, ax=ax)

ax.set_title('WASH Access × Social Shock Prevalence (Burkina Faso)', fontsize=14)
ax.axis('off')

# Annotate regions using cleaned names
for idx, row in plot_df.iterrows():
    if row["region_clean"].lower() != "centre": 
        ax.annotate(
            text=row["region_clean"].replace("-", " ").title(),  # clean up hyphens if any
            xy=(row["geometry"].centroid.x, row["geometry"].centroid.y),
            ha='center',
            va='center',
            fontsize=8,
            color='black'
        )

# Legend
for key, color in bivariate_colors.items():
    ax.plot([], [], marker='s', linestyle='none', color=color, label=key)
ax.legend(
    title='WASH (↓) × Shocks (→)',
    loc='lower left',
    fontsize=8,
    title_fontsize=9,
    frameon=False
)

plt.tight_layout()
plt.show()


# In[ ]:





# In[303]:


# If these are all 0/1 binary indicators in merged_data
merged_data['wash_index'] = (
    merged_data[['toilet_improved', 'garbage_safe', 'wastewater_safe',
                 'child_feces_safe', 'excreta_safe']]
    .sum(axis=1)
)

# Normalize to 0–1 scale if needed
merged_data['wash_index'] = merged_data['wash_index'] / 5


# In[304]:


df_wash_all = merged_data[['region_clean', 'wash_index']].dropna()

# Plot
import seaborn as sns
import matplotlib.pyplot as plt

region_order = df_wash_all.groupby("region_clean")["wash_index"].median().sort_values().index.tolist()

plt.figure(figsize=(14, 6))
sns.violinplot(
    data=df_wash_all,
    x='region_clean',
    y='wash_index',
    inner='quartile',
    palette='coolwarm',
    order=region_order
)

plt.title('Distribution of WASH Access by Region (Burkina Faso)', fontsize=14)
plt.xticks(rotation=45, ha='right')
plt.xlabel('Region')
plt.ylabel('WASH Index (0–1)')
plt.tight_layout()
plt.show()


# In[308]:


# Pull region_clean from merged_data 
if 'wash' not in globals():
    wash = df.copy()  

# Merge region_clean using hhid as the key (or 'cluster' if that's more consistent)
wash = wash.merge(
    merged_data[['hhid', 'region_clean']],
    on='hhid',
    how='left'
)


# In[309]:


# Prepare plotting dataset
df_wash_all = wash[['region_clean', 'wash_index']].dropna()

# Sort regions by median WASH access
region_order = df_wash_all.groupby("region_clean")["wash_index"].median().sort_values().index.tolist()


# In[310]:


plt.figure(figsize=(14, 6))
sns.violinplot(
    data=df_wash_all,
    x='region_clean',
    y='wash_index',
    inner='quartile',
    palette='coolwarm',
    order=region_order
)

plt.title('Distribution of WASH Index by Region (Burkina Faso)', fontsize=14)
plt.xticks(rotation=45, ha='right')
plt.xlabel('Region')
plt.ylabel('WASH Index (0–1)')
plt.ylim(0, 1.05)
plt.tight_layout()
plt.show()


# In[323]:


# Sort regions by median WASH index
region_order = df_wash_all.groupby("region_clean")["wash_index"].median().sort_values().index.tolist()

plt.figure(figsize=(16, 6))
ax = sns.violinplot(
    data=df_wash_all,
    x='region_clean',
    y='wash_index',
    inner='quartile',
    palette='coolwarm',
    order=region_order,
    bw=0.2,
    cut=0
)

# National average line
national_avg = df_wash_all["wash_index"].mean()
plt.axhline(national_avg, linestyle='--', color='black', linewidth=1)
plt.text(-0.5, national_avg + 0.01, 'National Average', color='black')

# Remove Y-axis gridlines
ax.yaxis.grid(False)

# Formatting
plt.title('Distribution of WASH Index by Region (Burkina Faso)', fontsize=14)
plt.xlabel('Region')
plt.ylabel('WASH Index (0–1)')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()


# In[326]:


import seaborn as sns
import matplotlib.pyplot as plt

# Sort regions by median WASH index
region_order = merged_data.groupby("region_clean")["wash_index"].median().sort_values().index.tolist()

# replace 0/1 with labels for clarity
merged_data["urban_label"] = merged_data["urban"].map({0: "Rural", 1: "Urban"})

plt.figure(figsize=(14, 6))
sns.violinplot(
    data=merged_data,
    x='region_clean',
    y='wash_index',
    hue='urban_label',
    split=True,
    inner='quartile',
    palette='coolwarm',
    order=region_order
)

plt.title('Urban vs Rural WASH Index by Region (Burkina Faso)', fontsize=14)
plt.xticks(rotation=45, ha='right')
plt.xlabel('Region')
plt.ylabel('WASH Index (0–1)')
plt.tight_layout()
plt.show()


# In[327]:


import seaborn as sns
import matplotlib.pyplot as plt

# Sort regions by median WASH index
region_order = merged_data.groupby("region_clean")["wash_index"].median().sort_values().index.tolist()

# Label urban/rural
merged_data["urban_label"] = merged_data["urban"].map({0: "Rural", 1: "Urban"})

# Define better-looking colors: Blue for Rural, Orange for Urban
custom_palette = {"Rural": "#0072B2", "Urban": "#D55E00"}

plt.figure(figsize=(16, 6))
sns.violinplot(
    data=merged_data,
    x='region_clean',
    y='wash_index',
    hue='urban_label',
    split=True,
    inner='quartile',
    palette=custom_palette,
    order=region_order
)

plt.title('Urban vs Rural WASH Index by Region (Burkina Faso)', fontsize=14)
plt.xticks(rotation=45, ha='right')
plt.xlabel('Region')
plt.ylabel('WASH Index (0–1)')
plt.legend(title='', loc='upper right')
plt.tight_layout()
plt.show()


# In[328]:


import seaborn as sns
import matplotlib.pyplot as plt

# Define better colors: teal vs coral (flat & high contrast)
custom_palette = {"Rural": "#20B2AA", "Urban": "#FF6347"}  # LightSeaGreen and Tomato

# Ensure region order is by median WASH index
region_order = merged_data.groupby("region_clean")["wash_index"].median().sort_values().index.tolist()

# Label for plot
merged_data["urban_label"] = merged_data["urban"].map({0: "Rural", 1: "Urban"})

plt.figure(figsize=(18, 6))
sns.violinplot(
    data=merged_data,
    x='region_clean',
    y='wash_index',
    hue='urban_label',
    split=True,
    inner='quartile',
    palette=custom_palette,
    order=region_order
)

plt.title('Urban vs Rural WASH Index by Region (Burkina Faso)', fontsize=14)
plt.xticks(rotation=45, ha='right')
plt.xlabel('Region')
plt.ylabel('WASH Index (0–1)')
plt.legend(title='', loc='upper right')
plt.tight_layout()
plt.show()


# In[336]:


import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Point

# Fill missing values with mean
merged_data[["wash_index", "gps_latitude", "gps_longitude"]] = (
    merged_data[["wash_index", "gps_latitude", "gps_longitude"]]
    .fillna(merged_data[["wash_index", "gps_latitude", "gps_longitude"]].mean())
)

# Compute cluster-level averages
cluster_wash = (
    merged_data.groupby("cluster")[["wash_index", "gps_longitude", "gps_latitude"]]
    .mean()
    .reset_index()
)

# Convert to GeoDataFrame
cluster_wash["geometry"] = cluster_wash.apply(
    lambda row: Point(row["gps_longitude"], row["gps_latitude"]), axis=1
)
gdf_bubbles = gpd.GeoDataFrame(cluster_wash, geometry="geometry", crs="EPSG:4326")

# Plot bubble map
fig, ax = plt.subplots(figsize=(11, 9))
map_df.boundary.plot(ax=ax, color="gray", linewidth=0.5)
gdf_bubbles.plot(
    ax=ax,
    column="wash_index",
    markersize=gdf_bubbles["wash_index"] * 300,  # Adjust size as needed
    cmap="coolwarm",
    alpha=0.75,
    edgecolor="black",
    legend=True
)

ax.set_title("Cluster-Level WASH Access in Burkina Faso", fontsize=14)
ax.axis("off")
plt.tight_layout()
plt.show()


# In[ ]:





# In[ ]:




