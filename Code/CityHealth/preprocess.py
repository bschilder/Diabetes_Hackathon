"""
Functions to preprocess CityHealth data for concatenating additional
data and modeling
"""

import pandas as pd
import geopandas as gpd

def preprocess_cityhealth(cityhealth_df, path_geo_tract_data):
    """
    Input: Raw CityHealth dataframe
    Return: Pivoted NYC-specific data
    """

    nyc_counties = ['Bronx County',
                    'Kings County',
                    'New York County',
                    'Queens County',
                    'Richmond County']

    cityhealth_df['State, County, Tract FIPS  (leading 0)'] = \
        cityhealth_df['State, County, Tract FIPS  (leading 0)'].astype(str)
    cityhealth_df[cityhealth_df['County Name'].isin(nyc_counties)]

    # Change County code to 085 for Tract 990100 due to data error
    cityhealth_df.loc[cityhealth_df['State, County, Tract FIPS  (leading 0)'].apply(lambda x: x[-6:]) == '990100', 'CountyFIPS (leading 0)'] = 85
    cityhealth_df.loc[cityhealth_df['State, County, Tract FIPS  (leading 0)'].apply(lambda x: x[-6:]) == '990100', 'State, County, Tract FIPS  (leading 0)'] = '36085990100'



    # Pivot variables
    cols = ['State, County, Tract FIPS  (leading 0)', 'metric_name', 'est',]
    clean_df = pd.pivot_table(cityhealth_df[cols],
                                values='est',
                                index='State, County, Tract FIPS  (leading 0)',
                                columns='metric_name')

    # # Merge in census tract geometry data
    clean_df = add_census_tracts(clean_df, path_geo_tract_data)

    return clean_df

def create_fip_county_code(boro_code):

    boro_to_county_code = {
                            '1': '061',
                            '2': '005',
                            '3': '047',
                            '4': '081',
                            '5': '085',
                        }

    county_code = boro_to_county_code[boro_code]
    return str(county_code)

def create_fip_state_county_tract_code(row):
    code = '36' + row['fips_county_code'] + row['ct2010']
    return code

def add_census_tracts(cityhealth_df, path_geo_tract_data):
    """
    Adds in NYC census tract geometry data
    """

    data = cityhealth_df.reset_index()
    geo = gpd.read_file(path_geo_tract_data)

    geo['fips_county_code'] = geo['boro_code'].apply(create_fip_county_code)
    geo['fips_state_county_tract_code'] = geo.apply(lambda row: create_fip_state_county_tract_code(row), axis=1)

    # Merge NYC health data and geo tract data
    data = pd.merge(data,
                    geo[['geometry', 'fips_state_county_tract_code']],
                    how='left',
                    left_on='State, County, Tract FIPS  (leading 0)',
                    right_on='fips_state_county_tract_code',
                    right_index=False
                    ).set_index('State, County, Tract FIPS  (leading 0)')

    return data
