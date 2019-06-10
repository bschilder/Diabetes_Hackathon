"""
Functions to preprocess CityHealth data for concatenating additional
data and modeling
"""

import pandas as pd

def preprocess_cityhealth(cityhealth_df):
    """
    Input: Raw CityHealth dataframe
    Return: NYC-specific data pivoted as
    """

    nyc_counties = ['Bronx County',
                    'Kings County',
                    'New York County',
                    'Queens County',
                    'Richmond County']

    cityhealth_df['State, County, Tract FIPS  (leading 0)'] = \
        cityhealth_df['State, County, Tract FIPS  (leading 0)'].astype(str)
    cityhealth_df[cityhealth_df['County Name'].isin(nyc_counties)]

    # Pivot variables
    cols = ['State, County, Tract FIPS  (leading 0)', 'metric_name', 'est',]
    clean_df = pd.pivot_table(nyc_health[cols],
                                values='est',
                                index='State, County, Tract FIPS  (leading 0)',
                                columns='metric_name')

    return clean_df
