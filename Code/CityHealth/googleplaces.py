import random
import requests
import geopandas as gpd
import pandas as pd

from shapely.geometry import Polygon, Point
from time import sleep

import conf

class GooglePlacesSummary():

    def __init__(self, tract_data):
        self.endpoint = "https://maps.googleapis.com/maps/api/place/nearbysearch/json"
        self.key = conf.GOOGLE_MAPS_API_KEY
        self.tract_data = tract_data
        self.places_data = self.create_empty_dataframe()

    def create_empty_dataframe(self):
        '''
        Initialize empty dataframe for storing returned data from Places API
        '''

        places_data = gpd.GeoDataFrame(columns=['geometry', 'icon', 'id', 'name', 'opening_hours', 'photos', 'place_id',
           'plus_code', 'price_level', 'rating', 'reference', 'scope', 'types',
           'user_ratings_total', 'vicinity'])

         return places_data

    def request_nearby_places(self, location, place_type, radius=400):

        params = {
                    'key': self.key,
                    'location': location,
                    'radius': radius,
                    'type': place_type,

                }

        r = requests.get(self.endpoint, params=params)
        df = pd.DataFrame(r.json()['results'])

        results = pd.DataFrame(columns=['geometry', 'icon', 'id', 'name', 'opening_hours', 'photos', 'place_id',
           'plus_code', 'price_level', 'rating', 'reference', 'scope', 'types',
           'user_ratings_total', 'vicinity'])

        results = pd.concat([results, df])

        i = 0
        # If more than 20 results found continue making calls for 3 pages (API maxes at 60 results)
        while 'next_page_token' in r.json().keys() and i < 5:
            i += 1
            sleep(5)
            params['pagetoken'] = r.json()['next_page_token']
            r = requests.get(url, params=params)
            df = pd.DataFrame(r.json()['results'])
            results = pd.concat([results, df], ignore_index=True)

        return results

    def get_random_point_in_polygon(self, poly):
         minx, miny, maxx, maxy = poly.bounds
         while True:
             p = Point(random.uniform(minx, maxx), random.uniform(miny, maxy))
             if poly.contains(p):
                 return p

    def get_tract_data(self, tract_polygon, place_type, num_samples=3):
        '''
        Sample a random point in a the geometry of a tract num_samples times
        '''

        tract_data = gpd.GeoDataFrame(columns=['geometry', 'icon', 'id', 'name', 'opening_hours', 'photos', 'place_id',
           'plus_code', 'price_level', 'rating', 'reference', 'scope', 'types',
           'user_ratings_total', 'vicinity'])

        for i in range(num_samples):
            sleep(10)
            point_in_poly = self.get_random_point_in_polygon(tract_polygon)
            x, y = point_in_poly.x, point_in_poly.y
            coords = '{}, {}'.format(y, x)

            data_for_single_point = self.request_nearby_places(location=coords, place_type=place_type, radius=800)

            tract_data = pd.concat([tract_data, data_for_single_point])

        return tract_data.loc[~tract_data['id'].duplicated()]

    def get_tract_summary_data(self):

        # Create a new container to store summary data
        results = pd.DataFrame(columns=['fips_state_county_tract_code'])

        for i, row in self.tract_data.iterrows():
            geometry = row['geometry']
            tract_id = row['fips_state_county_tract_code']
            print(tract_id)

            results.loc[i, 'fips_state_county_tract_code'] = tract_id
    #         results.loc[i, 'geometry'] = geometry

            # Get summary statistics

            places = ['supermarket',
                               'restaurant',
                               'school',
                               'subway_station',
                               'taxi_stand',
                               'train_station',
                               'transit_stand',
                               'hospital',
                               'police',
                               'park',
                               'parking',
                               'meal_delivery',
                               'meal_takeaway',
                               'liquor_store',
                               'bar',
                               'bus_station',
                               'cafe',
                               'car_wash',
                               'car_dealear',
                               'car_repair',
                               'convenience_store',
                               'doctor',
                               'fire_station',
                               'gas_station',
                               'hospital',
                               'gym',
                              ]
            places = ['supermarket']

            for place_type in places:
                tract_data = self.get_tract_data(geometry, place_type=place_type, num_samples=3)
                results.loc[i, place_type] = tract_data.shape[0]

        return results

if __name__ == "__main__":

    geo = gpd.read_file('data/cityhealth-newyork-190506/CHDB_data_tract_NY v5_3_withgeo.geojson')

    places_summary = GooglePlacesSummary(tract_data=geo[:5])
    summary = places_summary.get_tract_summary_data()
    print(summary)
