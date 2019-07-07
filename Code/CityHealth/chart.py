import plotly.plotly as py
import plotly.graph_objs as go
import geopandas as gpd

class SpiderPlot():

    def __init__(self, cityhealth_data):
        self.cityhealth_data = cityhealth_data

    def plot(self, tract_id):
        """
        Create a spider plot for given variables
        """

        diabetes_vars = ['Frequent mental distress',
                        'Obesity',
                        'Physical inactivity',
                        'Preventive services',
                        'Children in Poverty',
                        ]

        # Normalized data between 0 and 1
        cityhealth_data = self.cityhealth_data.copy()
        cityhealth_data[diabetes_vars] = cityhealth_data[diabetes_vars].apply(normalize)


        cityhealth_data = cityhealth_data.loc[cityhealth_data['fips_state_county_tract_code'] == tract_id]
        cityhealth_data = cityhealth_data.fillna(0)
        r = cityhealth_data[diabetes_vars].values[0]

        col_names = cityhealth_data[diabetes_vars].columns

        # Create Spider plot
        data = [go.Scatterpolar(
          r = r,
          theta = col_names.insert(len(col_names), col_names[0]),
          fill = 'toself',
        )]

        layout = go.Layout(
          polar = dict(
            radialaxis = dict(
              visible = True,
              range = [0, 1]
            )
          ),
          showlegend = False
        )

        fig = go.Figure(data=data, layout=layout)
        return py.iplot(fig, filename = "radar-basic")

def normalize(row):

    return (row - row.min()) / (row.max() - row.min())

if __name__ == "__main__":

    cityhealth_data = gpd.read_file('data/cityhealth-newyork-190506/CHDB_data_tract_NY v5_3_withgeo.geojson')

    chart = SpiderPlot(cityhealth_data)
    chart.plot('36005000100')
