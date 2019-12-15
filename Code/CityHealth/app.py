import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
from dash.dependencies import Input, Output
import plotly_express as px
import json
from plotly.express.colors import sequential
from urllib.request import urlopen
import plotly.graph_objects as go
import geopandas as gpd

import Code.CityHealth.CityHealth as CH

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

df = pd.read_csv('https://plotly.github.io/datasets/country_indicators.csv')

available_indicators = df['Indicator Name'].unique()

# Variables
method="ridge_regression"

data = CH.preprocess_data()
model, coefs = CH.train_model(data, test_size=0.3, method=method, plot_weights=False)
weights = CH.feature_weights(data, coefs=coefs, weight_label=method + "_coef")
city_average = round(data["Diabetes"].mean(),1)

# tract_dict = [36085032300]
tract_dict = [dict(label=x, value=x) for x in data.index]
# Find mix/max for radar plot
min_max = CH.min_max_scores(data, weights)


# Load map dataset
geo = gpd.read_file('data/cityhealth-newyork-190506/CHDB_data_tract_NY v5_3_withgeo.geojson')
shapes = json.loads(geo['geometry'].apply(lambda x: x).to_json())

app.layout = html.Div([
    html.Div([

        html.Div([
            dcc.Dropdown(
                id='crossfilter-xaxis-column',
                options=[{'label': i, 'value': i} for i in available_indicators],
                value='Fertility rate, total (births per woman)'
            ),
            dcc.RadioItems(
                id='crossfilter-xaxis-type',
                options=[{'label': i, 'value': i} for i in ['Linear', 'Log']],
                value='Linear',
                labelStyle={'display': 'inline-block'}
            )
        ],
        style={'width': '49%', 'display': 'inline-block'}),

        html.Div([
            dcc.Dropdown(
                id='crossfilter-yaxis-column',
                options=[{'label': i, 'value': i} for i in available_indicators],
                value='Life expectancy at birth, total (years)'
            ),
            dcc.RadioItems(
                id='crossfilter-yaxis-type',
                options=[{'label': i, 'value': i} for i in ['Linear', 'Log']],
                value='Linear',
                labelStyle={'display': 'inline-block'}
            )
        ], style={'width': '49%', 'float': 'right', 'display': 'inline-block'})
    ], style={
        'borderBottom': 'thin lightgrey solid',
        'backgroundColor': 'rgb(250, 250, 250)',
        'padding': '10px 5px'
    }),


    # Graphing radial plot
    ## Tract dropdown
    html.Div([
        html.P([dcc.Dropdown(id="tract_id", options=tract_dict, value=36085032300)]),
    ]),

    ## Map
    html.Div([
            dcc.Graph(
                id='tract_map',
                hoverData={'points': [{'location': '0'}]}
            )
            ],
            style={'width': '49%', 'display': 'inline-block'}
            ),

    ## Polar plot
    html.Div([
        dcc.Loading([
            dcc.Graph(id="graph", style={"width": "75%", "display": "inline-block"})
        ]),
    ],
        style={'width': '49%', 'display': 'inline-block', 'padding': '0 20', 'float': 'right'}
    ),

    # Scatter plot
    html.Div([
        dcc.Graph(
            id='crossfilter-indicator-scatter',
            hoverData={'points': [{'customdata': 'Japan'}]}
        )
    ], style={'width': '49%', 'display': 'inline-block', 'padding': '0 20'}),
    html.Div([
        dcc.Graph(id='x-time-series'),
        dcc.Graph(id='y-time-series'),
    ], style={'display': 'inline-block', 'width': '49%'}),
])


@app.callback(
    dash.dependencies.Output('crossfilter-indicator-scatter', 'figure'),
    [dash.dependencies.Input('crossfilter-xaxis-column', 'value'),
     dash.dependencies.Input('crossfilter-yaxis-column', 'value'),
     dash.dependencies.Input('crossfilter-xaxis-type', 'value'),
     dash.dependencies.Input('crossfilter-yaxis-type', 'value'),
     # dash.dependencies.Input('crossfilter-year--slider', 'value')
     ])
def update_graph(xaxis_column_name, yaxis_column_name,
                 xaxis_type, yaxis_type,
                 # year_value
                 ):
    dff = df[df['Year'] == 2002]

    return {
        'data': [dict(
            x=dff[dff['Indicator Name'] == xaxis_column_name]['Value'],
            y=dff[dff['Indicator Name'] == yaxis_column_name]['Value'],
            text=dff[dff['Indicator Name'] == yaxis_column_name]['Country Name'],
            customdata=dff[dff['Indicator Name'] == yaxis_column_name]['Country Name'],
            mode='markers',
            marker={
                'size': 15,
                'opacity': 0.5,
                'line': {'width': 0.5, 'color': 'white'}
            }
        )],
        'layout': dict(
            xaxis={
                'title': xaxis_column_name,
                'type': 'linear' if xaxis_type == 'Linear' else 'log'
            },
            yaxis={
                'title': yaxis_column_name,
                'type': 'linear' if yaxis_type == 'Linear' else 'log'
            },
            margin={'l': 40, 'b': 30, 't': 10, 'r': 0},
            height=450,
            hovermode='closest'
        )
    }

# Function to make interactive Map
@app.callback(Output("tract_map", "figure"), [Input("tract_id", "value")])
def map_tracts(tract_id):
    ''' '''
    fig = go.Figure(go.Choroplethmapbox(geojson=shapes,
                                        locations = [str(i) for i in geo.index.values],
                                        z=geo['Diabetes'],
                                        colorscale=sequential.PuRd, zmin=0, zmax=12,
                                        marker_opacity=0.5, marker_line_width=0))
    fig.update_layout(
        mapbox_style="carto-positron",
                      mapbox_zoom=10, mapbox_center = {"lat": 40.7410224, "lon": -73.9939661})
    fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})

    return fig

# Function to make polar radial chart
dropdowns = ["tract_id"]
@app.callback(Output("graph", "figure"),
    [
        Input('tract_map', "hoverData"),
        # Input("tract_id", "value")
    ])
def make_figure(tract_map, n_factors=6):
    ''' '''

    location = int(tract_map['points'][0]['location'])
    # tract = [shape['id'] for shape in shapes['features'] if shape['id'] == location]
    tract_id = int(geo.iloc[location]['fips_state_county_tract_code'])
    print(location, tract_id)

    polar_data = CH.prepare_polar(data, weights, stcotr_fips=tract_id, n_factors=n_factors)
    predicted, actual = CH.predict_value(model, data, stcotr_fips=tract_id, label_var="Diabetes")
    title = "% of Community with Diabetes:<br>"+\
            "  Predicted = " + str(round(predicted,1)) + " %<br>"+ \
            "  Actual = " + str(actual) + " %<br>"+\
            "  City Average = "+str(city_average)+" %"
    return px.line_polar(
        polar_data,
        r="Risk Score",
        theta="metric_name",
        # color="Risk Score",
        # template="plotly_dark",
        height=700,
    ).update_traces(
        fill='toself',
        fillcolor="rgba(0, 255, 238, .9)",#"rgba(0,0,255, .9)",
        mode = "lines+markers",
        line_color = "magenta",
        marker=dict(
            color="magenta",
            symbol="circle",
            size=12 )
    ).update_layout(
    title = dict(
        text = title,
        x = 1,
        y = .925),
    font_size = 15,
    showlegend = False,
    polar = dict(
      bgcolor = "rgba(0,0,0, .85)",#"""rgb(223, 223, 223)",
      angularaxis = dict(
        linewidth = 3,
        showline=True,
        linecolor='rgba(0,0,0, .85)',
        color="black"
      ), radialaxis = dict(
            side = "clockwise",
            showline = True,
            linewidth = 2,
            gridcolor = "white",
            gridwidth = 2,
            color = "magenta",
            visible =True,
            range=[min_max["min"], min_max["max"]]
          )
    ),
    paper_bgcolor = "rgba(0,0,0,0)"
)

def create_time_series(dff, axis_type, title):
    return {
        'data': [dict(
            x=dff['Year'],
            y=dff['Value'],
            mode='lines+markers'
        )],
        'layout': {
            'height': 225,
            'margin': {'l': 20, 'b': 30, 'r': 10, 't': 10},
            'annotations': [{
                'x': 0, 'y': 0.85, 'xanchor': 'left', 'yanchor': 'bottom',
                'xref': 'paper', 'yref': 'paper', 'showarrow': False,
                'align': 'left', 'bgcolor': 'rgba(255, 255, 255, 0.5)',
                'text': title
            }],
            'yaxis': {'type': 'linear' if axis_type == 'Linear' else 'log'},
            'xaxis': {'showgrid': False}
        }
    }


@app.callback(
    dash.dependencies.Output('x-time-series', 'figure'),
    [dash.dependencies.Input('crossfilter-indicator-scatter', 'hoverData'),
     dash.dependencies.Input('crossfilter-xaxis-column', 'value'),
     dash.dependencies.Input('crossfilter-xaxis-type', 'value')])
def update_y_timeseries(hoverData, xaxis_column_name, axis_type):
    country_name = hoverData['points'][0]['customdata']
    print(hoverData, country_name)
    dff = df[df['Country Name'] == country_name]
    dff = dff[dff['Indicator Name'] == xaxis_column_name]
    title = '<b>{}</b><br>{}'.format(country_name, xaxis_column_name)
    return create_time_series(dff, axis_type, title)


@app.callback(
    dash.dependencies.Output('y-time-series', 'figure'),
    [dash.dependencies.Input('crossfilter-indicator-scatter', 'hoverData'),
     dash.dependencies.Input('crossfilter-yaxis-column', 'value'),
     dash.dependencies.Input('crossfilter-yaxis-type', 'value')])
def update_x_timeseries(hoverData, yaxis_column_name, axis_type):
    dff = df[df['Country Name'] == hoverData['points'][0]['customdata']]
    dff = dff[dff['Indicator Name'] == yaxis_column_name]
    return create_time_series(dff, axis_type, yaxis_column_name)


if __name__ == '__main__':
    app.run_server(port=8080, debug=True)
