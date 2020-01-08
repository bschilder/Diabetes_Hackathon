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

import Code.cityhealth.CityHealth as CH

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

df = pd.read_csv('https://plotly.github.io/datasets/country_indicators.csv')

static_image_route = '/images/tracts/'

available_indicators = df['Indicator Name'].unique()

# Variables
method="ridge_regression"

data = CH.preprocess_data()
model, coefs = CH.train_model(data, test_size=0.3, method=method, plot_weights=False)
weights = CH.feature_weights(data, coefs=coefs, weight_label=method + "_coef")
city_average = round(data["Diabetes"].mean(),1)

tract_dict = [dict(label=x, value=x) for x in data.index]
# Find mix/max for radar plot
min_max = CH.min_max_scores(data, weights)

# Load map dataset
geo = gpd.read_file('data/cityhealth-newyork-190506/CHDB_data_tract_NY v5_3_withgeo_v2.geojson')
shapes = json.loads(geo['geometry'].apply(lambda x: x).to_json())
# Create text field in geo for hover information
geo['hovertext'] = 'Diabetes Rate: ' + geo['Diabetes'].astype(str) + '%' + '<br>' + \
    geo['Name'].astype(str) + '<br>' + \
    'FIP Tract: ' + geo['fips_state'].astype(str)

app.layout = html.Div([
    # Template section
    # html.Div([
        # html.Div([
        #     dcc.Dropdown(
        #         id='crossfilter-xaxis-column',
        #         options=[{'label': i, 'value': i} for i in available_indicators],
        #         value='Fertility rate, total (births per woman)'
        #     ),
        #     dcc.RadioItems(
        #         id='crossfilter-xaxis-type',
        #         options=[{'label': i, 'value': i} for i in ['Linear', 'Log']],
        #         value='Linear',
        #         labelStyle={'display': 'inline-block'}
        #     )
        # ],
        # style={'width': '49%', 'display': 'inline-block'}),

        # html.Div([
        #     dcc.Dropdown(
        #         id='crossfilter-yaxis-column',
        #         options=[{'label': i, 'value': i} for i in available_indicators],
        #         value='Life expectancy at birth, total (years)'
        #     ),
        #     dcc.RadioItems(
        #         id='crossfilter-yaxis-type',
        #         options=[{'label': i, 'value': i} for i in ['Linear', 'Log']],
        #         value='Linear',
        #         labelStyle={'display': 'inline-block'}
        #     )
        # ], style={'width': '49%', 'float': 'right', 'display': 'inline-block'})
    # ], style={
    #     'borderBottom': 'thin lightgrey solid',
    #     'backgroundColor': 'rgb(250, 250, 250)',
    #     'padding': '10px 5px'
    # }),

    html.H1('Diabetes Hackathon'),
    dcc.Tabs([
        dcc.Tab(label='App', id="App", children=[
        ## Tract dropdown
        html.Div([
            html.P([dcc.Dropdown(id="tract_id", options=tract_dict, value=36085032300)]),
        ]),

        ## Tract details
        html.Div([
                html.H2('Community View'),
                # html.Div(id='community_view_tract_shape'),
                html.Div([
                    html.Img(id='image', style={'height':'50%', 'width':'50%'})
                ]),
                html.Div([
                    html.H3(id='community_view_tract_id'),
                    html.H3(id='community_view_diabetes_rate'),
                ], style={'padding': '20 20'})
                ],
                style={'width': '29%', 'display': 'inline-block'}
                ),

        ## Map
        html.Div([
                dcc.Graph(
                    id='tract_map',
                    clickData={'points': [{'location': '1'}]}
                )
                ],
                style={'width': '69%', 'display': 'inline-block', 'float': 'right'}
                ),

        ## Polar plot
        html.Div([
            dcc.Loading([
                dcc.Graph(id="graph", style={"width": "75%", "display": "inline-block"})
            ]),
        ],
            # style={'width': '49%', 'display': 'inline-block', 'padding': '0 20', 'float': 'right'}
        ),
            ]), # END TAB1
    dcc.Tab(label='About', id='About', children=[
        html.Div(style={'margin':'100px'}, children=[
            html.Div(className="about_div", children=[
                html.H2('Why was this app made?'),
                dcc.Markdown('''
               - Type II Diabetes (T2D) is one of the most prevalent health conditions
                   in industrialized nations. In America alone, #### in #### people suffer with T2D
                   and is projected to reach #### by ####. T2D increases the risk for cardiovascular
                   disease by 2-4 times, which is by far the leading cause of death in industrialized nations.
                   The overall goal of this app is empower users of all kinds to better understand the various
                   factors underlying risk for T2D in many diverse communities across America.
               -  This app is ultimately meant to be used by anyone who has an interest in exploring T2D risk factors
                   across a wide range of communities. Some specific use cases could include (but are by no means limited to):
                   1. Public health officials
                   2. City planners
                   3. Hospital systems
                   4. Community health advocates and activists
                   5. Residents who are interested in learning about their own communities.
               ''')
            ]),
            html.Div(className="about_div", children=[
                html.H2('What can this app do?'),
                dcc.Markdown('''
                +	This app makes use of the large amount of publicly available data and analyzes it using state-of-art AI/machine learning algorithms.
                +	Through this approach, it can:
                    1.	Visualize communities at high risk for T2D.
                    2.	Identify the risk factors that are most likely driving T2D in each specific
                        community (i.e. provide a community-specific “risk profile”).
                    3.	Offer a list of recommendations and resources that our algorithm predicts are most
                        likely to have an impact on the health of that community. Each set of recommendations is
                        specifically tailored to each community based on their risk profile.
                ''')
            ]),
            html.Div(className="about_div", children=[
                html.H2('Where does this data come from?'),
                dcc.Markdown('''
                + [CityHealth Dashboard](https://www.cityhealthdashboard.com)
                + [IPUMS](https://ipums.org)
                    - [Health Surveys](https://healthsurveys.ipums.org)
                    - [NHGIS](https://www.nhgis.org)
                + Google Maps
                ''')
            ]),
            html.Div(className="about_div", children=[
                html.H2('Who are you?'),
                dcc.Markdown('''
                    + Matt is a magical unicorn that prances across the skies of the New York City skyline.
                     Is that a rainbow-colored sunset? No, it’s Matt.
                     He lives in Chinatown, New York City.
                        - [LinkedIn](https://www.linkedin.com/in/engmatthew)
                        - [Twitter](https://twitter.com/m3ngineer)
                        - [GitHub](https://github.com/m3ngineer)
                    + Brian is a bioinformatician currently at the Mount Sinai.
                     His research revolves around neuroscience, genomics, human evolution, machine learning, and more!
                     For some reason he likes to run in his free time.
                     Also, he lives in Harlem, New York City.
                        - [LinkedIn](https://www.linkedin.com/in/brian-schilder)
                        - [Twitter](https://twitter.com/BMSchilder)
                        - [GitHub](https://github.com/bschilder)
                ''')
            ]),
            html.Div(className="about_div", children=[
                html.H2('How exactly do you predict diabetes risk?'),
                dcc.Markdown('''
                    Programming and unicorn sparkles.
                ''')
            ])
        ])
    ])
    ])

])





# Update tract detail summary section
outputs = ['community_view_tract_id', 'community_view_diabetes_rate']
@app.callback(
    [Output(output, 'children') for output in outputs],
    [Input(component_id='tract_map', component_property='hoverData')]
)
def update_tract_detail_summary(tract_map):
    ''' Update the tract ID for the summary of the tract
    param: tract_map
    return: div
    '''

    try:
        location = int(tract_map['points'][0]['location'])
        tract_id = int(geo.iloc[location]['fips_state_county_tract_code'])
        diabetes_rate = geo.iloc[location]['Diabetes']
        # zipcode =
    except:
        tract_id, diabetes_rate = 'None', 'None'

    return 'Tract: {}'.format(tract_id),'Diabetes rate: {}%'.format(diabetes_rate)

# Show chosen tract shape in Community View
@app.callback(
    dash.dependencies.Output('image', 'src'),
    [Input(component_id='tract_map', component_property='hoverData')])
def update_image_src(tract_map):
    try:
        location = int(tract_map['points'][0]['location'])
        fip = geo.iloc[location]['fips_state_county_tract_code']
    except:
        fip = '36005000100'
    # return static_image_route + fip + '.png'
    return 'https://github.com/bschilder/Diabetes_Hackathon/blob/master/images/tracts/{}.png?raw=True'.format(fip)

# Add a static image route that serves images from desktop
# Be *very* careful here - you don't want to serve arbitrary files
# from your computer or server
@app.server.route('{}<image_path>.png'.format(static_image_route))
def serve_image(image_path):
    image_name = '{}.png'.format(image_path)
    # if image_name not in list_of_images:
    #     raise Exception('"{}" is excluded from the allowed static files'.format(image_path))
    return flask.send_from_directory(image_directory, image_name)

# Function to make interactive Map
@app.callback(Output("tract_map", "figure"), [Input("tract_id", "value")])
def map_tracts(tract_id):
    ''' '''
    fig = go.Figure(go.Choroplethmapbox(geojson=shapes,
            locations = [str(i) for i in geo.index.values],
            z=geo['Diabetes'],
            colorscale=sequential.PuRd, zmin=0, zmax=12,
            marker_opacity=0.5, marker_line_width=0,
            text = geo['hovertext'],
            )
        )

    fig.update_layout(
            mapbox_style="carto-positron",
            mapbox_zoom=10,
            mapbox_center = {"lat": 40.7410224, "lon": -73.9939661},
            margin={"r":0,"t":0,"l":0,"b":0}
        )

    return fig

# Function to make polar radial chart
dropdowns = ["tract_id"]
@app.callback(
        Output("graph", "figure"),
        [Input('tract_map', "clickData")]
    )
def generate_spider_plot(tract_map, n_factors=6):
    ''' '''

    location = int(tract_map['points'][0]['location'])
    tract_id = int(geo.iloc[location]['fips_state_county_tract_code'])

    polar_data = CH.prepare_polar(data, weights, stcotr_fips=tract_id, n_factors=n_factors)
    predicted, actual = CH.predict_value(model, data, stcotr_fips=tract_id, y_var="Diabetes")
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

if __name__ == '__main__':
    app.run_server(port=8081, debug=True)
