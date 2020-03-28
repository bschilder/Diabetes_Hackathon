
import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
from dash.dependencies import Input, Output
import plotly_express as px

from urllib.request import urlopen
from plotly.express.colors import sequential
import plotly.graph_objects as go

import Code.cityhealth.CityHealth as CH

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

# df = pd.read_csv('https://plotly.github.io/datasets/country_indicators.csv')
# available_indicators = df['Indicator Name'].unique()
static_image_route = '/images/tracts/'

def recompute():
    import pickle
    import json
    import geopandas as gpd

    data = CH.preprocess_data()
=======
    data = CH.preprocess_data()
>>>>>>> parent of d7de6f2... Update CityHealth data to V8. Update ML model
=======
    data = CH.preprocess_data()
>>>>>>> 92c02275586223aa59d8a725136525e5b967e347
    method = "ridge_regression"
    model, coefs = CH.train_model(data, test_size=0.3, method=method, plot_weights=False)
    nyc_data = CH.preprocess_data(data_path=data_path, NYC_only=True, impute=False, drop_na=False)

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
        geo['neighborhood_name'].astype(str) + '<br>' + \
        'FIP Tract: ' + geo['fips_state_county_tract_code'].astype(str)
    # Save all objects as a dict in a pickle
    pickle.dump({'data':nyc_data,
                 'model':model,
                 'method':method,
                 'coefs':coefs,
                 'weights':weights,
                 'city_average':city_average,
                 'tract_dict':tract_dict,
                 'min_max':min_max,
                 'geo':geo,
                 'shapes':shapes}, open("./data/saved_objects.pickle", 'wb'))

def import_precomputed(pickle_path="./data/saved_objects.pickle"):
    import pickle
    p = pickle.load(open(pickle_path, "rb"))
    return [p[x] for x in p]

data, model, method, coefs, weights, city_average, tract_dict, min_max, geo, shapes = import_precomputed(pickle_path="./data/saved_objects.pickle")
n_factors_dict =[{'label':x, 'value':x} for x in range(1,len(coefs))]

app.layout = html.Div([

    html.H1('Making health local'),
    dcc.Tabs([
        dcc.Tab(label='App', id="App", children=[
        ## Tract dropdown
        html.Div([
            html.P([dcc.Dropdown(id="tract_id", options=tract_dict, value=36085032300)]),
        ]),

        ## Tract details
        html.Div([
            html.Div([
                html.H2('Community View', style={'text-align': 'center'}),
                html.Div([
                    html.Img(id='image', style={'height':'50%', 'width':'50%', 'display': 'block', 'margin-left': 'auto', 'margin-right': 'auto'})
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
                    clickData={'points': [{'location': '1'}]},
                    style={'height': '100%'}
                )
                ],
                style={'width': '69%', 'display': 'inline-block', 'float': 'right'}
                ),
        ], style={'height': '100vh'}),

        ## Polar plot
        html.Div([
            html.Div([
                html.Div([
                    html.Div([
                                html.H4(id='radial_predicted'),
                                html.H4(id='radial_actual'),
                                html.H4(id='radial_city_average'),
                                html.H5('N factors:'),
                                html.P([dcc.Dropdown(id="n_factors", options=n_factors_dict, value=6)]),
                             ], style={'flex': '1'} ),
                    html.Div([]),
                         ],style={'display': 'flex', 'justify-content': 'center', 'align-items': 'center', 'height': '100%'},
                        )
                    ], style={'width': '29%', 'display': 'inline-block', 'height': '700px'}
                ),
            html.Div([
                dcc.Loading([
                            dcc.Graph(id="graph", style={"width": "90%", "display": "inline-block"})
                            ]),
                    ],
                style={'width': '69%', 'display': 'inline-block', 'float': 'right'}
                # style={'float': 'right'}
                ),
            ]),
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

                   In addition to clinical causes, risk of T2D has been widely associated with many
                   socialeconomic factors . Social determinants of health such as education,
                   employment, access to healthy food options, and health literacy may contribute to
                   an individual's risk level.

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
                        community (i.e. provide a community-specific \\"risk profile\\"").
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

                ''')
            ]),
            html.Div(className="about_div", children=[
                html.H2('How exactly do you predict diabetes risk?'),
                dcc.Markdown('''
                    Research has shown that the economic and social inequalities can contribute
                    to poorer health outcomes. Communities with poorer access to healthy food options,
                    transportation, medical resources, and educational resources are more likely
                    to have T2D. Many providers and healthcare workers do not have support tools
                    to screen for an identify social determinants of T2D.

                    Using neighborhood-level data, we have constructed an app that identifies
                    the contribution of various determinants for specific communities.
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
    return 'https://github.com/m3ngineer/Diabetes_Hackathon_Data/blob/master/images/tracts/{}.png?raw=True'.format(fip)

# Add a static image route that serves images from desktop
# Be *very* careful here - you don't want to serve arbitrary files
# from your computer or server
@app.server.route('{}<image_path>.png'.format(static_image_route))
def serve_image(image_path):
    image_name = '{}.png'.format(image_path)
    # if image_name not in list_of_images:
    #     raise Exception('"{}" is excluded from the allowed static files'.format(image_path))
    return flask.send_from_directory(image_directory, image_name)

# # Function to make interactive Map
@app.callback(Output("tract_map", "figure"), [Input("tract_id", "value")])
def map_tracts(tract_id):
    ''' Creates geographic heatmap of diabetes rates binned into quartiles '''

    bin_labels = [0, .25, .5, .75, 1]
    geo['Diabetes_bin'], bins_perc_diabetes = pd.qcut(geo['Diabetes'],
                              q=5,
                              retbins=True,
                              labels=bin_labels)
    # print(geo.columns)
    # import json
    # shapes = json.loads(geo['geometry'][:2].apply(lambda x: x).to_json())
    # print(geo[geo['Diabetes'].isnull()])
    # print('features')
    # print(len(shapes['features']))
    # print(geo[geo['geometry'].isnull()])
    # fig = go.Figure(go.Choroplethmapbox(geojson=shapes,
    #         locations = [str(i) for i in geo.index.values],
    #         z=geo['Diabetes_bin'],
    #         # z=geo['Diabetes'],
    #         colorscale=sequential.Blues,
    #         colorscale=[(0, "rgb(222,235,247)"),
    #                     (.25, "rgb(198,219,239)"),
    #                     (.5, "rgb(107,174,214)"),
    #                     (.75, "rgb(33,113,181)"),
    #                     (1, "rgb(8,48,107)")],
    #         zmin=0, zmax=1,
    #         marker_opacity=0.75, marker_line_width=0,
    #         text = geo['hovertext'],
    #         colorbar=dict(
    #             title="Diabetes Rate",
    #             tickvals=[0, 0.2, 0.4, 0.6, 0.8, 1],
    #             ticktext=['{}%'.format(label) for label in list(bins_perc_diabetes.astype(str))],
    #             )
    #         )
    #     )

# Function to make polar radial chart
dropdowns = ["tract_id"]
@app.callback(
        [Output("graph", "figure"),
        Output('radial_predicted', 'children'),
        Output('radial_actual', 'children'),
        Output('radial_city_average', 'children')],
        [Input('tract_map', "clickData"),
         Input('n_factors','value')]
    )
def generate_spider_plot(tract_map, n_factors=6):
    ''' '''

    location = int(tract_map['points'][0]['location'])
    tract_id = int(geo.iloc[location]['fips_state_county_tract_code'])
    check = int(n_factors)
    print(check)

    polar_data = CH.prepare_polar(data, weights, stcotr_fips=tract_id, n_factors=n_factors)
    predicted, actual = CH.predict_value(model, data, stcotr_fips=tract_id, y_var="Diabetes")
    radial_predicted = "Predicted = " + str(round(predicted,1)) + " %"
    radial_actual = "Actual = " + str(actual) + " %"
    radial_city_average = "City Average = "+str(city_average)+" %"

    return px.line_polar(
        polar_data,
        r="Risk Score",
        theta="metric_name",
        height=700,
        line_close=True
    ).update_traces(
        fill='toself',
        fillcolor='rgba(65, 105, 225, 0.60)',
        mode = "lines",
        line_color = "#F0F0F0",
        # marker=dict(
            # color="#4169e1",
            # symbol="circle",
            # size=12 )
    ).update_layout(
    title = dict(
        text = '',
        x = 0,
        y = .925),
    font_size = 15,
    showlegend = False,
    # margin=dict(l=50,r=50),
    # xaxis=dict(automargin=True),
    # yaxis=dict(automargin=True),
    polar = dict(
      bgcolor = "rgba(240,240,240, .85)",
      angularaxis = dict(
        linewidth = 3,
        showline=True,
        linecolor='rgba(255,255,255, .85)',
        color="black"
      ), radialaxis = dict(
            side = "clockwise",
            showline = True,
            linewidth = 2,
            gridcolor = "white",
            gridwidth = 2,
            color = "#4169e1",
            visible =True,
            range=[min_max["min"], min_max["max"]]
          )
    ),
    paper_bgcolor = "rgba(0,0,0,0)",
    annotations = [
        go.layout.Annotation(
            x=0.5,
            y=-0.15,
            showarrow=False,
            text="Custom x-axis title",
            xref="paper",
            yref="paper"
        )],
), radial_predicted, radial_actual, radial_city_average

if __name__ == '__main__':
    app.run_server(port=8080, debug=True)
