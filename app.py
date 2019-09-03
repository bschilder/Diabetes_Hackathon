# -*- coding: utf-8 -*-


import Code.CityHealth.CityHealth as CH
from os import system
import plotly_express as px
import dash
import dash_html_components as html
import dash_core_components as dcc
from dash.dependencies import Input, Output
from flask import Flask
from flask_flatpages import FlatPages
from flask_frozen import Freezer
from flask import request, after_this_request

# original example
## https://github.com/plotly/dash-px/blob/master/app.py

method="linear_regression"
# data_raw = CH.preprocess_data(raw=True)
data = CH.preprocess_data()
city_average = round(data["Diabetes"].mean(),1)
#Train model on data
model, coefs = CH.train_model(data, test_size=0.3, method=method, plot_weights=False)
# Label coefs
weights = CH.feature_weights(data, coefs=coefs, weight_label=method + "_coef")
polar_data = CH.prepare_polar(data, weights, n_factors=5)
# CH.weights_plot(weights)
# Find mix/max for radar plot
min_max = CH.min_max_scores(data, weights)

# Tract options
tract_dict = [dict(label=x, value=x) for x in data.index]
# N factors options
factor_range = list(range(1,len(data.columns)))
n_factors_dict = [dict(label=x, value=x) for x in factor_range]

col_options = [tract_dict, n_factors_dict]
dropdowns = ["Tract", "N_Factors"]


# By exposing this server variable, you can deploy Dash apps like you would any Flask app
server = Flask(__name__)
app = dash.Dash(
    __name__,
    external_stylesheets=["https://codepen.io/chriddyp/pen/bWLwgP.css"],
    server=server
)


app.layout = html.Div(
    [
        html.H1("Diabetes Risk Profile"),
        html.Div([
                 html.H5(["Select the community (a.k.a. Tract) of interest below."]),
                 html.H5(["On the right, a radar plot will display the Risk Profile of that Tract."])
                  ],
                 style={"width": "60%"}),
        html.Div(
            [html.Br(),
             html.Label(["Tract :"]),
             html.P([dcc.Dropdown(id="Tract", options=tract_dict, value=36085032300)]),
             html.Br(),
             html.Label(["Number of Risk Factors :"]),
             html.P([dcc.Dropdown(id="N_Factors", options=n_factors_dict, value=6)])],
            style={"width": "25%", "float": "left"},
        ),
        dcc.Graph(id="graph", style={"width": "75%", "display": "inline-block"}),
    ], style={"padding":"20px"}
)


@app.callback(Output("graph", "figure"), [Input(d, "value") for d in dropdowns])
def make_figure(Tract, n_factors):
    polar_data = CH.prepare_polar(data, weights, stcotr_fips=Tract, n_factors=n_factors)
    predicted, actual = CH.predict_value(model, data, stcotr_fips=Tract, label_var="Diabetes")
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


# Freeze flask!
# pages = FlatPages(server)
# freezer = Freezer(server)
# server.config.from_pyfile('project/settings.py')
# Save a list of all requirements for THIS repo only
# system("pipreqs --force ./")

if __name__ == '__main__':
    DEBUG = True
    # Debug mode
    if DEBUG:
        app.run_server(debug=True)
