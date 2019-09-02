# -*- coding: utf-8 -*-


import Code.CityHealth.CityHealth as CH
import dash
import dash_core_components as dcc
import dash_html_components as html

data_raw = CH.preprocess_data(raw=True)
data = CH.preprocess_data()
weights = CH.weight_features(data, test_size=0, method="multivariate_regression", plot_weights=False)
polar_data = CH.prepare_polar(data, weights, stcotr_fips=36119005702, n_factors=5)


import plotly_express as px
import dash
import dash_html_components as html
import dash_core_components as dcc
from dash.dependencies import Input, Output

# original example
## https://github.com/plotly/dash-px/blob/master/app.py

polar_data = CH.prepare_polar(data, weights, n_factors=5)
# Tract options
tract_dict = [dict(label=x, value=x) for x in data.index]
# N factors options
factor_range = list(range(1,len(data.columns)-1))
n_factors_dict = [dict(label=x, value=x) for x in factor_range]

col_options = [tract_dict, n_factors_dict]
dropdowns = ["Tract", "N_Factors"]


app = dash.Dash(
    __name__, external_stylesheets=["https://codepen.io/chriddyp/pen/bWLwgP.css"]
)

app.layout = html.Div(
    [
        html.H1("CityHealth: Diabetes Risk Profile"),
        html.Div(html.H5(["Select the community (a.k.a. Tract) of interest below. \
                On the right, a radar plot will display the Risk Profile of that Tract."]),
                 style={"width": "50%"}),
        html.Div(
            [html.P(["Tract:", dcc.Dropdown(id="Tract", options=tract_dict, value=36119005702)]),
             html.P(["N_Factors:", dcc.Dropdown(id="N_Factors", options=n_factors_dict, value=5)])],
            style={"width": "25%", "float": "left"},
        ),
        dcc.Graph(id="graph", style={"width": "75%", "display": "inline-block"}),
    ]
)


@app.callback(Output("graph", "figure"), [Input(d, "value") for d in dropdowns])
def make_figure(Tract=36119005702, n_factors=5):
    polar_data = CH.prepare_polar(data, weights, stcotr_fips=Tract, n_factors=n_factors)
    return px.line_polar(
        polar_data,
        r="Risk Score",
        theta="metric_name",
        height=700,
    ).update_traces(fill='toself')


app.run_server(debug=True)
# if __name__ == '__main__':
#     app.run_server(debug=True)
# import plotly.express as px
#     fig = px.line_polar(polar_data, r='Risk Score', theta='metric_name', line_close=True)
#     fig.update_traces(fill='toself')
#     fig.show()