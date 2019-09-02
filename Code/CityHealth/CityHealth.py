# *********************************************************************
# *********************************************************************
#******************************    CityHealth   ***********************
# *********************************************************************
# *********************************************************************



#=============== References ===============
## Feature selection:
### https://www.datacamp.com/community/tutorials/feature-selection-python

## Automated feature selection:
### https://www.kaggle.com/residentmario/automated-feature-selection-with-sklearn

## Deep Feature Synthesis:
### https://towardsdatascience.com/automated-feature-engineering-in-python-99baf11cc219

## (Properly) Installing Plotly 4.1.0
# https://github.com/plotly/plotly.py#jupyter-notebook-support

## Dash
### https://dash.plot.ly/getting-started

## Dash app w/ Plotly Express
### https://github.com/plotly/dash-px/blob/master/app.py
### https://medium.com/plotly/introducing-plotly-express-808df010143d

## Random Forest tutorial
### https://towardsdatascience.com/random-forest-in-python-24d0893d51c0

## Radar Plots
### xticks:
# https://plot.ly/javascript/reference/#layout-xaxis-ticklen

## Flask-nginx (Daniel's template for python app deployement)
### https://github.com/MaayanLab/flask-nginx

## Heroku setup
### https://devcenter.heroku.com/articles/getting-started-with-python#run-the-app-locally

## Static Flask site tutorial:
### https://stevenloria.com/hosting-static-flask-sites-on-github-pages/
## Flask Frozen documentation:
### https://pythonhosted.org/Frozen-Flask/

## Dash w/ WSGI
### https://dash.plot.ly/integrating-dash
#==========================================


import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


#--------------------------------------------------------------#
#------------------- Data Preprocessing -----------------------#
#--------------------------------------------------------------#

def geo_api(city, state):
    import requests
    base = "http://34.200.29.188/api/data/geographic-identifier?token="
    api_key = "6ad860a334109640f9abd53018d9bcb7"
    city_name  = "&city_name="+city
    state_abbr = "&state_abbr="+state
    full_api = (base+api_key+city_name+state_abbr).replace(" ","%20")
    r = requests.get(full_api)
    j = r.json()
    # j['fields']
    # j['metrics']
    data = pd.DataFrame(j['rows'])
    return data
# geo_data = geo_api("New York","NY")


def tract_api(state, city=False, token="6ad860a334109640f9abd53018d9bcb7"):
    import requests
    base = "http://34.200.29.188/api/data/tract-metric"
    token_key = "?token="+token
    # Get metrics
    j = requests.get(base+token_key).json()
    all_fields = j['tract-metric']['metric_options'].keys()
    if city:
        city_name  = "&city_name="+city
    else:
        city_name=""
    state_abbr = "&state_abbr="+state
    data=pd.DataFrame()
    for metric in all_fields:
        print("Extracting: "+ metric)
        full_api = (base+"/"+metric+token_key+city_name+state_abbr).replace(" ","%20")
        j = requests.get(full_api).json()
        # j['fields']
        # j['metrics']
        newDF = pd.DataFrame(j['rows'])
        data = pd.concat([data, newDF])
    return data
# data = tract_api(state="NY")

def normalize(data, label="Diabetes"):
    # Normalize all columns except the label ("Diabetes")
    # to maintain interpretability of Diabetes predictions.
    from sklearn.preprocessing import StandardScaler
    data_norm = data.copy()
    features = data.loc[:,data.columns!=label]
    data_norm.loc[:,data_norm.columns!=label] = StandardScaler().fit_transform(features)
    # MinMaxScaler()
    return data_norm

def pivot_dropna(data):
    from sklearn.preprocessing import StandardScaler
    # [1] Pivot
    data_pivot = data.loc[:, ["stcotr_fips", "metric_name", "est"]]\
        .pivot(index='stcotr_fips', columns='metric_name',values='est')
    # [2] Drop NAs
    data_pivot.drop(["Walkability"], axis=1, inplace=True)
    data_pivot = data_pivot.dropna()
    return data_pivot


def data_split(data, y_var="Diabetes", test_size=0.3):
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder
    lab_enc = LabelEncoder()
    # Tranform into array for computational speedup
    X = np.array(data.loc[:,data.columns!=y_var])
    y = np.array(data.loc[:,y_var])
    # Fix continuous issue
    # X = np.array([lab_enc.fit_transform(x) for x in X])
    # y = lab_enc.fit_transform(Y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=3)
    print(len(X_train), len(X_test))
    return X_train, X_test, y_train, y_test




#--------------------------------------------------------------#
################## Feature selection models ###################
#--------------------------------------------------------------#

def test_predictions(model, X_test, y_test):
    predictions = model.predict(X_test)
    # Calculate the absolute errors
    errors = abs(predictions - y_test)
    # Print out the mean absolute error (mae)
    print('Mean Absolute Error:', round(np.mean(errors), 4), '%.')

    # Calculate mean absolute percentage error (MAPE)
    mape = 100 * (errors / y_test)
    # Calculate and display accuracy
    accuracy = 100 - np.mean(mape)
    print('Accuracy:', round(accuracy, 4), '%.')
    return errors

def predict_value(model, data, stcotr_fips, label_var="Diabetes"):
    data_sub = data.loc[stcotr_fips].copy()
    X = np.array(data_sub[data_sub.index!=label_var]).reshape(1, -1)
    y = np.array(data_sub[label_var])
    actual = float(y)
    predicted = model.predict(X)
    return float(predicted), actual

# Lasso regression
def lasso(X_train, X_test, y_train, y_test, alpha=1):
    from sklearn.linear_model import Lasso
    # Lasso regression
    # for a in range(0,10):
    # print("Lasso: alpha = ",a)
    clf = Lasso(alpha=alpha)
    model = clf.fit(X = X_train, y = y_train)
    # model.intercept_
    errors = test_predictions(model, X_test, y_test)
    return model, model.coef_

# Mutual Information Criterion
def mutual_info(X_train, X_test, y_train, y_test):
    def mutual_info_subset(X, y, mode='percentile', param=50):
        from sklearn.feature_selection import GenericUnivariateSelect
        trans = GenericUnivariateSelect(score_func=mutual_info_regression,
                                        mode=mode, param=param)
        X_trans = trans.fit_transform(X, y)
        return X_trans

    from sklearn.feature_selection import mutual_info_regression
    coefs = mutual_info_regression(X_train, y_train, n_neighbors=3)
    return coef


def plot_decision_tree(tree_small,
                       feature_names=None,
                       file_prefix="rf_tree"):
    print("Creating decision tree plot ==> "+file_prefix+'.png')
    if type(feature_names) != "list":
        print("Creating default feature names.")
        feature_names = ["Feature." + str(x + 1) for x in range(len(tree_small.feature_importances_))]
    # Save the tree as a png image
    export_graphviz(tree_small, out_file=file_prefix+'.dot', feature_names=feature_names, rounded=True, precision=1)
    (graph,) = pydot.graph_from_dot_file(file_prefix+'.dot')
    graph.write_png(file_prefix+'.png');

def decision_tree(X_train, X_test, y_train, y_test,
                  max_depth=6,
                  feature_names=None,
                  dt_plot=True):
    from sklearn.tree import DecisionTreeRegressor, export_graphviz
    clf =  DecisionTreeRegressor(max_depth=max_depth)
    model = clf.fit(X_train, y_train)
    Gini_importance = model.feature_importances_
    errors = test_predictions(model, X_test, y_test)
    if dt_plot:
        plot_decision_tree(model, feature_names, file_prefix="dt_tree")
    # regressor.decision_path(X)
    return model, Gini_importance

def random_forest(X_train, X_test, y_train, y_test,
                  n_estimators=1000,
                  max_depth=6,
                  feature_names=None,
                  dt_plot=True):
    from sklearn.ensemble import RandomForestRegressor
    # Instantiate model with 1000 decision trees
    rf = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    model = rf.fit(X_train, y_train)
    errors = test_predictions(model, X_test, y_test)
    Gini_importance = model.feature_importances_
    if dt_plot:
        # Extract the small tree
        tree_small = model.estimators_[5]
        plot_decision_tree(tree_small, feature_names, file_prefix="rf_tree")
    return model, Gini_importance

def ridge_regression(X_train, X_test, y_train, y_test):
    # ridge regression == L2-Regularization
    # A helper method for pretty-printing the coefficients
    def pretty_print_coefs(coefs, names=None, sort=False):
        if names == None:
            names = ["X%s" % x for x in range(len(coefs))]
        lst = zip(coefs, names)
        if sort:
            lst = sorted(lst, key=lambda x: -np.abs(x[0]))
        return " + ".join("%s * %s" % (round(coef, 3), name)
                          for coef, name in lst)

    from sklearn.linear_model import Ridge
    ridge = Ridge(alpha=1.0)
    model = ridge.fit(X_train, y_train)
    print("Ridge model:", pretty_print_coefs(ridge.coef_))
    errors = test_predictions(model, X_test, y_test)
    return model, model.coef_

# Multivariate Linear Regression
def linear_regression(X_train, X_test, y_train, y_test):
    from sklearn.linear_model import LinearRegression
    # from sklearn.preprocessing import PolynomialFeatures
    # Create interaction term (not polynomial features)
    # interaction = PolynomialFeatures(degree=3, include_bias=False, interaction_only=True)
    # X = interaction.fit_transform(X)

    # Easy summary method
    # import statsmodels.api as sm
    # model  = sm.OLS(y, X).fit()
    # model.summary()
    LR = LinearRegression()
    model = LR.fit(X_train, y_train)
    errors = test_predictions(model, X_test, y_test)
    # reg.score(X, y)
    # model.intercept_
    return model, model.coef_




def feature_weights(data, coefs, weight_label="coef"):
    print("Creating feature weights dataframe.")
    weights = pd.DataFrame({weight_label: coefs},
                                       index=data.columns[data.columns != "Diabetes"])
    weights.sort_values(by=weights.columns[0], ascending=False, inplace=True)
    return weights

def weights_plot(weights):
    pal = sns.color_palette("Blues_d", len(weights))
    plt.subplots(1)
    sns.barplot(x=weights.iloc[:,0], y=weights.index, palette=pal)
    plt.show()


def min_max_scores(data, weights):
    from math import ceil, floor
    # min
    minScore_df = pd.concat([data.min()[weights.index], weights], axis=1)
    minScore_df["Score"] = minScore_df.iloc[:, 0] * minScore_df.iloc[:, 1]
    min_score = floor(minScore_df["Score"].min())
    # max
    maxScore_df = pd.concat([data.max()[weights.index], weights], axis=1)
    maxScore_df["Score"] = maxScore_df.iloc[:, 0] * maxScore_df.iloc[:, 1]
    max_score = ceil(maxScore_df["Score"].max())
    return {"min":min_score,"max":max_score}

def train_model(data, test_size=0.3, method="random_forest", plot_weights=True):
    print("Training Model:",method)
    X_train, X_test, y_train, y_test = data_split(data, y_var="Diabetes", test_size=test_size)
    feature_names = data.columns[data.columns != "Diabetes"]
    if method=="lasso":
        model, coefs = lasso(X_train, X_test, y_train, y_test, alpha=1)
    if method=="mutual_info":
        coefs = mutual_info(X_train, X_test, y_train, y_test)
        model=None
    if method=="decision_tree":
        model, coefs = decision_tree(X_train, X_test, y_train, y_test,
                                    max_depth=6,
                                    feature_names=feature_names,
                                    dt_plot=False)
    if method=="random_forest":
        # Accuracy plateua's around max_depth=10
        model, coefs = random_forest(X_train, X_test, y_train, y_test,
                              max_depth=8,
                              n_estimators=1000,
                              feature_names=feature_names,
                              dt_plot=False)
    if method=="ridge_regression":
        model, coefs = ridge_regression(X_train, X_test, y_train, y_test)
    if method=="linear_regression":
        model, coefs = linear_regression(X_train, X_test, y_train, y_test)
    return model, coefs



######################################
################## PLOTS #############
######################################

def prepare_polar(data, weights, stcotr_fips=36119005702, n_factors=5):
    print("Calculating Risk Scores...")
    dat_sub = pd.DataFrame(data.loc[stcotr_fips]).copy()
    merged = dat_sub.merge(weights, on="metric_name")
    # Create Risk Score col
    merged["Risk Score"] = merged.iloc[:, 0] * merged.iloc[:, 1]
    print("Sorting values by",merged.columns[1]+".")
    merged.sort_values(by=[merged.columns[1]], ascending=False, inplace=True)
    # Sort according to highest weight
    polar_data = merged.iloc[0:n_factors,]
    polar_data = polar_data.reset_index()
    return polar_data

def polar_plotly(polar_data):
    # Standard plotly
    import plotly.graph_objects as go
    fig = go.Figure(data=go.Scatterpolar(r=polar_data["Risk Score"],
                                         theta=polar_data.index,
                                         fill='toself'))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True
            ),
        ),
        showlegend=False
    )
    fig.show()
    return fig


def polar_plotly_express(polar_data):
    # Radar Charts Documentation: https://plot.ly/python/radar-chart/
    # Plotly Express in Dash: https://medium.com/plotly/introducing-plotly-express-808df010143d
    # polar_data = prepare_polar(data, weights, stcotr_fips=36119005702, n_factors=5)
    # Plotly express
    import plotly.express as px
    fig = px.line_polar(polar_data, r='Risk Score', theta='metric_name', line_close=True)
    fig.update_traces(fill='toself')
    fig.show()
    return fig

def cluster_heatmap(data_raw):
    # Compute metric-metric correlations
    corr_matrix  = data_raw.pivot(index='stcotr_fips',columns='metric_name',values='est').drop(["Walkability"], axis=1).corr()

    # Stack and sort correlations
    corr_stack = corr_matrix.stack()[corr_matrix.stack()<1].sort_values(ascending=False)
    corr_stack["Diabetes",:].sort_values(ascending=False)
    # Clustermap
    g = sn.clustermap(data=corr_matrix)
    # plt.setp(g.ax_heatmap.xaxis.get_majorticklabels(), rotation=30)
    plt.subplots_adjust(bottom=0.3, right=0.7)
    plt.show()

def facet_scatter(data_raw):
    g = sns.FacetGrid(data_raw, col="metric_name", row="metric_name")
    g = g.map(plt.scatter, "est", "est",  alpha=.7)
    plt.show()


def plot_PCA(data, hue_var="Diabetes"):
    from sklearn.decomposition import PCA
    # Pivot & clean
    data_clean = pivot_dropna(data)
    # Separating out the features
    # x = data_clean.drop(["Diabetes"], axis=1).values
    # # Separating out the target
    # y = data_clean.loc[:, ['Diabetes']].values
    # Standardizing the features
    x = StandardScaler().fit_transform(data_clean)

    pca = PCA(n_components=3)
    principalComponents = pca.fit_transform(x)
    principalDf = pd.DataFrame(principalComponents, columns = ["PC1","PC2","PC3"], index=data_clean.index)
    finalDf = pd.concat([principalDf, data_clean], axis=1)

    sn.scatterplot(data=finalDf, x="PC1", y="PC2", hue=hue_var)
    plt.show()

    loadings = pd.DataFrame(data=pca.components_,
                 columns=data_clean.columns,
                 index=["PC1","PC2","PC3"]).T
    print("PC Loadings:")
    loadings_sorted = loadings.sort_values("PC1", ascending=False)
    display(loadings_sorted)

    print("Variance explained:")
    print(pca.explained_variance_ratio_)
    return finalDf

def tsne(data, perplexity=30, hue_var="Diabetes"):
    from sklearn.manifold import TSNE
    data_clean = pivot_dropna(data)
    X_embedded = TSNE(n_components=3, perplexity=perplexity).fit_transform(data_clean)
    tsneDF = pd.DataFrame(X_embedded, index=data_clean.index, columns=["tSNE1","tSNE2","tSNE3"])
    tsneDF = pd.concat([tsneDF, data_clean], axis=1)
    sns.scatterplot(data=tsneDF, x="tSNE1", y="tSNE2", hue=hue_var)
    plt.show()
    return tsneDF




# if __name__ == "__main__":
def preprocess_data(data_path="./Code/CityHealth/Data/New York v6.0 - released June 12, 2019/CHDB_data_tract_NY v6_0.csv", raw=False):

    ##### stcotr_fips = (Concatenation of state, county and tract FIPS codes)
    data_raw = pd.read_csv(data_path)
    if raw:
        print("Returning raw CityHealth data.")
        return raw
    else:
        print("Preprocessing CityHealth data (pivot ==> dropna => normalize).")
        data = pivot_dropna(data_raw)
        # data.describe()
        data = normalize(data, label="Diabetes")
        # X_train, X_test, y_train, y_test = data_split(data)
        return data


# features = pd.get_dummies(["A","T","C","G"])