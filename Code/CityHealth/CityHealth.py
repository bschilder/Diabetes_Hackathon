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

## Dash w/ Heroku Template::
### https://github.com/plotly/dash-heroku-template
#==========================================


import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from multiprocessing.dummy import Pool as ThreadPool
import Code.cityhealth.ML_models as ML


#--------------------------------------------------------------#
#------------------- Data Preprocessing -----------------------#
#--------------------------------------------------------------#

def geo_api(state="all", city=False, token="6ad860a334109640f9abd53018d9bcb7"):
    import requests
    # base = "http://34.200.29.188/api/data/geographic-identifier?token=" # old API
    base="https://api.cityhealthdashboard.com/api/data/geographic-identifier?token="
    if city:
        city_name = "&city_name="+city
    else:
        city_name=""
    if state!="":
        state_abbr = "&state_abbr="+state
    else:
        state_abbr=""
    full_api = (base+token+city_name+state_abbr).replace(" ","%20")
    r = requests.get(full_api)
    j = r.json()
    # j['fields']
    # j['metrics']
    data = pd.DataFrame(j['rows'])
    return data
# geo_data = geo_api(city="New York", state="NY")

def all_cities_states(data_path="Code/CityHealth/Data/CHDB_data_city_all v7_1.csv.gz"):
    dat = pd.read_csv(data_path)
    return dat

def tract_api(state, city=False, token="6ad860a334109640f9abd53018d9bcb7"):
    print("Requesting data from",state)
    import requests
    # base = "http://34.200.29.188/api/data/tract-metric" # old API
    base = "https://api.cityhealthdashboard.com/api/data/tract-metric"
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


def multithread(func, items):
    import time
    start = time.time()
    pool = ThreadPool(4)
    results = pool.map(func, items)
    pool.close()
    pool.join()
    end = time.time()
    print(len(items),"items extracted in", round(end - start, 2), "seconds.")
    df = pd.concat(results)
    return df


def tract_api_multistate(states="all", save_path="./Code/CityHealth/Data/CityHealth_data.csv.gz"):
    if states=="all":
        geo_dat = all_cities_states()
        states = geo_dat.state_abbr.unique()
    df = multithread(func=tract_api, items=states)
    if save_path!=False:
        df.to_csv(save_path)
    return df


def pivot_handleNA(data_raw, impute=False, drop_na=True):
    from sklearn.preprocessing import StandardScaler
    # [1] Pivot
    data_pivot = data_raw.loc[:, ["state_abbr","city_name","stcotr_fips", "metric_name", "est"]].reset_index()\
        .pivot_table(index=['state_abbr','city_name','stcotr_fips'], columns='metric_name',values='est').reset_index()
    data_pivot.index = data_pivot['stcotr_fips']
    before = data_pivot.shape[0]
    # [2] Drop NAs
    all_fields = [x for x in data_raw.metric_name.unique() if x != "Walkability"]
    if impute:
        print("+ Imputing missing values...")

        # Tutorial: https://scikit-learn.org/stable/modules/impute.html
        from sklearn.impute import SimpleImputer
        imputer = SimpleImputer(strategy="mean")
        data_pivot[all_fields] = imputer.fit_transform( data_pivot[all_fields] )
    elif drop_na:
        print("+ Dropping missing values...")
        data_pivot = data_pivot.dropna()
        print(before-data_pivot.shape[0],"/",data_pivot.shape[0],"tracts dropped after filtering NAs.")
    return data_pivot


def normalize(data, y_var="Diabetes"):
    # Normalize all columns except the label ("Diabetes")
    # to maintain interpretability of Diabetes predictions.
    from sklearn.preprocessing import StandardScaler
    data_norm = data.copy()
    id_vars = [y_var, "state_abbr", "city_name", "stcotr_fips"]
    features = data.loc[:,~data.columns.isin(id_vars)]
    data_norm.loc[:,~data_norm.columns.isin(id_vars)] = StandardScaler().fit_transform(features)
    # MinMaxScaler()
    return data_norm



def feature_weights(data, coefs, weight_label="coef", y_var="Diabetes"):
    print("Creating feature weights dataframe.")
    id_vars = [y_var, "state_abbr", "city_name", "stcotr_fips"]
    weights = pd.DataFrame({weight_label: coefs},
                                       index=data.columns[~data.columns.isin(id_vars)])
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
    X_train, X_test, y_train, y_test = ML.data_split(data, y_var="Diabetes", test_size=test_size)
    feature_names = data.columns[data.columns != "Diabetes"]
    if method=="lasso":
        model, coefs = ML.lasso(X_train, X_test, y_train, y_test, alpha=1)
    if method=="mutual_info":
        coefs = ML.mutual_info(X_train, X_test, y_train, y_test)
        model=None
    if method=="decision_tree":
        model, coefs = ML.decision_tree(X_train, X_test, y_train, y_test,
                                    max_depth=6,
                                    feature_names=feature_names,
                                    dt_plot=False)
    if method=="random_forest":
        # Accuracy plateua's around max_depth=10
        model, coefs = ML.random_forest(X_train, X_test, y_train, y_test,
                              max_depth=8,
                              n_estimators=1000,
                              feature_names=feature_names,
                              dt_plot=False)
    if method=="ridge_regression":
        model, coefs = ML.ridge_regression(X_train, X_test, y_train, y_test)
    if method=="linear_regression":
        model, coefs = ML.linear_regression(X_train, X_test, y_train, y_test)
    return model, coefs



######################################
################## PLOTS #############
######################################

def prepare_polar(data, weights, stcotr_fips, n_factors=5):
    print("Calculating Risk Scores...")
    # stcotr_fips = 36085032300
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


def plot_PCA(data_raw, hue_var="Diabetes"):
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    # Pivot & clean
    data_clean = pivot_handleNA(data_raw)
    # Separating out the features
    # x = data_clean.drop(["Diabetes"], axis=1).values
    # # Separating out the target
    # y = data_clean.loc[:, ['Diabetes']].values
    # Standardizing the features
    id_vars = ["state_abbr", "city_name", "stcotr_fips"]
    x = StandardScaler().fit_transform( data_clean.loc[:,~data_clean.columns.isin(id_vars)] )

    pca = PCA(n_components=3)
    principalComponents = pca.fit_transform(x)
    principalDf = pd.DataFrame(principalComponents, columns = ["PC1","PC2","PC3"], index=data_clean.index)
    finalDf = pd.concat([principalDf, data_clean], axis=1)

    sns.scatterplot(data=finalDf, x="PC1", y="PC2", hue=hue_var)
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

def tsne(data_raw, perplexity=30, hue_var="Diabetes"):
    from sklearn.manifold import TSNE
    from sklearn.preprocessing import StandardScaler
    # Prep data
    data_clean = pivot_handleNA(data_raw)
    id_vars = ["state_abbr", "city_name", "stcotr_fips"]
    x = StandardScaler().fit_transform(data_clean.loc[:, ~data_clean.columns.isin(id_vars)])
    # t-SNE
    X_embedded = TSNE(n_components=3, perplexity=perplexity).fit_transform(x)
    tsneDF = pd.DataFrame(X_embedded, index=data_clean.index, columns=["tSNE1","tSNE2","tSNE3"])
    tsneDF = pd.concat([tsneDF, data_clean], axis=1)
    sns.scatterplot(data=tsneDF, x="tSNE1", y="tSNE2", hue=hue_var)
    plt.show()
    return tsneDF

def run_umap(data_raw, hue_var="Diabetes"):
    import umap
    from sklearn.preprocessing import StandardScaler
    # Prep data
    data_clean = pivot_handleNA(data_raw, impute=True)
    id_vars = ["state_abbr", "city_name", "stcotr_fips"]
    x = StandardScaler().fit_transform(data_clean.loc[:, ~data_clean.columns.isin(id_vars)])
    # UMAP
    reducer = umap.UMAP(n_components=3, random_state=2019)
    embedding = reducer.fit_transform(x)
    embedding.shape

    # sns.scatterplot(x=embedding[:, 0], y=embedding[:, 1], hue=data_clean["Diabetes"], )
    umap_dims = pd.DataFrame(embedding, columns=["UMAP"+str(i+1) for i in range(0,embedding.shape[1])]  )
    data_clean.index = range(0,data_clean.shape[0])
    dat = pd.concat([data_clean, umap_dims], axis=1)

    # Facet plot
    ## Seaborn
    g = sns.FacetGrid(data=dat, hue="est", col="metric_name")
    g = g.map(plt.scatter, x="UMAP1", y="UMAP2")
    plt.show()

    # Plotly
    ## https://plot.ly/python/line-and-scatter/
    import plotly.express as px
    from plotly.offline import plot

    fig = px.scatter(dat, x="UMAP1", y="UMAP2", color=hue_var, hover_data=['UMAP1','UMAP2','state_abbr','city_name','stcotr_fips',hue_var], opacity=.75)
    plot(fig)

    # import pyggplot
    # p = pyggplot.Plot(dat)
    # p.add_scatter('UMAP1', 'UMAP2', color="est").facet_grid(facets="metric_name~.")
    # p.render("image.png")

    # Single plot
    metrics = [x for x in data_raw.metric_name.unique() if x != "Walkability"]
    for i in range(1, 7):
        plt.subplot(2, 3, i)
        plt.text(0.5, 0.5, str((2, 3, i)),
                 fontsize=18, ha='center')
    fig = plt.figure()
    fig.subplots_adjust(hspace=0.4, wspace=0.4)
    for i,hue_var in enumerate(metrics):
        print(hue_var)
        ax = fig.add_subplot(2, 3, i)
        ax.scatter(x=embedding[:, 0], y=embedding[:, 1], c=data_clean[hue_var], s=.75)
        plt.gca().set_aspect('equal', 'datalim')
        plt.title('UMAP:'+hue_var);
        plt.colorbar()
        plt.show()


# if __name__ == "__main__":
def preprocess_data(data_path="data/cityhealth/CHDB_data_tract_all_v7_1.csv.gz",
                    raw=False,
                    NYC_only=False,
                    impute=False,
                    drop_na=True):
    ##### stcotr_fips = (Concatenation of state, county and tract FIPS codes)
    data_raw = pd.read_csv(data_path)
    if NYC_only:
        data_raw = data_raw.loc[data_raw.city_name=="New York",:]
    if raw:
        print("Returning raw CityHealth data.")
        return raw
    else:
        print("Preprocessing CityHealth data (pivot ==> dropna => normalize).")
        data = pivot_handleNA(data_raw, impute=impute, drop_na=drop_na)
        # data.describe()
        data = normalize(data, y_var="Diabetes")
        # X_train, X_test, y_train, y_test = data_split(data)
        return data


def find_homogenous_cities(data):
    data_melt = data.melt(id_vars=["state_abbr","city_name","stcotr_fips"])
    counts = data_melt.groupby(['state_abbr','city_name','metric_name']).nunique()
    counts.loc[counts["value"]==1,:]