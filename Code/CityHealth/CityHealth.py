

import seaborn as sn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from IPython.display import display


def geo_api(city, state):
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

# data = pd.read_csv("CityHealth/Data/New York v6.0 - released June 12, 2019/CHDB_data_tract_NY v6_0.csv")

# dat = pd.read_csv("./Code/CityHealth/New York v6.0 - released June 12, 2019/CHDB_data_tract_NY v6_0.csv")
# set(data.metric_name.unique()) - set(dat.metric_name.unique())
# stcotr_fips = (Concatenation of state, county and tract FIPS codes)

def pivot_dropna(data):
    data_pivot = data.loc[:, ["stcotr_fips", "metric_name", "est"]]\
        .pivot(index='stcotr_fips', columns='metric_name',values='est')
    data_pivot.drop(["Walkability"], axis=1, inplace=True)
    data_clean = data_pivot.dropna()
    return data_clean

def multivariate_regression(data):
    from sklearn import linear_model
    from sklearn.preprocessing import PolynomialFeatures, MinMaxScaler
    # Pivot & clean
    data_clean = pivot_dropna(data)

    # Normalize
    data_norm = pd.DataFrame(StandardScaler().fit_transform(data_clean), index=data_clean.index, columns=data_clean.columns)
    # MinMaxScaler()

    # Subset variables
    y = data_norm["Diabetes"]
    X = data_norm.drop("Diabetes", axis=1)

    # Create interaction term (not polynomial features)
    # interaction = PolynomialFeatures(degree=3, include_bias=False, interaction_only=True)
    # X = interaction.fit_transform(X)

    # Run model
    model  = sm.OLS(y, X).fit()
    model.summary()

    # Lasso regression


    clf = linear_model.Lasso(alpha=1)
    X = preprocessing.normalize(data_clean.drop("Diabetes", axis=1))
    y = preprocessing.normalize(np.asarray(data_clean["Diabetes"]).reshape(-1,1) )
    model = clf.fit(y = y, X = X)
    model.coef_
    model.intercept_

    data_clean.loc[:,["Diabetes","Obesity"]]

    #
    np.corrcoef(data_clean["Diabetes"], data_clean["Obesity"])
    plt.scatter(data_clean["Diabetes"], data_clean["Obesity"])
    plt.show()




def cluster_heatmap(data):
    # Compute metric-metric correlations
    corr_matrix  = data.pivot(index='stcotr_fips',columns='metric_name',values='est').drop(["Walkability"], axis=1).corr()

    # Stack and sort correlations
    corr_stack = corr_matrix.stack()[corr_matrix.stack()<1].sort_values(ascending=False)
    corr_stack["Diabetes",:].sort_values(ascending=False)
    # Clustermap
    g = sn.clustermap(data=corr_matrix)
    # plt.setp(g.ax_heatmap.xaxis.get_majorticklabels(), rotation=30)
    plt.subplots_adjust(bottom=0.3, right=0.7)
    plt.show()

def facet_scatter(data):
    g = sn.FacetGrid(data, col="metric_name", row="metric_name")
    g = g.map(plt.scatter, "est", "est")
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
    sn.scatterplot(data=tsneDF, x="tSNE1", y="tSNE2", hue=hue_var)
    plt.show()
    return tsneDF