

import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt



data = pd.read_csv("./CityHealth/New York v5.3 - released May 6, 2019/CHDB_data_tract_NY v5_3.csv",   na_values=-999)

metrics = data.metric_name.unique()
print(metrics)
len(metrics)


state = data.loc[:,["city_name",
                     "State, County, Tract FIPS  (leading 0)",
                     "metric_name",
                     "est"]]
state.columns = ["City","Location","Metric","Value"]

NYC = state.loc[subset["City"]=="New York",:]


# Barplots of metrics
## All cities in NYC
sn.barplot(data=state, x="Value", y="Metric", hue="City")
plt.show()
## NYC only
sn.barplot(data=NYC, x="Value", y="Metric")
plt.show()


# Compute metric-metric correlations
corr_matrix  = state.pivot(index='Location',columns='Metric',values='Value').corr()
# Stack and sort correlations
corr_stack = corr_matrix.stack()[corr_matrix.stack()<1].sort_values(ascending=False)
corr_stack["Binge drinking",:].sort_values(ascending=False)


# Clustermap
g = sn.clustermap(data=corr_matrix)
# plt.setp(g.ax_heatmap.xaxis.get_majorticklabels(), rotation=30)
plt.subplots_adjust(bottom=0.3, right=0.7)
plt.show()

