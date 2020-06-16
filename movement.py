"""
Kara Shibley and Griffin Reichmuth
CSE 160 AC
This file plots various sections of the data and runs a random forest
regressor to predict county level movement.
"""
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble.forest import RandomForestRegressor
import plotly.express as px


# Import data and keep columns for retail data

data = pd.read_csv("features_labels_rf")

data = data.drop(['Unnamed: 0',
                  'grocery_and_pharmacy_percent_change_from_baseline',
                  'parks_percent_change_from_baseline',
                  'transit_stations_percent_change_from_baseline',
                  'workplaces_percent_change_from_baseline',
                  'residential_percent_change_from_baseline'], axis=1)

rec_retail = 'retail_and_recreation_percent_change_from_baseline'
groc_pharm = 'grocery_and_pharmacy_percent_change_from_baseline'
parks = 'parks_percent_change_from_baseline'
transit = 'transit_stations_percent_change_from_baseline'
workplaces = 'workplaces_percent_change_from_baseline'
residential = 'residential_percent_change_from_baseline'


# Plot all the data

all_counties = px.line(data, x="date", y=rec_retail, color="County name",
                       line_group="County name", hover_name="County name")
all_counties.show()

# Plot data based on feature importance
gov_plot = px.scatter(data, x="date", y=rec_retail,
                      color='g_Candidate_Jay Inslee',
                      color_continuous_scale=[(0, "red"), (1, "blue")],
                      hover_name="County name",
                      title="Gubernatorial Race")
gov_plot.show()

pres_plot = px.scatter(data, x="date", y=rec_retail,
                       color='p_Candidate_Donald J. Trump / Michael R. Pence',
                       color_continuous_scale=[(0, "blue"), (1, "red")],
                       hover_name="County name",
                       title="Presidential Race")

pres_plot.show()


# data for US and WA
us_data = pd.read_csv("US")
wa_mobility = us_data[us_data["sub_region_1"] == 'Washington']
us_data['date'] = pd.to_datetime(us_data['date'])
wa_mobility['date'] = pd.to_datetime(wa_mobility['date'])


# Sort data by movement type and find the daily average movement
# from the baseline by grouping by date

# recreation and retail
rec_retail_us = us_data.groupby("date")[rec_retail].mean()
rec_retail_wa = wa_mobility.groupby("date")[rec_retail].mean()

# grocery and pharmacy
groc_pharm_us = us_data.groupby("date")[groc_pharm].mean()
groc_pharm_wa = wa_mobility.groupby("date")[groc_pharm].mean()

# parks
parks_us = us_data.groupby("date")[parks].mean()
parks_wa = wa_mobility.groupby("date")[parks].mean()

# transit
transit_us = us_data.groupby("date")[transit].mean()
transit_wa = wa_mobility.groupby("date")[transit].mean()

# workplaces
workplaces_us = us_data.groupby("date")[workplaces].mean()
workplaces_wa = wa_mobility.groupby("date")[workplaces].mean()

# residential
residential_us = us_data.groupby("date")[residential].mean()
residential_wa = wa_mobility.groupby("date")[residential].mean()

# plot mobility data US vs WA
fig, [[ax1, ax2, ax3], [ax4, ax5, ax6]] = \
      plt.subplots(2, figsize=(25, 15), ncols=3)

rec_retail_wa.plot(x='date', y=rec_retail, legend=True, label="WA", ax=ax1)
rec_retail_us.plot(x='date', y=rec_retail, legend=True, label="US", ax=ax1)
ax1.set_title('Recreation and Retail Percent Change from Baseline')

groc_pharm_wa.plot(x='date', y=groc_pharm, legend=True, label="WA", ax=ax2)
groc_pharm_us.plot(x='date', y=groc_pharm, legend=True, label="US", ax=ax2)
ax2.set_title('Grocery and Pharmacy Percent Change from Baseline')

parks_wa.plot(x='date', y=parks, legend=True, label="WA", ax=ax3)
parks_us.plot(x='date', y=parks, legend=True, label="US", ax=ax3)
ax3.set_title('Parks Percent Change from Baseline')

transit_wa.plot(x='date', y=transit, legend=True, label="WA", ax=ax4)
transit_us.plot(x='date', y=transit, legend=True, label="US", ax=ax4)
ax4.set_title('Transit Percent Change from Baseline')

workplaces_wa.plot(x='date', y=workplaces, legend=True, label="WA", ax=ax5)
workplaces_us.plot(x='date', y=workplaces, legend=True, label="US", ax=ax5)
ax5.set_title('Workplace Percent Change from Baseline')

residential_wa.plot(x='date', y=residential, legend=True, label="WA", ax=ax6)
residential_us.plot(x='date', y=residential, legend=True, label="US", ax=ax6)
ax6.set_title('Residential Percent Change from Baseline')

plt.savefig("US vs. WA.png")

# Set date as the index
data['date'] = pd.to_datetime(data['date'])
data = data.set_index('date')


train_dates = data.loc["2020-03-23": '2020-04-13'].drop("County name", axis=1)
test_dates = data.loc["2020-04-27": '2020-05-18'].drop("County name", axis=1)

# Random Forest model
RF_Model = RandomForestRegressor(n_estimators=100,
                                 max_features=1, oob_score=True)


# RF for group 2
features_train = train_dates.loc[:, train_dates.columns != rec_retail]
labels_train = train_dates[rec_retail]

rgr = RF_Model.fit(features_train, labels_train)

features_test = test_dates.loc[:, test_dates.columns != rec_retail]
labels_test = test_dates[rec_retail]

rgr_predict = rgr.predict(features_test)
rgr_error = abs(rgr_predict - labels_test)
print(rgr_error)
print(rgr_error.mean())
