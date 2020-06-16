"""
Kara Shibley and Griffin Reichmuth
CSE 160 AC
This final combines data from multiple data sets for export
"""

import pandas as pd

# read in each data set csv

metro_classification = pd.read_csv("CSE163FinalProj/CDC_test.csv")
election = pd.read_csv("CSE163FinalProj/20161108_AllCounties (1).csv")
ag = pd.read_csv("CSE163FinalProj/WA Counties Agriculture.csv")
mobility = pd.read_csv("CSE163FinalProj/Global_Mobility_Report.zip")

# Filter dataframes for data of interest

# Filter metro_classification for WA counties
cdc_wa = metro_classification[metro_classification["State Abr."] == 'WA']
cdc_metro = cdc_wa[["County name", "2013 code"]]
cdc_metro["County name"] = cdc_metro["County name"].str.replace(" County", "")


# Filter election for President
election = election[["County", "Race", "Candidate", "PercentageOfTotalVotes"]]
pres = election[election["Race"] == "United States President/Vice President"]

groups = ['County', 'Race']
pres_max_vote = pres.groupby(groups)["PercentageOfTotalVotes"].max()

pres_totals = []
for total in range(len(pres_max_vote)):
    pres_totals.append(pres_max_vote[total])

pres_race = pres[pres["PercentageOfTotalVotes"].isin(pres_totals)]
pres_race = pres_race.drop(["PercentageOfTotalVotes"], axis=1)
pres_data = pres_race.rename(columns={"Race": "p_Race",
                                      "Candidate": "p_Candidate"})

# Filter election data for Governor
gov = election[election["Race"] == "Washington State Governor"]

gov_max_vote = gov.groupby(['County', 'Race'])["PercentageOfTotalVotes"].max()

gov_totals = []
for total in range(len(gov_max_vote)):
    gov_totals.append(gov_max_vote[total])

gov_race = gov[gov["PercentageOfTotalVotes"].isin(gov_totals)]
gov_race = gov_race.drop(["PercentageOfTotalVotes"], axis=1)
gov_data = gov_race.rename(columns={"Race": "g_Race",
                                    "Candidate": "g_Candidate"})


# Filter ag for county level ag data
ag_data = ag[["County", "Market value of agricultural products sold ($1000)",
              "Total cropland (acres)"]]

market_val = ["Market value of agricultural products sold ($1000)"]
ag_market = ag_data.nlargest(10, market_val)
largest_market = list(ag_market["County"])

ag_cropland = ag_data.nlargest(10, ["Total cropland (acres)"])
largest_cropland = list(ag_cropland["County"])

ag_data["large market"] = ""
ag_data["large cropland"] = ""

for county in list(ag_data["County"]):
    if county in largest_market is True:
        ag_data.at[ag_data["County"] == county, 'large market'] = 1
    else:
        ag_data.at[ag_data["County"] == county, 'large market'] = 0

    if county in largest_cropland is True:
        ag_data.at[ag_data["County"] == county, 'large cropland'] = 1
    else:
        ag_data.at[ag_data["County"] == county, 'large cropland'] = 0

ag_data = ag_data.drop(["Market value of agricultural products sold ($1000)",
                        "Total cropland (acres)"], axis=1)


# Filter Google mobility data for WA data
us = mobility[mobility['country_region'] == 'United States']
wa_mobility = us[us["sub_region_1"] == 'Washington']
wa_test = wa_mobility.drop(["country_region", "country_region_code",
                            "sub_region_1"], axis=1)
wa = wa_test[wa_test['sub_region_2'].notna()]
wa["sub_region_2"] = wa["sub_region_2"].str.replace(" County", "")

# send US and WA dataframe to csv to be used in another file
us.to_csv("US")

#  Merge data
merge_1 = cdc_metro.merge(pres_data, left_on='County name',
                          right_on="County", how="outer")
merge_2 = merge_1.merge(gov_data, left_on='County name',
                        right_on="County", how="outer")
merge_3 = merge_2.merge(ag_data, left_on='County name',
                        right_on="County", how="outer")
merge_4 = merge_3.merge(wa, left_on='County name',
                        right_on="sub_region_2", how="outer")
merge_4 = merge_4.drop(["County_x", "County_y", "County",
                        "sub_region_2", "p_Race", "g_Race"], axis=1)


data = pd.get_dummies(merge_4, columns=['p_Candidate', 'g_Candidate'])
data.to_csv("data")

# Prepare non-time series data to be exported to other file
ft_labels = merge_3.drop(["County_x", "County_y", "County",
                          "p_Race", "g_Race"], axis=1)


ft_labels = pd.get_dummies(ft_labels, columns=['g_Candidate', 'p_Candidate'])

ft_labels = ft_labels.drop(['p_Candidate_Hillary Clinton / Tim Kaine',
                            'g_Candidate_Bill Bryant'], axis=1)
ft_labels.to_csv("features_labels")


# Data for random forests (contains the time series data with from merge 4)

ft_labels_rf = pd.get_dummies(merge_4, columns=['g_Candidate', 'p_Candidate',
                                                "2013 code"])

ft_labels_rf = ft_labels_rf.drop(['p_Candidate_Hillary Clinton / Tim Kaine',
                                  'g_Candidate_Bill Bryant', "large market",
                                  "large cropland"], axis=1)
ft_labels_rf.to_csv("features_labels_rf")
