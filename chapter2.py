import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

""" Example 1 """

data = {
    'year' : [2010,2011,2012,2010,2011,2012,2010,2011,2012],
    'team' : ['FCBarcelona','FCBarcelona','FCBarcelona','RMadrid','RMadrid','RMadrid','ValenciaCF','ValenciaCF','ValenciaCF'],
    'wins' : [30,28,32,29,32,26,21,17,19],
    'draws' : [6,7,4,5,4,7,8,10,8],
    'losses' : [2,3,2,4,2,5,9,11,11]
}

# print(data)

df = pd.DataFrame(data)
# print(df)

""" Example 2 """

edu = pd.read_csv("educ_figdp_1_Data.csv",
                  na_values=":",
                  usecols=["TIME","GEO","Value"])
# print(edu)
# print(edu.head())
# print(edu.tail())

print(edu["Value"])
print(edu[10:14])
print(edu.loc[90:94,["TIME","GEO"]])

print(edu[edu["Value"] > 6.5].tail())

print(edu[edu["Value"].isnull()].head())

print(edu.max(axis=0)) # the function should be applied to the rows for each column

print("Pandas maximum", edu["Value"].max())
print("Python max", max(edu["Value"]))

s = edu["Value"]/100
print(s.head())

s = edu["Value"].apply(np.sqrt)
print(s.head())

s = edu["Value"].apply(lambda d: d**2)
print(s.head())

edu["ValueNorm"] = edu["Value"]/edu["Value"].max()
print(edu.head())

eduAlt = edu.drop('ValueNorm', axis=1)
print(edu.head())
print(eduAlt.head())

edu = pd.concat([edu,
                 pd.DataFrame({"TIME":2000,"Value":5.00,"GEO":"a"},
                 index=[max(edu.index)+1])])
print(edu.tail())

""" Example 3 """

some_data = {
    "A":[1,None,3,None],
    "B":[None,None,6,None],
    "C":[7,8,9,None]
}

some_df = pd.DataFrame(some_data)
# print(some_df)

df_any = some_df.dropna(how="any")
# print(df_any)

df_all = some_df.dropna(how="all")
print(df_all)

print("Going back to example 2")

eduDrop = edu.dropna(how="any", subset=["Value"])
print(eduDrop.head())

eduFilled = edu.fillna(value={"Value":0,"ValueNorm":0})
print(eduFilled.head())

edu.sort_values(by="Value",ascending=True,inplace=True)
print(edu.head())

edu.sort_index(axis=0,ascending=True,inplace=True)
print(edu.head())

group = edu[["GEO","Value"]].groupby("GEO").mean()
print(group)

filtered_edu = edu[edu["TIME"]>2005]

pivedu = pd.pivot_table(filtered_edu,
                        values="Value",
                        index = ["GEO"],
                        columns = ["TIME"])

print(pivedu)

print(pivedu.loc[["Spain","Portugal"],[2006,2011]])

pivedu = pivedu.drop([
    "Euro area (13 countries)",
    "Euro area (15 countries)",
    "Euro area (17 countries)",
    "Euro area (18 countries)",
    "European Union (25 countries)",
    "European Union (27 countries)",
    "European Union (28 countries)",
], axis=0)

pivedu = pivedu.rename(index={"Germany (until 1990 former territory of the FRG)":"Germany"})

print(pivedu.head(15))
pivedu = pivedu.dropna()
print(pivedu.rank(ascending=False,method="first"))

sample_Data = pd.Series([50,30,20,50,40])
print(sample_Data.rank(method="first"))
