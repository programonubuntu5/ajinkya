
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

transactions = [
    ["Milk", "Bread", "Butter"],
    ["Bread", "Butter"],
    ["Milk", "Bread"],
    ["Milk", "Butter"],
    ["Milk", "Bread", "Butter"]
]

te = TransactionEncoder()
data = te.fit(transactions).transform(transactions)

import pandas as pd
df = pd.DataFrame(data, columns=te.columns_)

frequent = apriori(df, min_support=0.4, use_colnames=True)
print("Frequent Itemsets:")
print(frequent)

rules = association_rules(
    frequent, metric="confidence", min_threshold=0.5
)
print("Association Rules:")
print(rules)

# Outlier detection using IQR
values = [10, 12, 11, 13, 12, 100]

Q1 = pd.Series(values).quantile(0.25)
Q3 = pd.Series(values).quantile(0.75)
IQR = Q3 - Q1

for x in values:
    if x < Q1 - 1.5 * IQR or x > Q3 + 1.5 * IQR:
        print("Outlier:", x)
