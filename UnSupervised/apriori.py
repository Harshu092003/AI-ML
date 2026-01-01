# Association Rule learning algorithm

import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

dataset = [
    ['Milk', 'Bread', 'Butter'],
    ['Bread', 'Butter'],
    ['Milk', 'Bread'],
    ['Milk', 'Butter'],
    ['Bread', 'Butter']
]


te = TransactionEncoder()
te_array = te.fit(dataset).transform(dataset)
print(te_array)
df = pd.DataFrame(te_array, columns=te.columns_)
print(df)

frequent_itemsets = apriori(df, min_support=0.4, use_colnames=True)
print(frequent_itemsets)


rules = association_rules(
    frequent_itemsets,
    metric="confidence",
    min_threshold=0.6
)

print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])
