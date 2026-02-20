import pandas as pd

data = {
    "Name": ["Rahul", "Priya", "Aman", "Neha"],
    "Marks": [85, 92, 78, 88],
    "City": ["Mumbai", "Delhi", "Pune", "Chennai"]
}

df = pd.DataFrame(data)
print(df)
#     Name  Marks     City
# 0  Rahul     85   Mumbai
# 1  Priya     92    Delhi
# 2   Aman     78     Pune
# 3   Neha     88  Chennai

print(df.info())
# Shows column types, non-null counts

print(df.describe())
# count, mean, std, min, max stats

print(df["Name"])
# Rahul, Priya, Aman, Neha

print(df[["Name", "Marks"]])
#     Name  Marks
# 0  Rahul     85
# 1  Priya     92
# 2   Aman     78
# 3   Neha     88

print(df[df["Marks"] > 80])
# Rahul, Priya, Neha rows

df["Passed"] = df["Marks"] >= 50
print(df)
# Adds Passed column (True/False)

df.loc[df["Marks"] < 80, "Grade"] = "B"
df.loc[df["Marks"] >= 80, "Grade"] = "A"
print(df)
# Grade column added (A/B)

print(df.sort_values("Marks", ascending=False))
# Sorted by marks descending

print("Average:", df["Marks"].mean())  # ~85.75
print("Total:", df["Marks"].sum())     # 343

df["Class"] = ["A", "A", "B", "B"]
grouped = df.groupby("Class")["Marks"].mean()
print(grouped)
# A -> avg of Rahul, Priya
# B -> avg of Aman, Neha

df.loc[2, "Marks"] = None
print(df)
# Marks for Aman becomes NaN

print(df.fillna(0))
# NaN replaced with 0

df.to_csv("output_students.csv", index=False)
print("Saved output_students.csv")
