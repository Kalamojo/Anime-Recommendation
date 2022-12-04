#import libraries
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sb
import matplotlib.pyplot as plt

#Helper functions
def to_1D(series):
    return pd.Series([x for _list in series for x in _list])

def strDateToNum(date):
    date_list = date.split(" to ")
    newDate = date_list[0]
    year = int(newDate[-4:])
    return 2023 - year

def strDurToNum(dur):
    dur_list = dur.split(" hr.")
    if "sec." in dur:
        return 1
    if len(dur_list) > 1:
        hr = int(dur_list[0])
        if dur_list[1] == "" or dur_list[1] == " per ep.":
            return hr * 60
        mins = dur_list[1][1:]
        try:
            mins = int(mins.split(" min.")[0])
        except ValueError:
            print(dur)
        return mins + 60 * hr
    else:
        dur2 = dur_list[0].split(" min.")
        return int(dur2[0])

#Preprocessing and cleaning datasheet 
def preprocess(df):
    df = df.replace('Unknown', np.nan) #set all Unknowns tp NaN (Not a number)
    print("Filtering out data...")
    
    df["Score"] = df["Score"].ffill() #forward-filling NaN values in Score column

    for i, row in df.iterrows():
        if not isinstance(row["Aired"], str):
            continue
        df.at[i, "Aired"] = strDateToNum(row["Aired"])
    df["Aired"] = df["Aired"].ffill()

    for i, row in df.iterrows():
        if pd.isna(row["Episodes"]):
            if row["Type"] == "Movie":
                df.at[i, "Episodes"] = 1
            else:
                df.at[i, "Episodes"] = row["Aired"] * 96
    df["Episodes"] = df["Episodes"].ffill()

    for i, row in df.iterrows():
        if not isinstance(row["Duration"], str):
            continue
        df.at[i, "Duration"] = strDurToNum(row["Duration"])
    df["Duration"] = df["Duration"].ffill()

    df["Rating"] = df["Rating"].fillna("G - All Ages")
    df["Source"] = df["Source"].fillna("")

    columns_to_drop = ["English name", "Japanese name", "Premiered", "Ranked", "Score-10", "Score-9", 
                    "Score-8", "Score-7", "Score-6", "Score-5", "Score-4", "Score-3", "Score-2", 
                    "Score-1", "Licensors", "Producers", "Studios"]
    df = df.drop(columns_to_drop, axis=1)

    for i, row in df.iterrows():
        if pd.isna(row["Type"]):
            if row["Episodes"] == 1:
                df.at[i, "Type"] = "Movie"
            else:
                df.at[i, "Type"] = "TV"
    
    df["Genres"] = df["Genres"].fillna("")
    count2 = []

    categorical_columns = ["Genres", "Source", "Type", "Rating"]
    for col in categorical_columns:
        print(f"Processing {col}")
        for i, row in df.iterrows():
            df.at[i, col] = row[col].split(", ")

        counts = to_1D(df[col]).value_counts()
        if col == "Genres":
            count2 = counts.index

        for label in counts.index:
            for i,row in df.iterrows():
                if label in row[col]:
                    df.at[i, label] = 1
                else:
                    df.at[i, label] = 0

    columns_to_drop = ["Genres", "Name", "Source", "Type", "Rating"]
    df = df.drop(columns_to_drop, axis=1)

    return df, count2

#Now to the main stuff
url = "Anime-Recommendation/anime.csv"

anime_old = pd.read_csv(url)

anime, counts = preprocess(anime_old)

#Using Pearson Correlation
#to show how different shows are correlated to each other
plt.figure(figsize=(150,75))
cor = anime[counts].corr()
sb.heatmap(cor, annot=True, cmap=plt.cm.Reds)
plt.savefig('corr.png')

cosine_sim = cosine_similarity(anime, anime)

#Recommendation by Anime Title
name_indices = pd.Series(anime_old["Name"])

def recommend_by_title(title, cosine_sim = cosine_sim):
    recommended_anime = []
    idx = name_indices[name_indices == title].index[0]
    score_series = pd.Series(cosine_sim[idx]).sort_values(ascending = False)
    top_10_indices = list(score_series.iloc[1:11].index)
    
    for i in top_10_indices:
        recommended_anime.append(list(anime_old['Name'])[i])
        
    return np.vstack((recommended_anime,score_series.iloc[1:11].values)).T

