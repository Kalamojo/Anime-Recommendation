import pandas as pd
import numpy as np

def to_1D(series):
    return pd.Series([x for _list in series for x in _list])

# Loading Data

## Original User Dataset
#> To download directly (May take up to 5 minutes):
print("Downloading Users dataset....")
url1 = "https://drive.google.com/uc?export=download&id=1ziexNQvec0ilylv-U73nJoKuWenKCk5v&confirm=t&uuid=99001e73-0e6c-4793-b4ce-ac3033cbe37e&at=AHV7M3d_G0FOnB8Mm8zWlcZOSdsV:1670106934976"
users1 = pd.read_csv(url1)
users1.to_csv("users.csv")
print("Finished Downloading Users dataset")

## Original Anime Dataset
print("Downloading Anime dataset....")
url2 = "https://drive.google.com/uc?export=download&id=1dBY8fvIAdBzcn7ZeVooLJ04AAeLCj5lY"
anime1 = pd.read_csv(url2)
anime1.to_csv("anime.csv")
print("Finished downloading Anime dataset")

## List of Anime Names for reference
with open('anime_names.txt', 'w') as f:
    f.writelines([name + '\n' for name in anime1['Name']])

# Preprocessing Stage
## Anime
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
    
def preprocessA(df):
    print("Preprocessing Anime")
    df = df.replace('Unknown', np.nan)
    df["Score"] = df["Score"].ffill()

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
                    "Score-1", "Licensors", "Producers", "Studios", "MAL_ID"]
    df = df.drop(columns_to_drop, axis=1)

    for i, row in df.iterrows():
        if pd.isna(row["Type"]):
            if row["Episodes"] == 1:
                df.at[i, "Type"] = "Movie"
            else:
                df.at[i, "Type"] = "TV"
    
    df["Genres"] = df["Genres"].fillna("")

    categorical_columns = ["Genres", "Source", "Type", "Rating"]
    for col in categorical_columns:
        print(f"Processing {col}")
        for i, row in df.iterrows():
            df.at[i, col] = row[col].split(", ")

        counts = to_1D(df[col]).value_counts()

        for label in counts.index:
            for i,row in df.iterrows():
                if label in row[col]:
                    df.at[i, label] = 1
                else:
                    df.at[i, label] = 0

    columns_to_drop = ["Genres", "Name", "Source", "Type", "Rating"]
    df = df.drop(columns_to_drop, axis=1)

    return df

### May take up to 10 minutes
anime2 = preprocessA(anime1)
anime2.to_csv("anime2.csv")

# Preprocessing User
def preprocessU(df, numUsers=1000):
    print("Preprocessing Users")
    anime_list = df["anime_id"].value_counts()
    dfd = pd.DataFrame(columns=anime_list.index)
    dfd.loc[len(dfd)] = [0]*17562
    
    li = [0]*17562
    for i in range(325770):
        if (i % 5000 == 0):
            print(i)
        dfd.loc[i] = li
    
    user_names = df["user_id"].value_counts()
    dfd["user_id"] = user_names.index[:numUsers]

    prev = -1
    use = 0
    for i, row in df.iterrows():
        if row["user_id"] != prev:
            if (use % 100 == 0):
                print(use)
            if (use >= numUsers):
                print(f"First {use} users done")
                break;
            use += 1
            dfd.loc[(use-1), 'user_id'] = row['user_id']
            ind = dfd.index[dfd['user_id'] == row["user_id"]].tolist()[0]
            prev = row["user_id"]
        dfd.at[ind, row["anime_id"]] = row["rating"]
    
    dfd = dfd.fillna(0)
    return dfd

# This may take a REALLY Long Time (Just download the CSV for more practicality)
users2 = preprocessU(users1)
users2.to_csv("users2.csv")
