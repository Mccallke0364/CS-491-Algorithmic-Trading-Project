
import pandas as pd
import requests
import time
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.decomposition import PCA

def merge_dataframes(starting_df, dict_stock_dfs):
    """ merges the dataframes"""
    merged_data=starting_df
        #identifies the starting dataframe for subsequent merges
    for data_frame in dict_stock_dfs:
        #will iterate through the keys of the dictionaries of stock dataframes, these keys will be the tickers of the stocks
        df_to_add = dict_stock_dfs[data_frame]
            #accesses the current dataframe in the dictionary of stock dataframes  
        merged_data = pd.merge(merged_data, df_to_add, right_index=True, left_index=True)
            #merges the usa spending for each date with the corresponding stock data for that date
            #since the dates are the indicies, the merge occurs on the indicies
            #each stock dataframe has to have column titles that are unique to it's stock so that the stocks can all be in the same dataframe without overwriting eachothers data
                # ie every stock dataframe has data for o h l c and v so we add the stock ticker to the column name as an extra identifier
    merged_data.rename_axis("Date", inplace=True)
        #retains the original index identifier so that the index can be accessed using the keyword "Date" in future code
    return merged_data
    
def print_df(df, filename, location="dataframes/"):
    with open(f"{location}{filename}.txt", "w") as f:
        f.write(df.to_string())
    with open(f"{location}{filename}.csv", "w") as f_csv:
        f_csv.write(df.to_csv())

def get_data_for_all_stocks():
    dict={}
    for a in os.listdir("dataframes"):
        if a.endswith(".csv"):
            if a!="usa_spend.csv":
                date = "t"
                ticker = a[:-4]
                df = pd.read_csv(f"dataframes/{a}", parse_dates=[date], header=0, index_col=0)
                df.rename(columns={'vw': f'vw_{ticker}', "n": f"n_{ticker}"}, inplace=True)
<<<<<<< HEAD
                df.rename(columns={f'{ticker}_Returns': f'{ticker}'}, inplace=True)
                dict[ticker]= df[[f"{ticker}"]]
                # dict[a[:-4]] = df[[f'o_{ticker}', f'h_{ticker}', f'l_{ticker}', f'c_{ticker}',f'v_{ticker}', f"vw_{ticker}", f"n_{ticker}", f'{ticker}_SMA_10',f'{ticker}_SMA_50',f'{ticker}_Returns']]
            else:
                date = "Date"
                dataframe = pd.read_csv(f"dataframes/{a}", parse_dates=[date], header=0, index_col=0)
                
                continue
        
    return dict, dataframe[["awarding_agency_name", "total_outlayed_amount"]]

def PCA_create(PCA_dataframe, File_ext):
    PCA_dataframe_std = stats.zscore(PCA_dataframe)
    pca = PCA(n_components=14).fit(PCA_dataframe_std)
    # cum_var_explained = pca.explained_variance_ratio_.cumsum()
    # pca_needed = min([i for i in range(len(cum_var_explained)) if cum_var_explained[i]>.9])
    scores = pca.transform(PCA_dataframe)
    # print(scores.shape)
    # print("There are {} variables in the original dataset and {} PCs are needed to represent at least 90% of the variation in the data".format(PCA_dataframe.shape[0], pca_needed))
    # print_df(PCA_dataframe, "scores", location="created_dataframes/")
    type = PCA_dataframe.index.to_numpy()
    # print(type.shape)
    # print(scores[:,0].shape)
    to_graph = pd.DataFrame({"PC1": scores[:,0], "PC2":scores[:,1], "type":type})
    fig, ax = plt.subplots()
    sns.scatterplot(x="PC1", y="PC2", style="type", hue="type", data = to_graph)
    sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
    fig.savefig(f"PC1vsPC2_{File_ext}.png", dpi=500,bbox_inches="tight")

    fig_1, ax_1 = plt.subplots()
    contributions= pd.DataFrame({"PC1_contr":pca.components_.T[:,0], "PC2_contr": pca.components_.T[:,1]})
    a=sns.scatterplot(x="PC1_contr", y="PC2_contr", data=contributions, legend=True, ax=ax_1)
    # ax_1.legend(title="agency")
    # sns.move_legend(a, "upper left", bbox_to_anchor=(1, 1))
    fig_1.savefig(f"PC2_component_contributions_{File_ext}.png", bbox_inches="tight")

    max_arg=contributions.PC1_contr.abs().argmax()
    # print("the funding that contributes the most to PC1 is the {} fund".format(PCA_dataframe.loc[max_arg]))
    min_arg=contributions.PC1_contr.abs().argmin()
    # print("the funding that contributes the least to PC1 is the {} fund".format())
    PCA_fund= pd.DataFrame(PCA_dataframe.iloc[:,max_arg])
    PCA_fund[f"{File_ext}"]= type
    # print(PCA_fund)
    fig_2, ax_2 =plt.subplots()
    b=sns.histplot(x=max_arg, hue=f"{File_ext}", data=PCA_fund, ax=ax_2, element="bars", multiple="dodge")
    # ax_2.legend(title="agency", )
    sns.move_legend(b, "upper left", bbox_to_anchor=(1, 1))
    fig_2.savefig(f"PC1_fund_effect_{File_ext}.png", bbox_inches="tight")
    loadings = pd.DataFrame(pca.components_.T,
    columns=[f'PC{_}' for _ in range(14)],
    index=PCA_dataframe.columns)
    print(loadings)

    plt.plot(pca.explained_variance_ratio_)
    plt.ylabel('Explained Variance')
    plt.xlabel('Components')
    plt.savefig(f"a{File_ext}.png", bbox_inches="tight")

    PCA_fund_least = pd.DataFrame(PCA_dataframe.iloc[:, min_arg])
    PCA_fund_least[f"{File_ext}"] = type
    PCA_fund_least.replace([np.inf, -np.inf], np.nan, inplace=True)
    fig_3, ax_3 = plt.subplots()
    c=sns.histplot(x=min_arg, hue=f"{File_ext}", data=PCA_fund_least, ax = ax_3, element="bars", multiple="dodge")
    # ax_3.legend(title="agency")
    sns.move_legend(c, "upper left", bbox_to_anchor=(1, 1))
    fig_3.savefig(f"PC1_fund_least_exp_{File_ext}.png",bbox_inches="tight")
=======
                dict[a[:-4]] = df[[f'o_{ticker}', f'h_{ticker}', f'l_{ticker}', f'c_{ticker}',f'v_{ticker}', f"vw_{ticker}", f"n_{ticker}", f'{ticker}_SMA_10',f'{ticker}_SMA_50',f'{ticker}_Returns']]
            else:
                date = "Date"
                dataframe = pd.read_csv(f"dataframes/{a}", parse_dates=[date], header=0, index_col=0)
                continue
        
    return dict, dataframe
>>>>>>> main

dict_stock_dfs, usa_spending_data = get_data_for_all_stocks()

full_dataframe = merge_dataframes(usa_spending_data, dict_stock_dfs)
    #merges all the stock dataframes and usa spending based on the date the data was collected
<<<<<<< HEAD
# full_dataframe.dropna(inplace=True)
# full_data_frame_na=full_dataframe.set_index("awarding_agency_name")
# full_data_frame_na.dropna(inplace=True)
# full_dataframe=full_dataframe.groupby("awarding_agency_name")
# print(full_dataframe.head())
# print([a for a in full_dataframe.T])
# print_df(full_dataframe, "full_df")
# print_df(full_dataframe.head(50), "full_df_head", location="created_dataframes/")
# full_dataframe_grouped = full_dataframe.dropna().groupby(["awarding_agency_name"])
# PCA_create(full_dataframe,"full")
PCA_create(full_dataframe.dropna().groupby("awarding_agency_name").mean(), "grouped")
# for name, group in full_dataframe_grouped:
#     df=group.set_index(["total_outlayed_amount"]).drop(["awarding_agency_name"], axis="columns").dropna()
#     print_df(df, name, location="created_dataframes/")
#     PCA_create(df, f"{name[0]}")
# print_df(full_dataframe_grouped.head(50), "full_df_group_head", location="created_dataframes/")


# PCA_create(full_dataframe_grouped, "agency_sort")

# print_df(full_dataframe_grouped, "award_stock", location="created_dataframes/")
# # PCA_create(full_dataframe_grouped.T.groupby(["awarding_agency_name"]), "stock")
# full_dataframe.get_group(x) for x in full_dataframe.groups]
=======

# print_df(full_dataframe, "full_df")
print(full_dataframe.head())
>>>>>>> main
