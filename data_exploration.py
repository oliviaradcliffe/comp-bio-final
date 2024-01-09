"""
Data exploration file on lncRNA data for CISC471 project 

@author: Lizzy Klosa & Olivia Radcliffe
Date: November 27, 2023
"""

import matplotlib.pyplot as plt
import time
import os
import pickle
from matplotlib.table import Table
import numpy as np
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.decomposition import PCA

from CISC471_model_compare import true_labels, feature_select, best_feature


def ninetyPCA(standardized_df):  
    """ Function to perform PCA retaining 90% of the variance
    """
    # retain 90% of the variance 
    pca = PCA(n_components=0.9)

    # Fit the PCA model to the standardized training data
    ninetyPCA = pca.fit_transform(standardized_df)

    # Number of components/features selected
    n_selected_features = pca.n_components_
    print("Number of selected features for PCA retaining 90%: ", n_selected_features, "\n")

    return ninetyPCA

def pcaVisualization(standardized_df, class_labels):
    """ Function to visualize PCA plot
    """

    # use 3 PCA components
    pca = PCA(n_components=3)
    threeDpca = pca.fit_transform(standardized_df)

    # create a 3D scatter plot
    fig3 = plt.figure()
    ax = fig3.add_subplot(111, projection='3d')

    # extract the first three PCA components
    x = threeDpca[:, 0]
    y = threeDpca[:, 1]
    z = threeDpca[:, 2]

    label_color_dict = {1:'red',-1:'blue'}
    label_dict = {1:'Negative',-1:'Positive'}
    labels = [label_dict[label] for label in class_labels]

    # plot each point and assign color and label
    for i in range(len(x)):
        ax.scatter(x[i], y[i], z[i], color=label_color_dict[class_labels[i]], label=labels[i])

    # remove duplicates from legend
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())


    # set labels and title
    ax.set_xlabel('PCA 1 (%.2f%%)' % (pca.explained_variance_ratio_[0]*100))
    ax.set_ylabel('PCA 2 (%.2f%%)' % (pca.explained_variance_ratio_[1]*100)) 
    ax.set_zlabel('PCA 3 (%.2f%%)' % (pca.explained_variance_ratio_[2]*100)) 
    ax.set_title('PCA Plot of lncRNA Data')


def summarizeStats(X, labels):
    """ Function to compute summary statistics given data X and labels
    """
    # standardize instance
    scaler = StandardScaler()

    # standardizing train data
    standardized_data = scaler.fit_transform(X)
    # Convert the standardized data back to a DataFrame     
    standardized_df = pd.DataFrame(standardized_data, columns=labels)

    # display summary statistics
    summary_stats = standardized_df.describe()

    # Write the summary statistics to an Excel file
    excel_path = 'summary_statistics.xlsx'
    summary_stats.to_excel(excel_path, index=True)

    print(f'Summary statistics have been written to {excel_path}.')

    return standardized_df, summary_stats


if __name__ == '__main__':
    # ---- importing the dataset ----
    file_name = "yxv4i3btdbczetp4hyom1700683880.253802"
    data_file = 'bigtable.txt'

    df1, _, X, y = true_labels(file_name, data_file)

    # ----- Show the class distribution of data ----
    classes = list(['Positive', 'Negative'])
    counts = list([(y == 1).sum(), (y == -1).sum()])
    
    # creating the bar plot
    fig = plt.figure()
    plt.bar(classes, counts, color ='red')
    plt.xlabel("Cancer related")
    plt.ylabel("No. of genes")
    plt.title("Distibution of Data in Classes")


    # ----- Show the feature importance ----
    df, importances, best_features = feature_select(df1)

    # creating the bar plot
    fig2 = plt.figure()
    plt.bar(best_features[1:], importances, color ='blue')
    plt.subplots_adjust(top=0.96)
    plt.xticks(rotation=90)
    plt.xticks(fontsize=4)

    plt.xlabel("Features")
    plt.ylabel("Importance score")
    plt.title("Feature Importance Distribution")

    # ---- Compute Feature Ranking ----
    _, _, best_columns  = best_feature(df1)
    print("Best Features: ")
    for col in best_columns.columns:
        print("\t", col)


    # ---- Compute and display summary statistics for the data ----

    standardized_df, _ = summarizeStats(X, df1.columns[1:])

    # ---- Generate box plots for the standardized data ----
    fig4 = plt.figure()
    standardized_df.boxplot(rot=90)
    plt.subplots_adjust(top=0.96)
    plt.xticks(fontsize=4)
    plt.title("Box Plot of Standardized Data")


    #----- Select features that account for 90% of data variance -----

    pcaComponents = ninetyPCA(standardized_df)

    # ---- Visualize the trainset in 3D space when the first 3 PCA components are selected ----

    pcaVisualization(standardized_df, df1["label"])

    plt.show()

