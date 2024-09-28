# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 10:10:40 2024

@author: OlivervonGruenigen
"""

from pathlib import Path
import pandas as pd
import os
import numpy as np
import csv

def get_pix_to_um(magnification, resolution):
    
    # Define the pix_to_um value based on the magnification
    if magnification == 'X20':
        pix_to_um = 5.2288  
    elif magnification == 'X30':
        pix_to_um = 3.4858  
    elif magnification == 'X40':
        pix_to_um = 2.5991
    elif magnification == 'X50':
        pix_to_um = 2.0793
    elif magnification == 'X80':
        pix_to_um = 1.2952 
    elif magnification == 'X100':
        pix_to_um = 1.0336  
    
    # Adjust pix_to_um based on resolution (HD or Normal)
    if resolution == 'hd':
        pix_to_um *= (2880 / 4000)
    
    return pix_to_um

def merge_data(paths, session_folder):
    data = []

    for path in paths:
        csv_file = Path(path).parent / (Path(path).stem + "_Data.csv")
        temp_data = pd.read_csv(csv_file)
        data.append(temp_data)
        
    # Use pd.concat to combine all DataFrames in the list into a single DataFrame
    all_data = pd.concat(data, ignore_index=True)

    # Save all data file
    all_data_filename = "all_data.csv"
    all_data_path = os.path.join(session_folder, all_data_filename)
    all_data.to_csv(all_data_path, index=False, float_format='%.3f')
    
    # Summarize the data
    summary_data = summarize_data(all_data, session_folder)

    return all_data, summary_data


def summarize_data(data, session_folder):
    headers = ['Caps. Dia. [µm]', 'Circularity', 'Core/Shell Ratio [vol.]','Core Dia. [µm]', '----------',
               'Total Agg.[%]', 'Doublets [%]', 'Triplets [%]', '> Triplets [%]']
    
    dataframe = [
        ['mean'],
        ['std'],
        ['cv']
    ]

    value_list = ["sFeret", "sCircl", "sCore", "s_csRatio","s_cFeret"]
    aggTotal = 0
    doublets = 0
    triplets = 0
    more_triplets = 0

    for value in value_list:
        if value == "sCore":
            for cores in data[value]:
                if cores == 2:
                    aggTotal += 1
                    doublets += 1
                elif cores == 3:
                    aggTotal += 1
                    triplets += 1
                elif cores > 3:
                    aggTotal += 1
                    more_triplets += 1
        else:
            dataframe[0].append(np.round(np.mean(data[value]),2))
            dataframe[1].append(np.round(np.std(data[value]),2))
            dataframe[2].append(np.round(np.std(data[value]) / np.mean(data[value]),2))

    dataframe[0].append('----------')
    dataframe[1].append('----------')
    dataframe[2].append('----------')

    totalCaps = np.size(data[value_list[0]])
    aggTotal = aggTotal / totalCaps * 100
    dataframe[0].append(np.round(aggTotal,2))
    dataframe[1].append('----------')
    dataframe[2].append('----------')
    doublets = doublets / totalCaps * 100
    dataframe[0].append(np.round(doublets,2))
    dataframe[1].append('----------')
    dataframe[2].append('----------')
    triplets = triplets / totalCaps * 100
    dataframe[0].append(np.round(triplets,2))
    dataframe[1].append('----------')
    dataframe[2].append('----------')
    more_triplets = more_triplets / totalCaps * 100
    dataframe[0].append(np.round(more_triplets,2))
    dataframe[1].append('----------')
    dataframe[2].append('----------')

    # Create the CSV file
    summary_filename = "summary_data.csv"
    summary_path = os.path.join(session_folder, summary_filename)

    with open(summary_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        # Write the headers
        writer.writerow(['----------'] + headers)  # Add an empty cell at the beginning for the row names
        # Write the data
        writer.writerows(dataframe)

    return summary_path