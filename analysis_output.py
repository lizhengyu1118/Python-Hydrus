# -*- coding: utf-8 -*-
"""
Analysis Output Module.

This module provides standardized functions for saving analysis results,
including plots (PNG) and data (CSV).
It ensures that output paths are printed to the console upon saving.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt

def log_save_message(filepath):
    """
    Prints a standardized 'File saved to:' message to the console.
    
    Args:
        filepath (str): The full path to the saved file.
    """
    print(f"File saved to: {filepath}")

def save_plot_png(plt_object, task_name, output_dir, filename_suffix=""):
    """
    Saves the current matplotlib.pyplot figure to a PNG file and
    prints the save location to the console.
    
    Also shows and closes the plot.
    
    Args:
        plt_object (module): The matplotlib.pyplot module object (plt).
        task_name (str): The base name for the task (e.g., 'task_1').
        output_dir (str): The directory to save the file in.
        filename_suffix (str, optional): Suffix to differentiate plots 
                                         (e.g., '_daily_storage').
    """
    try:
        filename = f"{task_name}{filename_suffix}_plot.png"
        filepath = os.path.join(output_dir, filename)
        
        plt_object.savefig(filepath)
        log_save_message(filepath)
        
        plt_object.show()
        plt_object.close()
        
    except Exception as e:
        print(f"Error saving plot {filename}: {e}")

def save_results_csv(dataframe, task_name, output_dir, filename_suffix=""):
    """
    Saves a pandas DataFrame to a CSV file and prints the save
    location to the console.
    
    Args:
        dataframe (pd.DataFrame): The DataFrame to save.
        task_name (str): The base name for the task (e.g., 'task_1').
        output_dir (str): The directory to save the file in.
        filename_suffix (str, optional): Suffix to differentiate data 
                                         (e.g., '_daily_storage').
    """
    try:
        filename = f"{task_name}{filename_suffix}_data.csv"
        filepath = os.path.join(output_dir, filename)
        
        dataframe.to_csv(filepath)
        log_save_message(filepath)
        
    except Exception as e:
        print(f"Error saving CSV {filename}: {e}")

def log_custom_save(filepath):
    """
    A wrapper for logging file saves that don't use the
    standard plot/CSV functions (e.g., Task 3's TXT file).
    
    Args:
        filepath (str): The full path to the saved file.
    """
    log_save_message(filepath)
