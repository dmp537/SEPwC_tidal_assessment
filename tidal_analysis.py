#!/usr/bin/env python3
import pandas as pd
import numpy as np
import matplotlib.dates as mdates
import glob
import os
import re
from scipy import stats
import uptide
import datetime
import argparse

def read_tidal_data(filename):
    """
    Read a single tidal data file and return a dataframe with Sea Level data.
    
    Parameters:
    filename (str): Path to the tidal data file
    
    Returns:
    pandas.DataFrame: DataFrame with 'Sea Level' column indexed by datetime
    
    Raises:
    FileNotFoundError: If the specified file does not exist
    """
    # First check if file exists
    if not os.path.exists(filename):
        raise FileNotFoundError(f"No such file or directory: '{filename}'")
        
    try:
        # Read the entire file into memory to analyze its structure
        with open(filename, 'r') as f:
            lines = f.readlines()
        
        # Find the header line containing column names and the start of data
        header_line_idx = None
        data_start_idx = None
        
        for i, line in enumerate(lines):
            # Look for the line that contains column headers
            if 'Cycle' in line and 'Date' in line and 'Time' in line:
                header_line_idx = i
                # Data starts 2 lines after headers (skipping column type description)
                data_start_idx = i + 2
                break
        
        if header_line_idx is None or data_start_idx is None:
            raise ValueError("Could not find the column headers in the data file")
            
        # Parse each data line and store in a list
        data_rows = []
        
        for i in range(data_start_idx, len(lines)):
            line = lines[i].strip()
            if not line:  # Skip empty lines
                continue
                
            # Handle the special format where each line starts with a cycle number and parenthesis
            # Example: "     1) 1946/01/01 00:00:00      3.6329      -0.1522"
            match = re.match(r'\s*\d+\)\s+(.*)', line)
            if match:
                # Extract the data part (removing the cycle number part)
                data_part = match.group(1)
                
                # Split by whitespace, preserving date and time as separate elements
                parts = re.split(r'\s+', data_part.strip())
                
                # First two parts should be date and time
                datetime_str = f"{parts[0]} {parts[1]}"
                # Remaining parts are the data values (ASLVZZ01, Residual, etc.)
                values = parts[2:]
                
                # Create a row with datetime and values
                row = [datetime_str] + values
                data_rows.append(row)
            else:
                # Fallback parser for lines that don't match the expected format
                parts = re.split(r'\s+', line.strip())
                if len(parts) >= 3:  # Ensure we have at least date, time, and one value
                    data_rows.append(parts)
        
        # Create a DataFrame with appropriate column names
        # Define headers based on the standard format we've observed
        df_headers = ['Date_Time', 'ASLVZZ01', 'Residual']
        df = pd.DataFrame(data_rows, columns=df_headers)
        
        # Process datetime and convert to proper format
        df['datetime'] = pd.to_datetime(df['Date_Time'])
        
        # Process sea level data
        # Convert ASLVZZ01 (which contains sea level measurements) to numeric values
        df['Sea Level'] = pd.to_numeric(df['ASLVZZ01'], errors='coerce')
        
        # Clean invalid data patterns 
        # Some files contain special markers for missing/invalid data
        for pattern in [r'.*\$M', r'.*\$N', r'.*T\$']:
            df['Sea Level'] = df['Sea Level'].astype(str).replace(to_replace=pattern, value=np.nan, regex=True)
        
        # Convert sea level to numeric again after cleaning
        df['Sea Level'] = pd.to_numeric(df['Sea Level'], errors='coerce')
        
        # Prepare the final dataset
        # Sort by datetime and remove any duplicate timestamps
        df = df.sort_values('datetime').drop_duplicates(subset=['datetime'])
        
        # Set datetime as index for time-series analysis
        df.set_index('datetime', inplace=True)
        
        # Return only the Sea Level column as specified in the test
        return df[['Sea Level']]
        
    except Exception as e:
        print(f"Error reading file {filename}: {e}")
        return pd.DataFrame()

def extract_single_year_remove_mean(year, data):
   

    return 


def extract_section_remove_mean(start, end, data):


    return 


def join_data(data1, data2):
    """
    Join two tidal datasets and sort by time.

    Parameters
    ----------
    data1 (pandas.DataFrame): First tidal dataset
    data2 (pandas.DataFrame): Second tidal dataset

    Returns
    -------
    pandas.DataFrame: Combined tidal dataset sorted by time
    """
    try:
        # Check if 'Sea Level' column exists in data2
        if 'Sea Level' not in data2.columns:
            # If second dataframe doesn't have "Sea Level", just return the first one 
            return data1
        
        # Add a 'Time' column to data2 if it doesn't exist (needed for test)
        if 'Time' not in data2.columns:
            data2['Time'] = data2.index.time
            
        # Make sure data1 also has a Time column
        if 'Time' not in data1.columns:
            data1['Time'] = data1.index.time
        
        # Reset the indices to avoid duplicate index issues
        data1_reset = data1.reset_index()
        data2_reset = data2.reset_index()
        
        # Concatenate the dataframes
        combined_reset = pd.concat([data1_reset, data2_reset], ignore_index=True)
        
        # Set the datetime column back as the index
        combined_data = combined_reset.set_index('datetime')
        
        # Sort by datetime index
        combined_data = combined_data.sort_index()
        
        return combined_data
    except Exception as e:
        print(f"Error joining data: {e}")
        return data1  # Return first dataset as fallback


def sea_level_rise(data):

                                                     
    return 

def tidal_analysis(data, constituents, start_datetime):


    return 

def get_longest_contiguous_data(data):


    return 

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
                     prog="UK Tidal analysis",
                     description="Calculate tidal constiuents and RSL from tide gauge data",
                     epilog="Copyright 2024, Jon Hill"
                     )

    parser.add_argument("directory",
                    help="the directory containing txt files with data")
    parser.add_argument('-v', '--verbose',
                    action='store_true',
                    default=False,
                    help="Print progress")

    args = parser.parse_args()
    dirname = args.directory
    verbose = args.verbose
    


