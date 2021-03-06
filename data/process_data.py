import sys

import pandas as pd
from sqlalchemy import create_engine
import numpy as np



def load_data(messages_filepath, categories_filepath):
    """
    input arguments: 
          message_filepath: directory path of the message dataset
          categories_filepath: directory path of the categories dataset
    output: return the loaded data sets as a merged dataframe

    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = pd.merge(messages,categories,on='id')
    return df 


def clean_data(df):
    """
    input: the loaded dataframe
    output: a cleaned data dataframe
    """
    # Fix the categories columns name
    categories = df['categories'].str.split(pat=';', expand=True)
    row = categories.loc[0]
    colnames = []
    for entry in row:
        colnames.append(entry[:-2])
    category_colnames = colnames
    categories.columns = category_colnames
    
    for column in categories:
        categories[column] = categories[column].str[-1:]
        categories[column] = categories[column].astype(int)
        
    df.drop('categories', axis=1, inplace=True)
    df = pd.concat([df, categories], axis=1)
    df.drop_duplicates(inplace=True)
    # Removing entry that is non-binary, 'related' contians value as 2 
    df = df[df['related'] != 2]
    return df



def save_data(df, database_filename):
    """
    Save Data to SQLite Database Function
    
    Arguments:
        df -> Combined data containing messages and categories with categories cleaned up
        database_filename -> Path to SQLite destination database
    """
    
    engine = create_engine('sqlite:///'+ database_filename)
    #table_name = database_filename.replace(".db","") + "_table"
    df.to_sql('messages', engine, index=False, if_exists ='replace')
  
def main():
    """
    Main function which will kick off the data processing functions. There are three primary actions taken by this function:
        1) Load Messages Data with Categories
        2) Clean Categories Data
        3) Save Data to SQLite Database
    """
    
    # Print the system arguments
    # print(sys.argv)
    
    # Execute the ETL pipeline if the count of arguments is matching to 4
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:] # Extract the parameters in relevant variable

        print('Loading messages data from {} ...\nLoading categories data from {} ...'
              .format(messages_filepath, categories_filepath))
        
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning categories data ...')
        df = clean_data(df)
        
        print('Saving data to SQLite DB : {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data has been saved to database!')
    
    else: # Print the help message so that user can execute the script with correct parameters
        print("Please provide the arguments correctly: \nSample Script Execution:\n\
> python process_data.py disaster_messages.csv disaster_categories.csv disaster_response_db.db \n\
Arguments Description: \n\
1) Path to the CSV file containing messages (e.g. disaster_messages.csv)\n\
2) Path to the CSV file containing categories (e.g. disaster_categories.csv)\n\
3) Path to SQLite destination database (e.g. disaster_response_db.db)")

if __name__ == '__main__':
    main()
