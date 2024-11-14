'''
Goal: juxtapose LLMs' and human annotation for their variable
'''
from asyncio.subprocess import Process
from typing import Optional

import pandas as pd
import json
import os
import numpy as np

from sympy import false


def read_LLM_outputs(task_folder_name, llm_model, candidate):
    """

    :param task_folder_name: e.g "Debate Statements"
    :param llm_model: e.g "gpt", "claude"
    :param candidate: e.g "trump", "harris"
    :return: df
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(os.path.dirname(script_dir))

    llm_file= os.path.join(root_dir, "Content", "LLM outputs", task_folder_name,
                 f"{llm_model}_processed_{candidate}_statements.json")
    with open (llm_file, 'r') as f:
         data = json.load(f)
    df_output= pd.DataFrame(data)
    return df_output

def read_human_annotations(task_folder_name, annotation_num, candidate):
    """
    :param task_folder_name: e.g "Debate Statements"
    :param annotation_num: the index of that annotator. For this research, only "1","2","3"
    :param candidate: "trump", "harris"
    :return: df
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(os.path.dirname(script_dir))
    human_annotations_file= os.path.join(root_dir, "Content","Annotation",task_folder_name, candidate, f"annotation{annotation_num}.json")
    with open (human_annotations_file, 'r') as f:
        data = json.load(f)["statements"]
    df_output= pd.DataFrame(data)
    return df_output


def process_df_llm(df_llm):
    """
    Process statement dataframe based on multiple conditions:
    1. If Processed is False, set Importance, Classification, and Ideology Score to null
    2. For processed statements:
        a. If Statement Importance is '0', set Classification and Ideology Score to null
        b. If Statement Classification is '0' and Ideology Score is '-1', set Importance to '0'
        c. Convert remaining Classification and Ideology Score to integers

    Parameters:
    df (pandas.DataFrame): Input dataframe with statement data

    Returns:
    pandas.DataFrame: Processed dataframe
    """
    # Create a copy to avoid modifying the original dataframe
    processed_df = df_llm.copy()

    def clean_value(value):
        """Clean values by extracting numbers and handling special cases"""
        if pd.isna(value) or value == '':
            return '0'
        if isinstance(value, str):
            # Remove any leading/trailing whitespace
            value = value.strip()

            # Handle 'Label: number' format
            if ':' in value:
                parts = value.split(':')
                if len(parts) >= 2 and parts[1].strip().split()[0].isdigit():
                    return parts[1].strip().split()[0]

            # If the value starts with a number or negative sign followed by a number
            import re
            match = re.match(r'^-?\d+', value)
            if match:
                return match.group(0)

            # If it's just a plain number string
            if value.isdigit() or (value.startswith('-') and value[1:].isdigit()):
                return value

            return '0'
        return str(value)

    # Clean all relevant columns first
    columns_to_clean = ['Statement Importance', 'Statement Classification', 'Statement Ideology Score']
    for column in columns_to_clean:
        # Print debug information
        print(f"\nUnique values in {column} before cleaning:")
        print(processed_df[column].unique())

        processed_df[column] = processed_df[column].apply(clean_value)

        # Print debug information
        print(f"\nUnique values in {column} after cleaning:")
        print(processed_df[column].unique())

    # Convert 'Processed' to boolean, handling NaN values
    processed_df['Processed'] = processed_df['Processed'].fillna(False)
    processed_df['Processed'] = processed_df['Processed'].astype(bool)

    # Handle unprocessed statements first
    unprocessed_mask = processed_df['Processed'] == False
    processed_df.loc[unprocessed_mask, columns_to_clean] = np.nan

    # Only process the statements where Processed is True
    processed_mask = processed_df['Processed'] == True

    # First condition: If Classification is '0' and Ideology Score is '-1', set Importance to '0'
    special_case_mask = processed_mask & \
                        (processed_df['Statement Classification'] == '0') & \
                        (processed_df['Statement Ideology Score'] == '-1')
    processed_df.loc[special_case_mask, 'Statement Importance'] = '0'

    try:
        # Convert Statement Importance to integer for comparison (only for processed statements)
        processed_df.loc[processed_mask, 'Statement Importance'] = \
            processed_df.loc[processed_mask, 'Statement Importance'].astype(int)

        # Create mask for rows where Importance is 0 (among processed statements)
        zero_importance_mask = processed_mask & (processed_df['Statement Importance'] == 0)

        # Set values to null where Importance is 0
        processed_df.loc[zero_importance_mask, ['Statement Classification', 'Statement Ideology Score']] = np.nan

        # Convert remaining Classification and Ideology Score entries to integers
        # Only for processed statements with non-zero importance
        valid_entries_mask = processed_mask & ~zero_importance_mask
        processed_df.loc[valid_entries_mask, 'Statement Classification'] = \
            processed_df.loc[valid_entries_mask, 'Statement Classification'].astype(int)
        processed_df.loc[valid_entries_mask, 'Statement Ideology Score'] = \
            processed_df.loc[valid_entries_mask, 'Statement Ideology Score'].astype(int)
    except ValueError as e:
        print("\nError during conversion:")
        print(e)
        print("\nCurrent state of data:")
        for column in columns_to_clean:
            print(f"\nUnique values in {column}:")
            print(processed_df[column].unique())
        raise

    return processed_df


def create_llm_csv(model_name, candidate, df):
    """
    Create CSV file from the processed dataframe,
    only keeping the original columns

    Parameters:
    model_name (str): Name of the LLM model
    df (pandas.DataFrame): Processed dataframe
    """
    original_columns = [
        'Statement ID',
        'Statement Topic',
        'Statement Content',
        'Statement Importance',
        'Statement Classification',
        'Statement Ideology Score',
        'Processed'  # Keep the original Processed column
    ]

    df1 = df[original_columns].copy()
    df1.to_csv(f"{candidate}_{model_name}_output.csv", index=False)

def create_juxtapose_csv(df, candidate, variable_name):
    df1 = df["Statement ID"].astype(str).str.zfill(3)  # Pad to 3 digits
    df1.to_csv(f"{candidate}_{variable_name}_juxtapose.csv", index=False)

def read_processed_llm_csv(model_name, candidate):
    return pd.read_csv(f"{candidate}_{model_name}_output.csv")

def read_juxtapose_csv(candidate, variable_name):
    return pd.read_csv(f"{candidate}_{variable_name}_juxtapose.csv")

def add_data(task_folder_name, variable_name, source_type, candidate, annotation_num= None, llm_model=None ):
    """

    :param task_folder_name: e.g "Debate Statements"
    :param variable_name: "Statement Ideology Score", "Statement Classification"
    :param source_type: "llm" or "human"
    :param candidate: "trump", "harris"
    :param annotation_num: the index of that annotator. For this research, only "1","2","3"
    :param llm_model: e.g. "gpt", "claude"
    """
    df= read_juxtapose_csv(candidate, variable_name)

    if source_type == "llm":
        data= read_processed_llm_csv(llm_model, candidate)
        df[llm_model]= data[variable_name]
    else:
        data= read_human_annotations(task_folder_name, annotation_num, candidate)
        df[f"human{annotation_num}"] = data[variable_name]
    df["Statement ID"]= df["Statement ID"].astype(str).str.zfill(3)
    df.to_csv(f"{candidate}_{variable_name}_juxtapose.csv", index=False)






if __name__ == '__main__':
    llm_model= "gpt"
    # Go up to root directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(os.path.dirname(script_dir))

    #the type of task
    task_folder_name= "Debate Statements"

    # the candidate
    candidate= "harris"
    #the variable to analyze
    variable_name= "Statement Classification"
    #annotator number
    annotation_num= "3"
    #paths
    llm_output_path= os.path.join(root_dir, "Content","LLM outputs", task_folder_name, f"{llm_model}_processed_{candidate}_statements.json")
    human_annotation_path= os.path.join(root_dir, "Content","Annotation",task_folder_name, candidate, f"annotation{annotation_num}.json")
    df= read_processed_llm_csv(llm_model, candidate)
    add_data(task_folder_name, variable_name, source_type="human", candidate=candidate, annotation_num=annotation_num)





