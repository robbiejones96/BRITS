""" Usage:
    data_processing.py convert --data-folder=<data-folder> [options]
    data_processing.py validate --json-file=<json-file>
    data_processing.py info --json-file=<json-file>

    Options:
        --file-extension=<file-extension>                       File format for the data files [default: csv]
        --output-file=<output-file>                             JSON output file name [default: output]
        --file-regex=<file-regex>                               Regex to match files for extraction    
        --train-split=<train-split>                             Proportion (float) of files to use as training set   
        --seed=<seed>                                           Random seed                
"""

import pandas as pd
import numpy as np
import os
import json
import matplotlib.pyplot as plt
import re
from docopt import docopt
from pdb import set_trace

TIME_STEP = 8.3 #TODO: check if units are important, right now this is in milliseconds
MISSING_VALUE = 0.0 #i.e., a 0 indicates a missing value
COLUMN_TO_REMOVE = 30 #synthetically create missing data point in this column for evaluation purposes
MISSING_PROB = 0.3 # probability that we manually remove a datapoint for evaluation purposes
SUPPORTED_FILE_TYPES = set(["csv"])

def read_json(json_file):
    json_data = None
    print("Loading json data from {}".format(os.path.abspath(json_file)))
    with open(json_file, 'r') as input_file:
        json_data = json.load(input_file)
    return json_data

def save_json(json_data, data_folder, json_file_name):
    output_file_path = os.path.join(data_folder, json_file_name + ".json")
    print("Saving JSON to {}".format(os.path.abspath(output_file_path)))
    with open(output_file_path, 'w') as output_file:
        json.dump(json_data, output_file)

def validate_pair(pair):
    """
    Takes the forward and backward 
    """
    forward = pair["forward"]
    backward = pair["backward"]
    assert len(forward) == len(backward), "Length of forward and backward sequences don't match!"
    for i in range(len(forward)):
        forward_x = forward[i]
        backward_x = backward[-(i + 1)]
        assert forward_x == backward_x, "Forward and backward sequences don't match up! Forward is {} and backward is {}".format(forward_x,  backward_x)

def validate(input_file_path, print_every = 50):
    """
    Takes the path to the JSON file created from the convert method and
    validates its structure. This should be edited depending on your data
    and what you want to validate.
    

    :param input_file_path: the file path to the .json file created from convert
    :param print_every: int determining how often method prints status
    """
    json_data = read_json(input_file_path)
    print("{} sequences in this file.".format(len(json_data)))
    for i, pair in enumerate(json_data):
        if (i + 1) % print_every == 0:
            print("Validated {} sequences so far.".format(i + 1))
        validate_pair(pair)

def load_and_clean_df(file_path):
    """
    Loads a dataframe from the given csv file and does a little cleaning.
    This should be edited depending on how you want your dataframes to look.
    With this specific code, this function
        * Gives nice column names to the dataframe
        * Gets rid of columns 100 and 101

    :param file_path: the file path to the .csv file
    :returns: Pandas dataframe
    """
    df = pd.read_csv(file_path, header=None)

    # the following can be removed/edited depending on your data and needs
    new_column_names = ["Angle_" + str(num + 1) for num in range(30)]
    new_column_names += ["Marker_" + str(num + 1) for num in range(30)]
    new_column_names += [str(num + 1) for num in range(60, len(df.columns))]
    df.columns = new_column_names
    df = df.drop(labels=["100", "101"], axis="columns")

    return df

def generate_JSON_entry(row, prev_masks = None, prev_deltas = None):
    """
    Takes datapoint and extracts information into JSON-writable format

    :param row: Pandas dataframe row that stores the measurement at time t
    :param prev_mask: numpy array of 0s and 1s that represents m_(t - 1) in the paper
    :param prev_deltas: numpy array of floats that represents delta_(t - 1) in the paper
    :returns: tuple storing (ret, masks, deltas)
        * ret: a dict storing the following JSON information (all of type list):
            * x_t: this is the original measurement but with some dimensions removed (specified by eval_masks)
            * eval_masks: this specifies which values in evals to purposefully treat as missing for evaluation purposes
            * evals: the original sequence that we will use to evaluate our model
            * deltas: list of masks for timestep t, represents m_t in the paper
            * masks: list of floats that represents delta_t in the paper
        * masks: numpy array of masks for timestep t, represents m_t in the paper
        * deltas: numpy array of floats that represents delta_t in the paper
    """

    evals = row
    dim = len(row)
    
    # we want to introduce missing values so we can evaluate our model later
    # later should probably take in eval_masks as a parameter that we can specify
    eval_masks = np.zeros(dim, dtype=int)
    #eval_masks[COLUMN_TO_REMOVE] = np.random.choice([0, 1], p=[1 - MISSING_PROB, MISSING_PROB])

    x_t = row.copy()
    x_t[eval_masks == 1] = MISSING_VALUE
    masks = np.where(x_t == MISSING_VALUE, 0, 1)
    if eval_masks[COLUMN_TO_REMOVE]:
        assert masks[COLUMN_TO_REMOVE] == 0
    if prev_masks is None: # this is the first time step
        deltas = np.zeros(dim)
    else:
        deltas = np.full(dim, TIME_STEP) # s_t - s_(t-1)
        deltas[prev_masks == 0] += prev_deltas[prev_masks == 0]
    
    ret = {}
    ret["x_t"] = x_t.values.tolist()
    ret["eval_masks"] = eval_masks.tolist()
    ret["evals"] = evals.tolist()
    ret["deltas"] = deltas.tolist()
    ret["masks"] = masks.tolist()
    return (ret, masks, deltas)

def extract_forward_and_backward(df):
    """
    Takes dataframe for a file and extracts the forward and backward sequence to be saved to JSON

    :param df: Pandas dataframe where each row is a time step
    :returns: tuple storing (forward, backward)
        * forward: the JSON information in the original sequence order
        * backward: the JSON information in the reverse order (for bidirectional RNN)
    """
    num_rows = df.shape[0]
    masks, deltas = None, None
    forward = []
    for t in range(num_rows):
        x_t = df.iloc[t][:60] # for now just focusing on first 60 columns
        ret, masks, deltas = generate_JSON_entry(x_t, masks, deltas)
        forward.append(ret)
    backward = forward[::-1]
    return (forward, backward)

def extract_json_entry(file_path):
    """
    Takes path to data file and extracts the forward and
    backward sequences to be saved to JSON

    :param file_path: path to file storing sequence of measurements
    :returns: dict storing 
        * forward: the JSON information in the original sequence order
        * backward: the JSON information in the reverse order
                    (for bidirectional RNN)
    """
    forward, backward = None, None
    file_extension = os.path.splitext(file_path)[1]
    if file_extension == ".csv":
        df = load_and_clean_df(file_path)
        forward, backward = extract_forward_and_backward(df)
    return {"forward": forward, "backward": backward}

def plot_dim(dim, df = None):
    if df is None:
        df = pd.read_csv(SAMPLE_FILE_PATH, header=None)
    dim_array = df[dim].values
    plt.plot(np.arange(dim_array.size), dim_array)

def extract_json(data_folder, regex, file_extension):
    json_data = []
    print("Extracting files using regex: {}".format(regex))
    regex = re.compile(regex)
    
    for file in os.listdir(data_folder):
        if regex.search(file):
            print("Converting file {} to JSON".format(file))
            file_path = os.path.join(data_folder, file)
            json_entry = extract_json_entry(file_path)
            json_data.append(json_entry)
    return json_data

def convert(args):
    """
    Converts files into JSON format for BRITS models

    :param args: dict of command line arguments 
        - data_folder: path to folder storing data files to be converted
        - file_extension: the file extension of the data files to be converted
            - only the file types in SUPPORTED_FILE_TYPES (see constants at top)
              are supported
        - file_regex: regex to specify file names to be converted
            - if file_regex is None, then method will convert all files of
              type file_extension
        - train_split: proportion of files to use as training set
            - if None, then JSON data will be saved into one file
            - otherwise, will be saved into two files, one for train and one for test
        - output_file_name: file name to save JSON output file
            - if train_split is not None, then JSON data will be saved into two files
                - train set will be prefixed by "train_", test set prefixed by "test_"
        - seed: random seed
            - current randomness comes from random train/test split and random removal
              of values for imputation
    """

    data_folder = args["--data-folder"]
    file_extension = args["--file-extension"]
    file_regex = args["--file-regex"]
    output_file_name = args["--output-file"]
    train_split = float(args["--train-split"])
    seed = int(args["--seed"])

    if seed is not None:
        print("Seeding random generator with seed {}".format(seed))
        np.random.seed(seed)

    if file_extension not in SUPPORTED_FILE_TYPES:
        print(".{} is not currently a supported file type")
        return

    file_regex = file_regex if file_regex is not None else ".*"
    file_regex += r"\." + file_extension
    json_data = extract_json(data_folder, file_regex, file_extension)

    if train_split is None:
        save_json(json_data, data_folder, output_file_name)
    else:
        indices = np.random.permutation(len(json_data)) # randomly split train/test
        num_train_files = int(len(json_data) * train_split)
        print("Using {} files for training".format(num_train_files))
        train_indices = indices[:num_train_files]
        test_indices = indices[num_train_files:]
        train_data = [json_data[index] for index in train_indices]
        test_data = [json_data[index] for index in test_indices]
        save_json(train_data, data_folder, "train_" + output_file_name)
        save_json(test_data, data_folder, "test_" + output_file_name)
    
    

def print_info(json_file):
    json_data = read_json(json_file)
    print("{} sequences in this file.".format(len(json_data)))
    print("{} total measurements".format(sum(len(seq["forward"]) for seq in json_data)))

def main(args):
    if args["convert"]:
        
        convert(args)
    elif args["validate"]:
        input_file = args["--json-file"]
        validate(input_file)  
    elif args["info"]:
        json_file = args["--json-file"] 
        print_info(json_file)
    
if __name__ == "__main__":
    args = docopt(__doc__)
    main(args)
