""" Usage:
    data_processing.py convert --yaml-file=<yaml-file> [--validate --info] [options]
    data_processing.py validate --json-file=<json-file>
    data_processing.py info --json-file=<json-file>

    Options:
        --print-every=<print-every>         How often to print status during validation [default: 50]              
"""

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import re
from docopt import docopt
from utils import read_json, save_json, read_yaml

TIME_STEP = 8.3 #TODO: check if units are important, right now this is in milliseconds
MISSING_VALUE = 0.0 #i.e., a 0 indicates a missing value
COLUMN_TO_REMOVE = 30 #synthetically create missing data point in this column for evaluation purposes
MISSING_PROB = 0.3 # probability that we manually remove a datapoint for evaluation purposes
SUPPORTED_FILE_TYPES = set(["csv"])



def validate_pair(pair, dimension):
    """
    Takes the forward and backward sequences and runs some basic validation.
    This can be edited depending on your data and what you want to validate.

    :param pair: dict of {"forward": # forward sequence, "backward": # backward sequence}
    """
    forward = pair["forward"]
    backward = pair["backward"]
    assert len(forward) == len(backward), "Length of forward and backward sequences don't match!"
    for i in range(len(forward)):
        forward_x = forward[i]
        backward_x = backward[-(i + 1)]
        assert forward_x == backward_x, "Forward and backward sequences don't match up! Forward is {} and backward is {}".format(forward_x,  backward_x)
        assert len(forward_x["x_t"]) == dimension, "Dimension of measurement doesn't match! Expected {} but got {}".format(dimension, len(forward_x["x_t"]))

def validate(args):
    """
    Takes the path to the JSON file created from the convert method and
    validates its structure. This should be edited depending on your data
    and what you want to validate.
    

    :param input_file_path: the file path to the .json file created from convert
    :param print_every: int determining how often method prints status
    """

    print_banner()
    print("RUNNING VALIDATION\n")

    json_data = read_json(args["--json-file"])
    print("{} sequences in this file.".format(len(json_data)))
    print_every = int(args["--print-every"])
    dimension = len(json_data[0]["forward"][0]["x_t"])
    for i, pair in enumerate(json_data):
        if (i + 1) % print_every == 0:
            print("Validated {} sequences so far.".format(i + 1))
        validate_pair(pair, dimension)

    print("Validation passed!")
    print_banner()

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
    eval_masks[COLUMN_TO_REMOVE] = np.random.choice([0, 1], p=[1 - MISSING_PROB, MISSING_PROB])

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

def extract_forward_and_backward(df, max_len):
    """
    Takes dataframe for a file and extracts the forward and backward sequence to be saved to JSON

    :param df: Pandas dataframe where each row is a time step
    :param max_len (int): maximum sequence length (sequences longer than this get split)
        * If max_len is None, then no splitting is performed

    :returns: tuple storing (forward, backward)
        * forward (list): list of sequences (no longer than max_len) in the original order
        * backward (list): list of sequences (no longer than max_len) in the reverse order (for bidirectional RNN)
    """
    num_rows = df.shape[0]
    masks, deltas = None, None
    forward = []
    for t in range(num_rows):
        x_t = df.iloc[t][:60] # for now just focusing on first 60 columns
        ret, masks, deltas = generate_JSON_entry(x_t, masks, deltas)
        forward.append(ret)
    
    if max_len is None:
        backward = forward[::-1]
        return ([forward], [backward])
    else:
        num_splits = num_rows // max_len # NOTE: this discards sequences < max_len
        forward_split = [forward[i * max_len : (i + 1) * max_len] for i in range(num_splits)]
        backward_split = [forward_seq[::-1] for forward_seq in forward_split]
        return (forward_split, backward_split)

def extract_json_entries(file_path, max_len):
    """
    Takes path to data file and extracts the forward and
    backward sequences to be saved to JSON

    :param file_path (str): path to file storing sequence of measurements
    :param max_len (int): maximum sequence length (sequences longer than this get split)

    :returns: list of dicts storing 
        * forward: the JSON information in the original sequence order
        * backward: the JSON information in the reverse order
                    (for bidirectional RNN)
    """
    forward, backward = None, None
    file_extension = os.path.splitext(file_path)[1]
    if file_extension == ".csv":
        df = load_and_clean_df(file_path)
        forward, backward = extract_forward_and_backward(df, max_len)
    return [{"forward": f, "backward": b} for f, b in zip(forward, backward)]

def plot_dim(dim, df = None):
    if df is None:
        df = pd.read_csv(SAMPLE_FILE_PATH, header=None)
    dim_array = df[dim].values
    plt.plot(np.arange(dim_array.size), dim_array)

def extract_json(data_folder, regex, file_extension, max_len):
    json_data = []
    print("Extracting files using regex: {}".format(regex))
    regex = re.compile(regex)
    
    for file in os.listdir(data_folder):
        if regex.search(file):
            print("Converting file {} to JSON".format(file))
            file_path = os.path.join(data_folder, file)
            json_entries = extract_json_entries(file_path, max_len)
            json_data.extend(json_entries)
    return json_data

def print_banner():
    print("-" * 80)

def random_split(json_data, split):
    indices = np.random.permutation(len(json_data)) # randomly split train data
    num_train_seqs = int(len(json_data) * split)
    
    train_indices = indices[:num_train_seqs]
    test_indices = indices[num_train_seqs:]
    train_data = [json_data[index] for index in train_indices]
    test_data = [json_data[index] for index in test_indices]
    return train_data, test_data

def convert(args):
    """
    Converts files into JSON format for BRITS models

    :param yaml_file: configuration file with the following information
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
        - val_split: proportion of training sequences to use as validation set
            - if None, then no splitting will be done
            - otherwise, validation data will saved into separate file
        - output_folder: path to folder to save JSON data
        - output_file_name: file name to save JSON output file
            - if train_split is not None, then JSON data will be saved into two files
                - train set will be prefixed by "train_", test set prefixed by "test_"
        - max_len: maximum sequence length
            - if sequence is longer than max_len, gets split
            - can be None for no splitting
        - seed: random seed
            - current randomness comes from random train/test split and random removal
              of values for imputation

    :param validate_after: boolean determining whether to run validation at the end
        - validation can always be run from the command line

    :param info_after: boolean determining whether to print info about JSON data
        - info can always be run from the command line
    """

    print_banner()
    print("RUNNING CONVERSION\n")

    yaml_data = read_yaml(args["--yaml-file"])

    data_folder = yaml_data["data_folder"]
    file_extension = yaml_data["file_extension"]
    file_regex = yaml_data["file_regex"]
    output_file_name = yaml_data["output_file_name"]
    output_folder = yaml_data["output_folder"]
    train_split = yaml_data["train_split"]
    val_split = yaml_data["val_split"]
    max_len = yaml_data["max_len"]
    seed = yaml_data["seed"]

    if file_extension not in SUPPORTED_FILE_TYPES:
        print(".{} is not currently a supported file type")
        return

    if seed is not None:
        print("Seeding random generator with seed {}".format(seed))
        np.random.seed(seed)
    else:
        print("WARNING: The random number generator has not been seeded.")
        print("You are encouraged to run again with a random seed for reproducibility!")

    file_regex = file_regex if file_regex is not None else ".*"
    file_regex += r"\." + file_extension
    json_data = extract_json(data_folder, file_regex, file_extension, max_len)

    output_files = []

    train_data = json_data
    test_data = None 
    val_data = None

    if train_split is not None:
        train_data, test_data = random_split(train_data, train_split)
        save_path = save_json(test_data, output_folder, "test_" + output_file_name)
        output_files.append(save_path)

    if val_split is not None:
        train_data, val_data = random_split(train_data, 1 - val_split)
        save_path = save_json(val_data, output_folder, "val_" + output_file_name)
        output_files.append(save_path)

    print("Using {} sequences for training".format(len(train_data)))
    save_path = save_json(train_data, output_folder, "train_" + output_file_name)
    output_files.append(save_path)
        

    print_banner()

    if args["--validate"]: # run validation on newly saved JSON
        for output_file in output_files:
            args["--json-file"] = output_file
            validate(args)

    if args["--info"]: # print info about newly saved JSON
        for output_file in output_files:
            print_info(output_file)

def print_info(json_file):
    print_banner()
    print("PRINTING INFO\n".format(json_file))

    json_data = read_json(json_file)
    print("{} sequences in this file.".format(len(json_data)))
    print("{} total measurements".format(sum(len(seq["forward"]) for seq in json_data)))
    print("{} missing values".format(sum(sum(measure["eval_masks"]) for seq in json_data for measure in seq["forward"] )))
    print("Shortest sequence is of length {}".format(min(len(seq["forward"]) for seq in json_data)))
    print("Longest sequence is of length {}".format(max(len(seq["forward"]) for seq in json_data)))

    print_banner()

def main(args):
    if args["convert"]:
        convert(args)
    elif args["validate"]:
        validate(args)  
    elif args["info"]:
        json_file = args["--json-file"] 
        print_info(json_file)
    
if __name__ == "__main__":
    args = docopt(__doc__)
    main(args)

