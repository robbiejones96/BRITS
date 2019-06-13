# Description
The source code of RITS-I, RITS, BRITS-I, BRITS models for time-series imputation/classification. These are based on the original implementations in the repo from which this one is forked. The original paper can be found [here](https://arxiv.org/pdf/1805.10572.pdf).

To train the model, run:

`python main.py train --yaml-file=<path-to-training-yaml-file>`
   
We use YAML files to organize the experiments (e.g., keep track of hyperparameter values). The format of the YAML files are described in a section below.

The two Python files to run are data_processing.py (for data processing) and main.py (for training/validating/testing). They both have [docopt](https://github.com/docopt/docopt) strings at the top, so you can run `python data_processing.py -h` or `python main.py -h` to see the different command line options.

The Python version for this code is Python 3.6. There is a requirements.txt included for [pip](https://pip.pypa.io/en/stable/).

# Data Format
The training/validation/testing data needs to be in JSON format and saved to disk.
The data format is as follows:

* All of the data should be stored in one big JSON list.

* Each entry in the list is a dictionary which represents one time series (which will have multiple measurements).
   * Each dictionary has two keys
      * "forward"
      * "backward"
   * "forward" and "backward" each map to a list of dictionaries, which represents the time series in forward/backward directions. Each dictionary in the list encapsulates one time step in the time series. The key/value pairs in the dictionary are:
      * evals: (list) the original time step measurement with all values present
      * eval_masks: (list) this specifies which dimensions in evals to purposefully treat as missing for evaluation purposes
      * x_t: (list) this is the original measurement but with some dimensions set to missing (specified by eval_masks)
      * masks: (list) list of floats that represents delta_t in the paper
      * deltas: (list) list of masks for timestep t, represents m_t in the paper
    
# YAML File Format (subject to change)
We use YAML files to keep track of all of the experiments as well as help with processing data into the right JSON format. If you already have a way to get your data into the format specified above, then you only need to look at the YAML format for running training.

## Data Processing
* data_folder: # path to folder storing data files  
* file_extension: # file format of the data files (right now only supports CSV)
* output_folder: # path to folder to save JSON data  
* output_file_name: # JSON output file name (don’t include “.json”)  
* file_regex:  # regex to select only certain data files for processing   
   * If empty, defaults to .*  (i.e., chooses all files with the specified file extension)
* max_len: 128 # maximum sequence length
   * Sequence gets chunked into smaller sequences if longer than max_len  
      * Any sequence shorter than 128 gets discarded  
   * If empty, no chunking is performed  
* train_split: # proportion of sequences to use as training set  
   * Can be empty for no train/test split  
* val_split: # proportion of training sequences to use as validation set  
   * Can be empty for no train/val split  
* seed: # random seed (please use for reproducibility!)  

## Training
* experiment_name: # name of experiment
* train_data: # path to JSON storing training data
* val_data: # path to JSON storing validation data
   * Can be empy if no validation data
* model: # name of model to train
* batch_size: # size of minibatches
* hidden_size: # size of hidden layer in recurrent model
* results_folder: # path to folder to save trained model and statistics
* optimizer: # name of optimizer from torch.optim to use (e.g., Adam)
* learning_rate: # learning rate to use for optimizer (float)
* max_epoch: # number of epochs to train for (int)
* seed: # random seed (please use for reproducibility!)


# Data Download Links (from original repo)

* Air Quality Data:
URL: https://www.microsoft.com/en-us/research/wp-content/uploads/2016/06/STMVL-Release.zip

* Health-care Data:
URL: https://physionet.org/challenge/2012/
We use the test-a.zip in our experiment.

* Human Activity Data:
URL: https://archive.ics.uci.edu/ml/datasets/Localization+Data+for+Person+Activity
