import os
import pandas as pd
import numpy as np
import sklearn
from sklearn import preprocessing, model_selection
# from sklearn.model_selection import cross_validate
import random
import csv
import sys


def apply_RFECV_mask(mask, *args):
    """Applies a binary mask to a dataframe to remove columns. Binary mask is
    created from recursive feature elimination and cross validation and
    optimizes the generalization of the model

    Args:
        :param mask (string): text file containing the binary mask
        :param *args (pandas dataframe): Dataframes containing columns to mask
    Returns:
        :new dataframes (pandas df): new dataframes with columns removed
    """
    assert os.path.isfile(mask), "please pass a string specifying mask location"
    dir_path = os.path.dirname(os.path.realpath(__file__))
    mask = os.path.join(dir_path, mask)
    # get mask data
    updated_args = []
    with open(mask, 'r') as f:
        reader = csv.reader(f)
        column_mask = list(reader)[0]
    # apply mask to columns
    column_indexes = []
    for dataframe in args:
        assert len(column_mask) == len(list(dataframe)), 'mask length {} does not match dataframe length {}'.format(len(column_mask), len(list(dataframe)))
        for i, col in enumerate(column_mask):
            if col.strip() == 'False':
                column_indexes.append(i)

        updated_args.append(dataframe.drop(dataframe.columns[column_indexes], axis=1))

    return updated_args


class data_base(object):
    """Handles all data fetching and preparation. Attributes
       can be assigned to csv files with the assignment operator. Typical use
       case is to set raw_data to a csv file matching the format found in
       Input files and then calling clean_raw_data(). This sets the clean_X_data,
       y_enrichment and target values. From this point you can split the data
       to train/test the model using our data. To predict your own data, make sure your excel sheet
       matches the format in <Input_Files/database.csv>. Then you can
       call db.predict = <your_csv_path>. The X_test and Y_test data will now
       be your data. Just remove the stratified_data_split from the pipeline
       because you will now not need to split any data.

       Args:
            None
       Attributes:
            :self._raw_data (Pandas Dataframe): Holds raw data in the same form as excel file. initialized after fetch_raw_data() is called
            :self._clean_X_data (Pandas Dataframe): Holds cleaned and prepared X data.
            :self._Y_enrichment (numpy array): Holds continous Y values
            :self._X_train (Pandas Dataframe): Holds the X training data
            :self._X_test (Pandas Dataframe): Holds the X test data
            :self._Y_train (Pandas Dataframe): Holds the Y training data
            :self._Y_test (Pandas Dataframe): Holds the T testing data
            :self._test_accesion_numbers (list): holds the accesion_numbers
            in the test set
        """
    _ENRICHMENT_SPLIT_VALUE = 0.85 # enrichment threshold to classify as bound or unbound
    categorical_data = ['Enzyme Commission Number', 'Particle Size', 'Particle Charge', 'Solvent Cysteine Concentration', 'Solvent NaCl Concentration']
    columns_to_drop = ['Protein Length', 'Sequence', 'Enrichment', 'Accesion Number']

    def __init__(self):
        self._raw_data = None
        self._clean_X_data = None
        self._Y_enrichment = None
        self._target = None
        self._X_train = None
        self._Y_train = None
        self._X_test = None
        self._Y_test = None
        self._test_accesion_numbers = None
        # If you want to use our model set this to your csv file using the assignment operator
        self._predict = None

    def clean_raw_data(self):
        """ Cleans the raw data, drops useless columns, one hot encodes, and extracts
        class information

        Args, Returns: None
        """
        self.clean_X_data = self.raw_data
        # Categorize Interprot identifiers n hot encoding
        self.clean_X_data = multi_label_encode(self.clean_X_data, 'Interprot')
        # one hot encode categorical data
        for category in self.categorical_data:
            self.clean_X_data = one_hot_encode(self.clean_X_data, category)

        # Grab some useful data before dropping from independent variables
        self.Y_enrichment = self.clean_X_data['Enrichment']
        accesion_numbers = self.clean_X_data['Accesion Number']
        # drop useless columns
        for column in self.columns_to_drop:
            self.clean_X_data = self.clean_X_data.drop(column, axis=1)

        self.clean_X_data = fill_nan(self.clean_X_data, 'Protein Abundance')
        self.clean_X_data = normalize_and_reshape(self.clean_X_data, accesion_numbers)
        self._target = classify(self.Y_enrichment, self._ENRICHMENT_SPLIT_VALUE) #enrichment or nsaf

        self.X_train = self.clean_X_data
        self.Y_train = self.target

    def clean_user_test_data(self, user_data):
        """This method makes it easy for other people to make predictions
        on their data.
        called by assignment operator when users set db.predict = <path_to_csv>

        Args:
            :param user_data: users data they wish to predict
        Returns:
            None
        """
        # Categorize Interprot identifiers n hot encoding
        user_data = multi_label_encode(user_data, 'Interprot')
        # one hot encode categorical data
        for category in self.categorical_data:
            user_data = one_hot_encode(user_data, category)

        # Grab some useful data before dropping from independant variables
        self.Y_test = user_data['Enrichment']
        accesion_numbers = user_data['Accesion Number']

        for column in self.columns_to_drop:
            user_data = user_data.drop(column, 1)

        user_data = fill_nan(user_data, 'Protein Abundance')
        self.X_test = normalize_and_reshape(user_data, accesion_numbers)
        self.Y_test = classify(self.Y_test, self._ENRICHMENT_SPLIT_VALUE) # enrichment or nsaf
        # Get accession number
        self.test_accesion_numbers = self.X_test['Accesion Number']
        self.X_train = self.X_train.drop('Accesion Number', axis=1)
        self.X_test = self.X_test.drop('Accesion Number', axis=1)

    def stratified_data_split(self, test_size=0.0):
        """Randomized stratified shuffle split that sets training and testing data

        Args:
            :param test_size (float): The percentage of data to use for testing
        Returns:
            None
        """
        assert 1.0 >= test_size >= 0.0, "test_size must be between 0 and 1"
        assert self.predict is None, "Remove stratified_data_split() if using your own data"

        self.X_train, self.X_test, self.Y_train, self.Y_test = model_selection.train_test_split(self.clean_X_data, self.target, test_size = test_size, stratify=self.target, random_state=int((random.random()*100)))
        self.test_accesion_numbers = self.X_test['Accesion Number']
        self.X_train = self.X_train.drop('Accesion Number', axis=1)
        self.X_test = self.X_test.drop('Accesion Number', axis=1)

    @staticmethod
    def fetch_raw_data(enm_database):
        """Fetches enm-protein data from a csv file
        called by assignment operator for db.raw_data

        Args:
            :param enm_database (str): path to csv database
        Returns:
            None
        """
        assert os.path.isfile(enm_database), "please pass a string specifying database location"

        dir_path = os.path.dirname(os.path.realpath(__file__))
        enm_database = os.path.join(dir_path, enm_database)
        try:
            raw_data = pd.read_csv(enm_database)
        except ValueError:
            raise ValueError("File is not a valid csv")

        return raw_data

    @property
    def X_train(self):
        if self._X_train is None:
            raise ValueError("Initialize X_train by calling stratified_data_split()")
        else:
            return self._X_train

    @property
    def X_test(self):
        if self._X_test is None:
            raise ValueError("Initialize X_test by calling stratified_data_split()")
        else:
            return self._X_test

    @property
    def Y_train(self):
        if self._Y_train is None:
            raise ValueError("Initialize Y_train by calling stratified_data_split()")
        else:
            return self._Y_train

    @property
    def Y_test(self):
        return self._Y_test

    @property
    def raw_data(self):
        if self._raw_data is None:
            raise ValueError("Initialize raw data by setting raw_data=<path.csv>")
        return self._raw_data

    @property
    def clean_X_data(self):
        if self._clean_X_data is None:
            raise ValueError("Initialize clean_X_data by calling clean_data()")
        else:
            return self._clean_X_data

    @property
    def Y_enrichment(self):
        if self._Y_enrichment is None:
            raise ValueError("Initialize Y_enrichment by calling clean_data()")
        else:
            return self._Y_enrichment

    @property
    def target(self):
        if self._target is None:
            raise ValueError("Initialize target by calling clean_data()")
        else:
            return self._target

    @property
    def test_accesion_numbers(self):
        if self._test_accesion_numbers is None:
            raise ValueError("Initialize test_accesion_numbers by calling stratified_data_split()")
        else:
            return self._test_accesion_numbers

    @property
    def predict(self):
        return self._predict

    @X_train.setter
    def X_train(self, path):
        if isinstance(path, str) and os.path.isfile(path):
            # If trying to set to value from excel
            self._X_train = self.fetch_raw_data(path)
        else:
            # If trying to set to already imported array
            self._X_train = path

    @X_test.setter
    def X_test(self, path):
        if isinstance(path, str) and os.path.isfile(path):
            # If trying to set to value from excel
            self._X_test = self.fetch_raw_data(path)
        else:
            # If trying to set to already imported array
            self._X_test = path

    @Y_train.setter
    def Y_train(self, path):
        if isinstance(path, str) and os.path.isfile(path):
            # If trying to set to value from excel
            self._Y_train = self.fetch_raw_data(path)
        else:
            # If trying to set to already imported array
            self._Y_train = path

    @Y_test.setter
    def Y_test(self, path):
        if isinstance(path, str) and os.path.isfile(path):
            # If trying to set to value from excel
            self._Y_test = self.fetch_raw_data(path)
        else:
            # If trying to set to already imported array
            self._Y_test = path

    @raw_data.setter
    def raw_data(self, enm_database):
        if isinstance(enm_database, str) and os.path.isfile(enm_database):
            self._raw_data = self.fetch_raw_data(enm_database)
        else:
            self._raw_data = enm_database

    @clean_X_data.setter
    def clean_X_data(self, path):
        if isinstance(path, str) and os.path.isfile(path):
            # If trying to set to value from excel
            self.clean_X_data = self.fetch_raw_data(path)
        else:
            # If trying to set to already imported array
            self._clean_X_data = path

    @Y_enrichment.setter
    def Y_enrichment(self, path):
        if isinstance(path, str) and os.path.isfile(path):
            # If trying to set to value from excel
            self._Y_enrichment = self.fetch_raw_data(path)
        else:
            # If trying to set to already imported array
            self._Y_enrichment = path

    @test_accesion_numbers.setter
    def test_accesion_numbers(self, path):
        if isinstance(path, str) and os.path.isfile(path):
            # If trying to set to value from excel
            self._Y_enrichment = self.fetch_raw_data(path)
        else:
            # If trying to set to already imported array
            self._test_accesion_numbers = path

    @predict.setter
    def predict(self, path):
        if os.path.isfile(path):
            self._predict = self.fetch_raw_data(path)
            self._predict = self.clean_user_test_data(self._predict)
        else:
            self._predict = path


def normalize_and_reshape(data, labels):
    """
    Normalizes the data using MinMaxScaler and concatenates it with the label information.
    
    Args:
        data (pd.DataFrame): The data to normalize.
        labels (pd.Series): The labels to concatenate with normalized data.
    
    Returns:
        pd.DataFrame: Normalized DataFrame with labels concatenated.

    Raises:
        ValueError: If input data is not a DataFrame or labels is not a Series.
    """
    if not isinstance(data, pd.DataFrame):
        raise ValueError("Input data must be a pandas DataFrame.")
    
    if not isinstance(labels, pd.Series):
        raise ValueError("Labels must be a pandas Series.")

    try:
        # Normalize the data
        norm_df = preprocessing.MinMaxScaler().fit_transform(data)
        data = pd.DataFrame(norm_df, columns=data.columns)

        # Concatenate labels
        data = pd.concat([labels.reset_index(drop=True), data], axis=1)
        data.reset_index(drop=True, inplace=True)
    except Exception as e:
        raise ValueError(f"Error during normalization: {e}")
    
    return data

def classify(data, cutoff):
    """
    Classifies continuous data as bound or unbound based on a cutoff value.

    Args:
        data (array-like): Array or list of continuous data to classify.
        cutoff (float): Cutoff value for classification.

    Returns:
        np.ndarray: Binary array where 1 indicates bound and 0 indicates unbound.
    
    Raises:
        ValueError: If the input data cannot be converted to a NumPy array.
    """
    try:
        data = np.array(data, dtype=float)  # Ensure the data is a NumPy array of floats
    except ValueError as e:
        raise ValueError(f"Error converting data to NumPy array: {e}")

    # Vectorized classification using NumPy
    classified_data = np.where(data >= cutoff, 1, 0)

    return classified_data

def fill_nan(data, column):
    """
    Fills NaN values in the specified column with the mean value of that column.

    This function finds NaN values in a specified column and replaces them with 
    the mean of the non-NaN values in that column.

    Args:
        data (pd.DataFrame): The DataFrame containing the data.
        column (str): The name of the column to fill NaN values in.

    Returns:
        pd.DataFrame: The DataFrame with NaN values filled in the specified column.

    Raises:
        ValueError: If the specified column does not exist in the DataFrame.
        ValueError: If the column contains non-numeric data that cannot be averaged.
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame.")
    
    try:
        # Use pandas' fillna method with the mean of the column
        data[column] = data[column].fillna(data[column].mean())
    except TypeError as e:
        raise ValueError(f"Cannot compute mean for non-numeric data in column '{column}': {e}")
    
    return data

def one_hot_encode(dataframe, category):
    """
    Converts a categorical column into one-hot encoded columns.

    This function takes a DataFrame and a specified categorical column,
    and converts that column into one-hot encoded columns, which are 
    binary indicators for each category.

    Args:
        dataframe (pd.DataFrame): The DataFrame containing the data.
        category (str): The name of the column to be one-hot encoded.

    Returns:
        pd.DataFrame: The DataFrame with the specified column one-hot encoded 
        and the original column dropped.

    Raises:
        ValueError: If the input dataframe is not a pandas DataFrame or category is not a string.
    """
    if not isinstance(dataframe, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame.")
    
    dummy = pd.get_dummies(dataframe[category], prefix=category)
    dataframe = pd.concat([dataframe, dummy], axis=1)
    dataframe.drop(category, axis=1, inplace=True)
    
    return dataframe


def multi_label_encode(dataframe, column):
    """
    Encodes a column containing multiple labels into separate binary columns.

    This function encodes a column containing multiple labels separated by 
    semi-colons into individual binary columns. Each unique label becomes a 
    column with binary values indicating whether the label is present for a 
    particular row.

    Args:
        dataframe (pd.DataFrame): The DataFrame containing the data.
        column (str): The name of the column to be multi-label encoded.

    Returns:
        pd.DataFrame: The DataFrame with the multi-label column encoded into 
        separate binary columns.

    Raises:
        ValueError: If the column does not exist in the DataFrame or contains invalid data.
    """
    if column not in dataframe.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame.")
    
    # Initialize storage for unique labels
    interprot_identifiers = set()

    try:
        # Create a dictionary with multi-label lists from the column
        protein_ips = dataframe[column].str.split(';').apply(lambda x: [i for i in x if i]).to_dict()

        # Gather all unique labels from the entire column
        for labels in protein_ips.values():
            interprot_identifiers.update(labels)

        # Create a multi-hot encoded DataFrame
        categorical_df = pd.DataFrame(0, index=dataframe.index, columns=sorted(interprot_identifiers))

        for row, labels in protein_ips.items():
            categorical_df.loc[row, labels] = 1
    except Exception as e:
        raise ValueError(f"Error during multi-label encoding: {e}")
    
    # Drop the original column and concatenate the new encoded columns
    dataframe = dataframe.drop(column, axis=1)
    new_dataframe = pd.concat([dataframe, categorical_df], axis=1)

    return new_dataframe

def clean_print(obj):
    """
    Recursively prints nested data structures (like dicts and lists) in a readable format.

    This function is used to print objects such as dictionaries and lists in a 
    clean and structured format, making them more human-readable, especially useful 
    for nested structures.

    Args:
        obj (object): The object to be printed in a readable format.

    Returns:
        None
    """
    if isinstance(obj, dict):
        for key, val in obj.items():
            print("\n" + key)
            clean_print(val)
    elif isinstance(obj, list):
        for val in obj:
            clean_print(val)
    else:
        print(str(obj))


def to_excel(classification_information):
    """Prints model output to an Excel file (CSV format).

    Args:
        classification_information (dict): Information about results
            classification_information = {
                'all_predict_proba' : np.empty([TOTAL_TESTED_PROTEINS], dtype=float),
                'all_true_results' : np.empty([TOTAL_TESTED_PROTEINS], dtype=int),
                'all_accesion_numbers' : np.empty([TOTAL_TESTED_PROTEINS], dtype=str),
                'all_particle_information' : np.empty([2, TOTAL_TESTED_PROTEINS], dtype=int),
                'all_solvent_information' : np.empty([3, TOTAL_TESTED_PROTEINS], dtype=int)
            }
    Returns:
        None
    """
    required_keys = ['all_predict_proba', 'all_true_results', 'all_accesion_numbers',
                     'all_particle_information', 'all_solvent_information']

    # Ensure all required keys are present
    for key in required_keys:
        if key not in classification_information:
            raise KeyError(f"Missing required key '{key}' in classification_information")

    with open('prediction_probability.csv', 'w') as file:
        file.write('Protein Accesion Number, Particle Type, Solvent Conditions, '
                   'True Bound Value, Predicted Bound Value, Predicted Probability '
                   'of Being Bound, Properly Classified\n')

        # Unpack classification information
        predict_proba = classification_information['all_predict_proba']
        true_results = classification_information['all_true_results']
        accession_numbers = classification_information['all_accesion_numbers']
        particle_sizes, particle_charges = classification_information['all_particle_information']
        cys_conditions, salt8_conditions, salt3_conditions = classification_information['all_solvent_information']

        for proba, true_val, accession, particle_s, particle_c, cys, salt8, salt3 in zip(
                predict_proba, true_results, accession_numbers, particle_sizes, particle_charges,
                cys_conditions, salt8_conditions, salt3_conditions):

            # Set the default values for classification
            properly_classified = 'no'
            bound = 'no'
            predicted_bound = 'no'
            particle = ''
            solvent = '10 mM NaPi pH 7.4'  # Default solvent

            # Proper classification handling
            if int(round(proba)) == true_val:
                properly_classified = 'yes'

            # True and predicted bound value
            if true_val == 1:
                bound = 'yes'
            if int(round(proba)) == 1:
                predicted_bound = 'yes'

            # Particle size and charge classification
            if particle_s == 0:
                particle_size = '10nm'
            elif particle_s == 1:
                particle_size = '100nm'

            if particle_c == 1:
                particle_charge = 'positive'
            else:
                particle_charge = 'negative'

            # Combine particle size and charge
            if particle_size == '10nm' and particle_charge == 'positive':
                particle = '(+) 10 nm AgNP'
            elif particle_size == '10nm' and particle_charge == 'negative':
                particle = '(-) 10 nm AgNP'
            elif particle_size == '100nm' and particle_charge == 'positive':
                particle = '(+) 100 nm AgNP'
            elif particle_size == '100nm' and particle_charge == 'negative':
                particle = '(-) 100 nm AgNP'

            # Solvent conditions handling
            if cys == 1:
                solvent += ' + 0.1 mM cys'
            elif salt8 == 1:
                solvent += ' + 0.8 mM NaCl'
            elif salt3 == 1:
                solvent += ' + 3.0 mM NaCl'

            # Write the formatted row to the CSV
            file.write(f'{accession},{particle},{solvent},{bound},'
                       f'{predicted_bound},{proba:.3f},{properly_classified}\n')


def hold_in_memory(classification_information, metrics, iterations, test_size):
    """Holds classification data in memory to be exported to Excel.

    Args:
        classification_information (dict): Container for all the classification information from all runs.
        metrics (tuple): Information from the current test set to add to classification_information. Expected to be:
            - (numpy array: predicted probabilities, numpy array: true results,
              numpy array: accession numbers, dict: additional information)
        iterations (int): The current test iterations.
        test_size (int): The amount of values in the current test set.

    Returns:
        None
    """
    # Ensure that classification_information contains necessary keys
    required_keys = ['all_predict_proba', 'all_true_results', 'all_accesion_numbers',
                     'all_particle_information', 'all_solvent_information']

    for key in required_keys:
        if key not in classification_information:
            raise KeyError(f"Missing required key '{key}' in classification_information")

    i = iterations
    TEST_SIZE = test_size  # 10% of training data is used for testing

    # Information is placed into numpy arrays as blocks
    try:
        classification_information['all_predict_proba'][i * TEST_SIZE:(i * TEST_SIZE) + TEST_SIZE] = metrics[0]
        classification_information['all_true_results'][i * TEST_SIZE:(i * TEST_SIZE) + TEST_SIZE] = metrics[1]
        classification_information['all_accesion_numbers'][i * TEST_SIZE:(i * TEST_SIZE) + TEST_SIZE] = metrics[2]
        
        classification_information['all_particle_information'][0][i * TEST_SIZE:(i * TEST_SIZE) + TEST_SIZE] = metrics[3]['Particle Charge_1']
        classification_information['all_particle_information'][1][i * TEST_SIZE:(i * TEST_SIZE) + TEST_SIZE] = metrics[3]['Particle Size_10']
        
        classification_information['all_solvent_information'][0][i * TEST_SIZE:(i * TEST_SIZE) + TEST_SIZE] = metrics[3]['Solvent Cysteine Concentration_0.1']
        classification_information['all_solvent_information'][1][i * TEST_SIZE:(i * TEST_SIZE) + TEST_SIZE] = metrics[3]['Solvent NaCl Concentration_0.8']
        classification_information['all_solvent_information'][2][i * TEST_SIZE:(i * TEST_SIZE) + TEST_SIZE] = metrics[3]['Solvent NaCl Concentration_3.0']
    
    except IndexError as e:
        raise IndexError(f"Error indexing into classification_information: {e}")

if __name__ == "__main__":
    db = data_base()
    db.clean_data()