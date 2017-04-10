import unittest

import bacteriopop_utils
import load_data
import numpy as np
import pandas as pd
import pandas as pd
import requests

print "hello!"


class TestUrlsExist(unittest.TestCase):

    def test_raw_data_link(self):
        """
        Test for existence of raw_data.csv link in load_data module.
        """
        request = requests.get("https://raw.githubusercontent.com/"
                               "JanetMatsen/bacteriopop/master/raw_data/"
                               "raw_data.csv")
        self.assertEqual(request.status_code, 200)


    def test_sample_meta_info_link(self):
        """
        Test for existence of sample_meta_info.tsv link in load_data module.
        """
        request = requests.get("https://raw.githubusercontent.com/"
                               "JanetMatsen/bacteriopop/master/raw_data/"
                               "sample_meta_info.tsv")
        self.assertEqual(request.status_code, 200)


class TestDataframe(unittest.TestCase):

    def test_df_columns(self):
        """
        Test for output dataframe column count in load_data module.
        """
        df = load_data.load_data()
        cols = df.columns.tolist()
        num = len(cols)
        num_assert = len(['kingdom', 'phylum', 'class', 'order',
                          'family', 'genus', 'length', 'oxygen',
                          'replicate', 'week', 'abundance'])
        self.assertEqual(num, num_assert)

    def test_df_type(self):
        """
        Test for type of the output dataframe in load_data module.
        """
        df = load_data.load_data()
        self.assertEqual(type(df), pd.DataFrame)


class TestExtractFeatures(unittest.TestCase):

    def test_on_animal_df(self):
        """
        Simple example with expected numpy vector to compare to.
        Use fillna mode.
        """
        animal_df = pd.DataFrame({'animal': ['dog', 'cat', 'rat'],
                                  'color': ['white', 'brown', 'brown'],
                                  'gender': ['F', 'F', np.NaN],
                                  'weight': [25, 5, 1],
                                  'garbage': [0, 1, np.NaN],
                                  'abundance': [0.5, 0.4, 0.1]})
        extracted = bacteriopop_utils.extract_features(
            dataframe=animal_df,
            column_list=['animal', 'color', 'weight', 'abundance'],
            fillna=True
            )
        # check that the column names match what is expected
        self.assertEqual(extracted.columns.tolist(),
                         ['abundance', 'animal=cat', 'animal=dog',
                          'animal=rat', 'color=brown', 'color=white',
                          'weight'])
        # check that the values are what was expected.
        expected_result = np.array([[0.5, 0., 1., 0., 0., 1., 25.],
                                    [0.4, 1., 0., 0., 1., 0., 5.],
                                    [0.1, 0., 0., 1., 1., 0., 1.]])
        self.assertEqual(expected_result.tolist(),
                         extracted.as_matrix().tolist())


if __name__ == '__main__':
    unittest.main()
