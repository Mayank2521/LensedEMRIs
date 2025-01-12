"""
This module contains the class `GeneralUtils`, which provides some
useful utility functions.
"""

__author__ = ["Anuj Mishra <anuj.mishra@ligo.org>"]

import math
import collections
import numpy as np


class GeneralUtils:
    """
    Miscallaneous functions that can make your life simpler and effective.

    """

    def print_dict(self, a_dict):
        """
        Function to print a dictionary element by element.

        """
        for key, value in a_dict.items():
            print(f"{key} : {repr(value)}")

    def distribute(self, array, rank, ncpus):
        """
        Function to efficiently distribute an array/list among
        different processors.
        Useful for parallel computations, such as MPI.
        It maintains original order of the array while distributing.
        Caution: empty arrays can be checked using array.size == 0
        before doing any operation to avoid errors.

        Example
        -------

        .. code-block:: python

            # Example 1: Distributing the array among 3 ranks
            distribute(array=[1, 2, 3, 4, 5, 6, 7], rank=0, ncpus=3)
            # Output: [1, 2, 3]  # Elements for rank 0

            distribute(array=[1, 2, 3, 4, 5, 6, 7], rank=1, ncpus=3)
            # Output: [4, 5]     # Elements for rank 1

            distribute(array=[1, 2, 3, 4, 5, 6, 7], rank=2, ncpus=3)
            # Output: [6, 7]     # Elements for rank 2

        Each rank gets its assigned portion of the array,
        with excess elements allocated sequentially
        starting from `rank=0` up to `ncpus-1`.

        Parameters
        ----------
        array : numpy.array or list
            Array to be distributed.
        rank : int
            Rank of the process. An integer between [0, ncpus-1].
        ncpus : int
            Total number of processors.

        Returns
        -------
        numpy.array
            A segment of the provided array assigned to rank=rank.

        """

        array = np.array(array)
        array_len = len(array)
        q = int(array_len / ncpus)
        rem = array_len - q * ncpus

        if rank < rem:
            d_array = array[rank * (q + 1) : (rank + 1) * (q + 1)]
        elif rem <= rank <= ncpus - 1:
            d_array = array[rank * q + rem : (rank + 1) * q + rem]
        else:
            d_array = np.array([])
        return d_array

    def find_nearest(self, array, value):
        """
        Returns nearest element in an array to a given value.

        Parameters
        ----------
        array : numpy.array/list
        value : value to search for

        Returns
        -------
        int
            index of the nearest element.
        float
            Neearest element found in the array.

        """

        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return idx, array[idx]

    def is_num(self, x):
        """
        Function to check whether an input is a number or not.
        There is scope of improving the function as it can sometimes
        give false positives.

        Parameters
        ----------
        x : python object
            Object to check.

        Returns
        -------
        Bool

        """

        if isinstance(x, np.ndarray):
            # to take care of cases where a scalar is defined as np.array(num)
            try:
                len(x)
                return False
            except TypeError:
                return True
        else:
            is_list_type = isinstance(
                x, (np.ndarray, dict, collections.abc.Sequence)
            )  # [collections.abc.Sequence, np.ndarray, dict, np.nan])
            if is_list_type:
                return False
            if math.isnan(x):
                return False
            return isinstance(x, (float, int, np.floating, np.complexfloating))

    def str_num(self, n, r):
        """
        Function to covert an integer (n) to a string with
        specified number of digits (r).
        r < n will be treated as n itself.
        r > n will apply extra 0's in the beginning.

        Parameters
        ----------
        n : int
            Integer object to convert.
        r : int
            Number of digits in the string.

        Returns
        -------
        str
            String corresponding to the provided integer.

        """

        i = 1
        while n / 10**i >= 1:
            i += 1
        return "0" * (r - i) + str(n)

    def str_float(self, n, r):
        """
        Function to covert a float (n) to a string
        up to `r` decimal places.

        Parameters
        ----------
        n : float
            Float object to convert.
        r : int
            Number of decimal places to consider while conversion.

        Returns
        -------
        str
            String corresponding to the provided float
            valid upto r decimal places.

        """

        dec = round(n % 1, r)
        str_dec = self.str_num(int(round(10**r * dec)), r)
        if dec == 1:
            n += 1
            str_dec = str_dec[1:]
        return str(int(n)) + "p" + str_dec

    def str_y(self, n):
        """
        Returns string version for a number
        up to two decimal places.

        """

        return self.str_float(n, 2)

    def str_m(self, n):
        """
        Returns string version for a number
        up to one decimal place.

        """

        return self.str_float(n, 1)

    def str_to_float(self, sf, separater="p"):
        """
        This function takes a string representation of a number where the decimal point
        is denoted by a specified separator (default is 'p') and converts it into a float.
        It replaces the separator with a dot ('.') to ensure proper float conversion.

        Examples
        --------
        >>> str_to_float("123p45")
        123.45
        >>> str_to_float("678p90", separator="p")
        678.90
        """
        return float(sf.replace(separater, "."))

    def sync_keys_in_dict(self, key1, key2, default, **dictionary):
        """
        Ensures that both key1 and key2 in the dictionary have the same value.

        If either of the keys is present, assign the same value to the other key.
        If both keys are present, check if their values are the same.
        If neither key is present, set it to default value.

        Parameters:
        - key1: First key to check.
        - key2: Second key to check.
        - dictionary: The dictionary in which the keys are checked.

        Returns:
        - modified dictionary.

        Raises:
        - KeyError: If neither key is found in the dictionary.
        - ValueError: If both keys are present but their values are different.
        """
        val1 = dictionary.get(key1)
        val2 = dictionary.get(key2)

        if val1 is not None and val2 is not None:
            if val1 != val2:
                raise ValueError(
                    f"Values for `{key1}` and `{key2}` do not match.\
 Provide only one of the keys, not both."
                )
        elif val1 is not None:
            dictionary[key2] = val1
        elif val2 is not None:
            dictionary[key1] = val2
        else:
            print(
                f"Neither `{key1}` nor `{key2}` found in dictionary.\
 Setting it to `{default}` by default."
            )
            dictionary[key1] = dictionary[key2] = default

        return dictionary

    def check_if_dict_is_subset(self, dict_1, dict_2):
        """
        Check if one dictionary is a subset of another.

        This function checks whether all key-value pairs in the
        smaller dictionary exist in the larger dictionary,
        and if they do, checks if their values match.

        Parameters:
        - dict_1 (dict): First dictionary to compare.
        - dict_2 (dict): Second dictionary to compare.

        Returns:
          bool: True if all key-value pairs in the smaller dictionary
          are in the larger dictionary and equal.
          False otherwise.
        """
        # Identify the smaller and larger dictionaries
        smaller_dict, larger_dict = (
            (dict_1, dict_2) if len(dict_1) <= len(dict_2) else (dict_2, dict_1)
        )

        # Check if all key-value pairs in the smaller dictionary are in the larger dictionary,
        # and whether they match.
        for key, value in smaller_dict.items():
            if key not in larger_dict:
                return False

            larger_value = larger_dict.get(key)

            # If the value is a numpy array, compare using array comparison
            if isinstance(value, np.ndarray):
                # Check if the arrays are equal within a tolerance
                if (
                    not isinstance(larger_value, np.ndarray)
                    or value.shape != larger_value.shape
                    or not np.allclose(value, larger_value, rtol=1e-5, atol=1e-8)
                ):
                    return False

            # Direct comparison for non-array values
            elif isinstance(larger_value, np.ndarray) or larger_value != value:
                return False

        return True
