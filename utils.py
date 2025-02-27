import torch

class Utils:
    """
    Class eith static methods to perform common general operations.
    """

    @staticmethod
    def flatten_list_of_lists(list_of_lists):
        """
        Flatten a list of lists into a single list.

        Parameters:
        - list_of_lists: list of lists to be flattened.

        Returns:
        - flat_list: single list with all elements in the input list of lists.
                     Elements are ordered in the following way:
                     [list_of_lists[0][0], list_of_lists[0][1], ..., list_of_lists[0][len(list_of_lists[0]) - 1], list_of_lists[1][0], ...,
                      list_of_lists[1][len(list_of_lists[1]) - 1], ..., list_of_lists[len(list_of_lists) - 1][len(list_of_lists[len(list_of_lists) - 1]) - 1].
        """

        return [elem for sublist in list_of_lists for elem in sublist]