import torch

class Utils:
    """
    Class eith static methods to perform common general operations.
    """

    @staticmethod
    def get_device(device=None):
        """
        Get the device to use for tensor operations.

        Parameters:
        - device: string with the name of the device to use. Can be 'cuda', 'mps' or 'cpu'.
                  If None, the device is selected among 'cuda', 'mps' and 'cpu' in decreasing order of preference.

        Returns:
        - torch.device to be used for tensor operations.
          Device is selected among 'cuda', 'mps' and 'cpu' in decreasing order of preference if None is provided as input.
        """
        if device in ['cuda', 'mps', 'cpu']:
            return torch.device(device)
        if torch.cuda.is_available():
            return torch.device('cuda')
        elif torch.backends.mps.is_available():
            return torch.device('mps')
        return torch.device('cpu')
    
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
    
    @staticmethod
    def get_n_items_list_of_lists(list_of_lists):
        """
        Get the overall number of items in the input list of lists.

        Parameters:
        - list_of_lists: list of lists.

        Returns:
        - n_items_list: int representing the overall number of items in list_of_lists.
        """

        return sum([len(sublist) for sublist in list_of_lists])

    @staticmethod
    def select_loss(string_loss):
        """
        Returns the loss function identified by the input string.

        Parameters:
        - string_loss: string with the name of the loss function to use.

        Returns:
        - loss function identified by the input string.
        """

        if string_loss == 'MSE_loss':
            return torch.nn.MSELoss()
        elif string_loss == 'MAE_loss':
            return torch.nn.L1Loss()
        
        raise ValueError('Invalid loss function name.')
    
    @staticmethod
    def select_optimizer(string_opt):
        """
        Returns the optimizer identified by the input string.

        Parameters:
        - string_opt: string with the name of the optimizer to use.

        Returns:
        - optimizer identified by the input string.
        """

        if string_opt == 'Adam':
            return torch.optim.Adam
        elif string_opt == 'SGD':
            return torch.optim.SGD
        
        raise ValueError('Invalid optimizer name.')