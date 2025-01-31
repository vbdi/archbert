from typing import Union, List


class InputExample:
    """
    Structure for one input example with texts, the label and a unique id
    """
    def __init__(self, guid: str = '', texts: List[str] = None, graph=None, arch = None,  label: Union[int, float] = 0, unique_layers=None, n_params=None, n_layers=None):
        """
        Creates one InputExample with the given texts, guid and label


        :param guid
            id for the example
        :param texts
            the texts for the example.
        :param label
            the label for the example
        """
        self.guid = guid
        self.unique_layers = unique_layers
        self.n_params = n_params
        self.n_layers = n_layers
        self.texts = texts
        # ni
        self.graph = graph
        self.arch = arch
        self.label = label

    def __str__(self):
        return "<InputExample> label: {}, texts: {}".format(str(self.label), "; ".join(self.texts))
