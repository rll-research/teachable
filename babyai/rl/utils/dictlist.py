import numpy as np
import torch

class DictList(dict):
    """A dictionary of lists of same size. Dictionary items can be
    accessed using `.` notation and list items using `[]` notation.

    Example:
        >>> d = DictList({"a": [[1, 2], [3, 4]], "b": [[5], [6]]})
        >>> d.a
        [[1, 2], [3, 4]]
        >>> d[0]
        DictList({"a": [1, 2], "b": [5]})
    """

    __setattr__ = dict.__setitem__

    def __getattr__(self, index):
        try:
            return dict.__getitem__(self, index)
        except KeyError:
            raise AttributeError(index)

    def __len__(self):
        return len(next(iter(dict.values(self))))

    def __getitem__(self, index):
        return DictList({key: value[index] for key, value in dict.items(self)})

    def __setitem__(self, index, d):
        try:
            for key, value in d.items():
                dict.__getitem__(self, key)[index] = value
        except:
            setattr(self, index, d)


def merge_dictlists(list_of_dictlists):
    batch = list_of_dictlists[0]
    for k in batch.keys():
        try:
            vec_type = type(getattr(list_of_dictlists[0], k))
            if vec_type is list:
                v = [step for dict_list in list_of_dictlists for step in getattr(dict_list, k)]
            elif vec_type is torch.Tensor:
                v = torch.cat([getattr(dict_list, k) for dict_list in list_of_dictlists])
            elif vec_type is np.ndarray:
                v = np.concatenate([getattr(dict_list, k) for dict_list in list_of_dictlists])
            elif vec_type is DictList:
                v = merge_dictlists([getattr(dict_list, k) for dict_list in list_of_dictlists])
            else:
                raise NotImplementedError(vec_type)
            setattr(batch, k, v)
        except AttributeError as e:
            print(f"Error setting {k}, {e}")
    return batch
