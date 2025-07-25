import sys
from abc import ABCMeta, abstractmethod
from torch import nn
from copy import copy
import inspect
from huggingface_hub import hf_hub_download


class BaseModel(nn.Module, metaclass=ABCMeta):
    default_conf = {}
    required_inputs = []

    def __init__(self, conf):
        """Perform some logic and call the _init method of the child model."""
        super().__init__()
        self.conf = conf = {**self.default_conf, **conf}
        self.required_inputs = copy(self.required_inputs)
        self._init(conf)
        sys.stdout.flush()

    def forward(self, data):
        """Check the data and call the _forward method of the child model."""
        for key in self.required_inputs:
            assert key in data, "Missing key {} in data".format(key)
        return self._forward(data)

    @abstractmethod
    def _init(self, conf):
        """To be implemented by the child class."""
        raise NotImplementedError

    @abstractmethod
    def _forward(self, data):
        """To be implemented by the child class."""
        raise NotImplementedError

    def _download_model(self, repo_id=None, filename=None, **kwargs):
        """Download model from hf hub and return the path."""
        return hf_hub_download(
            repo_type="model",
            repo_id=repo_id,
            filename=filename,
        )


def dynamic_load(root, model):
    module_path = f"{root.__name__}.{model}" # imcui.hloc.matches.matchanything
    module = __import__(module_path, fromlist=[""]) # imcui.hloc.matchers.matchanything from '/home/sebastiangarcia/projects/swappr/src/MatchAnything/imcui/hloc/matchers/matchanything.py'
    classes = inspect.getmembers(module, inspect.isclass)
    # Filter classes defined in the module
    classes = [c for c in classes if c[1].__module__ == module_path] # [('BaseModel', <class 'imcui.hloc.utils.base_model.BaseModel'>), ('MatchAnything', <class 'imcui.hloc.matchers.matchanything.MatchAnything'>), ('PL_LoFTR', <class 'MatchAnything.src.lightning.lightning_loftr.PL_LoFTR'>), ('Path', <class 'pathlib.Path'>)]
    # Filter classes inherited from BaseModel
    classes = [c for c in classes if issubclass(c[1], BaseModel)] # [('MatchAnything', <class 'imcui.hloc.matchers.matchanything.MatchAnything'>)]
    assert len(classes) == 1, classes
    return classes[0][1]
    # return getattr(module, 'Model')
