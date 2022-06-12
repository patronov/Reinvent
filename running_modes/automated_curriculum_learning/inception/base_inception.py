from abc import ABC, abstractmethod
from typing import Union

from reinvent_chemistry import Conversions
from reinvent_models.model_factory.generative_model_base import GenerativeModelBase
from reinvent_models.model_factory.link_invent_adapter import LinkInventAdapter

from running_modes.configurations import ProductionStrategyInputConfiguration, InceptionConfiguration


class BaseInception(ABC):
    def __init__(self, configuration: Union[ProductionStrategyInputConfiguration], scoring_function, prior: Union[LinkInventAdapter, GenerativeModelBase]):
        self._chemistry = Conversions()
        self._load_to_memory(scoring_function, prior)

    def _load_to_memory(self, scoring_function, prior: LinkInventAdapter):
        raise NotImplementedError("_load_to_memory method is not implemented")

    @abstractmethod
    def add(self, *args, **kwargs):
        raise NotImplementedError("add method is not implemented")

    @abstractmethod
    def sample(self):
        raise NotImplementedError("sample method is not implemented")
