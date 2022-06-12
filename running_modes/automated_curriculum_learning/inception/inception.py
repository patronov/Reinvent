from typing import Union

from reinvent_models.model_factory.generative_model_base import GenerativeModelBase
from reinvent_models.model_factory.link_invent_adapter import LinkInventAdapter

from running_modes.automated_curriculum_learning.inception.base_inception import BaseInception
from running_modes.automated_curriculum_learning.inception.link_invent_inception import LinkInventInception
from running_modes.automated_curriculum_learning.inception.reinvent_inception import ReinventInception
from running_modes.configurations import ProductionStrategyInputConfiguration, CurriculumStrategyInputConfiguration
from running_modes.enums.production_strategy_enum import ProductionStrategyEnum


class Inception:
    def __new__(cls, configuration: Union[ProductionStrategyInputConfiguration, CurriculumStrategyInputConfiguration], scoring_function, prior: Union[LinkInventAdapter, GenerativeModelBase]) -> BaseInception:
        production_strategy_enum = ProductionStrategyEnum()
        if configuration.name == production_strategy_enum.LINK_INVENT:
            return LinkInventInception(configuration, scoring_function, prior)
        if configuration.name == production_strategy_enum.STANDARD:
            return ReinventInception(configuration.inception, scoring_function, prior)
