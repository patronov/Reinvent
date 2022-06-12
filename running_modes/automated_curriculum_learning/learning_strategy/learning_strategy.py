from running_modes.automated_curriculum_learning.learning_strategy import DAPStrategy, MAULIStrategy, MASCOFStrategy, \
    SDAPStrategy, DAPSingleQueryStrategy, DAPPatformerStrategy
from running_modes.automated_curriculum_learning.learning_strategy.base_learning_strategy import BaseLearningStrategy
from running_modes.automated_curriculum_learning.learning_strategy.learning_strategy_configuration import \
    LearningStrategyConfiguration
from running_modes.automated_curriculum_learning.learning_strategy.learning_strategy_enum import LearningStrategyEnum
from running_modes.automated_curriculum_learning.learning_strategy.linkinvent_dap_strategy import LinkInventDAPStrategy


class LearningStrategy:

    def __new__(cls, critic_model, optimizer, configuration: LearningStrategyConfiguration, logger=None) \
            -> BaseLearningStrategy:
        learning_strategy_enum = LearningStrategyEnum()
        if learning_strategy_enum.DAP_SINGLE_QUERY == configuration.name:
            return DAPSingleQueryStrategy(critic_model, optimizer, configuration, logger)
        if learning_strategy_enum.DAP == configuration.name:
            return DAPStrategy(critic_model, optimizer, configuration, logger)
        if learning_strategy_enum.MAULI == configuration.name:
            return MAULIStrategy(critic_model, optimizer, configuration, logger)
        if learning_strategy_enum.MASCOF == configuration.name:
            return MASCOFStrategy(critic_model, optimizer, configuration, logger)
        if learning_strategy_enum.SDAP == configuration.name:
            return SDAPStrategy(critic_model, optimizer, configuration, logger)
        if learning_strategy_enum.DAP_PATFORMER == configuration.name:
            return DAPPatformerStrategy(critic_model, optimizer, configuration, logger)
        if learning_strategy_enum.DAP_LINK_INVENT == configuration.name:
            return LinkInventDAPStrategy(critic_model, optimizer, configuration, logger)
