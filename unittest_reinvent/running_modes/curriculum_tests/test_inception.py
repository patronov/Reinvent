import os
import shutil
import unittest

import torch as ts
from reinvent_models.lib_invent.enums.generative_model_regime import GenerativeModelRegimeEnum
from reinvent_models.model_factory.configurations.model_configuration import ModelConfiguration
from reinvent_models.model_factory.generative_model import GenerativeModel

import running_modes.utils.general as utils_general
from reinvent_models.reinvent_core.models.model import Model
from reinvent_scoring.scoring.component_parameters import ComponentParameters
from reinvent_scoring.scoring.function import CustomSum

from running_modes.automated_curriculum_learning.inception.inception import Inception
from running_modes.automated_curriculum_learning.inception.link_invent_inception import LinkInventInception
from running_modes.configurations import InceptionConfiguration, ProductionStrategyInputConfiguration
from running_modes.enums.curriculum_strategy_enum import CurriculumStrategyEnum
from running_modes.enums.model_type_enum import ModelTypeEnum
from running_modes.enums.production_strategy_enum import ProductionStrategyEnum
from unittest_reinvent.fixtures.paths import PRIOR_PATH, MAIN_TEST_PATH, LINK_INVENT_PRIOR_PATH
from unittest_reinvent.fixtures.test_data import ETHANE, HEXANE, BUTANE, METHOXYHYDRAZINE, ASPIRIN, WARHEAD_PAIR
from reinvent_scoring.scoring.enums.scoring_function_component_enum import ScoringFunctionComponentNameEnum


class TestCLInceptionAdd(unittest.TestCase):

    def setUp(self):
        self.sf_enum = ScoringFunctionComponentNameEnum()
        self.ps_enum = ProductionStrategyEnum()
        self.cs_enum = CurriculumStrategyEnum()
        utils_general.set_default_device_cuda()
        self.log_path = MAIN_TEST_PATH

        if not os.path.isdir(self.log_path):
            os.makedirs(self.log_path)
        smiles = [HEXANE]

        model_type = ModelTypeEnum()
        model_regime = GenerativeModelRegimeEnum()
        prior_config = ModelConfiguration(model_type.LINK_INVENT, model_regime.INFERENCE, LINK_INVENT_PRIOR_PATH)
        prior = GenerativeModel(prior_config)
        warhead_pair = ['*c1ncnc2[nH]ncc12|*c1n(c2ccc(C)cc2)nc(C(C)(C)C)c1']
        inception_config = InceptionConfiguration(smiles=smiles, fragments=['[*]CC(=C)C(O)C=CCNC(N[*])=O'], memory_size=4, sample_size=4)
        scoring = ComponentParameters(component_type=self.sf_enum.JACCARD_DISTANCE,
                                      name=self.sf_enum.JACCARD_DISTANCE,
                                      weight=1.,
                                      specific_parameters={"smiles":[METHOXYHYDRAZINE, ASPIRIN]})
        scoring_function = CustomSum(parameters=[scoring])

        production_config = ProductionStrategyInputConfiguration(name=self.ps_enum.STANDARD,
                                                                 input=warhead_pair,
                                                                 scoring_function=None,
                                                                 diversity_filter=None,
                                                                 inception=inception_config,
                                                                 retain_inception=False, number_of_steps=3,
                                                                 learning_strategy=None)

        self.inception_model = LinkInventInception(configuration=production_config, scoring_function=scoring_function, prior=prior)
        # self.inception_model.add(smiles, score, prior_likelihood)

    def tearDown(self):
        if os.path.isdir(self.log_path):
            shutil.rmtree(self.log_path)

    def test_eval_and_add_tanimoto(self):
        self.assertEqual(len(self.inception_model.memory), 1)