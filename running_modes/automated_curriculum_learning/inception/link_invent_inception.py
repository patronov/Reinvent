from typing import List, Union

import pandas as pd
from reinvent_chemistry.conversions import Conversions
from reinvent_models.link_invent.dto import SampledSequencesDTO
from reinvent_models.model_factory.generative_model_base import GenerativeModelBase
from reinvent_models.model_factory.link_invent_adapter import LinkInventAdapter
from reinvent_scoring import FinalSummary

from running_modes.automated_curriculum_learning.inception.base_inception import BaseInception
from running_modes.automated_curriculum_learning.inception.inception_sample_dto import InceptionSampleDTO
from running_modes.configurations import ProductionStrategyInputConfiguration


class LinkInventInception(BaseInception):
    def __init__(self, configuration: Union[ProductionStrategyInputConfiguration], scoring_function, prior: Union[LinkInventAdapter, GenerativeModelBase]):
        super().__init__(configuration, scoring_function, prior)
        self.configuration = configuration
        self._chemistry = Conversions()
        self.memory: pd.DataFrame = pd.DataFrame(columns=['smiles', 'score', 'likelihood', 'fragments'])
        self._load_to_memory(scoring_function, prior)


    def _load_to_memory(self, scoring_function, prior: LinkInventAdapter):
        if len(self.configuration.input) != 1:
            raise IOError(f"LinkInventInception works only with a single input sequence only!")
        standardized_and_nulls = [self._chemistry.mol_to_smiles(self._chemistry.smile_to_mol(smile)) for smile in
                                  self.configuration.inception.smiles]
        standardized = [smile for smile in standardized_and_nulls if smile is not None]
        if self.configuration.inception.fragments and len(standardized) == len(self.configuration.inception.fragments):
            dtos = [SampledSequencesDTO(self.configuration.input[0], fragment, 0.) for fragment in self.configuration.inception.fragments]
            self._evaluate_and_add(standardized, dtos, scoring_function, prior)
        else:
            raise IOError(f"Invalid inception input!")

    def _purge_memory(self):
        unique_df = self.memory.drop_duplicates(subset=["smiles"])
        sorted_df = unique_df.sort_values('score', ascending=False)
        self.memory = sorted_df.head(self.configuration.inception.memory_size)

    def _evaluate_and_add(self, smiles: List[str], dtos: List[SampledSequencesDTO], scoring_function, prior: LinkInventAdapter):
        score = scoring_function.get_final_score(smiles)
        batch_likelihood_dto = prior.likelihood_smiles(dtos)
        likelihood = -batch_likelihood_dto.likelihood
        fragments = [dto.output for dto in dtos]
        df = pd.DataFrame(
            {"smiles": smiles, "score": score.total_score, "likelihood": likelihood.detach().cpu().numpy(), 'fragments': fragments})
        self.memory = self.memory.append(df)
        self._purge_memory()

    def add(self,score_summary: FinalSummary, sampled_sequences: List[SampledSequencesDTO], neg_likelihood):
        fragments = [dto.output for dto in sampled_sequences]
        df = pd.DataFrame({"smiles": score_summary.scored_smiles, "score": score_summary.total_score, "likelihood": neg_likelihood.detach().cpu().numpy(), 'fragments': fragments})
        self.memory = self.memory.append(df)
        self._purge_memory()

    def sample(self) -> InceptionSampleDTO:
        sample_size = min(len(self.memory), self.configuration.inception.sample_size)
        if sample_size > 0:
            sampled = self.memory.sample(sample_size)
            smiles = sampled["smiles"].values
            scores = sampled["score"].values
            prior_likelihood = sampled["likelihood"].values
            fragments = sampled["fragments"].values
            #TODO: Fix this. It will fail for multiple inputs
            inputs = [self.configuration.input[0] for _ in enumerate(fragments)]
            dto = InceptionSampleDTO(smiles, inputs, fragments, scores, prior_likelihood)
            return dto
