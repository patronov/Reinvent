from typing import List

import numpy as np
import torch.utils.data as tud
from reinvent_chemistry import Conversions
from reinvent_models.model_factory.generative_model_base import GenerativeModelBase
from reinvent_models.patformer.dataset.dataset import Dataset
from reinvent_models.patformer.dto.sampled_sequence_dto import SampledSequencesDTO
from reinvent_models.patformer.models.vocabulary import SMILESTokenizer

from running_modes.automated_curriculum_learning.actions import BaseSampleAction


class PatformerSampleModel(BaseSampleAction):
    def __init__(self, model: GenerativeModelBase, batch_size: int, logger=None, randomize=False, sample_uniquely=True):
        """
        Creates an instance of SampleModel.
        :params model: A model instance.
        :params batch_size: Batch size to use.
        :return:
        """
        super().__init__(logger)
        self.model = model
        self._batch_size = batch_size
        self._randomize = randomize
        self._sample_uniquely = sample_uniquely
        self._conversions = Conversions()

    def run(self, smiles: List[str]) -> List[SampledSequencesDTO]:
        smiles = [self._randomize_smile(smile) for smile in smiles] if self._randomize else smiles
        smiles = smiles * self._batch_size
        tokenizer = SMILESTokenizer()
        dataset = Dataset(smiles, self.model.get_vocabulary(), tokenizer)
        dataloader = tud.DataLoader(dataset, batch_size=len(dataset), shuffle=False, collate_fn=Dataset.collate_fn)
        sampled_sequences = []

        for batch in dataloader:
            src, src_mask = batch
            sampled_sequences = self.model.sample(src, src_mask)

        unique_sequences = self._sample_unique_sequences(sampled_sequences)

        return unique_sequences

    def _sample_unique_sequences(self, sampled_sequences: List[SampledSequencesDTO]) -> List[SampledSequencesDTO] :
        smiles = [dto.output for dto in sampled_sequences]
        unique_idxs = self._get_indices_of_unique_smiles(smiles)
        sampled_sequences_np = np.array(sampled_sequences)
        unique_sampled_sequences = sampled_sequences_np[unique_idxs]
        return unique_sampled_sequences.tolist()

    def _randomize_smile(self, smile: str):
        input_mol = self._conversions.smile_to_mol(smile)
        randomized_smile = self._conversions.mol_to_random_smiles(input_mol)
        return randomized_smile