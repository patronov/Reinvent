from typing import List

import torch
from reinvent_models.link_invent.dto import SampledSequencesDTO
from reinvent_models.patformer.dto import BatchLikelihoodDTO
from reinvent_scoring import FinalSummary

from running_modes.automated_curriculum_learning.dto import UpdatedLikelihoodsDTO
from running_modes.automated_curriculum_learning.inception.link_invent_inception import LinkInventInception
from running_modes.automated_curriculum_learning.learning_strategy.base_learning_strategy import BaseLearningStrategy
from running_modes.automated_curriculum_learning.learning_strategy.learning_strategy_configuration import \
    LearningStrategyConfiguration


class LinkInventDAPStrategy(BaseLearningStrategy):

    def __init__(self, critic_model, optimizer, configuration: LearningStrategyConfiguration, logger=None):
        """
        TODO: Provide description of the current strategy
        """
        super().__init__(critic_model, optimizer, configuration, logger)

        self._sigma = self._configuration.parameters.get("sigma", 120)

    def run(self, likelihood_dto: BatchLikelihoodDTO, score_summary: FinalSummary,
            sampled_sequences: List[SampledSequencesDTO], agent, inception) -> UpdatedLikelihoodsDTO:
        dto = self._calculate_loss(likelihood_dto, score_summary, sampled_sequences, agent, inception)
        self.optimizer.zero_grad()
        dto.loss.backward()

        self.optimizer.step()
        return dto

    def _calculate_loss(self, likelihood_dto: BatchLikelihoodDTO, score_summary: FinalSummary,
                        sampled_sequences: List[SampledSequencesDTO], agent, inception: LinkInventInception) -> UpdatedLikelihoodsDTO:
        batch = likelihood_dto.batch
        critic_nlls = self.critic_model.likelihood(*batch.input, *batch.output)
        negative_critic_nlls = -critic_nlls
        negative_actor_nlls = -likelihood_dto.likelihood
        augmented_nlls = negative_critic_nlls + self._sigma * self._to_tensor(score_summary.total_score)
        loss = torch.pow((augmented_nlls - negative_actor_nlls), 2)
        loss, agent_likelihood = self._inception_filter(agent, loss, negative_actor_nlls, negative_critic_nlls,
                                                        sampled_sequences, score_summary, inception)
        loss = loss.mean()
        dto = UpdatedLikelihoodsDTO(negative_actor_nlls, negative_critic_nlls, augmented_nlls, loss)
        return dto

    def _inception_filter(self, agent, loss, agent_likelihood, prior_likelihood,
                          sampled_sequences: List[SampledSequencesDTO], score_summary: FinalSummary, inception: LinkInventInception):
        if inception:
            dtos = inception.sample()
            if len(dtos.smiles) > 0:
                sampled_sequence_list = [SampledSequencesDTO(dtos.input[i], dtos.output[i], dtos.likelihood[i]) for i, _
                                         in enumerate(dtos.smiles)]
                exp_agent_likelihood = -agent.likelihood_smiles(sampled_sequence_list)
                exp_augmented_likelihood = dtos.likelihood + self._sigma * dtos.scores
                exp_loss = torch.pow((self._to_tensor(exp_augmented_likelihood) - exp_agent_likelihood), 2)
                loss = torch.cat((loss, exp_loss), 0)
                agent_likelihood = torch.cat((agent_likelihood, exp_agent_likelihood), 0)
            inception.add(score_summary, sampled_sequences, prior_likelihood)
        return loss, agent_likelihood
