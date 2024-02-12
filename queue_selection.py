from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import cached_property
from typing import List, Dict

import numpy as np
import torch

from utils import get_client_iou_matrix


@dataclass
class QueueMatcher(ABC):
    @abstractmethod
    def match_client_queues(
            self,
            queues: List[torch.Tensor]
    ) -> List[torch.Tensor]:
        ...


@dataclass
class NoOpQueueMatcher(QueueMatcher):

    def match_client_queues(self, queues: List[torch.Tensor]):
        return queues


@dataclass
class OracleIOUQueueMatcher(QueueMatcher, ABC):
    ious: np.ndarray


@dataclass
class OracleMostSimilarClientsQueueMatcher(OracleIOUQueueMatcher):
    n_queues_to_match: int
    take_most_similar: bool

    @cached_property
    def selection(self):
        result = []
        for c_id, ious in enumerate(self.ious):
            other_client_ids = [c for c in range(len(self.ious)) if c != c_id]
            sorted_client_ids = sorted(other_client_ids, key=lambda c: ious[c], reverse=self.take_most_similar)
            result.append(
                [c_id] + sorted_client_ids[:self.n_queues_to_match]
            )
        return result

    def match_client_queues(self, queues: List[torch.Tensor]):
        return [
            torch.cat([queues[s] for s in self.selection[c]], dim=1)
            for c in range(len(self.ious))
        ]


def get_queue_matcher(args) -> QueueMatcher:
    if args.qmatching == "noop":
        return NoOpQueueMatcher()
    elif "oracle" in args.qmatching:
        client_to_labels = {
            c: i["labels"]
            for (c, i) in args.client_info.items()
        }
        ious = get_client_iou_matrix(client_to_labels)

        return OracleMostSimilarClientsQueueMatcher(
            ious=ious,
            n_queues_to_match=args.qmatching_nqueues,
            take_most_similar=("most" in args.qmatching)
        )

    else:
        raise NotImplementedError(args.qmatching)


@dataclass
class GradientMatcher(ABC):
    @abstractmethod
    def match_gradient_dict(self, gradient_dict: Dict[int, torch.Tensor]) ->  Dict[int, torch.Tensor]:
        ...

@dataclass
class NoOpGradientMatcher:
    @abstractmethod
    def match_gradient_dict(self, gradient_dict: Dict[int, torch.Tensor]) ->  Dict[int, torch.Tensor]:
        return gradient_dict

@dataclass
class OracleIOUGradientMatcher(GradientMatcher, ABC):
    ious: np.ndarray


@dataclass
class OracleMostSimilarClientsGradientMatcher(OracleIOUGradientMatcher):
    n_queues_to_match: int
    take_most_similar: bool

    @cached_property
    def selection(self):
        result = []
        for c_id, ious in enumerate(self.ious):
            other_client_ids = [c for c in range(len(self.ious)) if c != c_id]
            sorted_client_ids = sorted(other_client_ids, key=lambda c: ious[c], reverse=self.take_most_similar)
            result.append(
                [c_id] + sorted_client_ids[:self.n_queues_to_match]
            )
        return result

    def match_gradient_dict(self, gradient_dict: Dict[int, torch.Tensor]) -> Dict[int, torch.Tensor]:
        result = dict()
        for k, v in gradient_dict.items():
            g = torch.zeros_like(v)
            for c_id in self.selection[k]:
                if c_id in gradient_dict:
                    g = g + gradient_dict[c_id]

            result[k] = g

        return result


def get_gradient_matcher(args) -> GradientMatcher:
    if args.gmatching == "noop":
        return NoOpGradientMatcher()
    elif "oracle" in args.gmatching:
        client_to_labels = {
            c: i["labels"]
            for (c, i) in args.client_info.items()
        }
        ious = get_client_iou_matrix(client_to_labels)

        return OracleMostSimilarClientsGradientMatcher(
            ious=ious,
            n_queues_to_match=args.gmatching_ngrads,
            take_most_similar=("most" in args.gmatching)
        )

    else:
        raise NotImplementedError(args.gmatching)