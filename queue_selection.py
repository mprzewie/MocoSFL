from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import cached_property
from typing import List

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
            torch.cat([queues[s] for s in self.selection[c]])
            for c in range(len(self.ious))
        ]


def get_queue_matcher(args) -> QueueMatcher:
    if args.qmatching == "noop":
        return NoOpQueueMatcher()
    elif "oracle" in args.qmatching:
        client_to_labels = {
            c: i["labels"]
            for (c, i) in args.client_info
        }
        ious = get_client_iou_matrix(client_to_labels)

        return OracleMostSimilarClientsQueueMatcher(
            ious=ious,
            n_queues_to_match=args.qmatching_nqueues,
            take_most_similar=("most" in args.qmatching)
        )

    else:
        raise NotImplementedError(args.qmatching)
