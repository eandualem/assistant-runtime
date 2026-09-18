"""Decisions: typed questions about a state, answered by a decision model.

A decision model is not a chat model. The application sends program state
and a set of typed questions (choice, score, noul) and receives typed answers
with calibrated probabilities from one provider call; nothing is generated
or parsed. The runtime holds the provider key; the application never does.
"""

from assistant_runtime.services.decisions.config import DecisionsConfig
from assistant_runtime.services.decisions.exceptions import DecisionError
from assistant_runtime.services.decisions.interface import DecisionProvider, DecisionService
from assistant_runtime.services.decisions.models import DecisionRequest, DecisionResponse

__all__ = [
    "DecisionError",
    "DecisionProvider",
    "DecisionRequest",
    "DecisionResponse",
    "DecisionService",
    "DecisionsConfig",
]
