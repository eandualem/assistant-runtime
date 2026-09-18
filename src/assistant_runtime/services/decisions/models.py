"""The decision call: one state, typed questions, typed answers.

The question and answer shapes follow the System One contract documented at
https://docs.typesafe.ai/api: a ``choice`` picks one option from its
``criteria`` map, a ``score`` picks one level from its ``criteria`` list, and
a ``noul`` answers whether a statement is true with a value from 0 to 1.
The runtime validates structure; the provider validates meaning.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

Instructions = str | dict[str, Any] | list[Any] | None
"""What the question asks: text, or the provider's structured form."""

CriteriaEntry = str | dict[str, Any] | None
"""One option or level: a description, ``{"what", "not_for", "examples"}``, or nothing."""


class ChoiceQuestion(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["choice"]
    instructions: Instructions = None
    criteria: dict[str, CriteriaEntry] = Field(min_length=2)


class ScoreQuestion(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["score"]
    instructions: Instructions = None
    criteria: list[CriteriaEntry] = Field(min_length=2)


class NoulQuestion(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["noul"]
    instructions: Instructions = None
    criteria: dict[Literal["true", "false"], CriteriaEntry] | None = None


Question = Annotated[ChoiceQuestion | ScoreQuestion | NoulQuestion, Field(discriminator="type")]


class DecisionRequest(BaseModel):
    """What an application sends. Provider credentials are never request fields."""

    model_config = ConfigDict(extra="forbid")

    state: Any
    questions: dict[Annotated[str, Field(min_length=1, max_length=64)], Question] = Field(
        min_length=1
    )
    profile: str | None = Field(default=None, pattern=r"^[a-z][a-z0-9_]{0,63}$")


Probability = Annotated[float, Field(ge=0, le=1)]
"""A calibrated value from 0 to 1; NaN and infinity are not answers."""


class ChoiceAnswer(BaseModel):
    model_config = ConfigDict(extra="allow")

    type: Literal["choice"]
    choice: str
    probabilities: dict[str, Probability]
    confidence: Probability


class ScoreAnswer(BaseModel):
    model_config = ConfigDict(extra="allow")

    type: Literal["score"]
    score: Annotated[float, Field(allow_inf_nan=False)]
    probabilities: dict[str, Probability]
    confidence: Probability
    legend: dict[str, Any] | None = None


class NoulAnswer(BaseModel):
    model_config = ConfigDict(extra="allow")

    type: Literal["noul"]
    noul: Probability


Answer = Annotated[ChoiceAnswer | ScoreAnswer | NoulAnswer, Field(discriminator="type")]


class Timing(BaseModel):
    """Milliseconds spent on this call: the provider round trip and the runtime's total."""

    total_ms: int
    provider_ms: int


class DecisionResponse(BaseModel):
    """The provider's answers, keyed like the questions, with usage and timing."""

    model_config = ConfigDict(extra="forbid")

    model: str | None = None
    answers: dict[str, Answer]
    usage: dict[str, Any] | None = None
    timing: Timing


__all__ = [
    "Answer",
    "ChoiceAnswer",
    "ChoiceQuestion",
    "DecisionRequest",
    "DecisionResponse",
    "NoulAnswer",
    "NoulQuestion",
    "Question",
    "ScoreAnswer",
    "ScoreQuestion",
    "Timing",
]
