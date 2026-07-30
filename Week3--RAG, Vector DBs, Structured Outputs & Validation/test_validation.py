#fail when output breaks the schema
import json

import pytest
from pydantic import ValidationError

from structured_output import StructuredAnswer


def test_missing_field_is_rejected():
    """Leaving out a required field (confidence) must fail."""
    with pytest.raises(ValidationError):
        StructuredAnswer(answer="x", source_files=[])


def test_wrong_type_is_rejected():
    """source_files must be a list, not a string."""
    with pytest.raises(ValidationError):
        StructuredAnswer(answer="x", source_files="lewis.pdf", confidence="low")


def test_bad_enum_is_rejected():
    """confidence must be one of high/medium/low - anything else fails."""
    with pytest.raises(ValidationError):
        StructuredAnswer(answer="x", source_files=[], confidence="maybe")


def test_extra_field_is_rejected():
    """extra='forbid' means an invented field must fail."""
    with pytest.raises(ValidationError):
        StructuredAnswer(answer="x", source_files=[], confidence="low", made_up="oops")



def test_non_json_string_is_rejected():
    """Raw text that isn't valid JSON must fail at the parse gate."""
    with pytest.raises(json.JSONDecodeError):
        json.loads("Sure! Here is your answer: {not json}")


def test_fenced_json_is_rejected_by_parser():
    """A common LLM failure: wrapping JSON in ```json fences breaks json.loads."""
    fenced = '```json\n{"answer": "x", "source_files": [], "confidence": "low"}\n```'
    with pytest.raises(json.JSONDecodeError):
        json.loads(fenced)


