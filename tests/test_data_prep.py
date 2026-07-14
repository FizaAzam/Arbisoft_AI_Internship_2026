import pandas as pd
import pytest

from data_prep import clean_text, prepare_frame


def test_removes_html_tags():
    assert clean_text("a great movie<br /><br />really") == "a great movie really"


def test_removes_urls():
    assert clean_text("see http://example.com/review now") == "see now"


def test_removes_punctuation():
    assert clean_text("wow!!! it's great, truly.") == "wow it s great truly"


def test_collapses_whitespace_and_strips():
    assert clean_text("  too    many \n spaces  ") == "too many spaces"


@pytest.mark.parametrize("text", ["", "   ", "<br />", "!!!"])
def test_returns_empty_string_when_nothing_survives(text):
    assert clean_text(text) == ""


def test_is_idempotent():
    once = clean_text("<b>Great</b> film -- see http://x.com !")
    assert clean_text(once) == once


def test_prepare_frame_lowercases_and_cleans_text():
    df = pd.DataFrame({"text": ["<br />GREAT Movie!"], "label": [1]})
    assert prepare_frame(df)["text"].tolist() == ["great movie"]


def test_prepare_frame_preserves_labels():
    df = pd.DataFrame({"text": ["good!", "bad!"], "label": [1, 0]})
    assert prepare_frame(df)["label"].tolist() == [1, 0]


def test_prepare_frame_does_not_mutate_input():
    df = pd.DataFrame({"text": ["<br />GREAT Movie!"], "label": [1]})
    prepare_frame(df)
    assert df["text"].tolist() == ["<br />GREAT Movie!"]
