import pytest
from src.dataset import load_dataset, get_labels, tokenize_annotation


def test_load_dataset():
    annotations = load_dataset()
    assert len(annotations)
    assert isinstance(annotations[0], dict)
    assert isinstance(annotations[0]['text'], str)
    assert isinstance(annotations[0]['labels'], list)
    assert isinstance(annotations[0]['spans'], list)


@pytest.mark.parametrize('spans, indices, expected', [
    ([(0, 6), (6, 13)], [1], [1,0]),
    ([(0, 6), (6, 13)], [6], [0,1]),
    ([(0, 6), (6, 13)], [12], [0,1]),
    ([(0, 6), (6, 13)], [13], [0,0]),
    ([(0, 6), (8, 13)], [6], [0,0]),
    ([(2, 6), (8, 13)], [1], [0,0]),
])
def test_get_labels(spans, indices, expected):
    """For each span checks if any indices belong to it."""

    get_labels(spans, indices) == expected


@pytest.mark.parametrize('ann', [
    {'text': 'Hello, World! Good world.', 'spans': [(0, 13), (13,25)], 'labels': ['Sentence', 'Sentence']}
])
def test_tokenize_annotation(ann):
    """For each span checks if any indices belong to it."""
    tann = tokenize_annotation(ann)
    assert len(tann['tokens']) == len(tann['start_labels']) == len(tann['end_labels'])
    assert tann['start_labels'] == [1, 0, 1, 0]
    assert tann['end_labels'] == [0, 1, 0, 1]
    assert tann['spans'] == [[0,2],[2,4]]
