import numpy as np
from src.dataset import load_dataset
from src.dummy_model import DummyModel

from src.evaluate import evaluate_segmentation, evaluate_from_text


def test_evalute_segmentation():
    # TODO: Write numerically precise tests

    # Test case 1: Perfect predictions
    labels_1      = np.array([[1, 1], [0, 0], [1, 1], [0, 0], [0, 1]])  # True labels
    predictions_1 = np.array([[1, 1], [0, 0], [1, 1], [0, 0], [0, 1]])  # Predicted labels

    results_1 = evaluate_segmentation(labels_1, predictions_1)
    assert results_1['start_f1'] == 1
    assert results_1['end_f1'] == 1

    # Test case 2: Imperfect predictions
    labels_2      = np.array([[1, 1], [0, 0], [1, 1]])  # True labels
    predictions_2 = np.array([[0, 1], [0, 1], [1, 0]])  # Predicted labels

    results_2 = evaluate_segmentation(labels_2, predictions_2)
    assert results_2['start_f1'] < 1
    assert results_2['end_f1'] < 1

    # Test case 3: Empty sequence
    labels_3 = np.array([])  # True labels
    predictions_3 = np.array([])  # Predicted labels

    results_3 = evaluate_segmentation(labels_3, predictions_3)
    assert results_3['start_f1'] == 0
    assert results_3['end_f1'] == 0


def test_evaluate_from_text():
    annotations = load_dataset()[:10]
    model = DummyModel()
    results = evaluate_from_text(model, annotations)
    assert results['start_f1'] > 0
    assert results['end_f1'] > 0

