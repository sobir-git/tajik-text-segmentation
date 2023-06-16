import numpy as np
from sklearn.metrics import precision_recall_fscore_support

from tajik_text_segmentation.annotated import Annotated, load_dataset, tokenize_annotation
from tajik_text_segmentation.text_segmenter import load_predictor


def evaluate_segmentation(labels: np.ndarray, predictions: np.ndarray):
    """
    labels: 2d-tensor of shape (N, 2) where N is the number of samples and 2 is the number of classes (is_sentence_start, is_sentence_end)
    predictions: 2d-tensor of shape (N, 2) where N is the number of samples and 2 is the number of classes (is_sentence_start, is_sentence_end)
    """

    assert labels.shape == predictions.shape

    # if both empty
    if labels.shape[0] == 0 and predictions.shape[0] == 0:
        return {
            "start_precision": 0,
            "start_recall": 0,
            "start_f1": 0,
            "end_precision": 0,
            "end_recall": 0,
            "end_f1": 0
        }    

    # Extract start and end predictions from the second column of the predictions tensor
    start_preds = predictions[:, 0]
    end_preds = predictions[:, 1]

    # Extract start and end labels from the second column of the labels tensor
    start_labels = labels[:, 0]
    end_labels = labels[:, 1]

    # Calculate precision, recall, and F1 score for start of sentence
    start_precision, start_recall, start_f1, _ = precision_recall_fscore_support(start_labels, start_preds, average='binary')

    # Calculate precision, recall, and F1 score for end of sentence
    end_precision, end_recall, end_f1, _ = precision_recall_fscore_support(end_labels, end_preds, average='binary')

    # Return precision, recall, and F1 score for each category
    return {
        "start_precision": start_precision,
        "start_recall": start_recall,
        "start_f1": start_f1,
        "end_precision": end_precision,
        "end_recall": end_recall,
        "end_f1": end_f1
    }


def evaluate_from_text(model, annotations: "list[Annotated]", display_report: bool=False):
    predictions = []
    start_labels = []
    end_labels = []
    for ann in annotations:
        ann = tokenize_annotation(ann)
        sl = ann.start_labels
        el = ann.end_labels
        pred = model.predict(ann.tokens)['preds']
        pred = np.asarray(pred, dtype=np.int32)  # (len(tokens),2)
        sl = np.asarray(sl, dtype=np.int32)
        el = np.asarray(el, dtype=np.int32)

        assert len(sl) == len(el) == len(pred), (len(sl), len(el), len(pred))
        start_labels.append(sl)
        end_labels.append(el)
        predictions.append(pred)

        if display_report:
            generate_report(ann.tokens, pred, sl, el)
    

    predictions = np.concatenate(predictions)  # (N,2)
    start_labels = np.concatenate(start_labels)
    end_labels = np.concatenate(end_labels)
    labels = np.vstack([start_labels, end_labels]).T   # (N,2)
    
    return evaluate_segmentation(labels, predictions)


def display_annotation(tokens: "list[str]", start_labels: "list[bool]", end_labels: "list[bool]"):
    buffer = []
    for token, sl, el in zip(tokens, start_labels, end_labels):
        sl = '{{' if sl else ''
        el = '}}' if el else ''
        buffer.append("{}{}{}".format(sl, repr(token)[1:-1], el))
    
    return ''.join(buffer)


def generate_report(tokens: "list[str]", pred: np.ndarray, start_labels: np.ndarray, end_labels: np.ndarray):
    start_pred = pred[:, 0]
    end_pred = pred[:, 1]

    # locate all places where prediction is wrong
    wrong = (start_pred != start_labels) | (end_pred != end_labels)
    wrong = np.nonzero(wrong)[0].tolist()

    # group all indices that have difference < 3 together
    groups = []
    for i in range(len(wrong)):
        if i == 0:
            groups.append([wrong[i]])
        elif wrong[i] - wrong[i - 1] < 3:
            groups[-1].append(wrong[i])
        else:
            groups.append([wrong[i]])

    # for each group select window
    for group in groups:
        start = max(0, group[0] - 2)
        end = min(group[-1] + 3, len(tokens))
        print('<<<')
        print(display_annotation(tokens[start:end], start_labels[start:end], end_labels[start:end]))
        print('||||')
        print(display_annotation(tokens[start:end], start_pred[start:end], end_pred[start:end]))
        print('>>>')


if __name__ == '__main__':
    annotations = load_dataset()
    
    # predictor = load_predictor('heuristic')
    # results = evaluate_from_text(predictor, annotations, display_report=True)
    # print("Heuristic model results:")
    # print(results)

    # neural network results
    predictor = load_predictor('nn')
    results = evaluate_from_text(predictor, annotations, display_report=True)
    print("Neural network results:")
    print(results)

