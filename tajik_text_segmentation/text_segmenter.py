import re
import json
from tajik_text_segmentation.annotated import Annotated
from tajik_text_segmentation.heuristic_model import HeuristicModel, HeuristicPredictor
from tajik_text_segmentation.boundary_resolver import SentenceBoundaryResolver
from tajik_text_segmentation.nn_model.model import Checkpoint, NNPredictor

from tajik_text_segmentation.tokenizer import tokenize_text

import pkg_resources


def preprocess_text(text: str):
    # remove extra spaces
    text = text.strip()
    text = re.sub(r'\n+', '\n', text)
    text = re.sub(r' +', ' ', text)
    text = text.replace(' \n', '\n')
    text = text.replace('\n ', '\n')
    return text


def load_config():

    resource_package = 'tajik_text_segmentation'
    resource_path = 'config.json'  # Relative path to config.json within the package

    config_string = pkg_resources.resource_string(resource_package, resource_path)
    config = json.loads(config_string.decode())

    return config


def load_predictor(model_name):
    config = load_config()['models'][model_name]
    if model_name == "nn":
        checkpoint = Checkpoint.init_from_path(config['checkpoint_path'])
        resolver = SentenceBoundaryResolver(**config['resolver_kwargs'])
        return NNPredictor(checkpoint.config, checkpoint.model, checkpoint.vocab, resolver=resolver)
    else:
        model = HeuristicModel()
        resolver = SentenceBoundaryResolver(**config['resolver_kwargs'])
        return HeuristicPredictor(model=model, resolver=resolver, **config['predictor_kwargs'])


class TextSegmenter:
    def __init__(self, model_name):
        self.model_name = model_name
        self.predictor = load_predictor(model_name)
    
    def segment_text(self, text, preprocess=True):
        
        # preproces text
        if preprocess:
            text = preprocess_text(text)

        # tokenize
        tokens, token_spans = tokenize_text(text)
        d = self.predictor.predict(tokens)
        preds = d['preds']

        ann = Annotated(tokens, start_labels=preds[:, 0], end_labels=preds[:,1])
        d['tokens'] = tokens
        d['spans'] = ann.spans
        d['annotated_text'] = ann.annotated_text
        d['sentences'] = ann.get_sentences()
        d['token_spans'] = token_spans
        d['text'] = text

        return d


if __name__ == '__main__':
    text = '''Осоишгоҳҳои умумӣ (барои калонсолон) ва махсус (оилавӣ, барои занҳои ҳомила, ҷавонон, байнихоҷагӣ, соҳили дарёию баҳрӣ ва ғ.) мешаванд. Осоишгоҳҳо барои дамгирии якрӯза, 6 -рӯза, 12-рӯза ва 24-рӯза таъйин шудаанд.
Дар Тоҷикистон осоишгоҳҳои:
«Қаротоғ», «Явроз» дар водии Ҳисор;
Осоишгоҳи Зумрад, «Баҳористон», «Конибодом», «Ҳавотоғ», ва «Ӯротеппа» ва диг. дар вилояти Суғд;
«Чилучорчашма», «Сари Хосор» ва диг. дар вилояти Хатлон;
«Гармчашма» ва диг. дар ВМКБ амал карда истодаанд.'''
    segmenter = TextSegmenter('nn')  # or 'heuristic'
    result = segmenter.segment_text(text)
    print('Sentences:', result['sentences'])
    print('Per token probabilities:')
    for t, (sp, ep) in zip(result['tokens'], result['probs']):
        print(f"{repr(t):20s}  start: {sp:.2f}  end: {ep:.2f}")
