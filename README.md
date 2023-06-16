# Tajik text segmentation

Usage:
```python
from src.text_segmenter import TextSegmenter

text = '''Осоишгоҳҳои умумӣ (барои калонсолон) ва махсус (оилавӣ, барои занҳои ҳомила, ҷавонон, байнихоҷагӣ, соҳили дарёию баҳрӣ ва ғ.) мешаванд. Осоишгоҳҳо барои дамгирии якрӯза, 6 -рӯза, 12-рӯза ва 24-рӯза таъйин шудаанд.
Дар Тоҷикистон осоишгоҳҳои:
«Қаротоғ», «Явроз» дар водии Ҳисор;
Осоишгоҳи Зумрад, «Баҳористон», «Конибодом», «Ҳавотоғ», ва «Ӯротеппа» ва диг. дар вилояти Суғд;
«Чилучорчашма», «Сари Хосор» ва диг. дар вилояти Хатлон;
«Гармчашма» ва диг. дар ВМКБ амал карда истодаанд.'''

segmenter = TextSegmenter('heuristic')
ann = segmenter.segment_text(text)
print("Annotated text:", ann.annotated_text)
print("Tokens:", ann.tokens)
print("Sentence spans:", ann.spans)
print("Sentences:", ann.get_sentences())
```