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

segmenter = TextSegmenter('nn')  # or 'heuristic'
result = segmenter.segment_text(text)
print('Sentences:', result['sentences'])
print('Per token probabilities:')
for t, (sp, ep) in zip(result['tokens'], result['probs']):
    print(f"{repr(t):20s}  start: {sp:.2f}  end: {ep:.2f}")
```

Output:
```python
Sentences: ['Осоишгоҳҳои умумӣ (барои калонсолон) ва махсус (оилавӣ, барои занҳои ҳомила, ҷавонон, байнихоҷагӣ, соҳили дарёию баҳрӣ ва ғ.) мешаванд.', ' Осоишгоҳҳо барои дамгирии якрӯза, 6 -рӯза, 12-рӯза ва 24-рӯза таъйин шудаанд.', '\nДар Тоҷикистон осоишгоҳҳои:', '\n«Қаротоғ», «Явроз» дар водии Ҳисор;', '\nОсоишгоҳи Зумрад, «Баҳористон», «Конибодом», «Ҳавотоғ», ва «Ӯротеппа» ва диг. дар вилояти Суғд;', '\n«Чилучорчашма», «Сари Хосор» ва диг. дар вилояти Хатлон;', '\n«Гармчашма» ва диг. дар ВМКБ амал карда истодаанд.']
Per token probabilities:
'Осоишгоҳҳои'         start: 1.00  end: 0.00
' умумӣ'              start: 0.00  end: 0.00
' (барои'             start: 0.00  end: 0.00
' калонсолон)'        start: 0.00  end: 0.00
' ва'                 start: 0.00  end: 0.00
' махсус'             start: 0.00  end: 0.00
' (оилавӣ,'           start: 0.00  end: 0.00
' барои'              start: 0.00  end: 0.00
' занҳои'             start: 0.00  end: 0.00
' ҳомила,'            start: 0.00  end: 0.00
' ҷавонон,'           start: 0.00  end: 0.00
' байнихоҷагӣ,'       start: 0.00  end: 0.00
' соҳили'             start: 0.00  end: 0.00
' дарёию'             start: 0.00  end: 0.00
' баҳрӣ'              start: 0.00  end: 0.00
' ва'                 start: 0.00  end: 0.00
' ғ.)'                start: 0.00  end: 0.00
' мешаванд.'          start: 0.02  end: 0.99
' Осоишгоҳҳо'         start: 0.70  end: 0.00
' барои'              start: 0.00  end: 0.00
' дамгирии'           start: 0.00  end: 0.00
' якрӯза,'            start: 0.00  end: 0.00
' 6'                  start: 0.00  end: 0.00
' -рӯза,'             start: 0.00  end: 0.00
' 12-рӯза'            start: 0.00  end: 0.00
' ва'                 start: 0.00  end: 0.00
' 24-рӯза'            start: 0.00  end: 0.00
' таъйин'             start: 0.00  end: 0.00
' шудаанд.'           start: 0.00  end: 1.00
'\nДар'               start: 1.00  end: 0.00
' Тоҷикистон'         start: 0.00  end: 0.00
' осоишгоҳҳои:'       start: 0.00  end: 1.00
'\n«Қаротоғ»,'        start: 1.00  end: 0.00
' «Явроз»'            start: 0.00  end: 0.00
' дар'                start: 0.00  end: 0.00
' водии'              start: 0.00  end: 0.00
' Ҳисор;'             start: 0.00  end: 1.00
'\nОсоишгоҳи'         start: 0.99  end: 0.00
' Зумрад,'            start: 0.00  end: 0.00
' «Баҳористон»,'      start: 0.00  end: 0.00
' «Конибодом»,'       start: 0.00  end: 0.00
' «Ҳавотоғ»,'         start: 0.00  end: 0.00
' ва'                 start: 0.00  end: 0.00
' «Ӯротеппа»'         start: 0.00  end: 0.00
' ва'                 start: 0.00  end: 0.00
' диг.'               start: 0.00  end: 0.22
' дар'                start: 0.12  end: 0.00
' вилояти'            start: 0.00  end: 0.00
' Суғд;'              start: 0.00  end: 1.00
'\n«Чилучорчашма»,'   start: 1.00  end: 0.00
' «Сари'              start: 0.00  end: 0.00
' Хосор»'             start: 0.00  end: 0.00
' ва'                 start: 0.00  end: 0.00
' диг.'               start: 0.00  end: 0.29
' дар'                start: 0.17  end: 0.00
' вилояти'            start: 0.00  end: 0.00
' Хатлон;'            start: 0.00  end: 1.00
'\n«Гармчашма»'       start: 1.00  end: 0.00
' ва'                 start: 0.00  end: 0.00
' диг.'               start: 0.00  end: 0.31
' дар'                start: 0.25  end: 0.00
' ВМКБ'               start: 0.00  end: 0.00
' амал'               start: 0.00  end: 0.00
' карда'              start: 0.00  end: 0.00
' истодаанд.'         start: 0.00  end: 1.00
```