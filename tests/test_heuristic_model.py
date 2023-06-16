
from tajik_text_segmentation.heuristic_model import predict_token


def test_predict_token():
    assert predict_token('', '\nСалом', ' ҷаҳон!') == (True,False)
    assert predict_token('\nМо', ' камбағалонем', '') == (False,True)
    assert predict_token(' Муҳаммад', ' с.а.в.', ' фармуданд:') == (False,False)
    assert predict_token(' А.', ' Имомзода', ' гузориши') == (False,False)
    assert predict_token(' 29', ' млн.', ' тон') == (False,False)
    assert predict_token(' аз', ' Восеъ', ' хабари') == (False,False)
    assert predict_token(' рафт.', ' Имрӯз', ' хабари') == (True,False)
    assert predict_token(' дирӯз', ' рафт.', ' Имрӯз') == (False,True)
    assert predict_token(' дирӯз', ' рафт...', ' вале') == (False,False)
    assert predict_token(' чаро', ' рафтӣ?', ' Бо') == (False,True)
    assert predict_token(' чаро', ' рафтӣ!', ' Бо') == (False,True)
    assert predict_token(' (тоҷ. ', ' Боймурод', ' Валиев') == (False,False)
    assert predict_token(' мудир', ' (тоҷ. ', ' Боймурод') == (False,False)
    assert predict_token(' ҷаз.', ' Шимолӣ)', ' ба') == (False,False)
    assert predict_token(' гуфтаанд:', ' Ҳар', ' ки') == (True,False)
    assert predict_token(' инҳоянд:', ' 1.', ' Манту') == (True,False)
    # assert predict_token(' ва', ' 1948.', ' Дар') == (False,True)
    assert predict_token(' расид.', ' 22', ' майи') == (True,False)
    assert predict_token(' расид.', ' 22-уми', ' майи') == (True,False)
    assert predict_token(' хабар,', '\nКи', ' бар') == (True,False)
    assert predict_token(' гуфто:', ' «Чӣ', ' гунаӣ') == (True,False)
    assert predict_token(' бедодгарона', '\nгум', 'шуд') == (True,False)
    assert predict_token(' кӯҳнаи', ' бедодгарона', '\nгум') == (False,True)
    assert predict_token(' эътибори', ' мардӣ!', '') == (False,True)
    assert predict_token(' мардӣ', ' !', '') == (False,True)
    assert predict_token(' шоистагон', ' соз!»', ' (101)') == (False,True)
    assert predict_token(' соз!»', ' (101)', ' Инҳо') == (True,False)
    assert predict_token(' ва', ' М.', ' Маданӣ') == (False,False)
    assert predict_token(' кунед!', ' — гӯён', ' фарёд') == (False,False)
    assert predict_token(' дар', ' куҷост?', ' – пурсид') == (False,False)
    assert predict_token(' эронӣ', ' форс-тоҷик).', ' Яке') == (False,True)
    assert predict_token(' эронӣ', ' (форс-тоҷик).', ' Яке') == (False,True)
    assert predict_token(' эронӣ', ' форс-тоҷик.)', ' Яке') == (False,True)
    assert predict_token(' 3', ' млн.', ' 865') == (False,False)
    assert predict_token(' куҷост?', ' - пурсид', ' командир') == (False,False)
    assert predict_token(' писарбача.', ' – Товариш,', ' поям') == (True,False)
    assert predict_token(' мекунад?', '\n- Товариш!', ' Товариш!') == (True,True)
    assert predict_token(' (=медоли', ' тиллои)', ' Улампиёди') == (False,False)
    assert predict_token(' намекушодам', ' 13.', '') == (False,True)
    assert predict_token(' мепурсам,', ' ки: »', '-«Обиву') == (False,True)
    assert predict_token(' яҳудон).', '\n3.', ' Онҳоеанд,') == (True,False)
    assert predict_token('\n3.', ' Онҳоеанд,', ' ки') == (False,False)
    assert predict_token(' наҳвӣ', ' \nОн,', ' яке') == (True,False)

'''
# Unsolved cases:
душман бираси.>]\t»\n«[<\tДил гарчи дар
1908 ва 1948. Дар июли
Барои чӣ ?" суоли кори баҳс
Киштибон ва наҳвӣ \nОн яке наҳвӣ
'''
