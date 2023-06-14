from src.tokenizer import tokenize_text


def test_tokenize_text():
    text = '''Салом, ҷаҳон!'''
    tokens, spans = tokenize_text(text)
    assert tokens == ['Салом,', ' ҷаҳон!']
    assert spans == [(0,6), (6,13)]
    
    text = '''\nГурбаи🐈\tбемеҳр.'''
    tokens, spans = tokenize_text(text)
    assert tokens == ['\nГурбаи🐈', '\tбемеҳр.']

    text = '''Салом, ҷаҳон!\nГурбаи🐈\tбемеҳр.'''
    tokens, spans = tokenize_text(text)
    assert tokens == ['Салом,', ' ҷаҳон!', '\nГурбаи🐈', '\tбемеҳр.']

    text = '''зинда мекунанд,\n Аз муҳаббат'''
    tokens, spans = tokenize_text(text)
    assert tokens == ['зинда', ' мекунанд,', '\n Аз', ' муҳаббат']

    text = '''дар куҷост? – пурсид'''
    tokens, spans = tokenize_text(text)
    assert tokens == ['дар', ' куҷост?', ' – пурсид']

    text = '''Муҳаммадъалӣ ( 1347);'''
    tokens, spans = tokenize_text(text)
    assert tokens == ['Муҳаммадъалӣ', ' ( 1347);']

    text = '''мепурсам, ки: » -«Обиву'''
    tokens, spans = tokenize_text(text)
    assert tokens == ['мепурсам,', ' ки: »', ' -«Обиву']

    text = '''фароҳам шуд.\nФарҳангсарои ориёӣ'''
    tokens, spans = tokenize_text(text)
    assert tokens == ['фароҳам', ' шуд.', '\nФарҳангсарои', ' ориёӣ']

    text = '''яҳудон).\n3. Онҳоеанд, ки'''
    tokens, spans = tokenize_text(text)
    assert tokens == ['яҳудон).', '\n3.', ' Онҳоеанд,', ' ки']

    text = '''Барои чӣ ?" суоли кори'''
    tokens, spans = tokenize_text(text)
    assert tokens == ['Барои', ' чӣ ?', '"', ' суоли', ' кори']

