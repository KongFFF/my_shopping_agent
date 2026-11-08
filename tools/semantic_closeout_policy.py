"""User-confirmed colloquial money policy. No query or task-ID exceptions."""
import re


def budget_amount(phrase, parser, fallback):
    # Only explicit monetary context is eligible; quantities such as 十几包
    # are not prices. Unsupported multi-amount clauses stay with the validator.
    token = r'(?:[二三四五六七八九]?十几|三四十)'
    hits = list(re.finditer(token + r'(?=\s*(?:元|块钱|块))', phrase))
    if len(hits) != 1:
        return fallback(phrase)
    hit = hits[0]
    rest = phrase[:hit.start()] + phrase[hit.end():]
    if re.search(r'\d|[零〇一二两三四五六七八九十百千万]|每|/|单价|以上|不少于|至少|高于|超过|到|至|低于', rest):
        return fallback(phrase)
    amount = 49 if hit[0] == '三四十' else int(parser(hit[0][:-1])) + 9
    return 'at_most', {'type':'number', 'value':float(amount), 'unit':'CNY'}, 'user_confirmed_colloquial_upper'
