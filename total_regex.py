from unidecode import unidecode
import scipy.stats

# FR Locale

GOOD_DECIMAL_SEPS = ['', ',', '.']

empty_sep = lambda name, max_size=2: f'(?P<{name}>[^\\n\d]{{0,{max_size}}})'

adjacent_texts = ['total', 'ttc', 'montant', 'a payer', 'montant du', 'total avec avantages']
adjacent = lambda name: f'(?P<{name}>(?:{"|".join(adjacent_texts)}))'

currecy_indicators = ['€', 'euros', 'euro', 'eur', 'e', '€uros', '€uro', '€ur']
currency = lambda name: f'(?P<{name}>(?:{"|".join(currecy_indicators)}))'

integer_part = '(?P<integer_part>(?:[1-9]\d+)|\d)'
decimal_part = '(?P<decimal_part>\d{2})'

regex_prefix = f"({adjacent('adjacent')}{empty_sep('sep_prefix', 14)})?({currency('currency_prefix')}{empty_sep('sep_currency_prefix')})?(?:{integer_part}{empty_sep('sep_decimal')}{decimal_part})({empty_sep('sep_currency_suffix')}{currency('currency_suffix')})?"
regex_suffix = f"({currency('currency_prefix')}{empty_sep('sep_currency_prefix')})?(?:{integer_part}{empty_sep('sep_decimal')}{decimal_part})({empty_sep('sep_currency_suffix')}{currency('currency_suffix')})?({empty_sep('sep_suffix', 14)}{adjacent('adjacent')})?"

MAX_REASONABLE = 250
def naive_score_total(possibility, len_matches):
    return round(100 * possibility["has_adjacent"] + 50 * possibility['has_currency'] + 50 * (possibility["sep_decimal"] in GOOD_DECIMAL_SEPS) + (1 / (1 + len_matches - possibility["index"])) * 30 + (0 if possibility['total'] > MAX_REASONABLE else scipy.stats.norm(MAX_REASONABLE, MAX_REASONABLE/4) * possibility['total'] * 70), 2) / 300