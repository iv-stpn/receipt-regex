from datetime import datetime
from unidecode import unidecode
import re

# FR Locale

GOOD_DATE_SEPS = ['-', '/', '']
GOOD_TIME_SEPS = [':', 'h', '']

empty_sep = lambda name, min_sep=0: f'(?P<{name}>[^\\n\d]{{{min_sep},3}})'

day = '(?P<day>[1-9]|0[1-9]|[12]\d|3[01])'
month = '(?P<month>[1-9]|0[1-9]|1[012])'

def current_year_regex():
    current_year = datetime.now().year
    this_decade = current_year // 10 % 10
    this_year = current_year // 10 % 10
    return f'(?P<year>(?:19)?[789]\d|(?:20)?[{"".join(map(str, range(this_decade)))}]\d|(?:20)?{this_decade}[{"".join(map(str, range(this_year+1)))}])'

year = current_year_regex()

hour = '(?P<hour>[01]\d|2[0-3])'
minute = '(?P<minute>[0-5]\d)'
second = '(?P<second>[0-5]\d)'

monthNames = ['january', 'february', 'march', 'april', 'may', 'june', 'july', 'august', 'september', 'october', 'november', 'december']
monthNamesFr = ['janvier', 'février', 'mars', 'avril', 'mai', 'juin', 'juillet', 'août', 'septembre', 'octobre', 'novembre', 'décembre']
specialMonths = {
    'jun': 'juin',
    'jul': 'juillet',
}
monthNamesFrNoAccent = list(map(unidecode, monthNamesFr))

def monthShort(month):
    if (len(month) == 3):
        return f"(?:{month}\.?)"
    elif (len(month) == 4):
        return f'(?:(?:{month[:3]}(?:{month[3]})?)\.?)'
    else:
        return f'(?:(?:{month[:3]}(?:{month[3]}(?:{month[4:]})?)?)\.?)'

regex_date_prefix_time = f"(?P<time>{hour}{empty_sep('sep_hour')}{minute}({empty_sep('sep_minute')}{second})?{empty_sep('sep_time', 1)})?{day}{empty_sep('sep_day')}((?P<month_name>{'|'.join(list(map(monthShort, monthNamesFrNoAccent)) + list(specialMonths.keys()))})|{month}){empty_sep('sep_month')}({year})"
regex_date_suffix_time = f"{day}{empty_sep('sep_day')}((?P<month_name>{'|'.join(list(map(monthShort, monthNamesFrNoAccent)) + list(specialMonths.keys()))})|{month}){empty_sep('sep_month')}({year})(?P<time>{empty_sep('sep_time', 1)}{hour}{empty_sep('sep_hour')}{minute}({empty_sep('sep_minute')}{second})?)?"

def get_sep_score(sep1, sep2, good_seps):
    if sep1 in good_seps:
        if (sep1 == sep2):
            return 30
        
        if sep2 in good_seps:
            return 5
        else:
            return 10
    
    if sep2 in good_seps:
        return 10
    
    return 0

def get_digit_score(is_day_one_digit, is_month_one_digit):
    if (is_day_one_digit and is_month_one_digit):
        return 20
    
    if (not is_day_one_digit and not is_month_one_digit):
        return 40

    if (is_day_one_digit):
        return 10
    
    return 0

def naive_score_date(possibility, len_matches):
    # Generate score based on whether the date is close to the current date (in an exponential )
    
    n_days_diff = (datetime.now() - possibility['date']).days + 1
    score = (1 / (1 + n_days_diff)) * 100
    score += 100 if possibility['has_time'] else 0
    score += (50 + 20 * possibility['is_day_one_digit']) if possibility['is_month_name'] else (get_sep_score(possibility['sep_day'], possibility['sep_month'], GOOD_DATE_SEPS) + get_digit_score(possibility['is_day_one_digit'], possibility['is_month_one_digit']))
    score += get_sep_score(possibility['sep_hour'], possibility['sep_minute'], GOOD_TIME_SEPS) if possibility['has_time'] else 0
    # Prioritize small and large indices with two normal distributions
    # Prioritize small indices
    score += (1 / (1 + possibility["index"])) * 40

    # Prioritize large indices
    score += (1 / (1 + len_matches - possibility["index"])) * 30

    return round(score, 2) / 300

def get_month_from_name(name):
    if name in specialMonths:
        name = specialMonths[name]

    possible_months = [i+1 for i, monthName in enumerate(monthNamesFrNoAccent) if name in monthName]
    if len(possible_months) > 0:
        return possible_months[0]
    
    raise Exception(f'Invalid month name {name}')

def parse_datetime(match):
    is_day_one_digit = len(match.group('day')) == 1
    day = int(match.group('day'))
    is_month_name = match.group('month') is None
    month = 6 if match.group('month_name') == 'jun' else (get_month_from_name(match.group('month_name')) if is_month_name else int(match.group('month')))
    

    is_month_one_digit = len(match.group('month')) == 1 if not is_month_name else False
    year = int(match.group('year'))

    if (year < 100):
        if (year < 70):
            year = 2000 + year
        else:
            year = 1900 + year
    
    # print(f'Parsed date {year}-{month}-{day} (day one digit: {is_day_one_digit}, month name: {is_month_name})')
    if match.group('time') is not None:
        hours = int(match.group('hour'))
        minutes = int(match.group('minute'))
        seconds = match.group('second') is not None and int(match.group('second')) or 0

        return datetime(year, month, day, hours, minutes, seconds), is_day_one_digit, is_month_one_digit, is_month_name, True

    return datetime(year, month, day), is_day_one_digit, is_month_one_digit, is_month_name, False


print(regex_date_suffix_time)
