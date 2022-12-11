import re
from unidecode import unidecode
from total_regex import naive_score_total, regex_suffix, regex_prefix
from date_regex import naive_score_date, regex_date_suffix_time, regex_date_prefix_time, parse_datetime

def test_regex(text: str, verbose=False):
    text = unidecode(text).lower()

    date_matches = [(match, i) for i, match in enumerate(re.finditer(regex_date_prefix_time, text))] + [(match, i) for i, match in enumerate(re.finditer(regex_date_suffix_time, text))]
    total_matches = [(match, i) for i, match in enumerate(re.finditer(regex_prefix, text))] + [(match, i) for i, match in enumerate(re.finditer(regex_suffix, text))]

    possible_dates = []
    for match, i in date_matches:
        try:
            parsed_date, is_day_one_digit, is_month_one_digit, is_month_name, has_time = parse_datetime(match)
            sep_day = match.group('sep_day').strip()
            sep_month = match.group('sep_month').strip()

            sep_hour = match.group('sep_hour').strip() if match.group('sep_hour') else None
            sep_minute = match.group('sep_minute').strip() if match.group('sep_minute') else None

            possible_dates.append((naive_score_date({
                "date": parsed_date,
                "is_day_one_digit": is_day_one_digit,
                "is_month_one_digit": is_month_one_digit,
                "sep_day": sep_day,
                "sep_month": sep_month,
                "sep_hour": sep_hour,
                "sep_minute": sep_minute,
                "is_month_name": is_month_name,
                "has_time": has_time,
                "index": i,
            }, len(date_matches)), parsed_date))
        
        except Exception as e:
            pass # Impossible date found

    possible_total = []
    for match, i in total_matches:
        total_amount = round(int(match.group('integer_part')) + int(match.group('decimal_part')) / 100, 2)
        sep_decimal = match.group('sep_decimal').strip()

        has_adjacent = match.group('adjacent') is not None
        has_currency = (match.group('currency_prefix') is not None) ^ (match.group('currency_suffix') is not None)

        possible_total.append((naive_score_total({ 
            "total": total_amount,
            "has_adjacent": has_adjacent,
            "has_currency": has_currency,
            "sep_decimal": sep_decimal,
            "index": i,
        }, len(total_matches)), total_amount))


    date_scores = sorted(possible_dates, key=lambda x: x[0], reverse=True)
    total_scores = sorted(possible_total, key=lambda x: x[0], reverse=True)

    if (verbose):
        print("Date matches:")
        print(date_scores)
        print("Total matches:")
        print(total_scores)

    date = date_scores[0]
    total = total_scores[0]

    return date, total

test_regex("""
u -‘|L Rue l‘{ÔTenr 13 lâzà—-°‘ä-"’3"œ “Ï‘.'.“.:: ,}'yr_u.m« t‘O1(6,+ 3, "":.',n 4, \ P— ‘."““‘.»;«::e.?'ÈFÈË... d‘on éééc ”IICT EA … 71 95 'ﬂMÈL“ n”ä}ü erI £ : 7 MESER :«:£ e + — +4 : 9753 '—"OŒ.A d wup . féféc rt \ » 2.50— —L‘FIt BŒue 1 n e EM EROLLETTE > + 9,55 3 IHÎ54"ÈW I . POLLET . mr :1 VOLATLLE FRANÇATEE ‘, 14 tn 3 ”TOJAL ÎIÜ(E[ d » % 27.35 F ur rrre t r rmc ﬁe » . . … tp tm= c ” s— de rine os TTLe r DT/|& ‘T‘rre qée —-.;‘;,--‘;.__ 27 . 35— . FUR ATL‘TVA s pu ‘“'":«"'"—’--._--—« * l’°“" "" , î\ à WA‘ vfg !. â4 ® 3 Ë; 133 , 0 Bdﬂc.,ll‘l E(I‘°' tri g k2, 437 35‘È\ä 1__‘î_Pr lt(:.)oV Sip. Â‘r.kot ©94325 — .( __,.- _l .............. Vois : resté 2 v131t954 d pOX èbl0qœr votre œuinté fluth” 2..) ” Ces.ta tusn io lft” 1
""", True
)