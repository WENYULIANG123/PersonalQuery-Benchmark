def smart_split_sentences(text):
    abbreviations = {
        'dr', 'mr', 'mrs', 'ms', 'sr', 'jr', 'ph.d', 'b.a', 'm.a', 'm.s',
        'i.e', 'e.g', 'etc', 'vs', 'cf', 'et al', 'al', 'fig', 'vol',
        'no', 'pp', 'p', 'sec', 'para', 'chap', 'vols', 'ed', 'eds',
        'co', 'corp', 'inc', 'ltd', 'llc', 'a.m', 'p.m'
    }
    sentences = []
    current_sentence = ""
    i = 0
    while i < len(text):
        char = text[i]
        current_sentence += char
        if char == '.':
            remaining_text = text[i + 1:].strip()
            if not remaining_text:
                sentences.append(current_sentence.strip())
                current_sentence = ""
                break
            next_char = remaining_text[0] if remaining_text else ''
            starts_new_sentence = next_char.isupper()
            sentence_lower = current_sentence.lower().strip()
            without_period = sentence_lower.rstrip('.')
            words = without_period.split()
            last_word = words[-1] if words else ""
            is_abbreviation = last_word in abbreviations
            is_abbrev_with_period = False
            for abbrev in abbreviations:
                if sentence_lower.endswith(abbrev + '.'):
                    is_abbrev_with_period = True
                    break
            next_is_digit = next_char.isdigit()
            if starts_new_sentence and not (is_abbreviation or is_abbrev_with_period) and not next_is_digit:
                sentences.append(current_sentence.strip())
                current_sentence = ""
        i += 1
    if current_sentence.strip():
        sentences.append(current_sentence.strip())
    return [s for s in sentences if s.strip()]

# Test cases
test_text = 'Dr. Smith went to the store. He bought 2.5 pounds of apples. This costs $5.99. Mrs. Johnson said hello. The meeting is at 3 p.m. etc.'
sentences = smart_split_sentences(test_text)
print(f'Split into {len(sentences)} sentences:')
for i, s in enumerate(sentences, 1):
    print(f'{i}. {s}')
