from nltk.tokenize import wordpunct_tokenize as word_tokenize
from nltk.tokenize import sent_tokenize

import re
import six
import textwrap

_whitelist = r"[0-9a-z\,\.\/\<\>]+"
_regex = "0-9a-z\,\.\/\<\>"


def filter_by_lang_regex(text, ratio=0.7, regex="0-9a-z\,\.\/\<\>"):
    candidate_text = re.sub(r"[^" + regex + "]+", " ", six.ensure_str(text), flags=re.IGNORECASE).replace(" ", "")
    text = text.replace(" ", "")

    return (len(candidate_text) / len(text)) > ratio


def filter_by_num_tokens(text, gt=64):
    return len(word_tokenize(text)) > gt


def filter_by_num_sents(text, gt=2):
    return len(sent_tokenize(text)) > gt


def filter_by_steps(text):
    return re.search('(step|mix all)', text, re.IGNORECASE) is not None


def filter_by_length(text, gt=40):
    return len(text) > gt


def filter_by_item(item_list, gt=4):
    return len(item_list) > gt


def chars_to_preserve(sentence, whitelist):
    try:
        tokenized = re.findall(whitelist, sentence, re.IGNORECASE)
        return " ".join(tokenized)
    except Exception as error:
        print(
            textwrap.dedent(
                f"""
                Bad characters range {whitelist},
                {error}
                """
            )
        )
        raise


def normalizer(text, whitelist=r"[0-9a-z\,\.\/\<\>]+", do_lowercase=False):
    if do_lowercase:
        text = text.lower()

    text = chars_to_preserve(text, whitelist=whitelist)
    text = " ".join([word.strip() for word in text.split() if word.strip()])
    text = text.strip()

    return text

# _text = "Crust, Peanut Butter}Melt <sep> 1/2Butter, 2 c. Eggs, Filling, Semi- Sweet Chocolate Chips, Milk, Butter, " \
#         "Frosting"
# out = normalizer(_text)
# print(out)
#
# _text = "step ... "
# print(re.search('(step|mix all)', _text, re.IGNORECASE) != None)
