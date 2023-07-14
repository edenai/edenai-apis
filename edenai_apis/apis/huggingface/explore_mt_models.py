import re

# import ipdb
from .mt_models_huggingface import models

normal_models = [
    m for m in models if re.match("Helsinki-NLP/opus-mt-[a-zA-Z]+-[a-zA-Z]+$", m)
]
other_models = [m for m in models if m not in normal_models]
other_two_models = [mod for mod in other_models if "tatoeba" in mod or "tc-big-" in mod]


def find_if_ok(i, j):
    return re.findall(i, j)[0] if re.findall(i, j) else j


normal_languages = [find_if_ok("[a-zA-Z]+-[a-zA-Z]+$", m) for m in normal_models]
other_languages = [find_if_ok("[a-zA-Z]+-[a-zA-Z]+$", m) for m in other_models]
other_two_languages = [find_if_ok("[a-zA-Z]+-[a-zA-Z]+$", m) for m in other_two_models]

diff_languages = [l for l in other_languages if l not in normal_languages]

src_normal = [lang.split("-")[0] for lang in normal_languages]
trg_normal = [lang.split("-")[1] for lang in normal_languages]

src_two = [lang.split("-")[0] for lang in other_two_languages]
trg_two = [lang.split("-")[1] for lang in other_two_languages]

src = set(src_normal + src_two)
trg = set(trg_normal + trg_two)

# ipdb.set_trace()
