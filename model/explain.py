from difflib import SequenceMatcher


def _explain(output, descriptor):
    score = []
    for item in descriptor:
        score.append(SequenceMatcher(None, str(output.lower()), str(" ".join(list(item.keys()))).lower()).ratio())
    ind = score.index(max(score))
    return " ".join(list(descriptor[ind].values()))