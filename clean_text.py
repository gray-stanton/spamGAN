import re
def clean(s,):
    ns = s.lower()
    ns = re.sub('[0-9]+', 'N', ns)
    ns = re.sub('[^a-zA-Z0-9 \-.,\'\"!?()]', ' ', ns) # Eliminate all but these chars
    ns = re.sub('([.,!?()\"\'])', r' \1 ', ns) # Space out punctuation
    ns = re.sub('\s{2,}', ' ', ns) # Trim ws
    str.strip(ns)
    return ns


