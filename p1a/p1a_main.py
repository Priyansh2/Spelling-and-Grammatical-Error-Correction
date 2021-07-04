from __future__ import unicode_literals
import sys
import re
import array
import os
import numpy as np
import pickle
from collections import Counter
from collections import defaultdict
def tokenize(text):
    STARTING_QUOTES = [
        (re.compile(r'^\"'), r'``'),
        (re.compile(r'(``)'), r' \1 '),
        (re.compile(r"([ \(\[{<])(\"|\'{2})"), r'\1 `` '),
    ]
    # punctuation
    PUNCTUATION = [
        (re.compile(r'([:,])([^\d])'), r' \1 \2'),
        (re.compile(r'([:,])$'), r' \1 '),
        (re.compile(r'\.\.\.'), r' ... '),
        (re.compile(r'[;@#$%&]'), r' \g<0> '),
        (re.compile(r'([^\.])(\.)([\]\)}>"\']*)\s*$'), r'\1 \2\3 '),  # Handles the final period.
        (re.compile(r'[?!]'), r' \g<0> '),

        (re.compile(r"([^'])' "), r"\1 ' "),
    ]
    # Pads parentheses
    PARENS_BRACKETS = (re.compile(r'[\]\[\(\)\{\}\<\>]'), r' \g<0> ')
    # Optionally: Convert parentheses, brackets and converts them to PTB symbols.
    CONVERT_PARENTHESES = [
        (re.compile(r'\('), '-LRB-'), (re.compile(r'\)'), '-RRB-'),
        (re.compile(r'\['), '-LSB-'), (re.compile(r'\]'), '-RSB-'),
        (re.compile(r'\{'), '-LCB-'), (re.compile(r'\}'), '-RCB-')
    ]
    DOUBLE_DASHES = (re.compile(r'--'), r' -- ')
    # ending quotes
    ENDING_QUOTES = [
        (re.compile(r'"'), " '' "),
        (re.compile(r'(\S)(\'\')'), r'\1 \2 '),
        (re.compile(r"([^' ])('[sS]|'[mM]|'[dD]|') "), r"\1 \2 "),
        (re.compile(r"([^' ])('ll|'LL|'re|'RE|'ve|'VE|n't|N'T) "), r"\1 \2 "),
    ]
    # List of contractions adapted from Robert MacIntyre's tokenizer.
    CONTRACTIONS2 = list(map(re.compile, [r"(?i)\b(can)(?#X)(not)\b",
                         r"(?i)\b(d)(?#X)('ye)\b",
                         r"(?i)\b(gim)(?#X)(me)\b",
                         r"(?i)\b(gon)(?#X)(na)\b",
                         r"(?i)\b(got)(?#X)(ta)\b",
                         r"(?i)\b(lem)(?#X)(me)\b",
                         r"(?i)\b(mor)(?#X)('n)\b",
                         r"(?i)\b(wan)(?#X)(na)\s"]))
    CONTRACTIONS3 = list(map(re.compile, [r"(?i) ('t)(?#X)(is)\b", r"(?i) ('t)(?#X)(was)\b"]))
    for regexp, substitution in STARTING_QUOTES:
        text = regexp.sub(substitution, text)
    for regexp, substitution in PUNCTUATION:
        text = regexp.sub(substitution, text)
    # Handles parentheses.
    regexp, substitution = PARENS_BRACKETS
    text = regexp.sub(substitution, text)
    # Optionally convert parentheses
    for regexp, substitution in CONVERT_PARENTHESES:
        text = regexp.sub(substitution, text)
    # Handles double dash.
    regexp, substitution = DOUBLE_DASHES
    text = regexp.sub(substitution, text)
    # add extra space to make things easier
    text = " " + text + " "
    for regexp, substitution in ENDING_QUOTES:
        text = regexp.sub(substitution, text)
    for regexp in CONTRACTIONS2:
        text = regexp.sub(r' \1 \2 ', text)
    for regexp in CONTRACTIONS3:
        text = regexp.sub(r' \1 \2 ', text)
    tokens=text.split()
    return tokens


def find_sentences(text):
    # Given text, tokenise into sentences and words. Take into consideration various delimiters.
    #tokens=re.findall('[A-Z]?[a-z]+|[A-Z]+|[0-9]+th|[0-9]+st|[0-9]+rd|[0-9]+nd|[a-z]+-[a-z]+|Dr\.|Mr\.|Mrs\.|\'s|\'d|\.|,|&|\d{1,3}\.\d{1,3}\.\d{1,3}|[\w\.-]+@[\w\.\w]|http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+|&',text)
    caps = "([A-Z])"
    digits = "([0-9])"
    abbreviations=['Ph.D.','e.g.','i.e.','dr.','mr.','bro.','bro','mrs.','ms.','jr.','sr.','vs.','ph.d.']
    prefixes = "(Mr|St|Mrs|Ms|Dr|Prof|Capt|Cpt|Lt|Mt)[.]"
    suffixes = "(Inc|Ltd|Jr|Sr|Co)"
    starters = "(Mr|Mrs|Ms|Dr|He\s|She\s|It\s|They\s|Their\s|Our\s|We\s|But\s|However\s|That\s|This\s|Wherever)"
    acronyms = "([A-Z][.][A-Z][.](?:[A-Z][.])?)"
    websites = "[.](com|net|org|io|gov|me|edu)"
    text = " " + text + "  "
    text = text.replace("\n"," ")
    text = re.sub(prefixes,"\\1<prd>",text)
    text = re.sub(websites,"<prd>\\1",text)
    for abbrev in abbreviations:
        if abbrev in text:
            mod_abbrev=''
            for ch in list(abbrev):
                if ch!=".":
                    mod_abbrev+=ch
                else:
                    mod_abbrev+="<prd>"
            text = text.replace(abbrev,mod_abbrev)
    text = re.sub(digits + "[.]" + digits,"\\1<prd>\\2",text)
    text = re.sub("\s" + caps + "[.] "," \\1<prd> ",text)
    text = re.sub(acronyms+" "+starters,"\\1<stop> \\2",text)
    text = re.sub(caps + "[.]" + caps + "[.]" + caps + "[.]","\\1<prd>\\2<prd>\\3<prd>",text)
    text = re.sub(caps + "[.]" + caps + "[.]","\\1<prd>\\2<prd>",text)
    text = re.sub(" "+suffixes+"[.] "+starters," \\1<stop> \\2",text)
    text = re.sub(" "+suffixes+"[.]"," \\1<prd>",text)
    text = re.sub(" " + caps + "[.]"," \\1<prd>",text)
    if "”" in text: text = text.replace(".”","”.")
    if "\"" in text: text = text.replace(".\"","\".")
    if "!" in text: text = text.replace("!\"","\"!")
    if "?" in text: text = text.replace("?\"","\"?")
    if "..." in text: text = text.replace("...","<prd><prd><prd>")
    text = text.replace(".",".<stop>")
    text = text.replace("?","?<stop>")
    text = text.replace("!","!<stop>")
    text = text.replace("<prd>",".")
    sentences = text.split("<stop>")
    sentences = sentences[:-1]
    sentences = [s.strip() for s in sentences]
    return sentences

def ml_prob(ngram,counts): ## ngram :- [w1,w2,w3]
    ngram=tuple(ngram)
    if len(ngram)==1:
        unigrams=counts[0]
        if ngram in unigrams:
            prob=unigrams[ngram]/float(sum(unigrams.values()))
        else:
            prob=0
    elif len(ngram)==2:
        unigrams=counts[0]
        bigrams=counts[1]
        w1=ngram[0]
        w2=ngram[1]
        if ngram in bigrams:
            prob = bigrams[ngram]/float(unigrams[ngram[0:1]])
        else:
            prob=0
    else:
        bigrams=counts[1]
        trigrams=counts[2]
        if ngram in trigrams:
            prob=trigrams[ngram]/float(bigrams[ngram[0:2]])
        else:
            prob=0
    return prob

def laplace_smoothing(ngram,counts):
    ngram=tuple(ngram)
    if len(ngram)==1:
        unigrams=counts[0]
        N=sum(unigrams.values())
        V=len(unigrams)
        if ngram in unigrams:
            prob = (unigrams[ngram]+1)/float(N+V)
        else:
            prob = 1/float(N+V)

    elif len(ngram)==2:
        unigrams=counts[0]
        bigrams=counts[1]
        V=len(unigrams)
        if ngram in bigrams:
            prob = (bigrams[ngram]+1)/float(unigrams[ngram[0:1]]+V)
        else:
            if ngram[0:1] in unigrams:
                prob = 1/float(unigrams[ngram[0:1]]+V)
            else:
                prob = 1/float(V)
    else:
        unigrams=counts[0]
        bigrams=counts[1]
        trigrams=counts[2]
        V=len(unigrams)
        if ngram in trigrams:
            prob = (trigrams[ngram]+1)/float(bigrams[ngram[0:2]]+V)
        else:
            if ngram[0:2] in bigrams:
                prob = 1/float(bigrams[ngram[0:2]]+V)
            else:
                prob = 1/float(V)
    return prob

def good_turing(ngram,counts):
    V=sum(counts[0].values())
    ngram=tuple(ngram)
    l=len(ngram)
    ngrams=counts[l-1]
    N=sum(ngrams.values())
    nrs={}
    for k,v in ngrams.items():
        if v not in nrs:
            nrs[v].append(k) ## dictionary of (key=freq of ngram) and (value = list of all ngram tuple whose freq. is key)
    nr_counts={k:len(v) for k,v in nrs.items()}
    if 0 not in nr_counts:
    	nr_counts[0]=V**l - N
    else:
    	nr_counts[0]+=V**l - N
    MAX=sorted(nr_counts.items())[0][1] ##max freq of ngram in corpus
    new_nrs={}
    for r, nr in nr_counts.items():
        if (r+1) in nr_counts:
            new_nr=(r+1)*nr_counts[r+1]/float(N)
        else:
            new_nr=MAX*r**-2/float(N)
        new_nrs[r]=new_nr

    if ngrams[ngram]>5:
        prob=ml_prob(ngram,counts)
    else:
        denominator=(1 - 6*new_nrs[6]/float(new_nrs[1]))/float(N)
        if ngram in ngrams:
            numerator=(ngrams[ngram]+1)*new_nrs[ngrams[ngram]+1]/float(new_nrs[ngrams[ngram]])
            mod_num=num - ngrams[ngram]*6*new_nrs[6]/float(new_nrs[1])
            prob = mod_num/float(denominator)
        else:
            prob = new_nrs[1]/(new_nrs[0]*float(denominator)) ##not considering singleton ngrams as unseen
    return prob

def witten_bell(ngram,counts):
    ngram=tuple(ngram)
    if len(ngram)==2:
        unigrams=counts[0]
        bigrams=counts[1]
        ngram_count=bigrams[ngram]
        prior_count=unigrams[ngram[0:1]]
        type_count=0
        s=0
        for bigram in bigrams:
            if bigram[0]==ngram[0:1]:
                type_count+=1
                s+=bigrams[bigram]
        vocab_size=len(bigrams)
        z = vocab_size - type_count
        if ngram_count==0:
            prob = type_count/float(z*(prior_count+type_count))
        else:
            prob = ngram_count/float(prior_count + type_count)
        wb_lambda=1-bigrams[bigram]/float(bigrams[bigram]+s)
        prob=(wb_lambda)*prob+(1-wb_lambda)*unigrams[ngram[1:2]]/float(sum(unigrams.values()))
    else:
        unigrams=counts[0]
        bigrams=counts[1]
        trigrams=counts[2]
        ngram_count = trigrams[ngram]
        prior_count = bigrams[ngram[0:2]]
        type_count=0
        s=0
        for trigram in trigrams:
            if ngram[0:1]==trigram[0] and ngram[1:2]==trigram[1]:
                type_count+=1
                s+=trigrams[trigram]
        vocab_size = len(trigrams)
        z = vocab_size - type_count
        if ngram_count == 0:
            prob = type_count/float(z*(prior_count + type_count))
        else:
            prob = ngram_count/float(prior_count + type_count)
        wb_lambda=1-trigrams[trigram]/float(trigrams[trigram]+s)
        prob=(wb_lambda)*prob+(1-wb_lambda)*bigrams[ngram[1:3]]/float(unigrams[ngram[1:2]])
    return prob

def stupid_backoff(ngram,counts):
    ngram=tuple(ngram)
    ans=0
    if len(ngram)==2:
        unigrams=counts[0]
        bigrams=counts[1]
        if ngram in bigrams:
            denominator=0
            for bigram in bigrams:
                if ngram[0:1]==bigram[0]:
                    denominator+=bigrams[bigram]
            ans=bigram[ngram]/float(denominator)
        if ans==0:
            if ngram[1:2] in unigrams:
                ans = 0.4 * unigrams[ngram[1:2]]/float(sum(unigrams.values()))
    else:
        unigrams=counts[0]
        bigrams=counts[1]
        trigrams=counts[2]
        if ngram in trigrams:
            denominator=0
            for trigram in trigrams:
                if ngram[0:1]==trigram[0] and ngram[1:2]==trigram[1]:
                    denominator+=trigrams[trigram]
            ans=trigrams[ngram]/float(denominator)
        if ans==0:
            if ngram[1:3] in bigrams:
                denominator=0
                for bigram in bigrams:
                    if ngram[1:2]==bigram[0]:
                        denominator+=bigrams[bigram]
                ans=0.4*bigrams[ngram[1:3]]/float(denominator)
        if ans==0:
            if ngram[2:3] in unigrams:
                ans = 0.16*unigrams[ngram[2:3]]/float(sum(unigrams.values()))
    return ans

def katz_backoff(ngram,counts):
    ngram=tuple(ngram)
    if len(ngram)==2:
        unigrams=counts[0]
        bigrams=counts[1]
        r = bigrams[ngram]
        k=5
        ngrams=bigrams
        N=sum(ngrams.values())
        nrs={}
        for k,v in ngrams.items():
            if v not in nrs:
                nrs[v].append(k) ## dictionary of key as freq of ngram and value as list of all ngram tuple
        nr_counts={k:len(v) for k,v in nrs.items()}
        MAX=sorted(nr_counts.items())[0][1] ##max freq of ngram in corpus
        new_nrs={}
        for r, nr in nr_counts.items():
            if (r+1) in nr_counts:
                new_nr=(r+1)*nr_counts[r+1]/float(N)
            else:
                new_nr=MAX*r**-2/float(N)
            new_nrs[r]=new_nr
        num1 = (r+1)*new_nrs[r+1]/float(r*new_nrs[r])
        num2 = (k+1)*new_nrs[k+1]/float(new_nrs[1])
        deno =  1 - (k+1)*new_nrs[k+1]/float(new_nrs[1])
        dr = (num1 - num2)/float(deno)
        s=0
        for bigram in bigrams:
            if bigram[0]==ngram[0:1]:
                s+=good_turing([bigram[0],bigram[1]],counts)
        numerator=1-s
        s=0
        for unigram in unigrams:
            s+=good_turing([unigram[0]],counts)
        denominator=1-s
        alpha = numerator/float(denominator)
        if r>k:
            prob = good_turing([ngram[0:1][0],ngram[1:2][0]],counts)
        elif r>0 and r<=k:
            prob = dr*good_turing([ngram[0:1][0],ngram[1:2][0]],counts)
        else:
            prob = alpha*good_turing([ngram[1:2][0]],counts)
    else:
        unigrams=counts[0]
        bigrams=counts[1]
        trigrams=counts[2]
        r1 = trigrams[ngram]
        r2= bigrams[ngram[1:3]]
        s1=0
        for trigram in trigrams:
            if trigram[0]==ngram[0:1] and trigram[1]==ngram[1:2]:
                s1+=good_turing([trigram[0],trigram[1],trigram[2]],counts)
        num=1-s1
        s1=0
        for bigram in bigrams:
            if bigram[0]==ngram[1:2]:
                s1+=good_turing([bigram[0],bigram[1]],counts)
        deno=1-s1
        alpha1=num/float(deno)
        s2=0
        for bigram in bigrams:
            if bigram[0]==ngram[1:2]:
                s+=good_turing([bigram[0],bigram[1]],counts)
        numerator=1-s
        s2=0
        for unigram in unigrams:
            s2+=good_turing([unigram[0]],counts)
        denominator=1-s
        alpha2 = numerator/float(denominator)
        if r1>0:
            prob=good_turing([ngram[0:1][0],ngram[1:2][0],ngram[2:3][0]],counts)
        elif r1==0 and r2>0:
            prob=alpha1*good_turing([ngram[1:2][0],ngram[2:3][0]],counts)
        else:
            prob=alpha2*good_turing([ngram[2:3][0]],counts)
    return prob

def find_ngrams(text, n):
    return Counter(zip(*[text[i:] for i in range(n)]))

def get_ngrams(n,tokens,level="word"):
    if level=="character":
        temp=find_ngrams(list(tokens[0]),n)
        for x in range(1,len(tokens)):
            temp+=find_ngrams(list(tokens[x]),n)
    else:
        temp = find_ngrams(tokens,n)
    ngrams=temp
    return ngrams

def language_model(model_name,counts,n):
    ##model_name :- "lap","gt","wb", "sbo", "kbo" [laplace/add-1,good-turing,witten_bell,stupid_backoff,katz-backoff]
    unigrams=counts[0]
    bigrams=counts[1]
    trigrams=counts[2]
    unigram_prob=defaultdict(float)
    bigram_prob=defaultdict(float)
    trigram_prob=defaultdict(float)
    if n==1:
        for unigram in unigrams:
            if model_name=="lap":
                unigram_prob[unigram]=laplace_smoothing([unigram[0]],counts)
            elif model_name=="gt":
                unigram_prob[unigram]=good_turing([unigram[0]],counts)
        return unigram_prob

    elif n==2:
        for bigram in bigrams:
            if model_name=="lap":
                bigram_prob[bigram]=laplace_smoothing([bigram[0],bigram[1]],counts)
            elif model_name=="gt":
                bigram_prob[bigram]=good_turing([bigram[0],bigram[1]],counts)
            elif model_name=="wb":
                bigram_prob[bigram]=witten_bell([bigram[0],bigram[1]],counts)
            elif model_name=="sbo":
                bigram_prob[bigram]=stupid_backoff([bigram[0],bigram[1]],counts)
            else:
                bigram_prob[bigram]=katz_backoff([bigram[0],bigram[1]],counts)
        return bigram_prob

    else:
        for trigram in trigrams:
            if model_name=="lap":
                trigram_prob[trigram]=laplace_smoothing([trigram[0],trigram[1],trigram[2]],counts)
            elif model_name=="gt":
                trigram_prob[trigram]=good_turing([trigram[0],trigram[1],trigram[2]],counts)
            elif model_name=="wb":
                trigram_prob[trigram]=witten_bell([trigram[0],trigram[1],trigram[2]],counts)
            elif model_name=="sbo":
                trigram_prob[trigram]=stupid_backoff([trigram[0],trigram[1],trigram[2]],counts)
            else:
                trigram_prob[trigram]=katz_backoff([trigram[0],trigram[1],trigram[2]],counts)
        return trigram_prob


def complete_ngram(ngram,counts,smoothing_method): ##ngram = [w1,w2] or [w1]
# Given (N-1) gram, and the value 'N', print the possibilities that complete the n-gram
# and plot them in decresing order of frequency
    next_gram=[]
    unigrams=counts[0]
    for word in unigrams:
        if len(ngram)==2:
            tokens=[ngram[0],ngram[1],word[0]]
        else:
            tokens=[ngram[0],word[0]]
        if smoothing_method=="lap":
            prob = laplace_smoothing(tokens,counts)
        elif smoothing_method=="gt":
            prob = good_turing(tokens,counts)
        elif smoothing_method=="wb":
            prob = witten_bell(tokens,counts)
        elif smoothing_method=="sbo":
            prob=stupid_backoff(tokens,counts)
        else:
            prob=katz_backoff(token,counts)
        next_gram.append((prob,word[0]))
    next_gram.sort(key=lambda tup: tup[0], reverse = True)
    return next_gram[0][1]

def edit_distance(word,letters):
    #letters    = 'abcdefghijklmnopqrstuvwxyz' ##add additional characters
    splits     = [(word[:i], word[i:])    for i in range(len(word) + 1)]
    deletes    = [L + R[1:]               for L, R in splits if R]
    transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R)>1]
    replaces   = [L + c + R[1:]           for L, R in splits if R for c in letters]
    inserts    = [L + c + R               for L, R in splits for c in letters]
    return set(deletes + transposes + replaces + inserts)

def find_candidates(word,unique_tokens):
    ##Fusion words can also be computed in order to enhance the list
    letters=[]
    for token in unique_tokens:
        for ch in list(token):
            if ch not in letters:
                letters.append(ch)
    letters="".join(str(ch) for ch in letters)
    c4=set(word)
    if word in unique_tokens:
        c1=set(word)
    edit1=edit_distance(word,letters)
    tmp1=[]
    tmp2=[]
    for w1 in edit1:
        for w2 in edit_distance(w1,letters):
            if w2 in unique_tokens:
                tmp2.append(w2)
        if w1 in unique_tokens:
            tmp1.append(w1)
    c2=set(tmp1)
    c3=set(temp2)
    return list(c1|c2|c3|c4)

def find_unique_tokens(unique_tokens,sentence_tokens):
    for token in sentence_tokens:
        if token!="\0":
            if token not in unique_tokens:
                unique_tokens.append(token)
    return unique_tokens

def spell_checker(word,n,model_name,counts):
# Given a word check if the spelling is correct using your Language model
# Feel free to add more helper functions for this task
# Giving candidate word with maximum probability using character L.M
# word level L.M is not computed and comapred against character level which is left for future task
    unique_tokens=[]
    script_path=os.path.dirname(os.path.abspath(__file__))
    fl = open(script_path+"/saved_items/unique_tokens.pickle",'rb')
    unique_tokens = pickle.load(fl)
    fl.close()
    candidates=find_candidates(word,unique_tokens)
    cand_probs=[]
    for cand in candidates:
        cand_ch_grams=[cand[i:i+n] for i in range(len(cand)-n+1)]
        if n==1:
            prod=1
            for unigram in cand_ch_grams:
                if model_name=="lap":
                    prod*=laplace_smoothing([unigram],counts)
                elif model_name=="gt":
                    prod*=good_turing([unigram],counts)
        elif n==2:
            prod=1
            first_ch = list(cand_ch_grams[0])[0]
            prod*=good_turing([first_ch],counts)
            for bigram in cand_ch_grams:
                if model_name=="lap":
                    prod*=laplace_smoothing([list(bigram)[0],list(bigram)[1]],counts)
                elif model_name=="gt":
                    prod*=good_turing([list(bigram)[0],list(bigram)[1]],counts)
                elif model_name=="wb":
                    prod*=witten_bell([list(bigram)[0],list(bigram)[1]],counts)
                elif model_name=="sbo":
                    prod*=stupid_backoff([list(bigram)[0],list(bigram)[1]],counts)
                else:
                    prod*=katz_backoff([list(bigram)[0],list(bigram)[1]],counts)

        else:
            prod=1
            first_ch=list(cand_ch_grams[0])[0]
            second_ch=list(cand_ch_grams[0])[1]
            prod*=good_turing([first_ch],counts)
            prod*=good_turing([first_ch,second_ch],counts)
            for trigram in cand_ch_grams:
                if model_name=="lap":
                    prod*=laplace_smoothing([list(trigram)[0],list(trigram)[1],list(trigram)[2]],counts)
                elif model_name=="gt":
                    prod*=good_turing([list(trigram)[0],list(trigram)[1],list(trigram)[2]],counts)
                elif model_name=="wb":
                    prod*=witten_bell([list(trigram)[0],list(trigram)[1],list(trigram)[2]],counts)
                elif model_name=="sbo":
                    prod*=stupid_backoff([list(trigram)[0],list(trigram)[1],list(trigram)[2]],counts)
                else:
                    prod*=katz_backoff([list(trigram)[0],list(trigram)[1],list(trigram)[2]],counts)

        cand_probs.append((prob,cand))
    cand_probs.sort(key=lambda tup: tup[0], reverse = True)
    return cand_probs[0][1]


def score_grammaticality(sentence,n,model_name,counts):
  # Given a sentence, Build a model from the data which can give a score of grammaticality.
  # More grammatical the sentence, better the score
  # Feel free to add helper functions
    n_grams = list(zip(*[sentence.split()[i:] for i in range(n)]))
    if n==1:
        prod=1
        for unigram in n_grams:
            if model_name=="lap":
                prod*=laplace_smoothing([unigram[0]],counts)
            elif model_name=="gt":
                prod*=good_turing([unigram[0]],counts)
    elif n==2:
        prod=1
        first_word = n_grams[0][0]
        prod*=good_turing([first_word],counts)
        for bigram in n_grams:
            if model_name=="lap":
                prod*=laplace_smoothing([bigram[0],bigram[1]],counts)
            elif model_name=="gt":
                prod*=good_turing([bigram[0],bigram[1]],counts)
            elif model_name=="wb":
                prod*=witten_bell([bigram[0],bigram[1]],counts)
            elif model_name=="sbo":
                prod*=stupid_backoff([bigram[0],bigram[1]],counts)
            else:
                prod*=katz_backoff([bigram[0],bigram[1]],counts)

    else:
        prod=1
        first_word= n_grams[0][0]
        second_word=ngrams[0][1]
        prod*=good_turing([first_word],counts)
        prod*=good_turing([first_word,second_word],counts)
        for trigram in n_grams:
            if model_name=="lap":
                prod*=laplace_smoothing([trigram[0],trigram[1],trigram[2]],counts)
            elif model_name=="gt":
                prod*=good_turing([trigram[0],trigram[1],trigram[2]],counts)
            elif model_name=="wb":
                prod*=witten_bell([trigram[0],trigram[1],trigram[2]],counts)
            elif model_name=="sbo":
                prod*=stupid_backoff([trigram[0],trigram[1],trigram[2]],counts)
            else:
                prod*=katz_backoff([trigram[0],trigram[1],trigram[2]],counts)

    return prod

def main():
    script_path=os.path.dirname(os.path.abspath(__file__))
    main_data=script_path+"/Gutenberg/txt/"
    sample_data=script_path+"/Gutenberg/sample_text/"
    unique_tokens_file=script_path+"/saved_items/unique_tokens.txt"
    corpus=''
    all_tokens_len_path=script_path+"/saved_items/token_lengths/"
    all_tokens_path=script_path+"/saved_items/tokens/"
    #data_path=sample_data
    '''data_path=main_data
    total=0
    for file in sorted(os.listdir(data_path)):
        if file=="Henry David Thoreau___A Week on the Concord and Merrimack Rivers.txt":
            continue
        tmp=''
        flag=0
        with open(data_path+file,'r') as fl:
            for line in fl:
                if line!="\n" and line!="":
                    flag=1
                    tmp+=line.rstrip().lstrip()
        fl.close()
        if flag!=0:
            total+=1
            corpus+=tmp+"\n"
        #break
    print("Total documents:- ",total)

    cnt=0
    for doc in corpus.split("\n"):
        if doc!='' and doc!="\n":
            fd=open(all_tokens_path+str(cnt)+".txt",'w')
            for sentence in find_sentences(doc):
                for token in tokenize(sentence):
                    fd.write(token+"\n")
                fd.write("\0\n")
            fd.close()
            cnt+=1
    for doc in sorted(os.listdir(all_tokens_path)):
        cnt=0
        fd=open(all_tokens_len_path+doc,'w')
        sentence_token_len=[]
        with open(all_tokens_path+doc,'r') as fl:
            for line in fl:
                if line.split()[0]!="\x00":
                    cnt+=1
                else:
                    sentence_token_len.append(cnt)
                    cnt=0
        fl.close()
        for len in sentence_token_len:
            fd.write(str(len)+"\n")
        fd.close()'''
    word_uni=Counter()
    word_bi=Counter()
    word_tri=Counter()
    ch_uni=Counter()
    ch_bi=Counter()
    ch_tri=Counter()
    unique_tokens=[]
    for doc in sorted(os.listdir(all_tokens_len_path)):
        tokens_len=[]
        with open(all_tokens_len_path+doc,'r') as fl:
            for line in fl:
                tokens_len.append(int(line.split()[0]))
        fl.close()
        index=0
        index1=0
        temp=np.empty(tokens_len[index1],dtype="object")
        with open(all_tokens_path+doc,'r') as fl:
            for line in fl:
                if line.split()[0]!="\x00":
                    temp[index]=line.split()[0].lower()
                    index+=1
                else:
                    if index!=tokens_len[index1]:
                        print("lol")
                    index=0
                    index1+=1
                    unique_tokens=find_unique_tokens(unique_tokens,temp)
                    word_uni+=get_ngrams(1,temp,level="word")
                    word_bi+=get_ngrams(2,temp,level="word")
                    word_tri+=get_ngrams(3,temp,level="word")
                    '''ch_uni+=get_ngrams(1,temp,level="character")
                    ch_bi+=get_ngrams(2,temp,level="character")
                    ch_tri+=get_ngrams(3,temp,level="character")'''
                    if index1<len(tokens_len):
                        temp=np.empty(tokens_len[index1],dtype="object")
        fl.close()
    with open(script_path+"/saved_items/word_uni.pickle",'wb') as fl:
        pickle.dump(word_uni,fl)
    fl.close()
    with open(script_path+"/saved_items/word_bi.pickle",'wb') as fl:
        pickle.dump(word_bi,fl)
    fl.close()
    with open(script_path+"/saved_items/word_tri.pickle",'wb') as fl:
        pickle.dump(word_tri,fl)
    fl.close()
    '''with open(script_path+"/saved_items/ch_uni.pickle",'wb') as fl:
        pickle.dump(ch_uni,fl)
    fl.close()
    with open(script_path+"/saved_items/ch_bi.pickle",'wb') as fl:
        pickle.dump(ch_bi,fl)
    fl.close()
    with open(script_path+"/saved_items/ch_tri.pickle",'wb') as fl:
        pickle.dump(ch_tri,fl)
    fl.close()
    '''
    with open(script_path+"/saved_items/unique_tokens.pickle",'wb') as fl:
        pickle.dump(unique_tokens,fl)
    fl.close()
    '''fl = open(script_path+"/saved_items/word_uni.pickle",'rb')
    word_uni = pickle.load(fl)
    fl.close()
    fl = open(script_path+"/saved_items/word_bi.pickle",'rb')
    word_bi = pickle.load(fl)
    fl.close()
    fl = open(script_path+"/saved_items/word_tri.pickle",'rb')
    word_tri = pickle.load(fl)
    fl.close()
    '''
    '''fl = open(script_path+"/saved_items/ch_uni.pickle",'rb')
    ch_uni = pickle.load(fl)
    fl.close()
    fl = open(script_path+"/saved_items/ch_bi.pickle",'rb')
    ch_bi = pickle.load(fl)
    fl.close()
    fl = open(script_path+"/saved_items/ch_tri.pickle",'rb')
    ch_tri = pickle.load(fl)
    fl.close()'''

    print("done")

    '''
    input_word="cad"
    model_name="gt"
    n=3
    counts=[ch_uni,ch_bi,ch_tri]
    correct_word=spell_checker(input_word,n,model_name,counts)
    print(correct_word)
    sentence1='I have a red apple'
    sentence2='apple a have I red'
    sentence=sentence1
    n=3
    model_name="gt"
    counts=[word_uni,word_bi,word_tri]
    score = score_grammaticality(sentence,n,model_name,counts)
    print(score)
    '''
main()
