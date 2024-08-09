import numpy as np
import copy
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction

def get_bleu_score(reference, hypothesis):
    ngram = 4
    weight = tuple((1. / ngram for _ in range(ngram)))
    return sentence_bleu([reference], hypothesis, weight, smoothing_function=SmoothingFunction().method1)

def calculate_selfBleu(sentences):
    bleu_scores = []

    for i, ref_sentence in enumerate(sentences):
        hypothesis_sentences = copy.deepcopy(sentences)
        hypothesis_sentences.pop(i)  
        
        if len(hypothesis_sentences) > 0:
            bleu_sum = sum(get_bleu_score(ref_sentence, hyp_sentence) for hyp_sentence in hypothesis_sentences)
            mean_bleu = bleu_sum / len(hypothesis_sentences)
            bleu_scores.append(mean_bleu)
        else:
            bleu_scores.append(0.0)

    return np.mean(bleu_scores)


