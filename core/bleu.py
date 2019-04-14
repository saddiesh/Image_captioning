import pickle as pickle
import os
import sys
sys.path.append('../coco-caption')
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.meteor.meteor import Meteor

def score(ref, hypo):
    scorers = [
        (Bleu(4),["Bleu_1","Bleu_2","Bleu_3","Bleu_4"]),
        (Rouge(),"ROUGE_L"),
        (Cider(),"CIDEr"),
    ]
    final_scores = {}
    for scorer,method in scorers:
        score,scores = scorer.compute_score(ref,hypo)
        print(method, score)
        if type(score)==list:
            for m,s in zip(method,score):
                final_scores[m] = s
        else:
            final_scores[method] = score
        print(final_scores)
    return final_scores
    

def evaluate(ref_path, cand_path, get_scores=False):

    # load caption data
    with open(ref_path, 'rb') as f:
        reff = pickle.load(f)

    # make ref dict:
    ref = {}
    for i, caption in enumerate(reff):
        ref[i] = [caption]

    with open(cand_path, 'rb') as f:
        cand = pickle.load(f)
    
    # make dictionary
    hypo = {}
    for i, caption in enumerate(cand):
        hypo[i] = [caption]
    
    # compute bleu score
    final_scores = score(ref, hypo)

    # print out scores
    print ('Bleu_1:\t',final_scores['Bleu_1'])
    print ('Bleu_2:\t',final_scores['Bleu_2'])
    print ('Bleu_3:\t',final_scores['Bleu_3'])
    print ('Bleu_4:\t',final_scores['Bleu_4'])
    #print ('METEOR:\t',final_scores['METEOR'])
    print ('ROUGE_L:',final_scores['ROUGE_L'])
    print ('CIDEr:\t',final_scores['CIDEr'])
    
    if get_scores:
        return final_scores
    
   
    
    
    
    
    
    
    
    
    
    


