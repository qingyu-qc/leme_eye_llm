import sys
import json
import evaluate
from BARTScore.bart_score import BARTScorer

rouge = evaluate.load("rouge")
bertscore = evaluate.load("evaluate-metric/bertscore")
bart_scorer = BARTScorer(device='mps', checkpoint="facebook/bart-large-cnn")
bart_scorer.load(path="bart_score.pth")

def process_json(json_file):
    with open(json_file, 'r') as file:
        data = json.load(file)
        golds = []
        preds = []
        for i in range(len(data)):
            golds.append(data[i]['truth'])
            preds.append(data[i]['logit_0'])
    return golds, preds

def compute_all_scores(json_file):
    # Process the JSON file to get golds and preds
    golds, preds = process_json(json_file)
    
    # Compute ROUGE scores
    rouge_results = rouge.compute(predictions=preds, references=golds)
    rouge1 = round(rouge_results["rouge1"], 4)
    rouge2 = round(rouge_results["rouge2"], 4)
    rougeL = round(rouge_results["rougeL"], 4)
    
    # Compute BERT scores
    bert_results = bertscore.compute(predictions=preds, references=golds, model_type="bert-base-multilingual-cased")
    bert_score_f1 = round(sum(bert_results["f1"]) / len(bert_results["f1"]), 4)
    
    # Compute BART scores
    bart_results = bart_scorer.score(srcs=list(preds), tgts=list(golds), batch_size=8)
    bart_score = round(sum(bart_results) / len(bart_results), 4)
    
    return {
        "rouge1": rouge1,
        "rouge2": rouge2,
        "rougeL": rougeL,
        "bert_score_f1": bert_score_f1,
        "bart_score": bart_score
    }

if __name__ == "__main__":
    json_file = sys.argv[1]
    scores = compute_all_scores(json_file)
    print(scores)

