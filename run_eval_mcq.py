import sys
import json
from sklearn.metrics import accuracy_score, f1_score

# This is to preprocess answers to obatin the first letter as the option. For example, "A. This is explanation ..." would be converted to a
def simple_match(pred):
    if len(pred) > 0:
        pred = pred.lstrip().lower()
        return pred[0]
    else:
        return 'missing'
    

def extract_first_letter(answer):
    """
    Extracts the first letter of a preprocessed answer to identify the chosen option.

    Example:
    For an input "A. This is because ... ", the function returns 'a'.
    """
    # Check if the answer is not empty
    if answer:
        # Remove leading spaces and convert to lowercase
        cleaned_answer = answer.lstrip().lower()
        # Return the first character of the cleaned answer
        return cleaned_answer[0]
    else:
        # Return 'missing' if the answer is empty
        return 'missing'


def extract_last_letter(answer):
    """
    Extracts the last letter of a preprocessed answer to identify the chosen option.
    
    Example:
    For an input "This is because ... The correct answer is A", the function returns 'a'.
    """
    # Check if the answer is not empty
    if answer:
        # Remove leading spaces and convert to lowercase
        cleaned_answer = answer.lstrip().lower()
        # Return the last character of the cleaned answer
        return cleaned_answer[-1]
    else:
        # Return 'missing' if the answer is empty
        return 'missing'


def convert_json_to_preds_gold(f, extraction_function):
    with open(f, 'r') as file:
        data = json.load(file)
    golds = [item['truth'].lower() for item in data]
    preds = [extraction_function(item['logit_0']) for item in data]
    return golds, preds


def compute_scores(golds, preds):
    accuracy = accuracy_score(golds, preds)
    weighted_f1 = f1_score(golds, preds, average='weighted')
    macro_f1 = f1_score(golds, preds, average='macro')
    return round(accuracy, 4), round(weighted_f1, 4), round(macro_f1, 4)


if __name__ == "__main__":
    
    json_file = sys.argv[1]
    extraction_method = sys.argv[2]

    if extraction_method == "extract_first_letter":
        extraction_function = extract_first_letter
    elif extraction_method == "extract_last_letter":
        extraction_function = extract_last_letter
    else:
        print("Invalid extraction method. Use 'extract_first_letter' or 'extract_last_letter'.")
        sys.exit(1)
    
    golds, preds = convert_json_to_preds_gold(json_file, extraction_function)
    scores = compute_scores(golds, preds)
    
    print("Accuracy:", scores[0])
    print("Weighted F1 Score:", scores[1])
    print("Macro F1 Score:", scores[2])