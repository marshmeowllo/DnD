import json
import csv

def save_feedback(prompt, res_a, res_b, vote):
    file_name = 'model_votes.csv'

    try:
        with open(file_name, 'r') as f:
            file_exists = True
    except FileNotFoundError:
        file_exists = False

    entry = {"prompt": prompt, "res_vanilla": res_a, "res_trained": res_b, "vote": "vanilla" if vote=='Model A' else "trained"}
    with open(file_name, 'a', newline='', encoding='utf-8') as f:
        fields = list(entry.keys())
        writer = csv.DictWriter(f, fieldnames=fields)
        if not file_exists:
            writer.writeheader()
        writer.writerow(entry)

def log_edit_response(prompt, original, edited, model):
    out = {
        "prompt": prompt,
        "original": original,
        "edited": edited,
        "model": model
    }
    with open("chat_edit_response.jsonl", "a") as f:
        f.write(json.dumps(out) + "\n")