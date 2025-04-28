import json

def save_feedback(model_name, index, session_state):
    session_state[f"history_{model_name}"][index]["feedback"] = session_state[f"feedback_{model_name}_{index}"]

def log_edit_response(prompt, original, edited, model):
    out = {
        "prompt": prompt,
        "original": original,
        "edited": edited,
        "model": model
    }
    with open("chat_edit_response.jsonl", "a") as f:
        f.write(json.dumps(out) + "\n")