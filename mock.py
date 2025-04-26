def mock_generate_response(user_input, model_name, temperature, top_p, top_k):
    return f'''{model_name}\n{user_input}\ntemperature: {temperature}\ntop_p: {top_p}\ntop_k: {top_k}'''

