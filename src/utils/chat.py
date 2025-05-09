import time

from config import CHAT_STREAM_DELAY

def chat_stream(response):
    for char in response:
        yield char
        time.sleep(CHAT_STREAM_DELAY)