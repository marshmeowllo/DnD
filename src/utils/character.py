import random

from langchain_core.documents import Document

from config import CHARACTER_STARTING_STATS, DEFAULT_LEVEL

def roll_ability_scores():
    stats = CHARACTER_STARTING_STATS
    random.shuffle(stats)
    return stats

def create_character_doc(player, name, race, char_class, background, stats):
    content = (
        f"Player: {player}\n"
        f"Name: {name}\n"
        f"Race: {race}\n"
        f"Class: {char_class}\n"
        f"Background: {background}\n"
        f"Stats: {stats}\n"
        f"Level: {DEFAULT_LEVEL}"
    )
    return Document(page_content=content, metadata={"player": player, "name": name})