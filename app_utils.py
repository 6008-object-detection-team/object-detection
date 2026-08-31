"""Small helpers shared by the application modules."""


def parse_prompts(text):
    """Turn comma, semicolon, or newline separated text into unique prompts."""
    separators = str.maketrans({"，": ",", "；": ",", ";": ",", "\n": ","})
    prompts = [item.strip() for item in text.translate(separators).split(",")]
    return list(dict.fromkeys(item for item in prompts if item))
