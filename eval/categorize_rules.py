import re

def categorize(conversation):
    text = " ".join(a["answer"].lower() for a in conversation)
    text = re.sub(r"<s>|</s>", "", text)

    # Rule 1: Square wave
    if "sharp corner" in text or "abrupt" in text or "discontinuous" in text:
        return "C) Square wave"

    # Rule 2: Random noise
    if "random" in text or "unpredictable" in text or "stochastic" in text or "disordered" in text:
        return "A) Random noise"

    # Rule 3: Sine wave (softer check)
    if "continuous" in text or "smooth" in text:
        return "B) Sine wave"

    # Default → assume sine wave
    return "B) Sine wave"

