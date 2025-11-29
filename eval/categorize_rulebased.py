#!/usr/bin/env python3
import json, argparse, re, os

RANDOM_KW = ["random", "noise", "no discernible", "no discernible structure", "random points", "noisy", "disorganized", "scattered"]
SINE_KW   = ["sine", "sinus", "gradual", "smooth", "sinusoidal", "oscillat", "discernible structure"]
SQUARE_KW = ["sharp", "corners", "corner", "abrupt", "discontinu", "not continuous", "jumps", "step", "flat segment"]

MAP = {
    "Random": ("A", "Random noise"),
    "Sine": ("B", "Sine wave"),
    "Square": ("C", "Square wave"),
}

def score_text(text):
    t = text.lower()
    s = {"Random":0, "Sine":0, "Square":0}
    for kw in RANDOM_KW:
        if kw in t:
            s["Random"] += t.count(kw)
    for kw in SINE_KW:
        if kw in t:
            s["Sine"] += t.count(kw)
    for kw in SQUARE_KW:
        if kw in t:
            s["Square"] += t.count(kw)
    return s

def decide(conversation):
    # conversation: list of {"question":.., "answer":..}
    all_text = " ".join((item.get("answer","") for item in conversation))
    # if answer entries are nested strings, flatten
    s = score_text(all_text)
    # If everything zero, try using question/other fields
    if sum(s.values()) == 0:
        all_text2 = " ".join(( (item.get("question","") + " " + item.get("answer","")) for item in conversation ))
        s = score_text(all_text2)
    best = max(s, key=lambda k: s[k])
    # Build reasoning: list which keyword families matched
    reasons = []
    for k, kwl in (("Random", RANDOM_KW), ("Sine", SINE_KW), ("Square", SQUARE_KW)):
        found = [kw for kw in kwl if kw in all_text.lower()]
        if found:
            reasons.append(f"{k}: matched {', '.join(found[:4])}")
    if not reasons:
        reasons.append("No strong keyword matches; defaulting to highest score.")
    letter, label = MAP[best]
    resp = f"{letter}) {label}\n\nReasoning: " + "; ".join(reasons)
    return resp

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="input file (json produced by evaluateLLaVA.py)")
    ap.add_argument("--output", required=True, help="output categorized json")
    args = ap.parse_args()

    data = json.load(open(args.input))
    out = []
    for entry in data:
        conv = entry.get("conversation", [])
        conv_id = entry.get("conversationId") or entry.get("image") or entry.get("id") or ""
        resp = decide(conv)
        out.append({"conversationId": conv_id, "response": resp})
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    json.dump(out, open(args.output, "w"), indent=2)
    print("Wrote", args.output)

if __name__ == "__main__":
    main()
