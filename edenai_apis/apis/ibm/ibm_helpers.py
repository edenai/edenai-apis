def generate_right_ssml_text(text, speaking_rate, speaking_pitch):
    attribs = {
        "rate": speaking_rate,
        "pitch": speaking_pitch
    }
    cleaned_attribs_string = ""
    for k,v in attribs.items():
        if not v:
            continue
        cleaned_attribs_string = f"{cleaned_attribs_string} {k}='{v}%'"
    if not cleaned_attribs_string.strip():
        return text
    smll_text = f"<speak><prosody {cleaned_attribs_string}>{text}</prosody></speak>"
    return smll_text