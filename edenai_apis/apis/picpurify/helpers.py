def content_processing(confidence):
    if confidence < 0.2:
        return 1
    elif confidence < 0.4:
        return 2
    elif confidence < 0.6:
        return 3
    elif confidence < 0.8:
        return 4
    elif confidence > 0.8:
        return 5
    else:
        return 0
