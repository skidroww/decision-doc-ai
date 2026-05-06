def clean_duplicate_text(text, similarity_threshold=0.9, min_length=15, window_size=10):
    
    lines = text.split('\n')
    cleaned_lines = []
    
    recent_lines = []

    for line in lines:
        line_stripped = line.strip()
        
        if not line_stripped:
            continue

        if len(line_stripped) < min_length:
            cleaned_lines.append(line_stripped)
            continue

        is_duplicate = False
        for recent in recent_lines:
            similarity = difflib.SequenceMatcher(None, recent, line_stripped).ratio()
            if similarity >= similarity_threshold:
                is_duplicate = True
                break
        
        if not is_duplicate: 
            cleaned_lines.append(line_stripped)
            recent_lines.append(line_stripped)
            
            if len(recent_lines) > window_size:
                recent_lines.pop(0)

    return '\n'.join(cleaned_lines)