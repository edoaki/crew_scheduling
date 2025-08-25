# utils/time.py

def hhmm_label(m: int) -> str:
    m = int(m)
    h = m // 60
    mi = m % 60
    return f"{h:02d}:{mi:02d}"
