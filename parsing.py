# --- Парсинг данных ---
def parse_input_data(input_str):
    """Разбираем строку, разделённую запятыми, в список чисел"""
    try:
        return [float(x.strip()) for x in input_str.split(',') if x.strip() and x.strip().lower() not in ['nan', '']]
    except:
        return []