import argparse


def parse_scalar(s):
    if not isinstance(s, str):
        return s
    t = s.strip()
    l = t.lower()
    if l == "true":
        return True
    if l == "false":
        return False
    if l in ("none", "null"):
        return None
    try:
        return int(t, 10)
    except Exception:
        pass
    try:
        return float(t)
    except Exception:
        return s


def fn_kv_csv(s: str) -> dict[str, dict[str, object]]:
    """
    Parse a string of comma-separated triplets: fn:key:value

    Returns:
        dict[fn_name] -> dict[key] = parsed_value

    Example:
        "fn1:width:1920,fn1:height:1080,fn2:quality:0.8"
        -> {"fn1": {"width": 1920, "height": 1080}, "fn2": {"quality": 0.8}}
    """
    result: dict[str, dict[str, object]] = {}
    if not s:
        return result

    for item in s.split(","):
        if not item:
            continue
        parts = item.split(":", 2)  # allow value to contain ":" beyond first two separators
        if len(parts) < 3:
            raise argparse.ArgumentTypeError(f"Bad item '{item}', expected FN:KEY:VALUE")
        fn, key, raw_val = parts[0], parts[1], parts[2]
        # If you need to allow colons in values, join leftover parts:
        # fn, key, raw_val = parts[0], parts[1], ":".join(parts[2:])

        if not fn:
            raise argparse.ArgumentTypeError(f"Bad item '{item}': empty function name")
        if not key:
            raise argparse.ArgumentTypeError(f"Bad item '{item}': empty key")

        val = parse_scalar(raw_val)
        bucket = result.setdefault(fn, {})
        bucket[key] = val
    return result
