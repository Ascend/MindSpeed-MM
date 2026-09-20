import re

def compile_extended_pattern(pattern: str):
    """
    Convert extended pattern (e.g., "*.layer.{0-4}") into a regex and range specs.
    Returns: (compiled_regex, specs)
      - specs: list of either (low, high) or None (for {*})
    """
    specs = []
    placeholder = "__NUM__"

    # Replace {*}, {a-b} with placeholder and record spec
    def replace_brace(match):
        inner = match.group(1)
        if inner == '*':
            specs.append(None)  # no range
            return placeholder
        elif '-' in inner:
            parts = inner.split('-')
            if len(parts) != 2:
                raise ValueError(f"Invalid brace pattern: {inner}")
            try:
                low = int(parts[0])
                high = int(parts[1])
            except ValueError as e:
                raise ValueError(f"Non-integer in range: {inner}") from e
            if low > high:
                raise ValueError(f"Invalid range: {inner} (low > high)")
            specs.append((low, high))
            return placeholder
        else:
            raise ValueError(f"Unrecognized brace pattern: {inner}")

    # Match {...} but avoid matching escaped or invalid braces
    temp_pattern = re.sub(r'\{([^}]*)\}', replace_brace, pattern)

    # Convert fnmatch wildcards to regex manually
    regex_parts = []
    i = 0
    while i < len(temp_pattern):
        if temp_pattern.startswith(placeholder, i):
            regex_parts.append(r'(\d+)')  # capture digits
            i += len(placeholder)
        else:
            c = temp_pattern[i]
            if c == '*':
                regex_parts.append(r'.*')
            elif c == '?':
                regex_parts.append(r'.')
            elif c == '.':
                regex_parts.append(r'\.')
            else:
                regex_parts.append(re.escape(c))
            i += 1

    full_regex = '^' + ''.join(regex_parts) + '$'
    return re.compile(full_regex), specs


def module_name_match(pattern: str, string):
    """
    Match a string against an extended pattern.
    """
    regex, specs = compile_extended_pattern(pattern)
    m = regex.match(string)
    if not m:
        return False

    groups = m.groups()
    for num_str, spec in zip(groups, specs):
        try:
            num = int(num_str)
        except ValueError:
            return False

        # If a range is specified, check bounds
        if spec is not None:
            low, high = spec
            if not (low <= num <= high):
                return False
    return True
