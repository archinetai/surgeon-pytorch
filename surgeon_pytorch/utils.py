from typing import Dict, List, Optional, Sequence, Union


def exists(val) -> bool:
    return val is not None


def to_list(value: Optional[Union[str, Sequence[str], Dict[str, str]]]):
    items: List[str] = []
    if isinstance(value, str):
        items = [value]
    elif isinstance(value, list) or isinstance(value, tuple):
        items = list(value)
    elif isinstance(value, dict):
        items = list(value.keys())
    return items
