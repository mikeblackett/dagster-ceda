import re
from collections.abc import Sequence
from pathlib import Path
from re import Pattern
from typing import Annotated, Final, TypeAlias

import dagster as dg
import upath as up

USER_STRING_DELIMITER: Final[str] = "/"


def _resolve_pattern(
    context: dg.ResolutionContext, raw_pattern: str
) -> Pattern:
    value = context.resolve_value(raw_pattern, as_type=str)
    return re.compile(value)


ResolvedPattern: TypeAlias = Annotated[
    Pattern,
    dg.Resolver(_resolve_pattern, model_field_type=str),
]


def _resolve_path(context: dg.ResolutionContext, raw_path: str) -> Path:
    value = context.resolve_value(raw_path, as_type=str)
    return Path(value)


ResolvedPath: TypeAlias = Annotated[
    Path,
    dg.Resolver(_resolve_path, model_field_type=str),
]


def _resolve_upath(context: dg.ResolutionContext, raw_path: str) -> up.UPath:
    value = context.resolve_value(raw_path, as_type=str)
    return up.UPath(value)


ResolvedUPath: TypeAlias = Annotated[
    up.UPath,
    dg.Resolver(_resolve_upath, model_field_type=str),
]


def _resolve_key_prefix(
    context: dg.ResolutionContext, raw_prefix: str | Sequence[str]
) -> list[str]:
    if isinstance(raw_prefix, str):
        value = context.resolve_value(raw_prefix, as_type=str)
        if USER_STRING_DELIMITER in value:
            return value.split(USER_STRING_DELIMITER)
        return [value]
    value = context.resolve_value(raw_prefix, as_type=Sequence[str])
    return list(value)


ResolvedKeyPrefix: TypeAlias = Annotated[
    str | Sequence[str],
    dg.Resolver(_resolve_key_prefix, model_field_type=str | Sequence[str]),
]
