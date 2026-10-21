"""Deeply immutable records, so verified content cannot change after validation."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, model_validator


def _immutable(*_args, **_kwargs):
    raise TypeError("FrozenDict is immutable")


class FrozenDict(dict):
    """A dict that serializes and compares like a dict but cannot be changed."""

    __slots__ = ()
    __setitem__ = __delitem__ = __ior__ = _immutable
    clear = pop = popitem = setdefault = update = _immutable

    def __hash__(self):
        return hash(frozenset(self.items()))

    def __copy__(self):
        return self

    def __deepcopy__(self, memo):
        # Values are frozen too, so a copy could never differ from the original.
        return self

    def __reduce__(self):
        return type(self), (dict(self),)


def freeze(value: Any) -> Any:
    """Return `value` with every nested dict made immutable; other values are unchanged."""
    if isinstance(value, FrozenDict):
        return value
    if isinstance(value, dict):
        return FrozenDict({key: freeze(item) for key, item in value.items()})
    if isinstance(value, tuple):
        items = tuple(freeze(item) for item in value)
        return value if all(a is b for a, b in zip(items, value)) else items
    return value


def _freeze_fields(model: BaseModel) -> BaseModel:
    for name in type(model).model_fields:
        value = model.__dict__[name]
        frozen = freeze(value)
        if frozen is not value:
            # Frozen models reject attribute assignment; this runs before anyone can observe them.
            model.__dict__[name] = frozen
    return model


class FrozenModel(BaseModel):
    """A frozen model whose mappings are frozen as well.

    Pydantic's `frozen=True` only blocks field reassignment. Verification results
    are cached per object, so nested dicts must not change after validation.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    @model_validator(mode="after")
    def freeze_mappings(self):
        return _freeze_fields(self)

    def model_copy(self, *, update=None, deep=False):
        return _freeze_fields(super().model_copy(update=update, deep=deep))

    @classmethod
    def model_construct(cls, _fields_set=None, **values):
        return _freeze_fields(super().model_construct(_fields_set, **values))
