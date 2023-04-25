from enum import Enum


class EnumWithAttrs(Enum):
    def __new__(cls, *args, **kwargs) -> 'EnumWithAttrs':
        value = len(cls.__members__) + 1
        obj = object.__new__(cls)
        obj._value_ = value
        return obj
