import functools

def affirm(value, predicate = None):
    """
    Assert the value according to the predicate, return it if it is true.

    Args:
        value (T): The value to be asserted.
        predicate (Callable[[T], bool] | bool | None): The predicate to be used for assertion. If None, evaluate the truthiness of the value.

    Returns:
        T: The value asserted.

    Raises:
        AssertionError: If the value is not true.
    """

    if __debug__:
        if predicate is None:
            predicate = bool

        if callable(predicate):
            if not predicate(value):
                raise AssertionError(f"Value {value} is not true according to the predicate {predicate}.")
        else:
            if not predicate:
                raise AssertionError(f"Value {value} is not true according to the predicate {predicate}.")

    return value

class non_null:
    """
    Assert whether the value is not None.
    """
    def __new__(cls, value) -> bool:
        return value is not None

class of_type:
    """
    Assert whether the value is of the specified type.
    """
    ty = object

    def __new__(cls, value) -> bool:
        return isinstance(value, cls.ty)

    @classmethod
    @functools.lru_cache
    def type(cls, ty):
        new_cls = type(cls.__name__, (cls,), {})
        new_cls.ty = ty
        return new_cls

    def __class_getitem__(cls, item):
        if isinstance(item, type):
            return cls.type(item)
        else:
            raise TypeError(f"Expected a type, got {type(item)}.")
