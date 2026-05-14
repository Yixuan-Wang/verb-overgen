import functools
from pathlib import Path
from typing import overload, TypeVar

T = TypeVar("T")

def adhoc(arg: T, /) -> T:
    """Assert the code is running with an environment variable ADHOC set to a truthy value, and return the argument.
    
    Args:
        arg (T): The argument to be returned if the assertion passes.
    
    Returns:
        T: The input argument, returned if the assertion passes.
    """
    import os
    if not os.environ.get("ADHOC"):
        raise RuntimeError("Set environment variable ADHOC=1 to proceed.")
    return arg

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

@overload
def ensure_dir(dir: str) -> str: ...
@overload
def ensure_dir(dir: Path) -> Path: ...
def ensure_dir(dir):
    if isinstance(dir, str):
        path_dir = Path(dir)
    elif isinstance(dir, Path):
        path_dir = dir
    else:
        raise TypeError("The directory must be a string or a Path object.")
        
    if path_dir.exists() and path_dir.is_file():
        raise AssertionError("The directory exists and is already a file.")
    
    if not path_dir.exists():
        path_dir.mkdir(parents=True, exist_ok=True)
        
    if isinstance(dir, str):
        return str(path_dir)
    else:
        return path_dir