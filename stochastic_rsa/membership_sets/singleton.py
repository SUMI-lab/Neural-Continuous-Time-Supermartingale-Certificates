"""Provides a singleton class."""


class Singleton(type):
    """A singleton class so that empty sets are not created many times."""
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(
                Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]
