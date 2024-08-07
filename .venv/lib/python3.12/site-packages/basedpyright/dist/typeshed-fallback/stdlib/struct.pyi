from _typeshed import ReadableBuffer, WriteableBuffer
from collections.abc import Iterator
from typing import Any

__all__ = ["calcsize", "pack", "pack_into", "unpack", "unpack_from", "iter_unpack", "Struct", "error"]

class error(Exception): ...

def pack(fmt: str | bytes, /, *v: Any) -> bytes:
    """
    pack(format, v1, v2, ...) -> bytes

    Return a bytes object containing the values v1, v2, ... packed according
    to the format string.  See help(struct) for more on format strings.
    """
    ...
def pack_into(fmt: str | bytes, buffer: WriteableBuffer, offset: int, /, *v: Any) -> None:
    """
    pack_into(format, buffer, offset, v1, v2, ...)

    Pack the values v1, v2, ... according to the format string and write
    the packed bytes into the writable buffer buf starting at offset.  Note
    that the offset is a required argument.  See help(struct) for more
    on format strings.
    """
    ...
def unpack(format: str | bytes, buffer: ReadableBuffer, /) -> tuple[Any, ...]:
    """
    Return a tuple containing values unpacked according to the format string.

    The buffer's size in bytes must be calcsize(format).

    See help(struct) for more on format strings.
    """
    ...
def unpack_from(format: str | bytes, /, buffer: ReadableBuffer, offset: int = 0) -> tuple[Any, ...]:
    """
    Return a tuple containing values unpacked according to the format string.

    The buffer's size, minus offset, must be at least calcsize(format).

    See help(struct) for more on format strings.
    """
    ...
def iter_unpack(format: str | bytes, buffer: ReadableBuffer, /) -> Iterator[tuple[Any, ...]]:
    """
    Return an iterator yielding tuples unpacked from the given bytes.

    The bytes are unpacked according to the format string, like
    a repeated invocation of unpack_from().

    Requires that the bytes length be a multiple of the format struct size.
    """
    ...
def calcsize(format: str | bytes, /) -> int:
    """Return size in bytes of the struct described by the format string."""
    ...

class Struct:
    """
    Create a compiled struct object.

    Return a new Struct object which writes and reads binary data according to
    the format string.

    See help(struct) for more on format strings.
    """
    @property
    def format(self) -> str:
        """struct format string"""
        ...
    @property
    def size(self) -> int:
        """struct size in bytes"""
        ...
    def __init__(self, format: str | bytes) -> None: ...
    def pack(self, *v: Any) -> bytes:
        """
        S.pack(v1, v2, ...) -> bytes

        Return a bytes object containing values v1, v2, ... packed according
        to the format string S.format.  See help(struct) for more on format
        strings.
        """
        ...
    def pack_into(self, buffer: WriteableBuffer, offset: int, *v: Any) -> None:
        """
        S.pack_into(buffer, offset, v1, v2, ...)

        Pack the values v1, v2, ... according to the format string S.format
        and write the packed bytes into the writable buffer buf starting at
        offset.  Note that the offset is a required argument.  See
        help(struct) for more on format strings.
        """
        ...
    def unpack(self, buffer: ReadableBuffer, /) -> tuple[Any, ...]:
        """
        Return a tuple containing unpacked values.

        Unpack according to the format string Struct.format. The buffer's size
        in bytes must be Struct.size.

        See help(struct) for more on format strings.
        """
        ...
    def unpack_from(self, buffer: ReadableBuffer, offset: int = 0) -> tuple[Any, ...]:
        """
        Return a tuple containing unpacked values.

        Values are unpacked according to the format string Struct.format.

        The buffer's size in bytes, starting at position offset, must be
        at least Struct.size.

        See help(struct) for more on format strings.
        """
        ...
    def iter_unpack(self, buffer: ReadableBuffer, /) -> Iterator[tuple[Any, ...]]:
        """
        Return an iterator yielding tuples.

        Tuples are unpacked from the given bytes source, like a repeated
        invocation of unpack_from().

        Requires that the bytes length be a multiple of the struct size.
        """
        ...
