import abc
import builtins
import codecs
import sys
from _typeshed import FileDescriptorOrPath, ReadableBuffer, WriteableBuffer
from collections.abc import Callable, Iterable, Iterator
from os import _Opener
from types import TracebackType
from typing import IO, Any, BinaryIO, Final, Generic, Literal, Protocol, TextIO, TypeVar, overload, type_check_only
from typing_extensions import Self

__all__ = [
    "BlockingIOError",
    "open",
    "open_code",
    "IOBase",
    "RawIOBase",
    "FileIO",
    "BytesIO",
    "StringIO",
    "BufferedIOBase",
    "BufferedReader",
    "BufferedWriter",
    "BufferedRWPair",
    "BufferedRandom",
    "TextIOBase",
    "TextIOWrapper",
    "UnsupportedOperation",
    "SEEK_SET",
    "SEEK_CUR",
    "SEEK_END",
]

if sys.version_info >= (3, 11):
    __all__ += ["DEFAULT_BUFFER_SIZE", "IncrementalNewlineDecoder", "text_encoding"]

_T = TypeVar("_T")

DEFAULT_BUFFER_SIZE: Final = 8192

SEEK_SET: Final = 0
SEEK_CUR: Final = 1
SEEK_END: Final = 2

open = builtins.open

def open_code(path: str) -> IO[bytes]:
    """
    Opens the provided file with the intent to import the contents.

    This may perform extra validation beyond open(), but is otherwise interchangeable
    with calling open(path, 'rb').
    """
    ...

BlockingIOError = builtins.BlockingIOError

class UnsupportedOperation(OSError, ValueError): ...

class IOBase(metaclass=abc.ABCMeta):
    def __iter__(self) -> Iterator[bytes]:
        """Implement iter(self)."""
        ...
    def __next__(self) -> bytes:
        """Implement next(self)."""
        ...
    def __enter__(self) -> Self: ...
    def __exit__(
        self, exc_type: type[BaseException] | None, exc_val: BaseException | None, exc_tb: TracebackType | None
    ) -> None: ...
    def close(self) -> None:
        """
        Flush and close the IO object.

        This method has no effect if the file is already closed.
        """
        ...
    def fileno(self) -> int:
        """
        Returns underlying file descriptor if one exists.

        OSError is raised if the IO object does not use a file descriptor.
        """
        ...
    def flush(self) -> None:
        """
        Flush write buffers, if applicable.

        This is not implemented for read-only and non-blocking streams.
        """
        ...
    def isatty(self) -> bool:
        """
        Return whether this is an 'interactive' stream.

        Return False if it can't be determined.
        """
        ...
    def readable(self) -> bool:
        """
        Return whether object was opened for reading.

        If False, read() will raise OSError.
        """
        ...
    read: Callable[..., Any]
    def readlines(self, hint: int = -1, /) -> list[bytes]:
        """
        Return a list of lines from the stream.

        hint can be specified to control the number of lines read: no more
        lines will be read if the total size (in bytes/characters) of all
        lines so far exceeds hint.
        """
        ...
    def seek(self, offset: int, whence: int = ..., /) -> int:
        """
        Change stream position.

        Change the stream position to the given byte offset. The offset is
        interpreted relative to the position indicated by whence.  Values
        for whence are:

        * 0 -- start of stream (the default); offset should be zero or positive
        * 1 -- current stream position; offset may be negative
        * 2 -- end of stream; offset is usually negative

        Return the new absolute position.
        """
        ...
    def seekable(self) -> bool:
        """
        Return whether object supports random access.

        If False, seek(), tell() and truncate() will raise OSError.
        This method may need to do a test seek().
        """
        ...
    def tell(self) -> int:
        """Return current stream position."""
        ...
    def truncate(self, size: int | None = ..., /) -> int:
        """
        Truncate file to size bytes.

        File pointer is left unchanged.  Size defaults to the current IO
        position as reported by tell().  Returns the new size.
        """
        ...
    def writable(self) -> bool:
        """
        Return whether object was opened for writing.

        If False, write() will raise OSError.
        """
        ...
    write: Callable[..., Any]
    def writelines(self, lines: Iterable[ReadableBuffer], /) -> None:
        """
        Write a list of lines to stream.

        Line separators are not added, so it is usual for each of the
        lines provided to have a line separator at the end.
        """
        ...
    def readline(self, size: int | None = -1, /) -> bytes:
        r"""
        Read and return a line from the stream.

        If size is specified, at most size bytes will be read.

        The line terminator is always b'\n' for binary files; for text
        files, the newlines argument to open can be used to select the line
        terminator(s) recognized.
        """
        ...
    def __del__(self) -> None: ...
    @property
    def closed(self) -> bool: ...
    def _checkClosed(self) -> None: ...  # undocumented

class RawIOBase(IOBase):
    def readall(self) -> bytes:
        """Read until EOF, using multiple read() call."""
        ...
    def readinto(self, buffer: WriteableBuffer, /) -> int | None: ...
    def write(self, b: ReadableBuffer, /) -> int | None: ...
    def read(self, size: int = -1, /) -> bytes | None: ...

class BufferedIOBase(IOBase):
    raw: RawIOBase  # This is not part of the BufferedIOBase API and may not exist on some implementations.
    def detach(self) -> RawIOBase:
        """
        Disconnect this buffer from its underlying raw stream and return it.

        After the raw stream has been detached, the buffer is in an unusable
        state.
        """
        ...
    def readinto(self, buffer: WriteableBuffer, /) -> int: ...
    def write(self, buffer: ReadableBuffer, /) -> int:
        """
        Write the given buffer to the IO stream.

        Returns the number of bytes written, which is always the length of b
        in bytes.

        Raises BlockingIOError if the buffer is full and the
        underlying raw stream cannot accept more data at the moment.
        """
        ...
    def readinto1(self, buffer: WriteableBuffer, /) -> int: ...
    def read(self, size: int | None = ..., /) -> bytes:
        """
        Read and return up to n bytes.

        If the argument is omitted, None, or negative, reads and
        returns all data until EOF.

        If the argument is positive, and the underlying raw stream is
        not 'interactive', multiple raw reads may be issued to satisfy
        the byte count (unless EOF is reached first).  But for
        interactive raw streams (as well as sockets and pipes), at most
        one raw read will be issued, and a short result does not imply
        that EOF is imminent.

        Returns an empty bytes object on EOF.

        Returns None if the underlying raw stream was open in non-blocking
        mode and no data is available at the moment.
        """
        ...
    def read1(self, size: int = ..., /) -> bytes:
        """
        Read and return up to n bytes, with at most one read() call
        to the underlying raw stream. A short result does not imply
        that EOF is imminent.

        Returns an empty bytes object on EOF.
        """
        ...

class FileIO(RawIOBase, BinaryIO):  # type: ignore[misc]  # incompatible definitions of writelines in the base classes
    """
    Open a file.

    The mode can be 'r' (default), 'w', 'x' or 'a' for reading,
    writing, exclusive creation or appending.  The file will be created if it
    doesn't exist when opened for writing or appending; it will be truncated
    when opened for writing.  A FileExistsError will be raised if it already
    exists when opened for creating. Opening a file for creating implies
    writing so this mode behaves in a similar way to 'w'.Add a '+' to the mode
    to allow simultaneous reading and writing. A custom opener can be used by
    passing a callable as *opener*. The underlying file descriptor for the file
    object is then obtained by calling opener with (*name*, *flags*).
    *opener* must return an open file descriptor (passing os.open as *opener*
    results in functionality similar to passing None).
    """
    mode: str
    # The type of "name" equals the argument passed in to the constructor,
    # but that can make FileIO incompatible with other I/O types that assume
    # "name" is a str. In the future, making FileIO generic might help.
    name: Any
    def __init__(
        self, file: FileDescriptorOrPath, mode: str = ..., closefd: bool = ..., opener: _Opener | None = ...
    ) -> None: ...
    @property
    def closefd(self) -> bool:
        """True if the file descriptor will be closed by close()."""
        ...
    def write(self, b: ReadableBuffer, /) -> int:
        """
        Write buffer b to file, return number of bytes written.

        Only makes one system call, so not all of the data may be written.
        The number of bytes actually written is returned.  In non-blocking mode,
        returns None if the write would block.
        """
        ...
    def read(self, size: int = -1, /) -> bytes:
        """
        Read at most size bytes, returned as bytes.

        Only makes one system call, so less data may be returned than requested.
        In non-blocking mode, returns None if no data is available.
        Return an empty bytes object at EOF.
        """
        ...
    def __enter__(self) -> Self: ...

class BytesIO(BufferedIOBase, BinaryIO):  # type: ignore[misc]  # incompatible definitions of methods in the base classes
    """Buffered I/O implementation using an in-memory bytes buffer."""
    def __init__(self, initial_bytes: ReadableBuffer = ...) -> None: ...
    # BytesIO does not contain a "name" field. This workaround is necessary
    # to allow BytesIO sub-classes to add this field, as it is defined
    # as a read-only property on IO[].
    name: Any
    def __enter__(self) -> Self: ...
    def getvalue(self) -> bytes:
        """Retrieve the entire contents of the BytesIO object."""
        ...
    def getbuffer(self) -> memoryview:
        """Get a read-write view over the contents of the BytesIO object."""
        ...
    def read1(self, size: int | None = -1, /) -> bytes:
        """
        Read at most size bytes, returned as a bytes object.

        If the size argument is negative or omitted, read until EOF is reached.
        Return an empty bytes object at EOF.
        """
        ...

class BufferedReader(BufferedIOBase, BinaryIO):  # type: ignore[misc]  # incompatible definitions of methods in the base classes
    """Create a new buffered reader using the given readable raw IO object."""
    def __enter__(self) -> Self: ...
    def __init__(self, raw: RawIOBase, buffer_size: int = ...) -> None: ...
    def peek(self, size: int = 0, /) -> bytes: ...

class BufferedWriter(BufferedIOBase, BinaryIO):  # type: ignore[misc]  # incompatible definitions of writelines in the base classes
    """
    A buffer for a writeable sequential RawIO object.

    The constructor creates a BufferedWriter for the given writeable raw
    stream. If the buffer_size is not given, it defaults to
    DEFAULT_BUFFER_SIZE.
    """
    def __enter__(self) -> Self: ...
    def __init__(self, raw: RawIOBase, buffer_size: int = ...) -> None: ...
    def write(self, buffer: ReadableBuffer, /) -> int: ...

class BufferedRandom(BufferedReader, BufferedWriter):  # type: ignore[misc]  # incompatible definitions of methods in the base classes
    """
    A buffered interface to random access streams.

    The constructor creates a reader and writer for a seekable stream,
    raw, given in the first argument. If the buffer_size is omitted it
    defaults to DEFAULT_BUFFER_SIZE.
    """
    def __enter__(self) -> Self: ...
    def seek(self, target: int, whence: int = 0, /) -> int: ...  # stubtest needs this

class BufferedRWPair(BufferedIOBase):
    """
    A buffered reader and writer object together.

    A buffered reader object and buffered writer object put together to
    form a sequential IO object that can read and write. This is typically
    used with a socket or two-way pipe.

    reader and writer are RawIOBase objects that are readable and
    writeable respectively. If the buffer_size is omitted it defaults to
    DEFAULT_BUFFER_SIZE.
    """
    def __init__(self, reader: RawIOBase, writer: RawIOBase, buffer_size: int = ...) -> None: ...
    def peek(self, size: int = ..., /) -> bytes: ...

class TextIOBase(IOBase):
    encoding: str
    errors: str | None
    newlines: str | tuple[str, ...] | None
    def __iter__(self) -> Iterator[str]:
        """Implement iter(self)."""
        ...
    def __next__(self) -> str:
        """Implement next(self)."""
        ...
    def detach(self) -> BinaryIO:
        """
        Separate the underlying buffer from the TextIOBase and return it.

        After the underlying buffer has been detached, the TextIO is in an
        unusable state.
        """
        ...
    def write(self, s: str, /) -> int:
        """
        Write string to stream.
        Returns the number of characters written (which is always equal to
        the length of the string).
        """
        ...
    def writelines(self, lines: Iterable[str], /) -> None:
        """
        Write a list of lines to stream.

        Line separators are not added, so it is usual for each of the
        lines provided to have a line separator at the end.
        """
        ...
    def readline(self, size: int = ..., /) -> str:
        """
        Read until newline or EOF.

        Returns an empty string if EOF is hit immediately.
        """
        ...
    def readlines(self, hint: int = -1, /) -> list[str]:
        """
        Return a list of lines from the stream.

        hint can be specified to control the number of lines read: no more
        lines will be read if the total size (in bytes/characters) of all
        lines so far exceeds hint.
        """
        ...
    def read(self, size: int | None = ..., /) -> str:
        """
        Read at most n characters from stream.

        Read from underlying buffer until we have n characters or we hit EOF.
        If n is negative or omitted, read until EOF.
        """
        ...

@type_check_only
class _WrappedBuffer(Protocol):
    # "name" is wrapped by TextIOWrapper. Its type is inconsistent between
    # the various I/O types, see the comments on TextIOWrapper.name and
    # TextIO.name.
    @property
    def name(self) -> Any: ...
    @property
    def closed(self) -> bool: ...
    def read(self, size: int = ..., /) -> ReadableBuffer: ...
    # Optional: def read1(self, size: int, /) -> ReadableBuffer: ...
    def write(self, b: bytes, /) -> object: ...
    def flush(self) -> object: ...
    def close(self) -> object: ...
    def seekable(self) -> bool: ...
    def readable(self) -> bool: ...
    def writable(self) -> bool: ...
    def truncate(self, size: int, /) -> int: ...
    def fileno(self) -> int: ...
    def isatty(self) -> bool: ...
    # Optional: Only needs to be present if seekable() returns True.
    # def seek(self, offset: Literal[0], whence: Literal[2]) -> int: ...
    # def tell(self) -> int: ...

_BufferT_co = TypeVar("_BufferT_co", bound=_WrappedBuffer, default=_WrappedBuffer, covariant=True)

class TextIOWrapper(TextIOBase, TextIO, Generic[_BufferT_co]):  # type: ignore[misc]  # incompatible definitions of write in the base classes
    r"""
    Character and line based layer over a BufferedIOBase object, buffer.

    encoding gives the name of the encoding that the stream will be
    decoded or encoded with. It defaults to locale.getpreferredencoding(False).

    errors determines the strictness of encoding and decoding (see
    help(codecs.Codec) or the documentation for codecs.register) and
    defaults to "strict".

    newline controls how line endings are handled. It can be None, '',
    '\n', '\r', and '\r\n'.  It works as follows:

    * On input, if newline is None, universal newlines mode is
      enabled. Lines in the input can end in '\n', '\r', or '\r\n', and
      these are translated into '\n' before being returned to the
      caller. If it is '', universal newline mode is enabled, but line
      endings are returned to the caller untranslated. If it has any of
      the other legal values, input lines are only terminated by the given
      string, and the line ending is returned to the caller untranslated.

    * On output, if newline is None, any '\n' characters written are
      translated to the system default line separator, os.linesep. If
      newline is '' or '\n', no translation takes place. If newline is any
      of the other legal values, any '\n' characters written are translated
      to the given string.

    If line_buffering is True, a call to flush is implied when a call to
    write contains a newline character.
    """
    def __init__(
        self,
        buffer: _BufferT_co,
        encoding: str | None = None,
        errors: str | None = None,
        newline: str | None = None,
        line_buffering: bool = False,
        write_through: bool = False,
    ) -> None: ...
    # Equals the "buffer" argument passed in to the constructor.
    @property
    def buffer(self) -> _BufferT_co: ...  # type: ignore[override]
    @property
    def closed(self) -> bool: ...
    @property
    def line_buffering(self) -> bool: ...
    @property
    def write_through(self) -> bool: ...
    def reconfigure(
        self,
        *,
        encoding: str | None = None,
        errors: str | None = None,
        newline: str | None = None,
        line_buffering: bool | None = None,
        write_through: bool | None = None,
    ) -> None:
        """
        Reconfigure the text stream with new parameters.

        This also does an implicit stream flush.
        """
        ...
    # These are inherited from TextIOBase, but must exist in the stub to satisfy mypy.
    def __enter__(self) -> Self: ...
    def __iter__(self) -> Iterator[str]:
        """Implement iter(self)."""
        ...
    def __next__(self) -> str:
        """Implement next(self)."""
        ...
    def writelines(self, lines: Iterable[str], /) -> None:
        """
        Write a list of lines to stream.

        Line separators are not added, so it is usual for each of the
        lines provided to have a line separator at the end.
        """
        ...
    def readline(self, size: int = -1, /) -> str: ...  # type: ignore[override]
    def readlines(self, hint: int = -1, /) -> list[str]:
        """
        Return a list of lines from the stream.

        hint can be specified to control the number of lines read: no more
        lines will be read if the total size (in bytes/characters) of all
        lines so far exceeds hint.
        """
        ...
    # Equals the "buffer" argument passed in to the constructor.
    def detach(self) -> _BufferT_co: ...  # type: ignore[override]
    # TextIOWrapper's version of seek only supports a limited subset of
    # operations.
    def seek(self, cookie: int, whence: int = 0, /) -> int:
        """
        Set the stream position, and return the new stream position.

          cookie
            Zero or an opaque number returned by tell().
          whence
            The relative position to seek from.

        Four operations are supported, given by the following argument
        combinations:

        - seek(0, SEEK_SET): Rewind to the start of the stream.
        - seek(cookie, SEEK_SET): Restore a previous position;
          'cookie' must be a number returned by tell().
        - seek(0, SEEK_END): Fast-forward to the end of the stream.
        - seek(0, SEEK_CUR): Leave the current stream position unchanged.

        Any other argument combinations are invalid,
        and may raise exceptions.
        """
        ...

class StringIO(TextIOWrapper):
    """
    Text I/O implementation using an in-memory buffer.

    The initial_value argument sets the value of object.  The newline
    argument is like the one of TextIOWrapper's constructor.
    """
    def __init__(self, initial_value: str | None = ..., newline: str | None = ...) -> None: ...
    # StringIO does not contain a "name" field. This workaround is necessary
    # to allow StringIO sub-classes to add this field, as it is defined
    # as a read-only property on IO[].
    name: Any
    def getvalue(self) -> str:
        """Retrieve the entire contents of the object."""
        ...

class IncrementalNewlineDecoder(codecs.IncrementalDecoder):
    r"""
    Codec used when reading a file in universal newlines mode.

    It wraps another incremental decoder, translating \r\n and \r into \n.
    It also records the types of newlines encountered.  When used with
    translate=False, it ensures that the newline sequence is returned in
    one piece. When used with decoder=None, it expects unicode strings as
    decode input and translates newlines without first invoking an external
    decoder.
    """
    def __init__(self, decoder: codecs.IncrementalDecoder | None, translate: bool, errors: str = ...) -> None: ...
    def decode(self, input: ReadableBuffer | str, final: bool = False) -> str: ...
    @property
    def newlines(self) -> str | tuple[str, ...] | None: ...
    def setstate(self, state: tuple[bytes, int], /) -> None: ...

if sys.version_info >= (3, 10):
    @overload
    def text_encoding(encoding: None, stacklevel: int = 2, /) -> Literal["locale", "utf-8"]:
        """
        A helper function to choose the text encoding.

        When encoding is not None, just return it.
        Otherwise, return the default text encoding (i.e. "locale").

        This function emits an EncodingWarning if encoding is None and
        sys.flags.warn_default_encoding is true.

        This can be used in APIs with an encoding=None parameter.
        However, please consider using encoding="utf-8" for new APIs.
        """
        ...
    @overload
    def text_encoding(encoding: _T, stacklevel: int = 2, /) -> _T:
        """
        A helper function to choose the text encoding.

        When encoding is not None, just return it.
        Otherwise, return the default text encoding (i.e. "locale").

        This function emits an EncodingWarning if encoding is None and
        sys.flags.warn_default_encoding is true.

        This can be used in APIs with an encoding=None parameter.
        However, please consider using encoding="utf-8" for new APIs.
        """
        ...
