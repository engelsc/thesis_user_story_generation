import sqlite3
import sys
from _typeshed import ReadableBuffer, StrOrBytesPath, SupportsLenAndGetItem, Unused
from collections.abc import Callable, Generator, Iterable, Iterator, Mapping
from datetime import date, datetime, time
from types import TracebackType
from typing import Any, Final, Literal, Protocol, SupportsIndex, TypeVar, final, overload
from typing_extensions import Self, TypeAlias

_T = TypeVar("_T")
_ConnectionT = TypeVar("_ConnectionT", bound=Connection)
_CursorT = TypeVar("_CursorT", bound=Cursor)
_SqliteData: TypeAlias = str | ReadableBuffer | int | float | None
# Data that is passed through adapters can be of any type accepted by an adapter.
_AdaptedInputData: TypeAlias = _SqliteData | Any
# The Mapping must really be a dict, but making it invariant is too annoying.
_Parameters: TypeAlias = SupportsLenAndGetItem[_AdaptedInputData] | Mapping[str, _AdaptedInputData]
_Adapter: TypeAlias = Callable[[_T], _SqliteData]
_Converter: TypeAlias = Callable[[bytes], Any]

paramstyle: str
threadsafety: int
apilevel: str
Date = date
Time = time
Timestamp = datetime

def DateFromTicks(ticks: float) -> Date: ...
def TimeFromTicks(ticks: float) -> Time: ...
def TimestampFromTicks(ticks: float) -> Timestamp: ...

version_info: tuple[int, int, int]
sqlite_version_info: tuple[int, int, int]
Binary = memoryview

# The remaining definitions are imported from _sqlite3.

PARSE_COLNAMES: Final[int]
PARSE_DECLTYPES: Final[int]
SQLITE_ALTER_TABLE: Final[int]
SQLITE_ANALYZE: Final[int]
SQLITE_ATTACH: Final[int]
SQLITE_CREATE_INDEX: Final[int]
SQLITE_CREATE_TABLE: Final[int]
SQLITE_CREATE_TEMP_INDEX: Final[int]
SQLITE_CREATE_TEMP_TABLE: Final[int]
SQLITE_CREATE_TEMP_TRIGGER: Final[int]
SQLITE_CREATE_TEMP_VIEW: Final[int]
SQLITE_CREATE_TRIGGER: Final[int]
SQLITE_CREATE_VIEW: Final[int]
SQLITE_CREATE_VTABLE: Final[int]
SQLITE_DELETE: Final[int]
SQLITE_DENY: Final[int]
SQLITE_DETACH: Final[int]
SQLITE_DONE: Final[int]
SQLITE_DROP_INDEX: Final[int]
SQLITE_DROP_TABLE: Final[int]
SQLITE_DROP_TEMP_INDEX: Final[int]
SQLITE_DROP_TEMP_TABLE: Final[int]
SQLITE_DROP_TEMP_TRIGGER: Final[int]
SQLITE_DROP_TEMP_VIEW: Final[int]
SQLITE_DROP_TRIGGER: Final[int]
SQLITE_DROP_VIEW: Final[int]
SQLITE_DROP_VTABLE: Final[int]
SQLITE_FUNCTION: Final[int]
SQLITE_IGNORE: Final[int]
SQLITE_INSERT: Final[int]
SQLITE_OK: Final[int]
if sys.version_info >= (3, 11):
    SQLITE_LIMIT_LENGTH: Final[int]
    SQLITE_LIMIT_SQL_LENGTH: Final[int]
    SQLITE_LIMIT_COLUMN: Final[int]
    SQLITE_LIMIT_EXPR_DEPTH: Final[int]
    SQLITE_LIMIT_COMPOUND_SELECT: Final[int]
    SQLITE_LIMIT_VDBE_OP: Final[int]
    SQLITE_LIMIT_FUNCTION_ARG: Final[int]
    SQLITE_LIMIT_ATTACHED: Final[int]
    SQLITE_LIMIT_LIKE_PATTERN_LENGTH: Final[int]
    SQLITE_LIMIT_VARIABLE_NUMBER: Final[int]
    SQLITE_LIMIT_TRIGGER_DEPTH: Final[int]
    SQLITE_LIMIT_WORKER_THREADS: Final[int]
SQLITE_PRAGMA: Final[int]
SQLITE_READ: Final[int]
SQLITE_REINDEX: Final[int]
SQLITE_RECURSIVE: Final[int]
SQLITE_SAVEPOINT: Final[int]
SQLITE_SELECT: Final[int]
SQLITE_TRANSACTION: Final[int]
SQLITE_UPDATE: Final[int]
adapters: dict[tuple[type[Any], type[Any]], _Adapter[Any]]
converters: dict[str, _Converter]
sqlite_version: str
version: str

if sys.version_info >= (3, 11):
    SQLITE_ABORT: Final[int]
    SQLITE_ABORT_ROLLBACK: Final[int]
    SQLITE_AUTH: Final[int]
    SQLITE_AUTH_USER: Final[int]
    SQLITE_BUSY: Final[int]
    SQLITE_BUSY_RECOVERY: Final[int]
    SQLITE_BUSY_SNAPSHOT: Final[int]
    SQLITE_BUSY_TIMEOUT: Final[int]
    SQLITE_CANTOPEN: Final[int]
    SQLITE_CANTOPEN_CONVPATH: Final[int]
    SQLITE_CANTOPEN_DIRTYWAL: Final[int]
    SQLITE_CANTOPEN_FULLPATH: Final[int]
    SQLITE_CANTOPEN_ISDIR: Final[int]
    SQLITE_CANTOPEN_NOTEMPDIR: Final[int]
    SQLITE_CANTOPEN_SYMLINK: Final[int]
    SQLITE_CONSTRAINT: Final[int]
    SQLITE_CONSTRAINT_CHECK: Final[int]
    SQLITE_CONSTRAINT_COMMITHOOK: Final[int]
    SQLITE_CONSTRAINT_FOREIGNKEY: Final[int]
    SQLITE_CONSTRAINT_FUNCTION: Final[int]
    SQLITE_CONSTRAINT_NOTNULL: Final[int]
    SQLITE_CONSTRAINT_PINNED: Final[int]
    SQLITE_CONSTRAINT_PRIMARYKEY: Final[int]
    SQLITE_CONSTRAINT_ROWID: Final[int]
    SQLITE_CONSTRAINT_TRIGGER: Final[int]
    SQLITE_CONSTRAINT_UNIQUE: Final[int]
    SQLITE_CONSTRAINT_VTAB: Final[int]
    SQLITE_CORRUPT: Final[int]
    SQLITE_CORRUPT_INDEX: Final[int]
    SQLITE_CORRUPT_SEQUENCE: Final[int]
    SQLITE_CORRUPT_VTAB: Final[int]
    SQLITE_EMPTY: Final[int]
    SQLITE_ERROR: Final[int]
    SQLITE_ERROR_MISSING_COLLSEQ: Final[int]
    SQLITE_ERROR_RETRY: Final[int]
    SQLITE_ERROR_SNAPSHOT: Final[int]
    SQLITE_FORMAT: Final[int]
    SQLITE_FULL: Final[int]
    SQLITE_INTERNAL: Final[int]
    SQLITE_INTERRUPT: Final[int]
    SQLITE_IOERR: Final[int]
    SQLITE_IOERR_ACCESS: Final[int]
    SQLITE_IOERR_AUTH: Final[int]
    SQLITE_IOERR_BEGIN_ATOMIC: Final[int]
    SQLITE_IOERR_BLOCKED: Final[int]
    SQLITE_IOERR_CHECKRESERVEDLOCK: Final[int]
    SQLITE_IOERR_CLOSE: Final[int]
    SQLITE_IOERR_COMMIT_ATOMIC: Final[int]
    SQLITE_IOERR_CONVPATH: Final[int]
    SQLITE_IOERR_CORRUPTFS: Final[int]
    SQLITE_IOERR_DATA: Final[int]
    SQLITE_IOERR_DELETE: Final[int]
    SQLITE_IOERR_DELETE_NOENT: Final[int]
    SQLITE_IOERR_DIR_CLOSE: Final[int]
    SQLITE_IOERR_DIR_FSYNC: Final[int]
    SQLITE_IOERR_FSTAT: Final[int]
    SQLITE_IOERR_FSYNC: Final[int]
    SQLITE_IOERR_GETTEMPPATH: Final[int]
    SQLITE_IOERR_LOCK: Final[int]
    SQLITE_IOERR_MMAP: Final[int]
    SQLITE_IOERR_NOMEM: Final[int]
    SQLITE_IOERR_RDLOCK: Final[int]
    SQLITE_IOERR_READ: Final[int]
    SQLITE_IOERR_ROLLBACK_ATOMIC: Final[int]
    SQLITE_IOERR_SEEK: Final[int]
    SQLITE_IOERR_SHMLOCK: Final[int]
    SQLITE_IOERR_SHMMAP: Final[int]
    SQLITE_IOERR_SHMOPEN: Final[int]
    SQLITE_IOERR_SHMSIZE: Final[int]
    SQLITE_IOERR_SHORT_READ: Final[int]
    SQLITE_IOERR_TRUNCATE: Final[int]
    SQLITE_IOERR_UNLOCK: Final[int]
    SQLITE_IOERR_VNODE: Final[int]
    SQLITE_IOERR_WRITE: Final[int]
    SQLITE_LOCKED: Final[int]
    SQLITE_LOCKED_SHAREDCACHE: Final[int]
    SQLITE_LOCKED_VTAB: Final[int]
    SQLITE_MISMATCH: Final[int]
    SQLITE_MISUSE: Final[int]
    SQLITE_NOLFS: Final[int]
    SQLITE_NOMEM: Final[int]
    SQLITE_NOTADB: Final[int]
    SQLITE_NOTFOUND: Final[int]
    SQLITE_NOTICE: Final[int]
    SQLITE_NOTICE_RECOVER_ROLLBACK: Final[int]
    SQLITE_NOTICE_RECOVER_WAL: Final[int]
    SQLITE_OK_LOAD_PERMANENTLY: Final[int]
    SQLITE_OK_SYMLINK: Final[int]
    SQLITE_PERM: Final[int]
    SQLITE_PROTOCOL: Final[int]
    SQLITE_RANGE: Final[int]
    SQLITE_READONLY: Final[int]
    SQLITE_READONLY_CANTINIT: Final[int]
    SQLITE_READONLY_CANTLOCK: Final[int]
    SQLITE_READONLY_DBMOVED: Final[int]
    SQLITE_READONLY_DIRECTORY: Final[int]
    SQLITE_READONLY_RECOVERY: Final[int]
    SQLITE_READONLY_ROLLBACK: Final[int]
    SQLITE_ROW: Final[int]
    SQLITE_SCHEMA: Final[int]
    SQLITE_TOOBIG: Final[int]
    SQLITE_WARNING: Final[int]
    SQLITE_WARNING_AUTOINDEX: Final[int]

if sys.version_info >= (3, 12):
    LEGACY_TRANSACTION_CONTROL: Final[int]
    SQLITE_DBCONFIG_DEFENSIVE: Final[int]
    SQLITE_DBCONFIG_DQS_DDL: Final[int]
    SQLITE_DBCONFIG_DQS_DML: Final[int]
    SQLITE_DBCONFIG_ENABLE_FKEY: Final[int]
    SQLITE_DBCONFIG_ENABLE_FTS3_TOKENIZER: Final[int]
    SQLITE_DBCONFIG_ENABLE_LOAD_EXTENSION: Final[int]
    SQLITE_DBCONFIG_ENABLE_QPSG: Final[int]
    SQLITE_DBCONFIG_ENABLE_TRIGGER: Final[int]
    SQLITE_DBCONFIG_ENABLE_VIEW: Final[int]
    SQLITE_DBCONFIG_LEGACY_ALTER_TABLE: Final[int]
    SQLITE_DBCONFIG_LEGACY_FILE_FORMAT: Final[int]
    SQLITE_DBCONFIG_NO_CKPT_ON_CLOSE: Final[int]
    SQLITE_DBCONFIG_RESET_DATABASE: Final[int]
    SQLITE_DBCONFIG_TRIGGER_EQP: Final[int]
    SQLITE_DBCONFIG_TRUSTED_SCHEMA: Final[int]
    SQLITE_DBCONFIG_WRITABLE_SCHEMA: Final[int]

# Can take or return anything depending on what's in the registry.
@overload
def adapt(obj: Any, proto: Any, /) -> Any:
    """adapt(obj, protocol, alternate) -> adapt obj to given protocol. Non-standard."""
    ...
@overload
def adapt(obj: Any, proto: Any, alt: _T, /) -> Any | _T:
    """adapt(obj, protocol, alternate) -> adapt obj to given protocol. Non-standard."""
    ...
def complete_statement(statement: str) -> bool:
    """
    complete_statement(sql)

    Checks if a string contains a complete SQL statement. Non-standard.
    """
    ...

if sys.version_info >= (3, 12):
    @overload
    def connect(
        database: StrOrBytesPath,
        timeout: float = 5.0,
        detect_types: int = 0,
        isolation_level: Literal["DEFERRED", "EXCLUSIVE", "IMMEDIATE"] | None = "DEFERRED",
        check_same_thread: bool = True,
        cached_statements: int = 128,
        uri: bool = False,
        *,
        autocommit: bool = ...,
    ) -> Connection:
        """
        Opens a connection to the SQLite database file database.

        You can use ":memory:" to open a database connection to a database that resides
        in RAM instead of on disk.
        """
        ...
    @overload
    def connect(
        database: StrOrBytesPath,
        timeout: float,
        detect_types: int,
        isolation_level: Literal["DEFERRED", "EXCLUSIVE", "IMMEDIATE"] | None,
        check_same_thread: bool,
        factory: type[_ConnectionT],
        cached_statements: int = 128,
        uri: bool = False,
        *,
        autocommit: bool = ...,
    ) -> _ConnectionT:
        """
        Opens a connection to the SQLite database file database.

        You can use ":memory:" to open a database connection to a database that resides
        in RAM instead of on disk.
        """
        ...
    @overload
    def connect(
        database: StrOrBytesPath,
        timeout: float = 5.0,
        detect_types: int = 0,
        isolation_level: Literal["DEFERRED", "EXCLUSIVE", "IMMEDIATE"] | None = "DEFERRED",
        check_same_thread: bool = True,
        *,
        factory: type[_ConnectionT],
        cached_statements: int = 128,
        uri: bool = False,
        autocommit: bool = ...,
    ) -> _ConnectionT:
        """
        Opens a connection to the SQLite database file database.

        You can use ":memory:" to open a database connection to a database that resides
        in RAM instead of on disk.
        """
        ...

else:
    @overload
    def connect(
        database: StrOrBytesPath,
        timeout: float = 5.0,
        detect_types: int = 0,
        isolation_level: Literal["DEFERRED", "EXCLUSIVE", "IMMEDIATE"] | None = "DEFERRED",
        check_same_thread: bool = True,
        cached_statements: int = 128,
        uri: bool = False,
    ) -> Connection:
        """
        connect(database[, timeout, detect_types, isolation_level,
                check_same_thread, factory, cached_statements, uri])

        Opens a connection to the SQLite database file *database*. You can use
        ":memory:" to open a database connection to a database that resides in
        RAM instead of on disk.
        """
        ...
    @overload
    def connect(
        database: StrOrBytesPath,
        timeout: float,
        detect_types: int,
        isolation_level: Literal["DEFERRED", "EXCLUSIVE", "IMMEDIATE"] | None,
        check_same_thread: bool,
        factory: type[_ConnectionT],
        cached_statements: int = 128,
        uri: bool = False,
    ) -> _ConnectionT:
        """
        connect(database[, timeout, detect_types, isolation_level,
                check_same_thread, factory, cached_statements, uri])

        Opens a connection to the SQLite database file *database*. You can use
        ":memory:" to open a database connection to a database that resides in
        RAM instead of on disk.
        """
        ...
    @overload
    def connect(
        database: StrOrBytesPath,
        timeout: float = 5.0,
        detect_types: int = 0,
        isolation_level: Literal["DEFERRED", "EXCLUSIVE", "IMMEDIATE"] | None = "DEFERRED",
        check_same_thread: bool = True,
        *,
        factory: type[_ConnectionT],
        cached_statements: int = 128,
        uri: bool = False,
    ) -> _ConnectionT:
        """
        connect(database[, timeout, detect_types, isolation_level,
                check_same_thread, factory, cached_statements, uri])

        Opens a connection to the SQLite database file *database*. You can use
        ":memory:" to open a database connection to a database that resides in
        RAM instead of on disk.
        """
        ...

def enable_callback_tracebacks(enable: bool, /) -> None:
    """
    enable_callback_tracebacks(flag)

    Enable or disable callback functions throwing errors to stderr.
    """
    ...

if sys.version_info < (3, 12):
    # takes a pos-or-keyword argument because there is a C wrapper
    def enable_shared_cache(enable: int) -> None:
        """
        enable_shared_cache(do_enable)

        Enable or disable shared cache mode for the calling thread.
        Experimental/Non-standard.
        """
        ...

if sys.version_info >= (3, 10):
    def register_adapter(type: type[_T], adapter: _Adapter[_T], /) -> None:
        """Register a function to adapt Python objects to SQLite values."""
        ...
    def register_converter(typename: str, converter: _Converter, /) -> None:
        """Register a function to convert SQLite values to Python objects."""
        ...

else:
    def register_adapter(type: type[_T], caster: _Adapter[_T], /) -> None:
        """
        register_adapter(type, callable)

        Registers an adapter with pysqlite's adapter registry. Non-standard.
        """
        ...
    def register_converter(name: str, converter: _Converter, /) -> None:
        """
        register_converter(typename, callable)

        Registers a converter with pysqlite. Non-standard.
        """
        ...

class _AggregateProtocol(Protocol):
    def step(self, value: int, /) -> object: ...
    def finalize(self) -> int: ...

class _SingleParamWindowAggregateClass(Protocol):
    def step(self, param: Any, /) -> object: ...
    def inverse(self, param: Any, /) -> object: ...
    def value(self) -> _SqliteData: ...
    def finalize(self) -> _SqliteData: ...

class _AnyParamWindowAggregateClass(Protocol):
    def step(self, *args: Any) -> object: ...
    def inverse(self, *args: Any) -> object: ...
    def value(self) -> _SqliteData: ...
    def finalize(self) -> _SqliteData: ...

class _WindowAggregateClass(Protocol):
    step: Callable[..., object]
    inverse: Callable[..., object]
    def value(self) -> _SqliteData: ...
    def finalize(self) -> _SqliteData: ...

class Connection:
    @property
    def DataError(self) -> type[sqlite3.DataError]: ...
    @property
    def DatabaseError(self) -> type[sqlite3.DatabaseError]: ...
    @property
    def Error(self) -> type[sqlite3.Error]: ...
    @property
    def IntegrityError(self) -> type[sqlite3.IntegrityError]: ...
    @property
    def InterfaceError(self) -> type[sqlite3.InterfaceError]: ...
    @property
    def InternalError(self) -> type[sqlite3.InternalError]: ...
    @property
    def NotSupportedError(self) -> type[sqlite3.NotSupportedError]: ...
    @property
    def OperationalError(self) -> type[sqlite3.OperationalError]: ...
    @property
    def ProgrammingError(self) -> type[sqlite3.ProgrammingError]: ...
    @property
    def Warning(self) -> type[sqlite3.Warning]: ...
    @property
    def in_transaction(self) -> bool: ...
    isolation_level: str | None  # one of '', 'DEFERRED', 'IMMEDIATE' or 'EXCLUSIVE'
    @property
    def total_changes(self) -> int: ...
    if sys.version_info >= (3, 12):
        @property
        def autocommit(self) -> int: ...
        @autocommit.setter
        def autocommit(self, val: int) -> None: ...
    row_factory: Any
    text_factory: Any
    if sys.version_info >= (3, 12):
        def __init__(
            self,
            database: StrOrBytesPath,
            timeout: float = ...,
            detect_types: int = ...,
            isolation_level: str | None = ...,
            check_same_thread: bool = ...,
            factory: type[Connection] | None = ...,
            cached_statements: int = ...,
            uri: bool = ...,
            autocommit: bool = ...,
        ) -> None: ...
    else:
        def __init__(
            self,
            database: StrOrBytesPath,
            timeout: float = ...,
            detect_types: int = ...,
            isolation_level: str | None = ...,
            check_same_thread: bool = ...,
            factory: type[Connection] | None = ...,
            cached_statements: int = ...,
            uri: bool = ...,
        ) -> None: ...

    def close(self) -> None:
        """Closes the connection."""
        ...
    if sys.version_info >= (3, 11):
        def blobopen(self, table: str, column: str, row: int, /, *, readonly: bool = False, name: str = "main") -> Blob:
            """
            Open and return a BLOB object.

            table
              Table name.
            column
              Column name.
            row
              Row index.
            readonly
              Open the BLOB without write permissions.
            name
              Database name.
            """
            ...

    def commit(self) -> None:
        """Commit the current transaction."""
        ...
    def create_aggregate(self, name: str, n_arg: int, aggregate_class: Callable[[], _AggregateProtocol]) -> None:
        """Creates a new aggregate. Non-standard."""
        ...
    if sys.version_info >= (3, 11):
        # num_params determines how many params will be passed to the aggregate class. We provide an overload
        # for the case where num_params = 1, which is expected to be the common case.
        @overload
        def create_window_function(
            self, name: str, num_params: Literal[1], aggregate_class: Callable[[], _SingleParamWindowAggregateClass] | None, /
        ) -> None:
            """
            Creates or redefines an aggregate window function. Non-standard.

            name
              The name of the SQL aggregate window function to be created or
              redefined.
            num_params
              The number of arguments the step and inverse methods takes.
            aggregate_class
              A class with step(), finalize(), value(), and inverse() methods.
              Set to None to clear the window function.
            """
            ...
        # And for num_params = -1, which means the aggregate must accept any number of parameters.
        @overload
        def create_window_function(
            self, name: str, num_params: Literal[-1], aggregate_class: Callable[[], _AnyParamWindowAggregateClass] | None, /
        ) -> None:
            """
            Creates or redefines an aggregate window function. Non-standard.

            name
              The name of the SQL aggregate window function to be created or
              redefined.
            num_params
              The number of arguments the step and inverse methods takes.
            aggregate_class
              A class with step(), finalize(), value(), and inverse() methods.
              Set to None to clear the window function.
            """
            ...
        @overload
        def create_window_function(
            self, name: str, num_params: int, aggregate_class: Callable[[], _WindowAggregateClass] | None, /
        ) -> None:
            """
            Creates or redefines an aggregate window function. Non-standard.

            name
              The name of the SQL aggregate window function to be created or
              redefined.
            num_params
              The number of arguments the step and inverse methods takes.
            aggregate_class
              A class with step(), finalize(), value(), and inverse() methods.
              Set to None to clear the window function.
            """
            ...

    def create_collation(self, name: str, callback: Callable[[str, str], int | SupportsIndex] | None, /) -> None:
        """Creates a collation function. Non-standard."""
        ...
    def create_function(
        self, name: str, narg: int, func: Callable[..., _SqliteData] | None, *, deterministic: bool = False
    ) -> None:
        """Creates a new function. Non-standard."""
        ...
    @overload
    def cursor(self, factory: None = None) -> Cursor:
        """Return a cursor for the connection."""
        ...
    @overload
    def cursor(self, factory: Callable[[Connection], _CursorT]) -> _CursorT:
        """Return a cursor for the connection."""
        ...
    def execute(self, sql: str, parameters: _Parameters = ..., /) -> Cursor:
        """Executes a SQL statement. Non-standard."""
        ...
    def executemany(self, sql: str, parameters: Iterable[_Parameters], /) -> Cursor:
        """Repeatedly executes a SQL statement. Non-standard."""
        ...
    def executescript(self, sql_script: str, /) -> Cursor:
        """Executes a multiple SQL statements at once. Non-standard."""
        ...
    def interrupt(self) -> None:
        """Abort any pending database operation. Non-standard."""
        ...
    if sys.version_info >= (3, 13):
        def iterdump(self, *, filter: str | None = None) -> Generator[str, None, None]: ...
    else:
        def iterdump(self) -> Generator[str, None, None]:
            """Returns iterator to the dump of the database in an SQL text format. Non-standard."""
            ...

    def rollback(self) -> None:
        """Roll back the current transaction."""
        ...
    def set_authorizer(
        self, authorizer_callback: Callable[[int, str | None, str | None, str | None, str | None], int] | None
    ) -> None:
        """Sets authorizer callback. Non-standard."""
        ...
    def set_progress_handler(self, progress_handler: Callable[[], int | None] | None, n: int) -> None:
        """Sets progress handler callback. Non-standard."""
        ...
    def set_trace_callback(self, trace_callback: Callable[[str], object] | None) -> None:
        """Sets a trace callback called for each SQL statement (passed as unicode). Non-standard."""
        ...
    # enable_load_extension and load_extension is not available on python distributions compiled
    # without sqlite3 loadable extension support. see footnotes https://docs.python.org/3/library/sqlite3.html#f1
    def enable_load_extension(self, enable: bool, /) -> None:
        """Enable dynamic loading of SQLite extension modules. Non-standard."""
        ...
    def load_extension(self, name: str, /) -> None:
        """Load SQLite extension module. Non-standard."""
        ...
    def backup(
        self,
        target: Connection,
        *,
        pages: int = -1,
        progress: Callable[[int, int, int], object] | None = None,
        name: str = "main",
        sleep: float = 0.25,
    ) -> None:
        """Makes a backup of the database. Non-standard."""
        ...
    if sys.version_info >= (3, 11):
        def setlimit(self, category: int, limit: int, /) -> int:
            """
            Set connection run-time limits.

              category
                The limit category to be set.
              limit
                The new limit. If the new limit is a negative number, the limit is
                unchanged.

            Attempts to increase a limit above its hard upper bound are silently truncated
            to the hard upper bound. Regardless of whether or not the limit was changed,
            the prior value of the limit is returned.
            """
            ...
        def getlimit(self, category: int, /) -> int:
            """
            Get connection run-time limits.

            category
              The limit category to be queried.
            """
            ...
        def serialize(self, *, name: str = "main") -> bytes:
            """
            Serialize a database into a byte string.

              name
                Which database to serialize.

            For an ordinary on-disk database file, the serialization is just a copy of the
            disk file. For an in-memory database or a "temp" database, the serialization is
            the same sequence of bytes which would be written to disk if that database
            were backed up to disk.
            """
            ...
        def deserialize(self, data: ReadableBuffer, /, *, name: str = "main") -> None:
            """
            Load a serialized database.

              data
                The serialized database content.
              name
                Which database to reopen with the deserialization.

            The deserialize interface causes the database connection to disconnect from the
            target database, and then reopen it as an in-memory database based on the given
            serialized data.

            The deserialize interface will fail with SQLITE_BUSY if the database is
            currently in a read transaction or is involved in a backup operation.
            """
            ...
    if sys.version_info >= (3, 12):
        def getconfig(self, op: int, /) -> bool:
            """
            Query a boolean connection configuration option.

            op
              The configuration verb; one of the sqlite3.SQLITE_DBCONFIG codes.
            """
            ...
        def setconfig(self, op: int, enable: bool = True, /) -> bool:
            """
            Set a boolean connection configuration option.

            op
              The configuration verb; one of the sqlite3.SQLITE_DBCONFIG codes.
            """
            ...

    def __call__(self, sql: str, /) -> _Statement:
        """Call self as a function."""
        ...
    def __enter__(self) -> Self:
        """For context manager. Non-standard."""
        ...
    def __exit__(
        self, type: type[BaseException] | None, value: BaseException | None, traceback: TracebackType | None, /
    ) -> Literal[False]:
        """For context manager. Non-standard."""
        ...

class Cursor(Iterator[Any]):
    arraysize: int
    @property
    def connection(self) -> Connection: ...
    # May be None, but using | Any instead to avoid slightly annoying false positives.
    @property
    def description(self) -> tuple[tuple[str, None, None, None, None, None, None], ...] | Any: ...
    @property
    def lastrowid(self) -> int | None: ...
    row_factory: Callable[[Cursor, Row], object] | None
    @property
    def rowcount(self) -> int: ...
    def __init__(self, cursor: Connection, /) -> None: ...
    def close(self) -> None:
        """Closes the cursor."""
        ...
    def execute(self, sql: str, parameters: _Parameters = (), /) -> Self:
        """Executes a SQL statement."""
        ...
    def executemany(self, sql: str, seq_of_parameters: Iterable[_Parameters], /) -> Self:
        """Repeatedly executes a SQL statement."""
        ...
    def executescript(self, sql_script: str, /) -> Cursor:
        """Executes a multiple SQL statements at once. Non-standard."""
        ...
    def fetchall(self) -> list[Any]:
        """Fetches all rows from the resultset."""
        ...
    def fetchmany(self, size: int | None = 1) -> list[Any]:
        """Fetches several rows from the resultset."""
        ...
    # Returns either a row (as created by the row_factory) or None, but
    # putting None in the return annotation causes annoying false positives.
    def fetchone(self) -> Any:
        """Fetches one row from the resultset."""
        ...
    def setinputsizes(self, sizes: Unused, /) -> None:
        """Required by DB-API. Does nothing in pysqlite."""
        ...
    def setoutputsize(self, size: Unused, column: Unused = None, /) -> None:
        """Required by DB-API. Does nothing in pysqlite."""
        ...
    def __iter__(self) -> Self:
        """Implement iter(self)."""
        ...
    def __next__(self) -> Any:
        """Implement next(self)."""
        ...

class Error(Exception):
    if sys.version_info >= (3, 11):
        sqlite_errorcode: int
        sqlite_errorname: str

class DatabaseError(Error): ...
class DataError(DatabaseError): ...
class IntegrityError(DatabaseError): ...
class InterfaceError(Error): ...
class InternalError(DatabaseError): ...
class NotSupportedError(DatabaseError): ...
class OperationalError(DatabaseError): ...

if sys.version_info < (3, 10):
    OptimizedUnicode = str

@final
class PrepareProtocol:
    def __init__(self, *args: object, **kwargs: object) -> None: ...

class ProgrammingError(DatabaseError): ...

class Row:
    def __init__(self, cursor: Cursor, data: tuple[Any, ...], /) -> None: ...
    def keys(self) -> list[str]:
        """Returns the keys of the row."""
        ...
    @overload
    def __getitem__(self, key: int | str, /) -> Any:
        """Return self[key]."""
        ...
    @overload
    def __getitem__(self, key: slice, /) -> tuple[Any, ...]:
        """Return self[key]."""
        ...
    def __hash__(self) -> int:
        """Return hash(self)."""
        ...
    def __iter__(self) -> Iterator[Any]:
        """Implement iter(self)."""
        ...
    def __len__(self) -> int:
        """Return len(self)."""
        ...
    # These return NotImplemented for anything that is not a Row.
    def __eq__(self, value: object, /) -> bool:
        """Return self==value."""
        ...
    def __ge__(self, value: object, /) -> bool:
        """Return self>=value."""
        ...
    def __gt__(self, value: object, /) -> bool:
        """Return self>value."""
        ...
    def __le__(self, value: object, /) -> bool:
        """Return self<=value."""
        ...
    def __lt__(self, value: object, /) -> bool:
        """Return self<value."""
        ...
    def __ne__(self, value: object, /) -> bool:
        """Return self!=value."""
        ...

@final
class _Statement: ...

class Warning(Exception): ...

if sys.version_info >= (3, 11):
    @final
    class Blob:
        def close(self) -> None:
            """Close the blob."""
            ...
        def read(self, length: int = -1, /) -> bytes:
            """
            Read data at the current offset position.

              length
                Read length in bytes.

            If the end of the blob is reached, the data up to end of file will be returned.
            When length is not specified, or is negative, Blob.read() will read until the
            end of the blob.
            """
            ...
        def write(self, data: ReadableBuffer, /) -> None:
            """
            Write data at the current offset.

            This function cannot change the blob length.  Writing beyond the end of the
            blob will result in an exception being raised.
            """
            ...
        def tell(self) -> int:
            """Return the current access position for the blob."""
            ...
        # whence must be one of os.SEEK_SET, os.SEEK_CUR, os.SEEK_END
        def seek(self, offset: int, origin: int = 0, /) -> None:
            """
            Set the current access position to offset.

            The origin argument defaults to os.SEEK_SET (absolute blob positioning).
            Other values for origin are os.SEEK_CUR (seek relative to the current position)
            and os.SEEK_END (seek relative to the blob's end).
            """
            ...
        def __len__(self) -> int:
            """Return len(self)."""
            ...
        def __enter__(self) -> Self:
            """Blob context manager enter."""
            ...
        def __exit__(self, type: object, val: object, tb: object, /) -> Literal[False]:
            """Blob context manager exit."""
            ...
        def __getitem__(self, key: SupportsIndex | slice, /) -> int:
            """Return self[key]."""
            ...
        def __setitem__(self, key: SupportsIndex | slice, value: int, /) -> None:
            """Set self[key] to value."""
            ...
