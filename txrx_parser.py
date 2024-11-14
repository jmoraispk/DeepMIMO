from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import re
from typing import Any, Generator, Iterator


RE_BOOL_TRUE = re.compile(r"yes")
RE_BOOL_FALSE = re.compile(r"no")
RE_BEGIN_NODE = re.compile(r"begin_<(?P<node_name>\S*)>")
RE_END_NODE = re.compile(r"end_<(?P<node_name>\S*)>")
RE_INT = re.compile(r"-?\d+")
RE_FLOAT = re.compile(r"-?\d+[.]\d+")
RE_LABEL = re.compile(r"\S+")

_EOF_TOKEN = object()
EOF_TYPE = object


class peekable[T]:
    def __init__(self, iterator: Iterator[T]):
        self.iterator = iterator
        self._sentinel = object()
        self._peek = self._sentinel

    def peek(self) -> T:
        if self._peek is self._sentinel:
            self._peek = next(self.iterator)
        return self._peek

    def __next__(self) -> T:
        if self._peek is not self._sentinel:
            value = self._peek
            self._peek = self._sentinel
            return value
        return next(self.iterator)

    def __iter__(self) -> Iterator[T]:
        return self

    def has_values(self) -> bool:
        if self._peek is self._sentinel:
            try:
                self.peek()
            except StopIteration:
                pass
        return self._peek != self._sentinel


@dataclass
class Node:
    """Node to represent a section delimited by begin_<...> / end_<...>."""

    name: str | None = None
    kind: str | None = None
    values: dict[str, Any] = field(default_factory=dict)
    labels: list[str] = field(default_factory=list)
    data: list[tuple[Any]] = field(default_factory=list)

    def __getitem__(self, key: str) -> Any:
        return self.values.__getitem__(key)

    def __setitem__(self, key: str, value: Any) -> None:
        return self.values.__setitem__(key, value)

    def __delitem__(self, key: str) -> None:
        return self.values.__delitem__(key)


def tokens(path: Path) -> Generator[str | EOF_TYPE, None, None]:
    """Breaks a file into space-separated tokens."""
    with open(path, "r") as f:
        for line in f:
            if line.startswith('Format'): continue # to skip "setup" file
            yield from line.split()
            yield "\n"
    yield _EOF_TOKEN


def eat(tokens: Iterator[str | EOF_TYPE], expected: Any) -> None:
    token = next(tokens)
    if token != expected:
        raise RuntimeError(f"Expected {expected!r}, got {token!r}.")


def parse_document(tokens: Iterator[str | EOF_TYPE]) -> dict[str, Any]:
    """
    document := node* EOF
    node := BEGIN_TAG NL value_lines END_TAG NL
    value_lines := (node | values)*
    values := (label | tuple) NL
    label := STR
    tuple := label | "yes" | "no" | INT | FLOAT
    """

    if not isinstance(tokens, peekable):
        tokens = peekable(tokens)
    document = {}
    while tokens.has_values() and (tok := tokens.peek()) is not _EOF_TOKEN:
        assert isinstance(tok, str)
        if RE_BEGIN_NODE.match(tok):
            node_name, node = parse_node(tokens)
            node.kind = node_name
            # node_name = node.values['project_id'] # How to name the class
            node_name = node.name
            if node_name in document:
                raise RuntimeError(f"Document already contains node {node_name}")
            document[node_name] = node
    return document


def parse_node(tokens: peekable[str | EOF_TYPE]) -> tuple[str, Node]:
    # Parse `begin_<...> XYZ`
    begin_tag = next(tokens)
    assert isinstance(begin_tag, str)
    begin_match = RE_BEGIN_NODE.match(begin_tag)
    assert begin_match is not None
    node_name = begin_match.group("node_name")
    node = Node()
    if tokens.peek() != "\n":
        optional_name = next(tokens)
        assert isinstance(optional_name, str)
        while tokens.peek() != '\n':
            optional_name += next(tokens)
        node.name = optional_name
    eat(tokens, "\n")

    # Parse the values between the two tags.
    for value_line in parse_value_lines(tokens):
        match value_line:
            case (str(label),):
                node.labels.append(label)
            case (str(label), value):
                node.values[label] = value
            case str(label), *rest:
                node.values[label] = rest
            case _:
                node.data.append(value_line)

    # Parse the closing tag.
    eat(tokens, f"end_<{node_name}>")
    eat(tokens, "\n")

    return node_name, node


def parse_value_lines(tokens: peekable[str | EOF_TYPE]) -> list[tuple[Any, ...]]:
    lines: list[tuple[Any, ...]] = []
    while tokens.has_values():
        peek = tokens.peek()
        if peek is _EOF_TOKEN:
            return lines

        assert isinstance(peek, str)
        if RE_END_NODE.match(peek):
            return lines
        elif RE_BEGIN_NODE.match(peek):
            lines.append(parse_node(tokens))
        else:
            lines.append(parse_values(tokens))
    return lines


def parse_values(tokens: peekable[str | EOF_TYPE]) -> tuple[Any, ...]:
    values: list[Any] = []
    while tokens.has_values() and (tok := tokens.peek()) != "\n":
        next(tokens)
        assert isinstance(tok, str)
        if RE_BOOL_TRUE.match(tok):
            values.append(True)
        elif RE_BOOL_FALSE.match(tok):
            values.append(False)
        elif RE_FLOAT.match(tok):
            values.append(float(tok))
        elif RE_INT.match(tok):
            values.append(int(tok))
        else:
            values.append(tok)
    eat(tokens, "\n")
    return tuple(values)


if __name__ == "__main__":
    tks = tokens(Path("simple_street_canyon_test.txrx"))
    document = parse_document(tks)
