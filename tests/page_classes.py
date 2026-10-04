"""Generated pages of the classes seen in real technical books.

Readable pages that trip the weak signals (tables of contents with dot
leaders, fully qualified class names, keys and hashes, regular expressions)
and broken text layers (replacement characters; NUL and other control
characters with almost no spaces). Each function returns the lines of one
page with at least 200 non-whitespace characters.
"""

import random

_TOPICS = (
    "Getting Started with Spring Boot",
    "Configuring Data Sources",
    "Securing the Web Layer",
    "Messaging with Kafka",
    "Testing Slices and Mocks",
    "Observability and Metrics",
    "Packaging and Deployment",
    "Reactive Endpoints",
)


def toc_glued(seed: int = 0) -> list[str]:
    """Table of contents whose dot leaders are glued to title and page."""
    return [
        f"{seed + i}.{i} {_TOPICS[(seed + i) % len(_TOPICS)]}"
        + "." * (60 - len(_TOPICS[(seed + i) % len(_TOPICS)]))
        + f"{10 + 7 * i}"
        for i in range(12)
    ]


def toc_spaced(seed: int = 0) -> list[str]:
    """Table of contents whose dot leaders are separated by spaces."""
    return [
        f"{seed + i}.{i} {_TOPICS[(seed + i) % len(_TOPICS)]} "
        + " .".join("" for _ in range(25))
        + f" {10 + 7 * i}"
        for i in range(12)
    ]


def class_names(seed: int = 0) -> list[str]:
    """Fully qualified class names and package paths, as in code listings."""
    parts = (
        "org.springframework.boot.autoconfigure.web.servlet",
        "com.example.inventory.persistence.repository.jpa",
        "io.micrometer.observation.contextpropagation.support",
        "org.apache.kafka.clients.consumer.internals.events",
    )
    names = (
        "WebMvcAutoConfiguration",
        "InventoryItemRepositoryImpl",
        "ObservationThreadLocalAccessor",
        "ApplicationEventProcessor",
    )
    return [
        f"import {parts[(seed + i) % 4]}.{names[(seed + i) % 4]};" for i in range(10)
    ]


def keys_and_hashes(seed: int = 0) -> list[str]:
    """Hex hashes, base58 addresses and keys up to 135 characters."""
    rng = random.Random(seed)
    hexdigits = "0123456789abcdef"
    base58 = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"
    lines = []
    for i in range(4):
        lines.append(
            "Transaction hash: " + "".join(rng.choice(hexdigits) for _ in range(64))
        )
        lines.append("Address: 1" + "".join(rng.choice(base58) for _ in range(33)))
        lines.append("xprv" + "".join(rng.choice(base58) for _ in range(131 - 4 * i)))
    return lines


def regular_expressions(seed: int = 0) -> list[str]:
    """A page of regular expressions: mostly symbols and spaces."""
    patterns = (
        r"^\s*(?:[-+]?\d+(?:\.\d*)?|\.\d+)\s*$",
        r"(?<=\b)(\w+)\s+\1\b",
        r"[^\s@]+@[^\s@]+\.[^\s@]{2,}",
        r"^(?!.*\.\.)[\w.%+-]+$",
        r"\{\{\s*(\w+)\s*\}\}",
        r"(?:\\[nrt\\]|[^\\\"])*",
        r"<([a-z]+)[^>]*>.*?</\1>",
        r"\$\{([^}:]+)(?::-([^}]*))?\}",
    )
    return [
        f"$ grep -P '{patterns[(seed + i) % len(patterns)]}' f{i}" for i in range(10)
    ]


def replacement_characters(seed: int = 0) -> list[str]:
    """A page of mostly U+FFFD, as from a font without a Unicode mapping."""
    return ["\ufffd" * (20 + (seed + i) % 7) + " ab " + "\ufffd" * 30 for i in range(8)]


def control_characters(seed: int = 0) -> list[str]:
    """NUL and other control characters among letters, almost no spaces."""
    controls = "\x00\x01\x0e\x10\x11"
    letters = "abcdefghijklmnopqrstuvwxyz"
    return [
        "".join(
            controls[(seed + step) % len(controls)]
            if step % 3 == 0
            else letters[(seed * 7 + line + step) % 26]
            for step in range(70)
        )
        for line in range(6)
    ]


READABLE = {
    "toc with glued dot leaders": toc_glued,
    "toc with spaced dots": toc_spaced,
    "fully qualified class names": class_names,
    "keys and hashes": keys_and_hashes,
    "regular expressions": regular_expressions,
}
BROKEN = {
    "replacement characters": replacement_characters,
    "control characters": control_characters,
}
