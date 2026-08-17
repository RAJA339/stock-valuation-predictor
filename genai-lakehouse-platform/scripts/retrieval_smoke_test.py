"""Post-deploy smoke test: does retrieval actually return anything?

A green Terraform apply and a READY index still do not prove the platform
works. This asks the deployed index a real question and asserts the shape of
what comes back -- the cheapest possible end-to-end check that Bronze ->
Silver -> Gold -> embeddings -> retrieval is connected.

    python scripts/retrieval_smoke_test.py --conf conf/dev.json

Exit codes: 0 healthy, 1 unhealthy, 2 misconfigured.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipelines.common.config import ConfigError, PipelineConfig  # noqa: E402
from pipelines.gold.vector_index import (  # noqa: E402
    SYNCED_COLUMNS,
    get_client,
    similarity_search,
)

logger = logging.getLogger("smoke")

DEFAULT_QUERIES = (
    "quarterly revenue growth",
    "risk factors and regulatory exposure",
)


def check(config: PipelineConfig, queries: tuple[str, ...], min_results: int) -> list[str]:
    """Return a list of failures. Empty means the deployment is healthy."""
    client = get_client()
    failures: list[str] = []

    for query in queries:
        try:
            hits = similarity_search(client, config, query, num_results=min_results)
        except Exception as exc:
            failures.append(f"query {query!r} raised {type(exc).__name__}: {exc}")
            continue

        if len(hits) < min_results:
            failures.append(f"query {query!r} returned {len(hits)} hits, expected >= {min_results}")
            continue

        missing = [c for c in SYNCED_COLUMNS if c not in hits[0]]
        if missing:
            # A hit without its metadata columns cannot be filtered or cited,
            # which makes it useless to the application even though the
            # similarity search "worked".
            failures.append(f"query {query!r}: hit is missing columns {missing}")

        empty = [h for h in hits if not (h.get("chunk_text") or "").strip()]
        if empty:
            failures.append(f"query {query!r}: {len(empty)} hit(s) have empty chunk_text")

        logger.info("query %r -> %d hits", query, len(hits))

    return failures


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)-8s %(message)s")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--conf", default=None)
    parser.add_argument("--query", action="append", default=None)
    parser.add_argument("--min-results", type=int, default=1)
    args = parser.parse_args(argv)

    try:
        config = PipelineConfig.resolve(["--conf", args.conf] if args.conf else [])
    except (ConfigError, FileNotFoundError) as exc:
        logger.error("configuration problem: %s", exc)
        return 2

    queries = tuple(args.query) if args.query else DEFAULT_QUERIES
    failures = check(config, queries, args.min_results)

    if failures:
        for failure in failures:
            logger.error("smoke test failed: %s", failure)
        return 1

    logger.info("retrieval smoke test passed against %s", config.vector_endpoint)
    return 0


if __name__ == "__main__":
    sys.exit(main())
