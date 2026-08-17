"""Gold: Databricks Vector Search endpoint + Delta Sync index.

Index type choice, because it is the decision that determines the operating
model:

``DELTA_SYNC`` with ``embedding_source_column`` (used here)
    Databricks reads the Change Data Feed of the Gold table, calls the
    embedding model endpoint itself, and keeps the index current. No embedding
    code, no embedding job, no drift between what is in Delta and what is in
    the index.

``DELTA_SYNC`` with ``embedding_vector_column``
    You compute the vectors. Only worth it for a bespoke or fine-tuned model
    that is not served as a Databricks endpoint.

``DIRECT_ACCESS``
    You own writes AND deletes. Choose it only when the source of truth is not
    a Delta table.

Pipeline mode ``TRIGGERED`` costs the sync job's compute per run; ``CONTINUOUS``
keeps compute warm for seconds-level freshness. Real-time retrieval over a
streaming corpus is what CONTINUOUS is for -- default to it in prod, TRIGGERED
in dev.

Run:
    python -m pipelines.gold.vector_index --conf conf/dev.json --wait
"""

from __future__ import annotations

import argparse
import logging
import time
from typing import Any

from ..common.config import PipelineConfig
from ..common.logging_utils import configure_logging, stage

logger = logging.getLogger(__name__)

PIPELINE_NAME = "gold_vector_index"
SOURCE_TABLE = "document_context"
PRIMARY_KEY = "chunk_id"
EMBEDDING_SOURCE_COLUMN = "chunk_text"

#: Columns kept alongside the vector so retrieval can filter without a second
#: round trip to Delta.
SYNCED_COLUMNS = [
    "chunk_id",
    "document_id",
    "chunk_index",
    "chunk_text",
    "symbol",
    "sector",
    "source_path",
    "published_date",
]

ENDPOINT_READY_STATES = {"ONLINE", "PROVISIONING"}
_POLL_SECONDS = 30
_DEFAULT_TIMEOUT_SECONDS = 3600


def get_client():
    """Vector Search client, authenticated from the notebook/job context."""
    from databricks.vector_search.client import VectorSearchClient

    # No credentials passed: on a Databricks cluster the client picks up the
    # job's service principal. Passing a PAT here is how tokens end up in logs.
    return VectorSearchClient(disable_notice=True)


def ensure_endpoint(client, endpoint_name: str, *, endpoint_type: str = "STANDARD") -> dict:
    """Create the endpoint if it does not exist; return its description."""
    try:
        endpoint = client.get_endpoint(endpoint_name)
        state = (endpoint.get("endpoint_status") or {}).get("state", "UNKNOWN")
        logger.info("endpoint.exists", extra={"endpoint": endpoint_name, "state": state})
        return endpoint
    except Exception:
        logger.info("endpoint.creating", extra={"endpoint": endpoint_name})
        return client.create_endpoint(name=endpoint_name, endpoint_type=endpoint_type)


def wait_for_endpoint(
    client, endpoint_name: str, timeout_seconds: int = _DEFAULT_TIMEOUT_SECONDS
) -> None:
    """Block until the endpoint is ONLINE. Endpoint creation takes minutes."""
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        endpoint = client.get_endpoint(endpoint_name)
        state = (endpoint.get("endpoint_status") or {}).get("state", "UNKNOWN")
        if state == "ONLINE":
            logger.info("endpoint.online", extra={"endpoint": endpoint_name})
            return
        if state not in ENDPOINT_READY_STATES:
            raise RuntimeError(f"Vector Search endpoint {endpoint_name} entered state {state}")
        time.sleep(_POLL_SECONDS)
    raise TimeoutError(f"Endpoint {endpoint_name} was not ONLINE within {timeout_seconds}s")


def ensure_index(
    client,
    config: PipelineConfig,
    *,
    source_table: str,
    index_name: str,
    pipeline_type: str,
) -> Any:
    """Create or return the Delta Sync index over ``source_table``."""
    try:
        index = client.get_index(endpoint_name=config.vector_endpoint, index_name=index_name)
        logger.info("index.exists", extra={"index": index_name})
        return index
    except Exception:
        logger.info(
            "index.creating",
            extra={
                "index": index_name,
                "source": source_table,
                "pipeline_type": pipeline_type,
                "embedding_model": config.embedding_model_endpoint,
            },
        )
        return client.create_delta_sync_index(
            endpoint_name=config.vector_endpoint,
            index_name=index_name,
            source_table_name=source_table,
            pipeline_type=pipeline_type,
            primary_key=PRIMARY_KEY,
            # Databricks embeds for us; see the module docstring.
            embedding_source_column=EMBEDDING_SOURCE_COLUMN,
            embedding_model_endpoint_name=config.embedding_model_endpoint,
            columns_to_sync=SYNCED_COLUMNS,
        )


def sync_index(index) -> None:
    """Kick a TRIGGERED index. A no-op for CONTINUOUS indexes."""
    try:
        index.sync()
        logger.info("index.sync.requested")
    except Exception as exc:  # CONTINUOUS indexes reject an explicit sync
        logger.info("index.sync.skipped", extra={"reason": str(exc)})


def wait_for_index(index, timeout_seconds: int = _DEFAULT_TIMEOUT_SECONDS) -> dict:
    """Block until the initial index build reports READY."""
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        status = index.describe().get("status", {})
        if status.get("ready"):
            logger.info(
                "index.ready",
                extra={"indexed_rows": status.get("indexed_row_count")},
            )
            return status
        logger.info(
            "index.building",
            extra={"detail": status.get("detailed_state") or status.get("message")},
        )
        time.sleep(_POLL_SECONDS)
    raise TimeoutError(f"Index was not READY within {timeout_seconds}s")


def similarity_search(
    client,
    config: PipelineConfig,
    query: str,
    *,
    index_name: str | None = None,
    num_results: int = 5,
    filters: dict | None = None,
) -> list[dict]:
    """Retrieve the nearest chunks. Used by the smoke test and by serving.

    Keeping retrieval in the same module as indexing means the column contract
    is defined once: if ``SYNCED_COLUMNS`` changes, this returns the new
    columns without a second edit somewhere else.
    """
    index_name = index_name or config.index_name(config.gold(SOURCE_TABLE))
    index = client.get_index(endpoint_name=config.vector_endpoint, index_name=index_name)
    response = index.similarity_search(
        query_text=query,
        columns=SYNCED_COLUMNS,
        num_results=num_results,
        filters=filters or {},
    )
    data = (response.get("result") or {}).get("data_array") or []
    columns = [c["name"] for c in (response.get("manifest") or {}).get("columns", [])]
    # strict=False: the API appends a trailing similarity score to each row that
    # has no corresponding entry in the manifest columns.
    return [dict(zip(columns, row, strict=False)) for row in data]


def run(config: PipelineConfig, *, wait: bool = False, pipeline_type: str | None = None) -> None:
    endpoint = config.vector_endpoint
    if not endpoint:
        raise ValueError("vector_endpoint is not configured")

    pipeline_type = pipeline_type or ("CONTINUOUS" if config.environment == "prod" else "TRIGGERED")
    source_table = config.gold(SOURCE_TABLE)
    index_name = config.index_name(source_table)

    with stage(logger, PIPELINE_NAME, endpoint=endpoint, index=index_name):
        client = get_client()
        ensure_endpoint(client, endpoint)
        if wait:
            wait_for_endpoint(client, endpoint)

        index = ensure_index(
            client,
            config,
            source_table=source_table,
            index_name=index_name,
            pipeline_type=pipeline_type,
        )

        if pipeline_type == "TRIGGERED":
            sync_index(index)
        if wait:
            wait_for_index(index)


def main() -> None:
    configure_logging()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--conf", dest="conf", default=None)
    parser.add_argument(
        "--pipeline-type",
        choices=["TRIGGERED", "CONTINUOUS"],
        default=None,
        help="Defaults to CONTINUOUS in prod, TRIGGERED elsewhere.",
    )
    parser.add_argument(
        "--wait", action="store_true", help="Block until the endpoint and index are ready."
    )
    args = parser.parse_args()

    config = PipelineConfig.resolve(["--conf", args.conf] if args.conf else [])
    run(config, wait=args.wait, pipeline_type=args.pipeline_type)


if __name__ == "__main__":
    main()
