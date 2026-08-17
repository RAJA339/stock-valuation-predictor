"""Apply the governance SQL to a Unity Catalog metastore and assert on drift.

Run against a SQL warehouse:

    python governance/apply.py \\
        --catalog genai_lakehouse_dev \\
        --environment dev \\
        --warehouse-id abc123 \\
        --assert

``--dry-run`` renders the substituted SQL to stdout without connecting, which
is what CI uses on a pull request: the statements are reviewable in the diff
before anyone has permission to run them.

Statement splitting is deliberately simple (semicolon at the end of a line
outside a string/comment). Do not put a multi-statement body inside a single
``CREATE FUNCTION`` here -- keep one statement per semicolon and it stays
correct.
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import sys
from pathlib import Path

logger = logging.getLogger("governance")

SCRIPT_DIR = Path(__file__).parent

#: Applied in order. Numbering is the dependency order, not decoration:
#: grants reference schemas, masks reference tables, audits reference both.
SCRIPTS = (
    "01_catalog_and_schemas.sql",
    "02_rbac_grants.sql",
    "03_pii_masking.sql",
    "04_audit_and_lineage.sql",
)

#: Views whose result set must be empty for the deploy to be considered clean.
ASSERTION_VIEWS = (
    "v_assert_pii_columns_masked",
    "v_assert_group_ownership",
    "v_assert_no_direct_user_grants",
)

#: (table, property, expected) triples checked with SHOW TBLPROPERTIES.
PROPERTY_ASSERTIONS = (
    ("gold.document_context", "delta.enableChangeDataFeed", "true"),
    ("silver.instrument_dim", "delta.enableChangeDataFeed", "true"),
)

_PLACEHOLDER_RE = re.compile(r"\$\{(\w+)\}")
_COMMENT_LINE_RE = re.compile(r"^\s*--")


class GovernanceError(RuntimeError):
    """Raised when a governance assertion fails."""


def render(sql: str, params: dict[str, str]) -> str:
    """Substitute ``${name}`` placeholders, failing loudly on a missing one.

    A silent empty substitution here would produce SQL that runs against the
    wrong catalog, which is far worse than an error.
    """
    missing = {m.group(1) for m in _PLACEHOLDER_RE.finditer(sql)} - params.keys()
    if missing:
        raise GovernanceError(f"missing SQL parameters: {sorted(missing)}")
    return _PLACEHOLDER_RE.sub(lambda m: params[m.group(1)], sql)


def split_statements(sql: str) -> list[str]:
    """Split a script into individual statements on top-level semicolons."""
    statements: list[str] = []
    buffer: list[str] = []
    for line in sql.splitlines():
        if _COMMENT_LINE_RE.match(line) and not buffer:
            continue
        buffer.append(line)
        if line.rstrip().endswith(";"):
            statement = "\n".join(buffer).strip().rstrip(";").strip()
            if statement:
                statements.append(statement)
            buffer = []
    tail = "\n".join(buffer).strip()
    if tail:
        statements.append(tail)
    return statements


def load_scripts(params: dict[str, str]) -> list[tuple[str, list[str]]]:
    """Read, render and split every governance script, in order."""
    out = []
    for name in SCRIPTS:
        path = SCRIPT_DIR / name
        if not path.exists():
            raise GovernanceError(f"missing governance script: {path}")
        out.append((name, split_statements(render(path.read_text(encoding="utf-8"), params))))
    return out


def _connect(server_hostname: str, http_path: str, token: str):
    from databricks import sql as dbsql

    return dbsql.connect(
        server_hostname=server_hostname,
        http_path=http_path,
        access_token=token,
    )


def apply_scripts(cursor, scripts: list[tuple[str, list[str]]]) -> int:
    applied = 0
    for name, statements in scripts:
        logger.info("applying %s (%d statements)", name, len(statements))
        for i, statement in enumerate(statements, start=1):
            try:
                cursor.execute(statement)
                applied += 1
            except Exception as exc:
                head = statement.splitlines()[0][:120]
                raise GovernanceError(f"{name} statement {i} failed ({head!r}): {exc}") from exc
    return applied


def run_assertions(cursor, catalog: str) -> list[str]:
    """Return a list of human-readable violations. Empty means healthy."""
    violations: list[str] = []

    for view in ASSERTION_VIEWS:
        cursor.execute(f"SELECT count(*) FROM {catalog}.governance.{view}")
        count = cursor.fetchone()[0]
        if count:
            cursor.execute(f"SELECT * FROM {catalog}.governance.{view} LIMIT 5")
            sample = cursor.fetchall()
            violations.append(f"{view}: {count} violation(s); first rows: {sample}")

    for table, prop, expected in PROPERTY_ASSERTIONS:
        cursor.execute(f"SHOW TBLPROPERTIES {catalog}.{table} ('{prop}')")
        rows = cursor.fetchall()
        value = str(rows[0][-1]).lower() if rows else "<unset>"
        if value != expected:
            violations.append(f"{table}: {prop} is {value!r}, expected {expected!r}")

    return violations


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)-8s %(message)s")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", required=True)
    parser.add_argument("--environment", required=True, choices=["dev", "staging", "prod"])
    parser.add_argument("--warehouse-id", default=os.environ.get("DATABRICKS_WAREHOUSE_ID"))
    parser.add_argument("--host", default=os.environ.get("DATABRICKS_HOST"))
    parser.add_argument("--token", default=os.environ.get("DATABRICKS_TOKEN"))
    parser.add_argument(
        "--pii-salt",
        default=os.environ.get("PII_HASH_SALT", ""),
        help="Salt for pseudonymising names. Must be stable per environment.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Render SQL, do not connect.")
    parser.add_argument("--assert", dest="do_assert", action="store_true")
    args = parser.parse_args(argv)

    if not args.dry_run and not args.pii_salt:
        # A rotating salt silently breaks every pseudonymous join; an empty one
        # makes the hashes guessable. Neither is acceptable outside a dry run.
        parser.error("--pii-salt (or PII_HASH_SALT) is required unless --dry-run is set")

    params = {
        "catalog": args.catalog,
        "environment": args.environment,
        "pii_salt": args.pii_salt or "DRY_RUN_SALT",
    }

    scripts = load_scripts(params)

    if args.dry_run:
        for name, statements in scripts:
            print(f"\n-- ===== {name} ({len(statements)} statements) =====")
            for statement in statements:
                print(f"{statement};\n")
        return 0

    missing = [n for n, v in (("--host", args.host), ("--token", args.token)) if not v]
    if missing or not args.warehouse_id:
        parser.error(f"missing connection settings: {missing or ['--warehouse-id']}")

    hostname = args.host.replace("https://", "").rstrip("/")
    http_path = f"/sql/1.0/warehouses/{args.warehouse_id}"

    with _connect(hostname, http_path, args.token) as connection:
        with connection.cursor() as cursor:
            applied = apply_scripts(cursor, scripts)
            logger.info("applied %d statements to %s", applied, args.catalog)

            if args.do_assert:
                violations = run_assertions(cursor, args.catalog)
                if violations:
                    for violation in violations:
                        logger.error("governance assertion failed: %s", violation)
                    return 1
                logger.info("all governance assertions passed")

    return 0


if __name__ == "__main__":
    sys.exit(main())
