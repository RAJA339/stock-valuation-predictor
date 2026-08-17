"""Governance SQL rendering and static policy checks.

No warehouse is involved: these tests assert that the SQL is well-formed,
fully parameterised, and that the policy invariants the platform claims to
enforce are actually present in the scripts. A governance file that silently
stopped masking a column would otherwise only be noticed by an audit.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from governance.apply import (
    ASSERTION_VIEWS,
    SCRIPTS,
    GovernanceError,
    load_scripts,
    render,
    split_statements,
)

GOVERNANCE_DIR = Path(__file__).resolve().parents[1] / "governance"
PARAMS = {"catalog": "genai_lakehouse_test", "environment": "dev", "pii_salt": "s3cr3t"}


@pytest.fixture(scope="module")
def rendered() -> dict[str, str]:
    return {
        name: render((GOVERNANCE_DIR / name).read_text(encoding="utf-8"), PARAMS)
        for name in SCRIPTS
    }


def _read(name: str) -> str:
    return (GOVERNANCE_DIR / name).read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def rbac() -> str:
    return _read("02_rbac_grants.sql")


@pytest.fixture(scope="module")
def masking() -> str:
    return _read("03_pii_masking.sql")


@pytest.fixture(scope="module")
def audit() -> str:
    return _read("04_audit_and_lineage.sql")


class TestRender:
    def test_substitutes_every_placeholder(self):
        assert render("USE CATALOG ${catalog};", PARAMS) == "USE CATALOG genai_lakehouse_test;"

    def test_missing_parameter_is_an_error(self):
        # Substituting an unknown placeholder with '' would produce SQL that
        # runs against the wrong catalog -- far worse than failing here.
        with pytest.raises(GovernanceError, match="missing SQL parameters"):
            render("USE CATALOG ${nope};", PARAMS)

    def test_leaves_ordinary_sql_untouched(self):
        sql = "SELECT * FROM t WHERE x = 'a';"
        assert render(sql, PARAMS) == sql


class TestSplitStatements:
    def test_splits_on_trailing_semicolons(self):
        assert len(split_statements("SELECT 1;\nSELECT 2;\n")) == 2

    def test_drops_leading_comment_blocks(self):
        statements = split_statements("-- a comment\n-- another\nSELECT 1;")
        assert statements == ["SELECT 1"]

    def test_keeps_multi_line_statements_intact(self):
        statements = split_statements("CREATE TABLE t (\n  a INT,\n  b INT\n);")
        assert len(statements) == 1
        assert "b INT" in statements[0]

    def test_trailing_statement_without_semicolon_is_kept(self):
        assert split_statements("SELECT 1") == ["SELECT 1"]

    def test_empty_input_yields_nothing(self):
        assert split_statements("\n\n  \n") == []


class TestScriptsAreLoadable:
    def test_every_declared_script_exists_and_parses(self):
        scripts = load_scripts(PARAMS)
        assert [name for name, _ in scripts] == list(SCRIPTS)
        for name, statements in scripts:
            assert statements, f"{name} produced no statements"

    def test_no_placeholder_survives_rendering(self, rendered):
        for name, sql in rendered.items():
            assert "${" not in sql, f"{name} still contains an unsubstituted placeholder"

    def test_scripts_are_ordered_by_dependency(self):
        # Grants reference schemas; masks reference tables; audits reference
        # both. The numeric prefix IS the dependency order.
        assert list(SCRIPTS) == sorted(SCRIPTS)


class TestRbacPolicy:
    def test_analysts_never_get_bronze_access(self, rbac):
        # Anchored to the start of a line: 'REVOKE ALL PRIVILEGES ON SCHEMA
        # bronze FROM `data_analysts`' contains the substring GRANT and is the
        # opposite of a violation.
        grants = re.findall(
            r"^GRANT[^;]+ON SCHEMA bronze[^;]+;", rbac, re.IGNORECASE | re.DOTALL | re.MULTILINE
        )
        for grant in grants:
            assert "data_analysts" not in grant, "raw, unvalidated data must not reach analysts"

    def test_analysts_cannot_modify_gold(self, rbac):
        gold_grants = re.findall(
            r"GRANT ([^;]+) ON SCHEMA gold TO `data_analysts`", rbac, re.IGNORECASE
        )
        for privileges in gold_grants:
            assert "MODIFY" not in privileges.upper()

    def test_service_principal_is_scoped_to_named_tables(self, rbac):
        # A schema-level grant would silently expose every future Gold table.
        assert "GRANT SELECT ON TABLE gold.document_context      TO `genai_app_service`;" in rbac
        schema_grants = re.findall(
            r"GRANT ([^;]*SELECT[^;]*) ON SCHEMA gold TO `genai_app_service`", rbac, re.IGNORECASE
        )
        assert not schema_grants

    def test_ownership_is_always_a_group(self, rbac):
        for owner in re.findall(r"OWNER TO `([^`]+)`", rbac):
            assert "@" not in owner, f"{owner} looks like a user; ownership must be a group"

    def test_every_role_gets_use_catalog(self, rbac):
        roles = {
            "data_engineers",
            "data_analysts",
            "data_scientists",
            "genai_app_service",
            "auditors",
        }
        granted = set(re.findall(r"GRANT USE CATALOG ON CATALOG \S+ TO `([^`]+)`", rbac))
        assert roles <= granted


class TestMaskingPolicy:
    def test_every_mask_function_checks_group_membership(self, masking):
        bodies = re.split(r"CREATE OR REPLACE FUNCTION ", masking)[1:]
        for body in bodies:
            name = body.split("(")[0].strip()
            if not name.startswith("governance.mask") and not name.startswith("governance.redact"):
                continue
            assert "is_account_group_member" in body, f"{name} does not check the caller"

    def test_every_mask_function_passes_null_through(self, masking):
        # A mask that turns NULL into a redaction string breaks IS NULL
        # predicates and every outer join downstream.
        bodies = re.split(r"CREATE OR REPLACE FUNCTION ", masking)[1:]
        for body in bodies:
            name = body.split("(")[0].strip()
            if "mask" not in name and "redact" not in name:
                continue
            assert "IS NULL THEN NULL" in body, f"{name} does not pass NULL through"

    def test_national_id_is_never_partially_revealed(self, masking):
        body = masking.split("FUNCTION governance.mask_national_id")[1].split(";")[0]
        assert "right(" not in body.lower(), "last-four of a national id is still identifying"
        assert "REDACTED" in body

    def test_name_pseudonymisation_is_salted(self, masking):
        body = masking.split("FUNCTION governance.mask_person_name")[1].split(";")[0]
        assert "sha2(" in body
        assert "${pii_salt}" in body, "an unsalted hash is rainbow-table reversible"

    def test_masks_are_bound_to_the_registered_columns(self, masking):
        # One row of the registry MERGE: (catalog, schema, table, column, class, policy)
        registered = re.findall(
            r"\('\$\{catalog\}',\s*'(\w+)',\s*'(\w+)',\s*'(\w+)',\s*'(\w+)',\s*'([\w.]+)'\)",
            masking,
        )
        assert registered, "the PII registry MERGE lost its VALUES block"

        bound = set(re.findall(r"ALTER COLUMN (\w+) SET MASK (\S+?);", masking))
        bound_columns = {column for column, _ in bound}
        bound_policies = dict(bound)

        for _schema, _table, column, _pii_class, policy in registered:
            assert column in bound_columns, f"{column} is registered as PII but never masked"
            # The registry must not drift from the binding: an audit report
            # naming the wrong function is worse than no report.
            assert bound_policies[column] == policy, (
                f"{column} is registered as masked by {policy} but bound to "
                f"{bound_policies[column]}"
            )

    def test_free_text_redaction_covers_the_common_identifiers(self, masking):
        body = masking.split("FUNCTION governance.redact_free_text")[1].split("END;")[0]
        for token in ("[EMAIL]", "[PHONE]", "[NATIONAL_ID]"):
            assert token in body


class TestAuditAssertions:
    def test_every_assertion_view_is_defined(self, audit):
        for view in ASSERTION_VIEWS:
            assert f"CREATE OR REPLACE VIEW governance.{view}" in audit

    def test_assertion_views_are_documented_as_must_be_empty(self, audit):
        for view in ASSERTION_VIEWS:
            block = audit.split(f"VIEW governance.{view}")[1][:300]
            assert "ASSERTION: must be empty" in block

    def test_audit_views_are_readable_by_auditors(self, audit):
        assert (
            "GRANT SELECT ON VIEW governance.v_pii_access_log               TO `auditors`;" in audit
        )


class TestMaintenance:
    def test_vacuum_never_drops_below_the_safe_retention(self):
        sql = (GOVERNANCE_DIR / "05_maintenance.sql").read_text(encoding="utf-8")
        retentions = [int(h) for h in re.findall(r"RETAIN (\d+) HOURS", sql)]
        assert retentions, "no VACUUM statements found"
        # Below 168h breaks concurrent readers and any time travel an incident
        # investigation would need.
        assert all(h >= 168 for h in retentions)
