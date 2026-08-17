-- ===========================================================================
-- 04 - Audit, lineage and governance assertions
--
-- Everything here reads Unity Catalog's own system tables. They are the only
-- source that cannot be bypassed by a job that "forgot" to log.
--
-- Requires: system.access and system.billing schemas enabled on the metastore
--   (one-off, per metastore, via the account console or the SystemSchemas API).
-- ===========================================================================

USE CATALOG ${catalog};
USE SCHEMA governance;

-- ---------------------------------------------------------------------------
-- Who read what, over the last 30 days.
-- ---------------------------------------------------------------------------

CREATE OR REPLACE VIEW governance.v_recent_table_access
COMMENT 'Read access to this catalog over the last 30 days, from system.access.audit.'
AS
SELECT
  event_time,
  user_identity.email                                   AS principal,
  action_name,
  request_params.full_name_arg                          AS securable,
  request_params.operation                              AS operation,
  source_ip_address
FROM system.access.audit
WHERE service_name = 'unityCatalog'
  AND action_name IN ('getTable', 'generateTemporaryTableCredential', 'createTable', 'deleteTable')
  AND event_date >= current_date() - INTERVAL 30 DAYS
  AND request_params.full_name_arg LIKE '${catalog}.%';

-- ---------------------------------------------------------------------------
-- Access to PII-bearing columns specifically. This is the report a regulator
-- asks for, and the reason the registry table exists.
-- ---------------------------------------------------------------------------

CREATE OR REPLACE VIEW governance.v_pii_access_log
COMMENT 'Every read of a table that carries at least one registered PII column.'
AS
SELECT
  a.event_time,
  a.principal,
  a.securable,
  r.column_name,
  r.pii_class,
  r.masking_policy,
  -- Membership is evaluated at query time, so this column answers "was the
  -- value actually visible", not merely "was the table touched".
  is_account_group_member('pii_readers') AS reader_is_privileged
FROM governance.v_recent_table_access a
JOIN governance.pii_column_registry r
  ON a.securable = concat_ws('.', r.catalog_name, r.schema_name, r.table_name);

-- ---------------------------------------------------------------------------
-- Lineage. system.access.table_lineage answers "if I change Bronze, what
-- breaks" without anyone maintaining a diagram.
-- ---------------------------------------------------------------------------

CREATE OR REPLACE VIEW governance.v_table_lineage
COMMENT 'Upstream/downstream table dependencies within the catalog.'
AS
SELECT DISTINCT
  source_table_full_name  AS upstream_table,
  target_table_full_name  AS downstream_table,
  entity_type,
  max(event_time)         AS last_seen
FROM system.access.table_lineage
WHERE (source_table_full_name LIKE '${catalog}.%' OR target_table_full_name LIKE '${catalog}.%')
  AND event_date >= current_date() - INTERVAL 90 DAYS
GROUP BY ALL;

-- ---------------------------------------------------------------------------
-- Governance drift assertions.
--
-- Every one of these is written to return ZERO ROWS when healthy, so CI can
-- assert on row count rather than parsing output.
-- ---------------------------------------------------------------------------

-- A1: a column is registered as PII but carries no mask binding.
CREATE OR REPLACE VIEW governance.v_assert_pii_columns_masked
COMMENT 'ASSERTION: must be empty. Registered PII columns that are not masked.'
AS
SELECT
  r.catalog_name, r.schema_name, r.table_name, r.column_name, r.masking_policy,
  'registered as PII but no mask is bound' AS violation
FROM governance.pii_column_registry r
LEFT JOIN system.information_schema.column_masks m
  ON  m.catalog_name = r.catalog_name
  AND m.schema_name  = r.schema_name
  AND m.table_name   = r.table_name
  AND m.column_name  = r.column_name
WHERE m.column_name IS NULL
  -- Tables that have not been created yet are not violations.
  AND EXISTS (
    SELECT 1 FROM system.information_schema.tables t
    WHERE t.table_catalog = r.catalog_name
      AND t.table_schema  = r.schema_name
      AND t.table_name    = r.table_name
  );

-- A2: a securable is owned by an individual rather than a group. Ownership by
-- a person is an offboarding incident waiting to happen.
CREATE OR REPLACE VIEW governance.v_assert_group_ownership
COMMENT 'ASSERTION: must be empty. Objects owned by a user instead of a group.'
AS
SELECT table_catalog, table_schema, table_name, table_owner,
       'owned by an individual, not a group' AS violation
FROM system.information_schema.tables
WHERE table_catalog = '${catalog}'
  AND table_owner LIKE '%@%';

-- A3: privileges granted directly to a user.
CREATE OR REPLACE VIEW governance.v_assert_no_direct_user_grants
COMMENT 'ASSERTION: must be empty. Privileges granted to users rather than groups.'
AS
SELECT grantor, grantee, table_catalog, table_schema, table_name, privilege_type,
       'privilege granted to a user, not a group' AS violation
FROM system.information_schema.table_privileges
WHERE table_catalog = '${catalog}'
  AND grantee LIKE '%@%';

-- A4 (Change Data Feed on the vector-index source) cannot be a view: table
-- properties are only reachable through SHOW TBLPROPERTIES / DESCRIBE DETAIL,
-- neither of which is allowed in a view body. It is asserted instead by
-- `python governance/apply.py --assert`, which runs:
--
--   SHOW TBLPROPERTIES gold.document_context ('delta.enableChangeDataFeed')
--
-- and fails the deploy when the value is not 'true'. Losing CDF is silent --
-- the index keeps serving stale answers rather than erroring -- which is
-- exactly why it is checked on every deploy.

-- ---------------------------------------------------------------------------
-- Cost attribution: which pipeline is actually spending the money.
-- ---------------------------------------------------------------------------

CREATE OR REPLACE VIEW governance.v_pipeline_cost_30d
COMMENT 'DBU consumption by job over the last 30 days, tagged by pipeline.'
AS
SELECT
  usage_metadata.job_name                    AS job_name,
  custom_tags['pipeline']                    AS pipeline,
  date_trunc('DAY', usage_start_time)        AS usage_day,
  sum(usage_quantity)                        AS dbus
FROM system.billing.usage
WHERE usage_start_time >= current_date() - INTERVAL 30 DAYS
  AND custom_tags['project'] = 'genai-lakehouse'
GROUP BY ALL;

GRANT SELECT ON VIEW governance.v_recent_table_access          TO `auditors`;
GRANT SELECT ON VIEW governance.v_pii_access_log               TO `auditors`;
GRANT SELECT ON VIEW governance.v_table_lineage                TO `auditors`;
GRANT SELECT ON VIEW governance.v_assert_pii_columns_masked    TO `auditors`;
GRANT SELECT ON VIEW governance.v_assert_group_ownership       TO `auditors`;
GRANT SELECT ON VIEW governance.v_assert_no_direct_user_grants TO `auditors`;
GRANT SELECT ON VIEW governance.v_pipeline_cost_30d            TO `data_platform_admins`;
