-- ===========================================================================
-- 01 - Catalog, schemas and object-level defaults
--
-- Terraform creates the catalog, schemas and volumes. This script is the
-- idempotent SQL companion: it is safe to run on every deploy and is what
-- guarantees the tables, tags and properties exist even when a table was
-- created by a pipeline rather than by Terraform.
--
-- Parameterised with ${catalog} / ${environment}. Substituted by CI:
--   databricks sql query --file 01_catalog_and_schemas.sql \
--     --param catalog=genai_lakehouse_dev --param environment=dev
-- ===========================================================================

USE CATALOG ${catalog};

CREATE SCHEMA IF NOT EXISTS bronze
  COMMENT 'Raw, append-only landing zone. Schema-on-read, no business logic.';

CREATE SCHEMA IF NOT EXISTS silver
  COMMENT 'Cleansed, deduplicated, conformed entities. SCD Type 2 history.';

CREATE SCHEMA IF NOT EXISTS gold
  COMMENT 'Business aggregates and the vector-search source tables.';

CREATE SCHEMA IF NOT EXISTS governance
  COMMENT 'Masking functions, row filters, audit views and access requests.';

-- ---------------------------------------------------------------------------
-- Tag taxonomy. Tags are the mechanism that makes governance scale: a policy
-- keyed on a tag applies to every column that carries it, including columns
-- that do not exist yet.
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS governance.pii_column_registry (
  catalog_name   STRING  NOT NULL COMMENT 'Catalog holding the column.',
  schema_name    STRING  NOT NULL COMMENT 'Schema holding the column.',
  table_name     STRING  NOT NULL COMMENT 'Table holding the column.',
  column_name    STRING  NOT NULL COMMENT 'Column classified as sensitive.',
  pii_class      STRING  NOT NULL COMMENT 'email | phone | national_id | name | account | free_text.',
  masking_policy STRING  NOT NULL COMMENT 'Function in governance.* applied to the column.',
  owner          STRING           COMMENT 'Data steward accountable for the classification.',
  registered_at  TIMESTAMP        COMMENT 'When the classification was recorded.'
)
USING DELTA
COMMENT 'System of record for which columns are PII and how they are masked.'
TBLPROPERTIES (delta.enableChangeDataFeed = true);

CREATE TABLE IF NOT EXISTS governance.access_audit (
  event_time     TIMESTAMP,
  principal      STRING,
  action_name    STRING,
  securable      STRING,
  request_params STRING,
  source_ip      STRING
)
USING DELTA
COMMENT 'Materialised slice of system.access.audit for the lakehouse catalog.'
PARTITIONED BY (DATE(event_time));

-- ---------------------------------------------------------------------------
-- Table-level properties that must hold regardless of who created the table.
-- CDF on the vector-index source is a hard requirement for Delta Sync.
-- ---------------------------------------------------------------------------

ALTER TABLE IF EXISTS gold.document_context
  SET TBLPROPERTIES (
    delta.enableChangeDataFeed = true,
    delta.autoOptimize.optimizeWrite = true,
    delta.autoOptimize.autoCompact = true
  );

ALTER TABLE IF EXISTS silver.instrument_dim
  SET TBLPROPERTIES (
    delta.enableChangeDataFeed = true,
    -- Deletion vectors make the SCD2 MERGE mark rows instead of rewriting
    -- whole files; on a wide dimension this is an order-of-magnitude saving.
    delta.enableDeletionVectors = true
  );

ALTER TABLE IF EXISTS silver.market_events
  SET TBLPROPERTIES (
    delta.autoOptimize.optimizeWrite = true,
    delta.autoOptimize.autoCompact = true,
    -- 7 days of time travel is enough to debug a bad batch; longer retention
    -- on a high-churn streaming table is pure storage cost.
    delta.logRetentionDuration = 'interval 7 days',
    delta.deletedFileRetentionDuration = 'interval 7 days'
  );
