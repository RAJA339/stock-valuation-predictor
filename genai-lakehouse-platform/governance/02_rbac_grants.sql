-- ===========================================================================
-- 02 - Role-based access control
--
-- The model, in one sentence: privileges are granted to GROUPS, never to
-- users, and the medallion layer decides who can see what.
--
--   data_platform_admins   own everything, manage the metastore
--   data_engineers         read/write Bronze+Silver, read Gold
--   data_analysts          read Gold only, PII masked
--   data_scientists        read Silver+Gold, PII masked, can create in a
--                          sandbox schema of their own
--   genai_app_service      read the vector-index source table only
--   auditors               read metadata and audit tables, no business data
--
-- Unity Catalog privileges are hierarchical: SELECT on a catalog implies
-- SELECT on every table in it, including tables created tomorrow. That is
-- powerful and dangerous -- grant broadly at the schema level for read roles,
-- and narrowly at the table level for service principals.
-- ===========================================================================

USE CATALOG ${catalog};

-- ---------------------------------------------------------------------------
-- Ownership. An owner has every privilege on the object implicitly and can
-- grant it onward -- so ownership always sits with a group, never a person
-- who might leave.
-- ---------------------------------------------------------------------------

ALTER CATALOG ${catalog} OWNER TO `data_platform_admins`;
ALTER SCHEMA bronze     OWNER TO `data_platform_admins`;
ALTER SCHEMA silver     OWNER TO `data_platform_admins`;
ALTER SCHEMA gold       OWNER TO `data_platform_admins`;
ALTER SCHEMA governance OWNER TO `data_platform_admins`;

-- ---------------------------------------------------------------------------
-- USE CATALOG is the gate: without it, nothing below matters. Every role that
-- touches the catalog needs it, including read-only ones.
-- ---------------------------------------------------------------------------

GRANT USE CATALOG ON CATALOG ${catalog} TO `data_engineers`;
GRANT USE CATALOG ON CATALOG ${catalog} TO `data_analysts`;
GRANT USE CATALOG ON CATALOG ${catalog} TO `data_scientists`;
GRANT USE CATALOG ON CATALOG ${catalog} TO `genai_app_service`;
GRANT USE CATALOG ON CATALOG ${catalog} TO `auditors`;

-- ---------------------------------------------------------------------------
-- Data engineers: build the pipelines, own Bronze and Silver.
-- ---------------------------------------------------------------------------

GRANT USE SCHEMA, CREATE TABLE, CREATE VOLUME, SELECT, MODIFY
  ON SCHEMA bronze TO `data_engineers`;

GRANT USE SCHEMA, CREATE TABLE, CREATE FUNCTION, SELECT, MODIFY
  ON SCHEMA silver TO `data_engineers`;

-- Gold is produced by CI, not by hand: engineers read it, they do not write it.
GRANT USE SCHEMA, SELECT ON SCHEMA gold TO `data_engineers`;

GRANT READ VOLUME, WRITE VOLUME ON VOLUME bronze.landing      TO `data_engineers`;
GRANT READ VOLUME, WRITE VOLUME ON VOLUME bronze.checkpoints  TO `data_engineers`;

-- ---------------------------------------------------------------------------
-- Analysts: Gold only. They never see Bronze -- unparsed, unvalidated data is
-- how a dashboard ends up reporting a corrupt record as a real trade.
-- ---------------------------------------------------------------------------

GRANT USE SCHEMA, SELECT ON SCHEMA gold TO `data_analysts`;

-- ---------------------------------------------------------------------------
-- Data scientists: Silver and Gold read access plus a sandbox they own.
-- ---------------------------------------------------------------------------

CREATE SCHEMA IF NOT EXISTS sandbox
  COMMENT 'Scratch space for data science. Not backed up, not governed, not a source for anything.';
ALTER SCHEMA sandbox OWNER TO `data_scientists`;

GRANT USE SCHEMA, SELECT ON SCHEMA silver TO `data_scientists`;
GRANT USE SCHEMA, SELECT ON SCHEMA gold   TO `data_scientists`;
GRANT ALL PRIVILEGES ON SCHEMA sandbox    TO `data_scientists`;

-- ---------------------------------------------------------------------------
-- The GenAI application service principal. Scoped to exactly the two objects
-- retrieval needs -- table-level, not schema-level, so a new Gold table is
-- not automatically exposed to the application.
-- ---------------------------------------------------------------------------

GRANT USE SCHEMA ON SCHEMA gold TO `genai_app_service`;
GRANT SELECT ON TABLE gold.document_context      TO `genai_app_service`;
GRANT SELECT ON TABLE gold.symbol_daily_metrics  TO `genai_app_service`;

-- ---------------------------------------------------------------------------
-- Auditors: metadata and audit trails, never business data. BROWSE grants
-- visibility of names and schemas WITHOUT the ability to read rows -- exactly
-- what an audit needs.
-- ---------------------------------------------------------------------------

GRANT BROWSE ON CATALOG ${catalog} TO `auditors`;
GRANT USE SCHEMA ON SCHEMA governance TO `auditors`;
GRANT SELECT ON TABLE governance.access_audit        TO `auditors`;
GRANT SELECT ON TABLE governance.pii_column_registry TO `auditors`;

-- ---------------------------------------------------------------------------
-- Quarantine tables hold rejected rows, which by definition failed validation
-- and may contain anything at all. Engineers and stewards only.
-- ---------------------------------------------------------------------------

GRANT SELECT ON TABLE silver.market_events_quarantine   TO `data_engineers`;
GRANT SELECT ON TABLE silver.document_chunks_quarantine TO `data_engineers`;

-- ---------------------------------------------------------------------------
-- Explicit denials. Unity Catalog has no DENY, so removal is the tool: revoke
-- anything a broader grant may have implied. These are no-ops on a clean
-- deploy and are the safety net on a dirty one.
-- ---------------------------------------------------------------------------

REVOKE ALL PRIVILEGES ON SCHEMA bronze FROM `data_analysts`;
REVOKE ALL PRIVILEGES ON SCHEMA bronze FROM `genai_app_service`;
REVOKE ALL PRIVILEGES ON SCHEMA silver FROM `data_analysts`;
REVOKE MODIFY ON SCHEMA gold FROM `data_analysts`;
REVOKE MODIFY ON SCHEMA gold FROM `data_scientists`;
