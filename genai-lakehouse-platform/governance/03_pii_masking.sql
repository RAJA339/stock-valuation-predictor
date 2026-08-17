-- ===========================================================================
-- 03 - Dynamic PII column masking and row-level security
--
-- A masking function runs at query time, per row, with the CALLER's identity.
-- One physical table therefore serves every audience: a steward sees the raw
-- email, an analyst sees j***@example.com, the GenAI service sees nothing at
-- all. No copies, no "_masked" views to keep in sync, no chance of the
-- unmasked copy leaking.
--
-- Two rules that decide whether this works or is theatre:
--   1. is_account_group_member() evaluates the QUERY's principal, so a view
--      owned by an admin does not launder access for its readers.
--   2. The masking function must not be droppable by the people it masks --
--      hence governance schema ownership sits with data_platform_admins.
-- ===========================================================================

USE CATALOG ${catalog};
USE SCHEMA governance;

-- ---------------------------------------------------------------------------
-- Masking functions. Each takes the column value and returns either the value
-- or a redacted form. Keep them total: a function that raises turns every
-- query on the table into an error.
-- ---------------------------------------------------------------------------

CREATE OR REPLACE FUNCTION governance.mask_email(email STRING)
RETURN CASE
  WHEN email IS NULL THEN NULL
  WHEN is_account_group_member('pii_readers') THEN email
  -- Keep the first character and the domain: enough to eyeball-match a record
  -- during support work, not enough to contact the person.
  WHEN email RLIKE '^[^@]+@[^@]+$'
    THEN concat(substr(email, 1, 1), '***@', split_part(email, '@', 2))
  ELSE '***REDACTED***'
END;

CREATE OR REPLACE FUNCTION governance.mask_phone(phone STRING)
RETURN CASE
  WHEN phone IS NULL THEN NULL
  WHEN is_account_group_member('pii_readers') THEN phone
  -- Last four digits only, the standard call-centre verification pattern.
  WHEN length(regexp_replace(phone, '[^0-9]', '')) >= 4
    THEN concat('***-***-', right(regexp_replace(phone, '[^0-9]', ''), 4))
  ELSE '***REDACTED***'
END;

CREATE OR REPLACE FUNCTION governance.mask_national_id(national_id STRING)
RETURN CASE
  WHEN national_id IS NULL THEN NULL
  -- National identifiers get NO partial reveal. Last-four of an SSN is still
  -- the most commonly abused "safe" identifier in existence.
  WHEN is_account_group_member('pii_readers') THEN national_id
  ELSE '***REDACTED***'
END;

CREATE OR REPLACE FUNCTION governance.mask_person_name(person_name STRING)
RETURN CASE
  WHEN person_name IS NULL THEN NULL
  WHEN is_account_group_member('pii_readers') THEN person_name
  WHEN is_account_group_member('data_scientists')
    -- Pseudonymisation rather than redaction: joins and cohort analysis still
    -- work, the identity does not survive. Salted so the hash cannot be
    -- rainbow-tabled back to a name.
    THEN concat('anon_', substr(sha2(concat(person_name, '${pii_salt}'), 256), 1, 12))
  ELSE '***REDACTED***'
END;

CREATE OR REPLACE FUNCTION governance.mask_account_number(account_number STRING)
RETURN CASE
  WHEN account_number IS NULL THEN NULL
  WHEN is_account_group_member('pii_readers') THEN account_number
  WHEN length(account_number) > 4 THEN concat(repeat('*', length(account_number) - 4), right(account_number, 4))
  ELSE '***REDACTED***'
END;

-- Free text is the hard case: a chunk of a document can contain anything.
-- Regex redaction is a mitigation, not a guarantee -- the durable control is
-- classifying the SOURCE and never letting unreviewed documents into an
-- audience-facing index.
CREATE OR REPLACE FUNCTION governance.redact_free_text(body STRING)
RETURN CASE
  WHEN body IS NULL THEN NULL
  WHEN is_account_group_member('pii_readers') THEN body
  ELSE regexp_replace(
         regexp_replace(
           regexp_replace(body, '[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\.[A-Za-z]{2,}', '[EMAIL]'),
           '\\b(?:\\+?1[-. ]?)?\\(?[0-9]{3}\\)?[-. ]?[0-9]{3}[-. ]?[0-9]{4}\\b', '[PHONE]'),
         '\\b[0-9]{3}-[0-9]{2}-[0-9]{4}\\b', '[NATIONAL_ID]')
END;

-- ---------------------------------------------------------------------------
-- Bind the functions to columns. SET MASK is idempotent; re-running replaces
-- the binding rather than stacking it.
-- ---------------------------------------------------------------------------

ALTER TABLE silver.instrument_dim
  ALTER COLUMN issuer_name SET MASK governance.mask_person_name;

ALTER TABLE gold.document_context
  ALTER COLUMN chunk_text SET MASK governance.redact_free_text;

-- Contact tables are created by the reference-data feed; guard the DDL so the
-- script stays runnable on an environment that has not ingested one yet.
-- (Databricks SQL ignores ALTER ... IF EXISTS on a missing table.)
ALTER TABLE IF EXISTS silver.counterparty_contacts
  ALTER COLUMN email SET MASK governance.mask_email;
ALTER TABLE IF EXISTS silver.counterparty_contacts
  ALTER COLUMN phone SET MASK governance.mask_phone;
ALTER TABLE IF EXISTS silver.counterparty_contacts
  ALTER COLUMN national_id SET MASK governance.mask_national_id;
ALTER TABLE IF EXISTS silver.counterparty_contacts
  ALTER COLUMN account_number SET MASK governance.mask_account_number;

-- ---------------------------------------------------------------------------
-- Row-level security. A row filter is a function returning a boolean; rows
-- for which it is false are invisible -- not an error, simply absent.
-- ---------------------------------------------------------------------------

CREATE OR REPLACE FUNCTION governance.filter_by_region(region STRING)
RETURN
  is_account_group_member('data_platform_admins')
  OR region IS NULL
  OR is_account_group_member(concat('region_reader_', lower(region)));

CREATE OR REPLACE FUNCTION governance.filter_restricted_documents(classification STRING)
RETURN
  coalesce(classification, 'public') IN ('public', 'internal')
  OR is_account_group_member('restricted_data_readers');

ALTER TABLE IF EXISTS silver.instrument_dim
  SET ROW FILTER governance.filter_by_region ON (country);

-- ---------------------------------------------------------------------------
-- Column classification tags. Tags are what let a scan answer "where is our
-- PII" without reading a single row, and what a future policy can key on.
-- ---------------------------------------------------------------------------

ALTER TABLE silver.instrument_dim
  ALTER COLUMN issuer_name SET TAGS ('pii' = 'name', 'sensitivity' = 'confidential');

ALTER TABLE gold.document_context
  ALTER COLUMN chunk_text SET TAGS ('pii' = 'free_text', 'sensitivity' = 'confidential');

ALTER TABLE silver.market_events
  SET TAGS ('data_domain' = 'market_data', 'sensitivity' = 'internal');

-- ---------------------------------------------------------------------------
-- Register the classifications so the registry, the tags and the actual
-- bindings can be reconciled by the audit script (05_audit_and_lineage.sql).
-- ---------------------------------------------------------------------------

MERGE INTO governance.pii_column_registry AS t
USING (
  SELECT * FROM VALUES
    ('${catalog}', 'silver', 'instrument_dim',       'issuer_name',    'name',      'governance.mask_person_name'),
    ('${catalog}', 'gold',   'document_context',     'chunk_text',     'free_text', 'governance.redact_free_text'),
    ('${catalog}', 'silver', 'counterparty_contacts','email',          'email',     'governance.mask_email'),
    ('${catalog}', 'silver', 'counterparty_contacts','phone',          'phone',     'governance.mask_phone'),
    ('${catalog}', 'silver', 'counterparty_contacts','national_id',    'national_id','governance.mask_national_id'),
    ('${catalog}', 'silver', 'counterparty_contacts','account_number', 'account',   'governance.mask_account_number')
  AS s(catalog_name, schema_name, table_name, column_name, pii_class, masking_policy)
) AS s
ON  t.catalog_name = s.catalog_name
AND t.schema_name  = s.schema_name
AND t.table_name   = s.table_name
AND t.column_name  = s.column_name
WHEN MATCHED THEN UPDATE SET
  t.pii_class      = s.pii_class,
  t.masking_policy = s.masking_policy
WHEN NOT MATCHED THEN INSERT
  (catalog_name, schema_name, table_name, column_name, pii_class, masking_policy, owner, registered_at)
VALUES
  (s.catalog_name, s.schema_name, s.table_name, s.column_name, s.pii_class, s.masking_policy,
   'data_platform_admins', current_timestamp());
