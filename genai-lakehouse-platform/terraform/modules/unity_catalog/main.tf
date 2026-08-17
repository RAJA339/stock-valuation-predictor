# ---------------------------------------------------------------------------
# Unity Catalog: metastore, storage credential, external locations, catalog and
# medallion schemas, plus the workspace binding.
#
# Two providers are in play and mixing them up is the #1 source of 401s:
#   databricks.account   -> accounts.cloud.databricks.com (metastore, assignment)
#   databricks.workspace -> <workspace>.cloud.databricks.com (credentials,
#                           external locations, catalogs, grants)
#
# A metastore is REGIONAL and there is at most one per region per account. Set
# create_metastore = false in every environment after the first, and pass the
# existing metastore_id instead.
# ---------------------------------------------------------------------------

resource "databricks_metastore" "this" {
  count    = var.create_metastore ? 1 : 0
  provider = databricks.account

  name = "${var.name_prefix}-metastore"
  # Deliberately NO storage_root: a metastore-level root creates managed-table
  # storage that every catalog silently inherits. Per-catalog storage roots
  # (below) keep Bronze/Silver/Gold on separate buckets and separate KMS grants.
  region        = var.region
  owner         = var.metastore_owner
  force_destroy = var.force_destroy
}

locals {
  metastore_id = var.create_metastore ? databricks_metastore.this[0].id : var.metastore_id
}

resource "databricks_metastore_assignment" "this" {
  provider = databricks.account

  metastore_id         = local.metastore_id
  workspace_id         = var.workspace_id
  default_catalog_name = var.catalog_name
}

# ---------------------------------------------------------------------------
# Storage credential + external locations
# ---------------------------------------------------------------------------

resource "databricks_storage_credential" "this" {
  provider = databricks.workspace

  name    = "${var.name_prefix}-storage-credential"
  comment = "IAM role granting Unity Catalog access to the ${var.name_prefix} lakehouse buckets."
  owner   = var.metastore_owner

  aws_iam_role {
    role_arn = var.storage_credential_role_arn
  }

  # A credential that can be used to read arbitrary S3 paths from a notebook
  # defeats the point of external locations. Force everything through them.
  isolation_mode = "ISOLATION_MODE_ISOLATED"

  depends_on = [databricks_metastore_assignment.this]
}

resource "databricks_external_location" "layer" {
  provider = databricks.workspace
  for_each = var.external_locations

  name            = "${var.name_prefix}-${each.key}"
  url             = each.value
  credential_name = databricks_storage_credential.this.name
  comment         = "${title(each.key)} layer root for ${var.name_prefix}."
  owner           = var.metastore_owner
  isolation_mode  = "ISOLATION_MODE_ISOLATED"

  # Protects against `terraform destroy` orphaning tables that still hold data.
  force_destroy = var.force_destroy
}

# ---------------------------------------------------------------------------
# Catalog + medallion schemas
# ---------------------------------------------------------------------------

resource "databricks_catalog" "this" {
  provider = databricks.workspace

  metastore_id = local.metastore_id
  name         = var.catalog_name
  comment      = "GenAI lakehouse catalog for the ${var.environment} environment."
  owner        = var.catalog_owner

  # Managed tables created in this catalog land under the Silver bucket root
  # unless the schema overrides it.
  storage_root = var.catalog_storage_root

  # ISOLATED means this catalog is only visible in workspaces it is explicitly
  # bound to (below) -- the control that stops a dev workspace from reading
  # prod Gold.
  isolation_mode = "ISOLATED"

  properties = {
    environment = var.environment
    managed_by  = "terraform"
  }

  force_destroy = var.force_destroy

  depends_on = [databricks_external_location.layer]
}

resource "databricks_workspace_binding" "this" {
  provider = databricks.workspace
  for_each = toset(var.bound_workspace_ids)

  securable_name = databricks_catalog.this.name
  securable_type = "catalog"
  workspace_id   = each.value
  binding_type   = var.environment == "prod" ? "BINDING_TYPE_READ_ONLY" : "BINDING_TYPE_READ_WRITE"
}

resource "databricks_schema" "medallion" {
  provider = databricks.workspace
  for_each = var.schemas

  catalog_name = databricks_catalog.this.name
  name         = each.key
  comment      = each.value.comment
  owner        = var.catalog_owner
  storage_root = each.value.storage_root

  properties = {
    medallion_layer = each.key
  }

  force_destroy = var.force_destroy
}

# Volumes are how unstructured source files (PDFs, transcripts, HTML) are
# governed by Unity Catalog rather than by raw S3 paths. Auto Loader reads
# from the volume path, so the file drop zone inherits UC permissions.
resource "databricks_volume" "landing" {
  provider = databricks.workspace

  name             = "landing"
  catalog_name     = databricks_catalog.this.name
  schema_name      = databricks_schema.medallion["bronze"].name
  volume_type      = "EXTERNAL"
  storage_location = "${var.external_locations["bronze"]}/landing"
  comment          = "Unstructured document drop zone consumed by Auto Loader."
  owner            = var.catalog_owner

  depends_on = [databricks_external_location.layer]
}

resource "databricks_volume" "checkpoints" {
  provider = databricks.workspace

  name             = "checkpoints"
  catalog_name     = databricks_catalog.this.name
  schema_name      = databricks_schema.medallion["bronze"].name
  volume_type      = "EXTERNAL"
  storage_location = "${var.external_locations["checkpoints"]}/checkpoints"
  comment          = "Structured Streaming checkpoint + schema-evolution location."
  owner            = var.catalog_owner

  depends_on = [databricks_external_location.layer]
}

# ---------------------------------------------------------------------------
# Baseline grants. Fine-grained RBAC lives in governance/*.sql, which is
# applied by CI -- this is only the minimum needed for the pipelines to run.
# ---------------------------------------------------------------------------

resource "databricks_grants" "catalog" {
  provider = databricks.workspace
  catalog  = databricks_catalog.this.name

  dynamic "grant" {
    for_each = var.data_engineer_groups
    content {
      principal  = grant.value
      privileges = ["USE_CATALOG", "USE_SCHEMA", "CREATE_SCHEMA", "SELECT", "MODIFY"]
    }
  }

  dynamic "grant" {
    for_each = var.data_analyst_groups
    content {
      principal  = grant.value
      privileges = ["USE_CATALOG", "USE_SCHEMA", "SELECT"]
    }
  }
}

resource "databricks_grants" "external_location" {
  provider          = databricks.workspace
  for_each          = databricks_external_location.layer
  external_location = each.value.name

  dynamic "grant" {
    for_each = var.data_engineer_groups
    content {
      principal  = grant.value
      privileges = ["READ_FILES", "WRITE_FILES", "CREATE_EXTERNAL_TABLE"]
    }
  }
}
