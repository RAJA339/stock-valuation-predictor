provider "aws" {
  region = var.aws_region

  default_tags {
    tags = local.tags
  }
}

# Account-level provider: metastore creation and workspace assignment.
# Authenticate with a service principal that holds the account admin role.
provider "databricks" {
  alias      = "account"
  host       = var.databricks_account_host
  account_id = var.databricks_account_id

  client_id     = var.databricks_client_id
  client_secret = var.databricks_client_secret
}

# Workspace-level provider: catalogs, schemas, credentials, grants.
provider "databricks" {
  alias = "workspace"
  host  = var.databricks_workspace_host

  client_id     = var.databricks_client_id
  client_secret = var.databricks_client_secret
}
