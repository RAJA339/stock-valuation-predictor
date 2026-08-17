terraform {
  required_version = ">= 1.10.0"

  required_providers {
    databricks = {
      source                = "databricks/databricks"
      version               = ">= 1.50.0"
      configuration_aliases = [databricks.account, databricks.workspace]
    }
  }
}
