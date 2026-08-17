variable "name_prefix" {
  description = "Prefix for metastore / credential / external location names."
  type        = string
}

variable "environment" {
  description = "Environment name: dev, staging or prod."
  type        = string
}

variable "region" {
  description = "AWS region of the metastore."
  type        = string
}

variable "create_metastore" {
  description = "Create the regional metastore. Exactly one environment per region may set this to true."
  type        = bool
  default     = false
}

variable "metastore_id" {
  description = "Existing metastore id, required when create_metastore is false."
  type        = string
  default     = ""

  validation {
    condition     = var.metastore_id == "" || can(regex("^[0-9a-f-]{36}$", var.metastore_id))
    error_message = "metastore_id must be empty or a UUID."
  }
}

variable "workspace_id" {
  description = "Numeric Databricks workspace id the metastore is assigned to."
  type        = string
}

variable "bound_workspace_ids" {
  description = "Workspace ids allowed to see the catalog (isolation_mode = ISOLATED)."
  type        = list(string)
  default     = []
}

variable "metastore_owner" {
  description = "Group owning the metastore, credential and external locations."
  type        = string
  default     = "account users"
}

variable "catalog_owner" {
  description = "Group owning the catalog and its schemas."
  type        = string
}

variable "catalog_name" {
  description = "Unity Catalog catalog name, e.g. genai_lakehouse_dev."
  type        = string

  validation {
    condition     = can(regex("^[a-z][a-z0-9_]{2,254}$", var.catalog_name))
    error_message = "catalog_name must be lowercase, start with a letter and contain only letters, digits and underscores."
  }
}

variable "catalog_storage_root" {
  description = "Default managed-table storage root for the catalog."
  type        = string
}

variable "storage_credential_role_arn" {
  description = "IAM role ARN backing the storage credential."
  type        = string
}

variable "external_locations" {
  description = "Map of location name -> s3:// url. Must include bronze, silver, gold and checkpoints."
  type        = map(string)

  validation {
    condition = length(setsubtract(
      ["bronze", "silver", "gold", "checkpoints"],
      keys(var.external_locations)
    )) == 0
    error_message = "external_locations must define bronze, silver, gold and checkpoints."
  }
}

variable "schemas" {
  description = "Medallion schemas to create."
  type = map(object({
    comment      = string
    storage_root = string
  }))
}

variable "data_engineer_groups" {
  description = "Groups granted read/write on the catalog."
  type        = list(string)
  default     = []
}

variable "data_analyst_groups" {
  description = "Groups granted read-only on the catalog."
  type        = list(string)
  default     = []
}

variable "force_destroy" {
  description = "Allow terraform destroy to remove non-empty catalogs/locations. Never enable in prod."
  type        = bool
  default     = false
}
