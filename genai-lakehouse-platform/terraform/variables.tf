variable "project" {
  description = "Project slug used to build every resource name."
  type        = string
  default     = "genai-lakehouse"
}

variable "environment" {
  description = "Environment name: dev, staging or prod."
  type        = string

  validation {
    condition     = contains(["dev", "staging", "prod"], var.environment)
    error_message = "environment must be one of dev, staging, prod."
  }
}

variable "owner" {
  description = "Team accountable for the platform, applied as a tag."
  type        = string
  default     = "data-platform"
}

variable "cost_center" {
  description = "Cost center tag for chargeback."
  type        = string
  default     = "unassigned"
}

# --- AWS ------------------------------------------------------------------

variable "aws_region" {
  description = "AWS region hosting the lakehouse and the Databricks workspace."
  type        = string
  default     = "us-east-1"
}

variable "force_destroy" {
  description = "Allow terraform destroy to delete non-empty buckets/catalogs. Dev only."
  type        = bool
  default     = false
}

# --- Kinesis --------------------------------------------------------------

variable "kinesis_stream_name" {
  description = "Suffix of the Kinesis stream carrying real-time events."
  type        = string
  default     = "market-events"
}

variable "kinesis_stream_mode" {
  description = "ON_DEMAND or PROVISIONED."
  type        = string
  default     = "ON_DEMAND"
}

variable "kinesis_shard_count" {
  description = "Shard count when kinesis_stream_mode is PROVISIONED."
  type        = number
  default     = 2
}

variable "kinesis_retention_hours" {
  description = "Hours of data retained in the Kinesis stream."
  type        = number
  default     = 168
}

variable "alarm_sns_topic_arns" {
  description = "SNS topics notified by streaming CloudWatch alarms."
  type        = list(string)
  default     = []
}

# --- Databricks -----------------------------------------------------------

variable "databricks_account_host" {
  description = "Databricks accounts console host."
  type        = string
  default     = "https://accounts.cloud.databricks.com"
}

variable "databricks_account_id" {
  description = "Databricks account id. Also used as the sts:ExternalId on the UC role."
  type        = string
}

variable "databricks_workspace_host" {
  description = "Workspace URL, e.g. https://dbc-1234abcd-5678.cloud.databricks.com."
  type        = string
}

variable "databricks_workspace_id" {
  description = "Numeric workspace id."
  type        = string
}

variable "databricks_client_id" {
  description = "Service principal application id used by Terraform (OAuth M2M)."
  type        = string
  sensitive   = true
}

variable "databricks_client_secret" {
  description = "Service principal OAuth secret. Injected from CI secrets, never committed."
  type        = string
  sensitive   = true
}

variable "create_metastore" {
  description = "Create the regional metastore. Only one environment per region may set this."
  type        = bool
  default     = false
}

variable "metastore_id" {
  description = "Existing metastore id, required when create_metastore is false."
  type        = string
  default     = ""
}

variable "bound_workspace_ids" {
  description = "Workspace ids allowed to see the catalog."
  type        = list(string)
  default     = []
}

variable "catalog_owner_group" {
  description = "Group owning the catalog and its schemas."
  type        = string
  default     = "data-platform-admins"
}

variable "metastore_owner_group" {
  description = "Group owning the metastore and storage credential."
  type        = string
  default     = "data-platform-admins"
}

variable "data_engineer_groups" {
  description = "Groups granted read/write on the catalog."
  type        = list(string)
  default     = ["data-engineers"]
}

variable "data_analyst_groups" {
  description = "Groups granted read-only on the catalog."
  type        = list(string)
  default     = ["data-analysts"]
}
