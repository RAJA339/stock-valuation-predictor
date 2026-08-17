variable "name_prefix" {
  description = "Prefix for every bucket name, e.g. genai-lakehouse-dev."
  type        = string

  validation {
    condition     = can(regex("^[a-z0-9][a-z0-9-]{2,40}$", var.name_prefix))
    error_message = "name_prefix must be lowercase alphanumeric/hyphen, 3-41 chars (S3 bucket naming rules)."
  }
}

variable "account_id" {
  description = "AWS account id, appended to bucket names to keep them globally unique."
  type        = string
}

variable "force_destroy" {
  description = "Allow terraform destroy to delete non-empty buckets. Never enable in prod."
  type        = bool
  default     = false
}

variable "kms_deletion_window_in_days" {
  description = "Waiting period before the lakehouse KMS key is actually deleted."
  type        = number
  default     = 30
}

variable "tags" {
  description = "Tags applied to every resource in the module."
  type        = map(string)
  default     = {}
}
