variable "name_prefix" {
  description = "Prefix for IAM role/policy names."
  type        = string
}

variable "account_id" {
  description = "AWS account id hosting the lakehouse."
  type        = string
}

variable "region" {
  description = "AWS region (used to scope SQS/SNS ARNs for Auto Loader)."
  type        = string
}

variable "aws_partition" {
  description = "AWS partition: aws, aws-us-gov or aws-cn."
  type        = string
  default     = "aws"
}

variable "databricks_account_id" {
  description = "Databricks account id. Used as the sts:ExternalId on the Unity Catalog trust policy."
  type        = string
}

variable "databricks_aws_account_id" {
  description = "AWS account id owned by Databricks that assumes the storage credential role. 414351767826 for commercial AWS."
  type        = string
  default     = "414351767826"
}

variable "lakehouse_bucket_arns" {
  description = "Bucket ARNs the Unity Catalog credential may read and write."
  type        = list(string)
}

variable "checkpoint_bucket_arn" {
  description = "Bucket ARN holding Structured Streaming checkpoints."
  type        = string
}

variable "kinesis_stream_arns" {
  description = "Kinesis stream ARNs the ingest role may consume."
  type        = list(string)
  default     = []
}

variable "kms_key_arn" {
  description = "KMS key encrypting the lakehouse buckets."
  type        = string
}

variable "kinesis_kms_key_arn" {
  description = "KMS key encrypting the Kinesis stream. Empty string when the stream is unencrypted."
  type        = string
  default     = ""
}

variable "tags" {
  description = "Tags applied to every resource in the module."
  type        = map(string)
  default     = {}
}
