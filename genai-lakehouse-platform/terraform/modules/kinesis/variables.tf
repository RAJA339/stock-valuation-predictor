variable "name_prefix" {
  description = "Prefix for the stream name."
  type        = string
}

variable "stream_name" {
  description = "Suffix identifying the stream, e.g. market-events."
  type        = string
  default     = "market-events"
}

variable "stream_mode" {
  description = "ON_DEMAND or PROVISIONED."
  type        = string
  default     = "ON_DEMAND"

  validation {
    condition     = contains(["ON_DEMAND", "PROVISIONED"], var.stream_mode)
    error_message = "stream_mode must be ON_DEMAND or PROVISIONED."
  }
}

variable "shard_count" {
  description = "Shard count. Only used when stream_mode is PROVISIONED."
  type        = number
  default     = 2
}

variable "retention_period_hours" {
  description = "Hours of data retained in the stream (24-8760)."
  type        = number
  default     = 168

  validation {
    condition     = var.retention_period_hours >= 24 && var.retention_period_hours <= 8760
    error_message = "retention_period_hours must be between 24 and 8760."
  }
}

variable "create_kms_key" {
  description = "Create a dedicated KMS key for the stream. Set false to reuse kms_key_arn."
  type        = bool
  default     = true
}

variable "kms_key_arn" {
  description = "Existing KMS key ARN, used when create_kms_key is false."
  type        = string
  default     = "alias/aws/kinesis"
}

variable "enable_alarms" {
  description = "Create CloudWatch alarms for iterator age and write throttling."
  type        = bool
  default     = true
}

variable "iterator_age_alarm_threshold_ms" {
  description = "Alarm when the consumer is this far behind real time, in milliseconds."
  type        = number
  default     = 600000 # 10 minutes
}

variable "alarm_sns_topic_arns" {
  description = "SNS topics notified by the CloudWatch alarms."
  type        = list(string)
  default     = []
}

variable "tags" {
  description = "Tags applied to every resource in the module."
  type        = map(string)
  default     = {}
}
