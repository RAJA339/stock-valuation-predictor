output "lakehouse_buckets" {
  description = "Medallion layer -> S3 bucket name."
  value       = module.storage.bucket_names
}

output "lakehouse_s3_uris" {
  description = "Medallion layer -> s3:// root registered as an external location."
  value       = module.storage.s3_uris
}

output "kms_key_arn" {
  description = "KMS key encrypting the lakehouse buckets."
  value       = module.storage.kms_key_arn
}

output "kinesis_stream_name" {
  description = "Kinesis stream consumed by the Bronze streaming pipeline."
  value       = module.streaming.stream_name
}

output "kinesis_stream_arn" {
  description = "ARN of the Kinesis stream."
  value       = module.streaming.stream_arn
}

output "unity_catalog_role_arn" {
  description = "IAM role backing the Unity Catalog storage credential."
  value       = module.iam.unity_catalog_role_arn
}

output "ingest_instance_profile_arn" {
  description = "Instance profile to attach to streaming ingest clusters."
  value       = module.iam.ingest_instance_profile_arn
}

output "catalog_name" {
  description = "Unity Catalog catalog created for this environment."
  value       = module.unity_catalog.catalog_name
}

output "metastore_id" {
  description = "Metastore id backing this environment."
  value       = module.unity_catalog.metastore_id
}

output "schema_names" {
  description = "Fully qualified medallion schema names."
  value       = module.unity_catalog.schema_names
}

output "landing_volume_path" {
  description = "UC volume Auto Loader reads unstructured documents from."
  value       = module.unity_catalog.landing_volume_path
}

output "checkpoint_volume_path" {
  description = "UC volume used for Structured Streaming checkpoints."
  value       = module.unity_catalog.checkpoint_volume_path
}

# Consumed by the pipelines at runtime via `terraform output -json > conf.json`
# so that job configuration never drifts from the deployed infrastructure.
output "pipeline_config" {
  description = "Runtime configuration block for the PySpark pipelines."
  value = {
    environment     = var.environment
    catalog         = module.unity_catalog.catalog_name
    bronze_schema   = "bronze"
    silver_schema   = "silver"
    gold_schema     = "gold"
    landing_path    = module.unity_catalog.landing_volume_path
    checkpoint_path = module.unity_catalog.checkpoint_volume_path
    kinesis_stream  = module.streaming.stream_name
    kinesis_region  = var.aws_region
    vector_endpoint = "${var.project}-${var.environment}-vs"
  }
}
