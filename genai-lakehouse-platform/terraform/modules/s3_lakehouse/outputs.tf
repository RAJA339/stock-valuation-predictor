output "bucket_names" {
  description = "Map of medallion layer -> bucket name."
  value       = { for k, b in aws_s3_bucket.layer : k => b.id }
}

output "bucket_arns" {
  description = "Map of medallion layer -> bucket ARN."
  value       = { for k, b in aws_s3_bucket.layer : k => b.arn }
}

output "bucket_regional_domain_names" {
  description = "Map of medallion layer -> regional domain name."
  value       = { for k, b in aws_s3_bucket.layer : k => b.bucket_regional_domain_name }
}

output "kms_key_arn" {
  description = "ARN of the KMS key encrypting every lakehouse bucket."
  value       = aws_kms_key.lakehouse.arn
}

output "s3_uris" {
  description = "s3:// roots used as Unity Catalog external locations."
  value = {
    bronze      = "s3://${aws_s3_bucket.layer["bronze"].id}/bronze"
    silver      = "s3://${aws_s3_bucket.layer["silver"].id}/silver"
    gold        = "s3://${aws_s3_bucket.layer["gold"].id}/gold"
    checkpoints = "s3://${aws_s3_bucket.layer["checkpoints"].id}"
  }
}
