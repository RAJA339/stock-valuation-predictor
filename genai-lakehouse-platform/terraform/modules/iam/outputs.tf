output "unity_catalog_role_arn" {
  description = "Role ARN registered as the Unity Catalog storage credential."
  value       = aws_iam_role.uc.arn
}

output "unity_catalog_role_name" {
  description = "Name of the Unity Catalog storage credential role."
  value       = aws_iam_role.uc.name
}

output "ingest_role_arn" {
  description = "Role ARN used by streaming ingest clusters."
  value       = aws_iam_role.ingest.arn
}

output "ingest_instance_profile_arn" {
  description = "Instance profile to register in the Databricks workspace for ingest clusters."
  value       = aws_iam_instance_profile.ingest.arn
}
