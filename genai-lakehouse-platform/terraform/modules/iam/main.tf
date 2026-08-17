# ---------------------------------------------------------------------------
# IAM for Unity Catalog storage credentials + streaming ingest.
#
# The Unity Catalog trust policy is the part everyone gets wrong. Databricks
# assumes the role twice: first as its own control-plane principal, then as
# THE ROLE ITSELF. That second hop means the role must appear in its own
# trust policy, which is a chicken-and-egg problem in Terraform:
#
#   1. create the role with a placeholder trust policy,
#   2. create the Databricks storage credential (which needs the role ARN),
#   3. update the trust policy so the role trusts itself.
#
# We short-circuit the cycle by constructing the role ARN from the account id
# and role name rather than referencing aws_iam_role.uc.arn, so the policy can
# be written in a single apply.
# ---------------------------------------------------------------------------

locals {
  uc_role_name = "${var.name_prefix}-uc-storage-credential"
  uc_role_arn  = "arn:${var.aws_partition}:iam::${var.account_id}:role/${local.uc_role_name}"
}

data "aws_iam_policy_document" "uc_assume_role" {
  # Hop 1: the Databricks control-plane account, constrained by the External ID
  # (your Databricks account id). Without the ExternalId condition this role is
  # open to every Databricks customer -- this is the classic confused deputy.
  statement {
    sid     = "DatabricksControlPlaneAssume"
    effect  = "Allow"
    actions = ["sts:AssumeRole"]

    principals {
      type        = "AWS"
      identifiers = ["arn:${var.aws_partition}:iam::${var.databricks_aws_account_id}:root"]
    }

    condition {
      test     = "StringEquals"
      variable = "sts:ExternalId"
      values   = [var.databricks_account_id]
    }
  }

  # Hop 2: the role assuming itself.
  statement {
    sid     = "UnityCatalogSelfAssume"
    effect  = "Allow"
    actions = ["sts:AssumeRole"]

    principals {
      type        = "AWS"
      identifiers = [local.uc_role_arn]
    }

    condition {
      test     = "StringEquals"
      variable = "sts:ExternalId"
      values   = [var.databricks_account_id]
    }
  }
}

resource "aws_iam_role" "uc" {
  name                 = local.uc_role_name
  description          = "Assumed by Databricks Unity Catalog to read/write the ${var.name_prefix} lakehouse."
  assume_role_policy   = data.aws_iam_policy_document.uc_assume_role.json
  max_session_duration = 3600
  tags                 = var.tags
}

data "aws_iam_policy_document" "uc_storage_access" {
  statement {
    sid    = "LakehouseObjectAccess"
    effect = "Allow"

    actions = [
      "s3:GetObject",
      "s3:PutObject",
      "s3:DeleteObject",
      "s3:GetObjectVersion",
      "s3:PutObjectAcl",
    ]

    resources = [for arn in var.lakehouse_bucket_arns : "${arn}/*"]
  }

  statement {
    sid    = "LakehouseBucketAccess"
    effect = "Allow"

    actions = [
      "s3:ListBucket",
      "s3:GetBucketLocation",
      "s3:ListBucketMultipartUploads",
      "s3:ListMultipartUploadParts",
      "s3:AbortMultipartUpload",
    ]

    resources = var.lakehouse_bucket_arns
  }

  statement {
    sid    = "LakehouseKmsAccess"
    effect = "Allow"

    actions = [
      "kms:Decrypt",
      "kms:Encrypt",
      "kms:GenerateDataKey*",
      "kms:DescribeKey",
    ]

    resources = [var.kms_key_arn]
  }

  # Required by the Unity Catalog credential validation ("Test connection")
  # so it can confirm the self-assume hop actually works.
  statement {
    sid       = "UnityCatalogCredentialValidation"
    effect    = "Allow"
    actions   = ["sts:AssumeRole"]
    resources = [local.uc_role_arn]
  }
}

resource "aws_iam_policy" "uc_storage_access" {
  name        = "${var.name_prefix}-uc-storage-access"
  description = "S3 + KMS access for the Unity Catalog storage credential."
  policy      = data.aws_iam_policy_document.uc_storage_access.json
  tags        = var.tags
}

resource "aws_iam_role_policy_attachment" "uc_storage_access" {
  role       = aws_iam_role.uc.name
  policy_arn = aws_iam_policy.uc_storage_access.arn
}

# ---------------------------------------------------------------------------
# Streaming ingest role: assumed by the Databricks data-plane (instance
# profile) so Structured Streaming can read the Kinesis stream and write its
# checkpoints. Kept separate from the Unity Catalog role so that a compromised
# job cluster cannot rewrite Gold tables.
# ---------------------------------------------------------------------------

data "aws_iam_policy_document" "ingest_assume_role" {
  statement {
    sid     = "DatabricksDataPlaneAssume"
    effect  = "Allow"
    actions = ["sts:AssumeRole"]

    principals {
      type        = "Service"
      identifiers = ["ec2.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "ingest" {
  name               = "${var.name_prefix}-streaming-ingest"
  description        = "Instance-profile role for Kinesis + Auto Loader ingest clusters."
  assume_role_policy = data.aws_iam_policy_document.ingest_assume_role.json
  tags               = var.tags
}

data "aws_iam_policy_document" "ingest_access" {
  statement {
    sid    = "KinesisRead"
    effect = "Allow"

    actions = [
      "kinesis:DescribeStream",
      "kinesis:DescribeStreamSummary",
      "kinesis:GetRecords",
      "kinesis:GetShardIterator",
      "kinesis:ListShards",
      "kinesis:ListStreams",
      "kinesis:SubscribeToShard",
    ]

    resources = var.kinesis_stream_arns
  }

  statement {
    sid    = "CheckpointReadWrite"
    effect = "Allow"

    actions = [
      "s3:GetObject",
      "s3:PutObject",
      "s3:DeleteObject",
      "s3:ListBucket",
    ]

    resources = concat(
      [var.checkpoint_bucket_arn],
      ["${var.checkpoint_bucket_arn}/*"],
    )
  }

  # Auto Loader's file-notification mode provisions and drains an SQS queue and
  # an SNS topic per stream. Scoped to the databricks-auto-ingest-* names that
  # Auto Loader itself generates.
  statement {
    sid    = "AutoLoaderFileNotification"
    effect = "Allow"

    actions = [
      "sqs:CreateQueue",
      "sqs:DeleteMessage",
      "sqs:DeleteQueue",
      "sqs:GetQueueAttributes",
      "sqs:GetQueueUrl",
      "sqs:ReceiveMessage",
      "sqs:SetQueueAttributes",
      "sqs:TagQueue",
      "sns:CreateTopic",
      "sns:DeleteTopic",
      "sns:GetTopicAttributes",
      "sns:SetTopicAttributes",
      "sns:Subscribe",
      "sns:Unsubscribe",
      "sns:TagResource",
    ]

    resources = [
      "arn:${var.aws_partition}:sqs:${var.region}:${var.account_id}:databricks-auto-ingest-*",
      "arn:${var.aws_partition}:sns:${var.region}:${var.account_id}:databricks-auto-ingest-*",
    ]
  }

  statement {
    sid    = "AutoLoaderBucketNotificationConfig"
    effect = "Allow"

    actions = [
      "s3:GetBucketNotification",
      "s3:PutBucketNotification",
    ]

    resources = var.lakehouse_bucket_arns
  }

  statement {
    sid    = "StreamingKmsAccess"
    effect = "Allow"

    actions = [
      "kms:Decrypt",
      "kms:GenerateDataKey*",
      "kms:DescribeKey",
    ]

    resources = compact([var.kms_key_arn, var.kinesis_kms_key_arn])
  }
}

resource "aws_iam_policy" "ingest_access" {
  name        = "${var.name_prefix}-streaming-ingest-access"
  description = "Kinesis + checkpoint + Auto Loader notification access."
  policy      = data.aws_iam_policy_document.ingest_access.json
  tags        = var.tags
}

resource "aws_iam_role_policy_attachment" "ingest_access" {
  role       = aws_iam_role.ingest.name
  policy_arn = aws_iam_policy.ingest_access.arn
}

resource "aws_iam_instance_profile" "ingest" {
  name = "${var.name_prefix}-streaming-ingest"
  role = aws_iam_role.ingest.name
  tags = var.tags
}
