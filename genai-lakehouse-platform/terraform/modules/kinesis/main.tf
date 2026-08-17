# ---------------------------------------------------------------------------
# Kinesis Data Stream feeding the Bronze real-time ingest pipeline.
#
# ON-DEMAND vs PROVISIONED: on-demand removes shard-count capacity planning and
# is the right default for spiky market/event traffic. Switch to PROVISIONED
# once the throughput profile is known and steady -- it is roughly 3-4x cheaper
# at sustained load.
# ---------------------------------------------------------------------------

resource "aws_kms_key" "stream" {
  count = var.create_kms_key ? 1 : 0

  description             = "${var.name_prefix} Kinesis SSE key"
  deletion_window_in_days = 30
  enable_key_rotation     = true
  tags                    = var.tags
}

resource "aws_kms_alias" "stream" {
  count = var.create_kms_key ? 1 : 0

  name          = "alias/${var.name_prefix}-kinesis"
  target_key_id = aws_kms_key.stream[0].key_id
}

locals {
  kms_key_arn = var.create_kms_key ? aws_kms_key.stream[0].arn : var.kms_key_arn
}

resource "aws_kinesis_stream" "events" {
  name = "${var.name_prefix}-${var.stream_name}"

  # retention_period is what makes a Structured Streaming job survive a
  # weekend outage. 24h is the AWS default and is not enough for anything
  # you would page someone about.
  retention_period = var.retention_period_hours

  encryption_type = "KMS"
  kms_key_id      = local.kms_key_arn

  stream_mode_details {
    stream_mode = var.stream_mode
  }

  # shard_count is only legal in PROVISIONED mode.
  shard_count = var.stream_mode == "PROVISIONED" ? var.shard_count : null

  shard_level_metrics = [
    "IncomingBytes",
    "IncomingRecords",
    "OutgoingBytes",
    "OutgoingRecords",
    "IteratorAgeMilliseconds",
    "ReadProvisionedThroughputExceeded",
    "WriteProvisionedThroughputExceeded",
  ]

  tags = var.tags
}

# Iterator age is THE health metric for a streaming consumer: it is how far
# behind real time the reader is. Alert on it, not on record counts.
resource "aws_cloudwatch_metric_alarm" "iterator_age" {
  count = var.enable_alarms ? 1 : 0

  alarm_name          = "${aws_kinesis_stream.events.name}-iterator-age"
  alarm_description   = "Bronze consumer is falling behind the ${aws_kinesis_stream.events.name} stream."
  namespace           = "AWS/Kinesis"
  metric_name         = "GetRecords.IteratorAgeMilliseconds"
  statistic           = "Maximum"
  period              = 300
  evaluation_periods  = 3
  threshold           = var.iterator_age_alarm_threshold_ms
  comparison_operator = "GreaterThanThreshold"
  treat_missing_data  = "notBreaching"

  dimensions = {
    StreamName = aws_kinesis_stream.events.name
  }

  alarm_actions = var.alarm_sns_topic_arns
  ok_actions    = var.alarm_sns_topic_arns
  tags          = var.tags
}

resource "aws_cloudwatch_metric_alarm" "write_throttle" {
  count = var.enable_alarms && var.stream_mode == "PROVISIONED" ? 1 : 0

  alarm_name          = "${aws_kinesis_stream.events.name}-write-throttled"
  alarm_description   = "Producers are being throttled writing to ${aws_kinesis_stream.events.name}; add shards."
  namespace           = "AWS/Kinesis"
  metric_name         = "WriteProvisionedThroughputExceeded"
  statistic           = "Sum"
  period              = 300
  evaluation_periods  = 2
  threshold           = 0
  comparison_operator = "GreaterThanThreshold"
  treat_missing_data  = "notBreaching"

  dimensions = {
    StreamName = aws_kinesis_stream.events.name
  }

  alarm_actions = var.alarm_sns_topic_arns
  tags          = var.tags
}
