-- 统计 youbi.yd_asr_segment.text 字符长度分布。
-- 分组：
--   1) 2026-06-01 23:59:59 及之前，即 created_at < '2026-06-02'
--   2) 2026-06-02 00:00:00 及以后，即 created_at >= '2026-06-02'
-- 档位宽度：20 个字符，0-20、21-40、41-60 ...

WITH segment_lengths AS (
  SELECT
    CASE
      WHEN created_at < '2026-06-02' THEN '2026-06-01_and_before'
      ELSE '2026-06-02_and_after'
    END AS date_group,
    CHAR_LENGTH(TRIM(COALESCE(text, ''))) AS text_len
  FROM youbi.yd_asr_segment
  WHERE created_at < '2026-06-02'
     OR created_at >= '2026-06-02'
),
bucketed AS (
  SELECT
    date_group,
    text_len,
    CASE
      WHEN text_len <= 20 THEN 0
      ELSE FLOOR((text_len - 1) / 20) * 20 + 1
    END AS bucket_start,
    CASE
      WHEN text_len <= 20 THEN 20
      ELSE FLOOR((text_len - 1) / 20) * 20 + 20
    END AS bucket_end
  FROM segment_lengths
),
bucket_counts AS (
  SELECT
    date_group,
    bucket_start,
    bucket_end,
    COUNT(*) AS segment_count
  FROM bucketed
  GROUP BY date_group, bucket_start, bucket_end
),
group_totals AS (
  SELECT
    date_group,
    COUNT(*) AS total_count
  FROM segment_lengths
  GROUP BY date_group
)
SELECT
  bc.date_group,
  CONCAT(bc.bucket_start, '-', bc.bucket_end) AS text_len_bucket,
  bc.segment_count,
  gt.total_count,
  ROUND(bc.segment_count / gt.total_count, 6) AS probability
FROM bucket_counts bc
JOIN group_totals gt ON gt.date_group = bc.date_group
ORDER BY
  CASE bc.date_group
    WHEN '2026-06-01_and_before' THEN 1
    ELSE 2
  END,
  bc.bucket_start;
