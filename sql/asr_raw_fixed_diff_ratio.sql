-- Compare raw and fixed ASR segments.
-- Compatible with MySQL 5.7+.
-- Run:
--   mysql -h 120.53.92.66 -P 3306 -u hoshuuch -p490229 youbi < sql/asr_raw_fixed_diff_ratio.sql

SELECT
  'overall' AS scope,
  COUNT(*) AS task_count,
  SUM(t.raw_segments) AS raw_segments,
  SUM(t.fixed_segments) AS fixed_segments,
  SUM(t.paired_segments) AS paired_segments,
  SUM(t.text_different_segments) AS text_different_segments,
  SUM(t.full_different_segments) AS full_different_segments,
  SUM(t.segment_count_mismatch) AS tasks_with_segment_count_mismatch,
  ROUND(
    SUM(t.text_different_segments) / NULLIF(SUM(t.paired_segments), 0),
    6
  ) AS text_different_ratio,
  ROUND(
    SUM(t.full_different_segments) / NULLIF(SUM(t.paired_segments), 0),
    6
  ) AS full_segment_different_ratio,
  ROUND(
    SUM(t.full_different_segments) / NULLIF(SUM(t.raw_segments), 0),
    6
  ) AS full_different_ratio_by_raw_segments
FROM (
  SELECT
    c.task_id,
    c.raw_segments,
    c.fixed_segments,
    COUNT(p.item_index) AS paired_segments,
    COALESCE(SUM(p.is_text_different), 0) AS text_different_segments,
    COALESCE(SUM(p.is_full_different), 0) AS full_different_segments,
    CASE
      WHEN c.raw_segments = c.fixed_segments AND c.raw_segments = COUNT(p.item_index)
      THEN 0
      ELSE 1
    END AS segment_count_mismatch
  FROM (
    SELECT
      task_id,
      SUM(segment_type = 'raw') AS raw_segments,
      SUM(segment_type = 'fixed') AS fixed_segments
    FROM yd_asr_segment
    WHERE segment_type IN ('raw', 'fixed')
    GROUP BY task_id
  ) c
  LEFT JOIN (
    SELECT
      r.task_id,
      r.item_index,
      CASE
        WHEN r.text <=> f.text THEN 0
        ELSE 1
      END AS is_text_different,
      CASE
        WHEN r.text <=> f.text
          AND r.start_time <=> f.start_time
          AND r.end_time <=> f.end_time
          AND r.speaker <=> f.speaker
          AND r.words_json <=> f.words_json
        THEN 0
        ELSE 1
      END AS is_full_different
    FROM yd_asr_segment r
    JOIN yd_asr_segment f
      ON f.task_id = r.task_id
     AND f.item_index = r.item_index
     AND f.segment_type = 'fixed'
    WHERE r.segment_type = 'raw'
  ) p ON p.task_id = c.task_id
  GROUP BY c.task_id, c.raw_segments, c.fixed_segments
) t;

SELECT
  c.task_id,
  c.raw_segments,
  c.fixed_segments,
  COUNT(p.item_index) AS paired_segments,
  COALESCE(SUM(p.is_text_different), 0) AS text_different_segments,
  COALESCE(SUM(p.is_full_different), 0) AS full_different_segments,
  ROUND(
    COALESCE(SUM(p.is_text_different), 0) / NULLIF(COUNT(p.item_index), 0),
    6
  ) AS text_different_ratio,
  ROUND(
    COALESCE(SUM(p.is_full_different), 0) / NULLIF(COUNT(p.item_index), 0),
    6
  ) AS full_segment_different_ratio,
  CASE
    WHEN c.raw_segments = c.fixed_segments AND c.raw_segments = COUNT(p.item_index)
    THEN 0
    ELSE 1
  END AS segment_count_mismatch
FROM (
  SELECT
    task_id,
    SUM(segment_type = 'raw') AS raw_segments,
    SUM(segment_type = 'fixed') AS fixed_segments
  FROM yd_asr_segment
  WHERE segment_type IN ('raw', 'fixed')
  GROUP BY task_id
) c
LEFT JOIN (
  SELECT
    r.task_id,
    r.item_index,
    CASE
      WHEN r.text <=> f.text THEN 0
      ELSE 1
    END AS is_text_different,
    CASE
      WHEN r.text <=> f.text
        AND r.start_time <=> f.start_time
        AND r.end_time <=> f.end_time
        AND r.speaker <=> f.speaker
        AND r.words_json <=> f.words_json
      THEN 0
      ELSE 1
    END AS is_full_different
  FROM yd_asr_segment r
  JOIN yd_asr_segment f
    ON f.task_id = r.task_id
   AND f.item_index = r.item_index
   AND f.segment_type = 'fixed'
  WHERE r.segment_type = 'raw'
) p ON p.task_id = c.task_id
GROUP BY c.task_id, c.raw_segments, c.fixed_segments
ORDER BY full_segment_different_ratio DESC, full_different_segments DESC, c.task_id;
