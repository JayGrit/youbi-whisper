ALTER TABLE whisper_run
  ADD COLUMN parent_run_id BIGINT NULL AFTER sub_stage,
  ADD COLUMN input_offset_ms INT NULL AFTER parent_run_id,
  ADD COLUMN input_end_ms INT NULL AFTER input_offset_ms,
  ADD INDEX idx_whisper_run_parent (parent_run_id);
