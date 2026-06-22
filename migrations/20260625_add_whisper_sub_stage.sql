ALTER TABLE whisper
  ADD COLUMN sub_stage VARCHAR(64) NOT NULL DEFAULT 'main' AFTER task_id,
  DROP PRIMARY KEY,
  ADD PRIMARY KEY (task_id, sub_stage);

ALTER TABLE whisper_run
  ADD COLUMN sub_stage VARCHAR(64) NOT NULL DEFAULT 'main' AFTER task_id;

ALTER TABLE video_info
  ADD COLUMN source_transcript_txt_url TEXT NULL;
