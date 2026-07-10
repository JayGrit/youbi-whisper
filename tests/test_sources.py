from ydbi_whisper.sources import detect_source


def test_detects_douyin_as_chinese_source() -> None:
    source = detect_source("https://www.douyin.com/video/7658678365918246058")

    assert source.name == "douyin"
    assert source.asr_language == "zh"
