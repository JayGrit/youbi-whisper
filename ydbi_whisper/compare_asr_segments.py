from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from .config import MYSQL_CONFIG


def _load_db_segments(task_id: str) -> list[str]:
    import mysql.connector

    with mysql.connector.connect(**MYSQL_CONFIG) as conn:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT text
            FROM yd_asr_segment
            WHERE task_id = %s
            ORDER BY item_index
            """,
            (task_id,),
        )
        return [str(row[0] or "").strip() for row in cur.fetchall()]


def _output_utterances(payload: dict[str, Any]) -> list[dict[str, Any]]:
    result = payload.get("result") if isinstance(payload.get("result"), dict) else {}
    utterances = result.get("utterances") or payload.get("utterances") or payload.get("segments") or []
    return utterances if isinstance(utterances, list) else []


def _load_output_segments(output_json: Path) -> list[str]:
    payload = json.loads(output_json.read_text(encoding="utf-8"))
    return [str(item.get("text") or "").strip() for item in _output_utterances(payload) if isinstance(item, dict)]


def write_compare_csv(task_id: str, output_json: Path, csv_path: Path) -> dict[str, int | str]:
    db_segments = _load_db_segments(task_id)
    output_segments = _load_output_segments(output_json)

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    total = max(len(db_segments), len(output_segments))
    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["序号", "数据库分句", "output分句"])
        for index in range(total):
            writer.writerow(
                [
                    index,
                    db_segments[index] if index < len(db_segments) else "",
                    output_segments[index] if index < len(output_segments) else "",
                ]
            )

    return {
        "task_id": task_id,
        "db_segments": len(db_segments),
        "output_segments": len(output_segments),
        "csv_path": str(csv_path.resolve()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare ASR segments from MySQL and a local ASR output JSON.")
    parser.add_argument("task_id", help="Pipeline task id, for example yaEagn27eGE.")
    parser.add_argument("output_json", type=Path, help="Local ASR payload JSON path.")
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="CSV output path. Default: test_outputs/<task_id>.asr_compare.csv",
    )
    args = parser.parse_args()

    output_json = args.output_json.expanduser().resolve()
    if not output_json.is_file():
        raise FileNotFoundError(f"output json not found: {output_json}")

    csv_path = args.csv.expanduser() if args.csv else Path("test_outputs") / f"{args.task_id}.asr_compare.csv"
    result = write_compare_csv(args.task_id, output_json, csv_path)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
