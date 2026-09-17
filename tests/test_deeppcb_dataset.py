from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from crackxnet_app.config import CLASS_ID_TO_NAME
from crackxnet_app.deeppcb.dataset import DeepPCBError, parse_annotation_file, train_val_test_samples


def test_class_mapping_matches_deeppcb_encoding() -> None:
    assert CLASS_ID_TO_NAME[1] == "Open Circuit"
    assert CLASS_ID_TO_NAME[2] == "Short Circuit"
    assert CLASS_ID_TO_NAME[3] == "Mouse Bite"
    assert CLASS_ID_TO_NAME[4] == "Spur"
    assert CLASS_ID_TO_NAME[5] == "Spurious Copper"
    assert CLASS_ID_TO_NAME[6] == "Pin Hole"


def test_annotation_parsing_and_invalid_handling(tmp_path: Path) -> None:
    ann = tmp_path / "sample.txt"
    ann.write_text("1 2 10 20 3\n", encoding="utf-8")
    parsed = parse_annotation_file(ann, (32, 32))
    assert parsed[0].class_id == 3
    assert parsed[0].x2 == 10

    bad = tmp_path / "bad.txt"
    bad.write_text("1 2 1 20 3\n", encoding="utf-8")
    with pytest.raises(DeepPCBError):
        parse_annotation_file(bad, (32, 32))


def test_split_reproducibility_with_minimal_fixture(tmp_path: Path) -> None:
    pcb = tmp_path / "PCBData"
    image_dir = pcb / "group00001" / "00001"
    ann_dir = pcb / "group00001" / "00001_not"
    image_dir.mkdir(parents=True)
    ann_dir.mkdir(parents=True)
    lines = []
    for index in range(6):
        stem = f"0000100{index}"
        Image.new("RGB", (640, 640), (0, 0, 0)).save(image_dir / f"{stem}_test.jpg")
        (ann_dir / f"{stem}.txt").write_text("1 2 10 20 1\n", encoding="utf-8")
        lines.append(f"group00001/00001/{stem}.jpg group00001/00001_not/{stem}.txt")
    (pcb / "trainval.txt").write_text("\n".join(lines[:4]), encoding="utf-8")
    (pcb / "test.txt").write_text("\n".join(lines[4:]), encoding="utf-8")

    a = train_val_test_samples(tmp_path, seed=7)
    b = train_val_test_samples(tmp_path, seed=7)
    assert [s.image_path for s in a[0]] == [s.image_path for s in b[0]]
    assert sum(len(part) for part in a) == 6
