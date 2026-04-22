from thai_plate_ocr.pipeline import CharBox, order_characters


def _c(cls: str, x: float, y: float, w: float = 20.0, h: float = 40.0) -> CharBox:
    return CharBox(cls=cls, conf=0.9, x1=x, y1=y, x2=x + w, y2=y + h)


def test_single_line_left_to_right():
    chars = [_c("1", 100, 10), _c("ก", 10, 10), _c("2", 60, 10), _c("3", 140, 10)]
    lines = order_characters(chars)
    assert [c.cls for line in lines for c in line] == ["ก", "2", "1", "3"]
    assert len(lines) == 1


def test_two_line_plate_splits_on_vertical_gap():
    # line 1 near y=10, line 2 near y=80 (gap=70, char_height=40 → gap > 0.6 * 40 = 24)
    chars = [
        _c("ก", 10, 10), _c("ข", 50, 10),
        _c("ก", 20, 80), _c("ร", 70, 80), _c("ช", 120, 80),
    ]
    lines = order_characters(chars)
    assert len(lines) == 2
    assert [c.cls for c in lines[0]] == ["ก", "ข"]
    assert [c.cls for c in lines[1]] == ["ก", "ร", "ช"]


def test_empty_detections_return_empty():
    assert order_characters([]) == []
