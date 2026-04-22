def test_cli_imports():
    from thai_plate_ocr.cli import app

    assert app is not None
    assert {c.name for c in app.registered_commands} >= {
        "download",
        "train-detector",
        "train-recognizer",
        "evaluate",
        "predict",
    }
