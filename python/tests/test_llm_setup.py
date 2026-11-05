from src.llm_setup import build_modelfile, weights_path


def test_weights_path_takes_the_first_from_and_skips_the_projector():
    shown = "# comment\nFROM /blobs/weights\nFROM /blobs/projector\nTEMPLATE {{ .Prompt }}\n"
    assert weights_path(shown) == "/blobs/weights"


def test_modelfile_has_no_projector():
    content = build_modelfile("/blobs/weights", 1.5)
    assert content.count("FROM ") == 1
    assert "RENDERER qwen3.5" in content
    assert "presence_penalty 1.5" in content
