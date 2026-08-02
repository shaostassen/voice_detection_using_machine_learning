from speechlens.metrics import cer, error_rate, wer


def test_wer_exact_after_normalization():
    assert wer("the cat sat", "The cat sat.") == 0.0


def test_wer_deletion():
    assert abs(wer("the cat sat", "the cat") - 1 / 3) < 1e-9


def test_wer_insertion():
    assert abs(wer("the cat sat", "the bad cat sat") - 1 / 3) < 1e-9


def test_wer_substitution():
    assert abs(wer("the cat sat", "the dog sat") - 1 / 3) < 1e-9


def test_wer_empty_cases():
    assert wer("", "") == 0.0
    assert wer("hello", "") == 1.0


def test_cer_cjk():
    assert cer("你好世界", "你好世界") == 0.0
    assert abs(cer("你好世界", "你好地球") - 0.5) < 1e-9


def test_error_rate_dispatch():
    assert error_rate("你好", "你好", language="zh")[0] == "cer"
    assert error_rate("hi there", "hi there", language="en")[0] == "wer"
