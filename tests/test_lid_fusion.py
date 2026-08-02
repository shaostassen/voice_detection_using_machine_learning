from speechlens.lid import entropy, fuse_distributions, language_name


def test_fusion_agreement():
    fused = fuse_distributions([
        {"en": 0.6, "de": 0.3, "fr": 0.1},
        {"en": 0.7, "de": 0.1, "fr": 0.2},
    ])
    assert max(fused, key=fused.get) == "en"
    assert abs(sum(fused.values()) - 1.0) < 1e-9


def test_fusion_vetoes_corrupted_chunk():
    """Two clean chunks say 'en'; one noise-corrupted chunk says 'nn'.
    Geometric-mean fusion keeps 'en' — a majority-of-good-windows property."""
    fused = fuse_distributions([
        {"en": 0.9, "nn": 0.1},
        {"en": 0.85, "nn": 0.15},
        {"en": 0.2, "nn": 0.8},
    ])
    assert max(fused, key=fused.get) == "en"


def test_entropy_orders_confidence():
    peaked = {"en": 0.98, "de": 0.02}
    flat = {"en": 0.5, "de": 0.5}
    assert entropy(peaked) < entropy(flat)


def test_language_name_fallback():
    assert language_name("en") == "English"
    assert language_name("xx") == "xx"
