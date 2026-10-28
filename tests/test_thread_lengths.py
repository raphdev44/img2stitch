"""Longueur de fil estimée par couleur, affichée dans les caractéristiques du motif."""
import pyembroidery as pe

from img2stitch.convert import THREAD_SETUP_MM, thread_lengths_mm


def _pattern(stitches, threads=2):
    pattern = pe.EmbPattern()
    for i in range(threads):
        pattern.add_thread({"color": 0xFF0000 + i})
    pattern.stitches = stitches
    return pattern


def test_lengths_follow_thread_order_and_skip_jumps():
    pattern = _pattern([
        [0, 0, pe.STITCH], [100, 0, pe.STITCH],        # fil 1 : 10 mm piqués
        [100, 0, pe.COLOR_CHANGE],
        [0, 0, pe.JUMP], [400, 0, pe.JUMP],            # déplacement : pas de fil posé
        [400, 0, pe.STITCH], [460, 0, pe.STITCH],      # fil 2 : 6 mm piqués
    ])
    lengths = thread_lengths_mm(pattern)
    assert len(lengths) == 2
    assert lengths == [93, 88]  # 13 mm + amorçage, 7,8 mm + amorçage


def test_setup_is_counted_for_every_thread():
    pattern = _pattern([[0, 0, pe.STITCH], [0, 0, pe.COLOR_CHANGE],
                        [10, 0, pe.STITCH], [20, 0, pe.STITCH]], threads=2)
    assert thread_lengths_mm(pattern) == [THREAD_SETUP_MM, THREAD_SETUP_MM + 1]


def test_empty_pattern():
    assert thread_lengths_mm(pe.EmbPattern()) == []
