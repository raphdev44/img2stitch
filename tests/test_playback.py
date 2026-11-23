import pyembroidery as pe

from img2stitch.playback import COLOR_CHANGE, JUMP, SEW, TRIM, playback_data


def _pattern():
    p = pe.EmbPattern()
    p.add_thread({"color": 0xFF0000})
    p.add_stitch_absolute(pe.STITCH, 100, 100)
    p.add_stitch_absolute(pe.STITCH, 120, 100)
    p.add_stitch_absolute(pe.JUMP, 300, 300)
    p.add_command(pe.TRIM)
    p.add_stitch_absolute(pe.STITCH, 320, 300)
    p.add_command(pe.COLOR_CHANGE)
    p.add_thread({"color": 0x00FF00})
    p.add_stitch_absolute(pe.STITCH, 330, 300)
    p.end()
    return p


def test_playback_keeps_the_order_of_the_machine():
    """Le film doit rejouer les commandes dans l'ordre : piqûres, saut, coupe, changement de fil."""
    data = playback_data(_pattern())
    assert [op[0] for op in data["ops"]] == [SEW, JUMP, TRIM, SEW, COLOR_CHANGE, SEW]
    assert data["threads"] == ["#ff0000", "#00ff00"]


def test_playback_coordinates_are_relative_to_the_stitched_area():
    data = playback_data(_pattern())
    assert data["bbox"] == [100, 100, 230, 200]

    def xy(op):
        return [(op[i], op[i + 1]) for i in range(2, len(op), 2)]

    assert xy(data["ops"][0]) == [(0, 0), (20, 0)]
    assert xy(data["ops"][1]) == [(200, 200)]
    # Après la coupe, la piqûre repart de la position du saut : la machine ne bouge pas sans fil.
    assert xy(data["ops"][3]) == [(200, 200), (220, 200)]
    assert data["ops"][4] == [COLOR_CHANGE, 1]
    assert xy(data["ops"][5]) == [(220, 200), (230, 200)]


def test_playback_counts_match_the_commands():
    data = playback_data(_pattern())
    assert data["counts"] == {"stitches": 4, "jumps": 1, "trims": 1, "color_changes": 1, "stops": 0}


def test_empty_pattern_has_no_operations():
    pattern = pe.EmbPattern()
    pattern.end()
    data = playback_data(pattern)
    assert data["ops"] == [] and data["bbox"] == [0, 0, 0, 0]
