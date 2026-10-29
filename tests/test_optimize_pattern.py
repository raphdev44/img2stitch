import pyembroidery as pe

from img2stitch.optimize import optimize_pattern


def _pattern(stitches):
    pattern = pe.EmbPattern()
    pattern.add_thread({"color": 0x123456})
    pattern.stitches = [list(stitch) for stitch in stitches]
    return pattern


def test_removes_duplicate_and_short_interior_stitches_without_mutating_input():
    pattern = _pattern([
        [0, 0, pe.STITCH],
        [0, 0, pe.STITCH],
        [1, 0, pe.STITCH],
        [10, 0, pe.STITCH],
        [20, 0, pe.STITCH],
    ])
    before = [stitch[:] for stitch in pattern.stitches]

    optimized, report = optimize_pattern(pattern)

    assert pattern.stitches == before
    assert optimized is not pattern
    assert optimized.stitches[0] == [0, 0, pe.STITCH]
    assert optimized.stitches[-1] == [20, 0, pe.STITCH]
    assert report["removed_duplicates"] == 1
    assert report["removed_micro_stitches"] == 1


def test_only_merges_trim_before_short_jump():
    pattern = _pattern([
        [0, 0, pe.STITCH],
        [0, 0, pe.TRIM],
        [20, 0, pe.JUMP],
        [20, 0, pe.STITCH],
        [20, 0, pe.TRIM],
        [50, 0, pe.JUMP],
        [50, 0, pe.STITCH],
    ])

    optimized, report = optimize_pattern(pattern)

    assert report["merged_trims"] == 1
    assert sum(s[2] & pe.COMMAND_MASK == pe.TRIM for s in optimized.stitches) == 1
    assert optimized.stitches[1][2] == pe.JUMP


def test_does_not_merge_a_trim_across_a_color_change():
    pattern = _pattern([
        [0, 0, pe.STITCH],
        [0, 0, pe.TRIM],
        [20, 0, pe.JUMP],
        [20, 0, pe.COLOR_CHANGE],
        [20, 0, pe.STITCH],
    ])

    optimized, report = optimize_pattern(pattern)

    assert report["merged_trims"] == 0
    assert any(s[2] & pe.COMMAND_MASK == pe.TRIM for s in optimized.stitches)


def test_splits_stitches_over_machine_length_limit():
    optimized, report = optimize_pattern(_pattern([
        [0, 0, pe.STITCH],
        [250, 0, pe.STITCH],
    ]))

    assert report["split_long_stitches"] == 2
    positions = [s[:2] for s in optimized.stitches]
    assert positions == [[0, 0], [83, 0], [167, 0], [250, 0]]
    assert max(abs(b[0] - a[0]) for a, b in zip(positions, positions[1:])) <= 120


def test_remaps_zone_start_indices_after_removing_stitches():
    pattern = _pattern([
        [0, 0, pe.STITCH],
        [0, 0, pe.STITCH],
        [10, 0, pe.STITCH],
        [10, 0, pe.COLOR_CHANGE],
        [20, 0, pe.STITCH],
    ])
    pattern.img2stitch_segments = [(0, 0), (4, 1)]

    optimized, _ = optimize_pattern(pattern)

    assert optimized.img2stitch_segments == [(0, 0), (3, 1)]
