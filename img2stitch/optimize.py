"""Conservative cleanup passes for existing embroidery patterns."""
from copy import deepcopy
from math import ceil, hypot

import pyembroidery as pe


def optimize_pattern(pattern, *, min_stitch_mm=0.3, max_stitch_mm=12.0, merge_travel_mm=2.5):
    """Return an optimized copy and a compact report without mutating ``pattern``.

    The application retains zone start positions, but not enough geometry and layer
    dependencies to safely reorder objects or change fill density.
    """
    optimized = pattern.copy()
    optimized.stitches = deepcopy(pattern.stitches)
    optimized.threadlist = deepcopy(pattern.threadlist)
    optimized.extras = deepcopy(pattern.extras)
    for name, value in pattern.__dict__.items():
        if name not in {"stitches", "threadlist", "extras", "_previousX", "_previousY"}:
            setattr(optimized, name, deepcopy(value))
    original = [list(stitch) for stitch in pattern.stitches]
    stitches = [(list(stitch), index) for index, stitch in enumerate(original)]
    segments = getattr(pattern, "img2stitch_segments", None) or []
    object_starts = {int(start) for start, _ in segments}
    report = {
        "removed_duplicates": 0,
        "removed_micro_stitches": 0,
        "merged_trims": 0,
        "split_long_stitches": 0,
        "warnings": [
            "Les dépendances de superposition et les couches ne sont pas décrites : l'ordre et la densité sont conservés."
        ],
    }

    min_units = min_stitch_mm * 10
    max_units = max_stitch_mm * 10
    merge_units = merge_travel_mm * 10
    cleaned = []
    index = 0
    while index < len(stitches):
        if stitches[index][0][2] & pe.COMMAND_MASK != pe.STITCH:
            cleaned.append(stitches[index])
            index += 1
            continue

        end = index + 1
        while end < len(stitches) and stitches[end][0][2] & pe.COMMAND_MASK == pe.STITCH:
            end += 1
        segment = stitches[index:end]
        unique = []
        for item in segment:
            if unique and _distance(unique[-1][0], item[0]) <= 0.5:
                report["removed_duplicates"] += 1
                continue
            unique.append(item)

        # Keep each segment's endpoints and cap geometric pruning at 25%.
        limit = len(unique) // 4
        kept = []
        removed = 0
        for position, item in enumerate(unique):
            stitch, source_index = item
            is_endpoint = (position == 0 or position == len(unique) - 1
                           or source_index in object_starts)
            if (not is_endpoint and removed < limit and kept
                    and _distance(kept[-1][0], stitch) < min_units):
                removed += 1
                report["removed_micro_stitches"] += 1
                continue
            kept.append(item)
        cleaned.extend(kept)
        index = end

    # A short trimmed travel is safe to leave connected; never cross color changes.
    untrimmed = []
    index = 0
    while index < len(cleaned):
        if cleaned[index][0][2] & pe.COMMAND_MASK != pe.TRIM:
            untrimmed.append(cleaned[index])
            index += 1
            continue
        end = index + 1
        while end < len(cleaned) and cleaned[end][0][2] & pe.COMMAND_MASK == pe.JUMP:
            end += 1
        previous = _last_position(untrimmed)
        travel_end = cleaned[end - 1][0] if end > index + 1 else None
        if (previous is not None and travel_end is not None
                and end < len(cleaned) and cleaned[end][0][2] & pe.COMMAND_MASK == pe.STITCH
                and _distance(previous, travel_end) <= merge_units):
            report["merged_trims"] += 1
            untrimmed.extend(cleaned[index + 1:end])
        else:
            untrimmed.extend(cleaned[index:end])
        index = end
    cleaned = untrimmed

    # pyembroidery coordinates are in tenths of a millimeter.
    bounded = []
    previous = None
    for stitch, source_index in cleaned:
        command = stitch[2] & pe.COMMAND_MASK
        if command == pe.STITCH and previous is not None:
            distance = _distance(previous, stitch)
            pieces = max(1, ceil(distance / max_units))
            if pieces > 1:
                report["split_long_stitches"] += pieces - 1
                for part in range(1, pieces):
                    ratio = part / pieces
                    bounded.append(([
                        round(previous[0] + (stitch[0] - previous[0]) * ratio),
                        round(previous[1] + (stitch[1] - previous[1]) * ratio),
                        stitch[2],
                    ], source_index))
        bounded.append((stitch, source_index))
        previous = stitch

    optimized.stitches = [stitch for stitch, _ in bounded]
    if segments:
        remapped = []
        starts = sorted(object_starts)
        for old_start, zone_index in segments:
            next_start = min((value for value in starts if value > old_start), default=None)
            new_start = next((
                output_index for output_index, (_, source_index) in enumerate(bounded)
                if source_index >= old_start and (next_start is None or source_index < next_start)
            ), None)
            if new_start is not None:
                remapped.append((new_start, zone_index))
        optimized.img2stitch_segments = remapped
    report["before"] = {
        "stitches": sum(1 for s in original if s[2] & pe.COMMAND_MASK == pe.STITCH),
        "trims": sum(1 for s in original if s[2] & pe.COMMAND_MASK == pe.TRIM),
    }
    report["after"] = {
        "stitches": sum(1 for s, _ in bounded if s[2] & pe.COMMAND_MASK == pe.STITCH),
        "trims": sum(1 for s, _ in bounded if s[2] & pe.COMMAND_MASK == pe.TRIM),
    }
    report["min_stitch_mm"] = min_stitch_mm
    report["max_stitch_mm"] = max_stitch_mm
    report["merge_travel_mm"] = merge_travel_mm
    return optimized, report


def _distance(first, second):
    return hypot(second[0] - first[0], second[1] - first[1])


def _last_position(stitches):
    for stitch, _ in reversed(stitches):
        if stitch[2] & pe.COMMAND_MASK in (pe.STITCH, pe.JUMP):
            return stitch
    return None
