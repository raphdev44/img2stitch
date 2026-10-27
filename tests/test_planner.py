import cv2
import numpy as np
import pyembroidery as pe
import pytest
from PIL import Image, ImageDraw

from img2stitch import Settings
from img2stitch.convert import image_to_pattern, pattern_stats


def _img(tmp_path, draw):
    im = Image.new("RGB", (400, 400), "white")
    draw(ImageDraw.Draw(im))
    f = tmp_path / "t.png"
    im.save(f)
    return str(f)


def test_thicken_lines_preserves_and_expands_hairline(tmp_path):
    from img2stitch.preprocess import load_and_quantize

    path = _img(tmp_path, lambda d: d.line((50, 200, 350, 200), fill="black", width=1))
    _, default_masks = load_and_quantize(path, 400, 2, background="none")
    palette, thickened_masks = load_and_quantize(path, 400, 2, background="none", thicken_lines=True)

    assert len(default_masks) == 1
    dark_index = min(range(len(palette)), key=lambda i: sum(palette[i]))
    assert thickened_masks[dark_index].sum() >= 900


def test_excluded_colors_are_reassigned_without_removing_pixels():
    from img2stitch.preprocess import _embroidered_pixels, load_and_quantize

    image = Image.new("RGB", (60, 40), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((10, 10, 24, 29), fill=(250, 20, 20))
    draw.rectangle((25, 10, 39, 29), fill=(200, 20, 20))
    draw.rectangle((40, 10, 49, 29), fill=(20, 20, 200))
    excluded = [((250, 20, 20), ((200, 20, 20), (20, 20, 200)))]
    _, original = _embroidered_pixels(image, 60, "auto", bg_scope="outside")
    rgb, included = _embroidered_pixels(image, 60, "auto", bg_scope="outside", exclude=excluded)
    np.testing.assert_array_equal(included, original)
    np.testing.assert_array_equal(rgb[15, 15], (200, 20, 20))
    palette, masks = load_and_quantize(image, 60, 2, "auto", bg_scope="outside",
                                      exclude=excluded, thicken_lines=True)
    np.testing.assert_array_equal(np.any(masks, axis=0), original)
    assert set(palette) == {(200, 20, 20), (20, 20, 200)}


def test_excluded_colors_preserve_transparency():
    from img2stitch.preprocess import _embroidered_pixels

    image = Image.new("RGBA", (3, 1))
    image.putpixel((1, 0), (250, 20, 20, 255))
    image.putpixel((2, 0), (20, 20, 200, 255))
    excluded = [((250, 20, 20), ((20, 20, 200),))]
    rgb, included = _embroidered_pixels(image, 3, background="none", exclude=excluded)
    np.testing.assert_array_equal(included, [[False, True, True]])
    np.testing.assert_array_equal(rgb[0, 1:], [[20, 20, 200], [20, 20, 200]])


def test_ring_with_covered_hole_needs_at_most_one_jump(tmp_path):
    """Anneau clair dont le trou sera rempli en bleu foncé (piqué après, car on pique du plus clair au plus foncé) :
    le fil de l'anneau traverse le trou à couvert. Un seul saut est inévitable : passer d'une moitié de
    l'anneau à l'autre oblige à franchir du fil déjà posé."""
    def draw(d):
        d.ellipse((50, 50, 350, 350), fill=(250, 210, 40))
        d.ellipse((130, 130, 270, 270), fill=(30, 60, 200))
    path = _img(tmp_path, draw)
    st = pattern_stats(image_to_pattern(path, Settings(width_mm=60, n_colors=2, outline=False)))
    assert st["jumps"] <= 1 and st["trims"] <= 1, st


def test_ring_with_background_hole_has_few_jumps(tmp_path):
    """Trou vide (fond) : on ne doit PAS sauter à chaque rangée (~40 sauts avec un parcours naïf),
    mais seulement quelques fois."""
    def draw(d):
        d.ellipse((50, 50, 350, 350), fill=(220, 30, 30))
        d.ellipse((130, 130, 270, 270), fill=(255, 255, 255))
    path = _img(tmp_path, draw)
    st = pattern_stats(image_to_pattern(path, Settings(width_mm=60, n_colors=1, outline=False)))
    assert st["jumps"] <= 6, st


def test_far_jump_is_trimmed(tmp_path):
    """Chaque saut long doit être précédé d'une coupe de fil."""
    def draw(d):
        d.rectangle((20, 20, 120, 120), fill=(220, 30, 30))
        d.rectangle((280, 280, 380, 380), fill=(220, 30, 30))
    path = _img(tmp_path, draw)
    pat = image_to_pattern(path, Settings(width_mm=60, n_colors=1))
    cmds = [c & pe.COMMAND_MASK for _, _, c in pat.stitches]
    assert pe.TRIM in cmds
    for i, c in enumerate(cmds):
        if c == pe.JUMP and cmds[i - 1] not in (pe.COLOR_CHANGE,) and i > 1:
            assert cmds[i - 1] == pe.TRIM, "saut long sans coupe"


@pytest.mark.parametrize("excluded", [(), (((220, 30, 30), ((30, 160, 70),)),)])
def test_every_filled_pixel_is_covered_by_thread(tmp_path, excluded):
    """L'optimisation du parcours ne doit oublier aucune zone à broder (et laisser le trou vide vide)."""
    from img2stitch.preprocess import load_and_quantize

    def draw(d):
        d.ellipse((30, 30, 330, 330), fill=(220, 30, 30))
        d.ellipse((110, 110, 250, 250), fill=(255, 255, 255))     # trou vide
        d.polygon([(330, 60), (390, 60), (360, 190)], fill=(30, 160, 70))
    path = _img(tmp_path, draw)
    s = Settings(width_mm=60, n_colors=2, exclude_colors=excluded)
    s.center_on_origin = False   # on compare aux masques : même repère que l'image
    pat = image_to_pattern(path, s)

    # Les coordonnées du motif (1/10 mm) valent des pixels à 10 px/mm : même repère que les masques
    _, masks = load_and_quantize(path, 600, 2, "auto", 40)
    target = np.any(masks, axis=0)
    canvas = np.zeros(target.shape, np.uint8)
    last = None
    for x, y, c in pat.stitches:
        cur = (round(x), round(y))
        if c & pe.COMMAND_MASK == pe.STITCH and last is not None:
            cv2.line(canvas, last, cur, 255, 5)   # fil de 0,5 mm
        last = cur
    core = cv2.erode(target.astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool)   # on ignore 0,3 mm de bord
    assert (canvas[core] > 0).mean() > 0.97, "zone oubliée par le parcours"
    hole = np.zeros_like(target)
    cv2.circle(hole.view(np.uint8), (270, 270), 40, 1, -1)    # centre du trou vide (image 400 px -> 600 px)
    assert (canvas[hole] > 0).mean() < 0.05, "du fil a été piqué dans le trou"


def test_pattern_is_centered_on_origin(tmp_path):
    """Les machines placent l'origine au centre du cerceau : le motif doit y être centré."""
    def draw(d):
        d.ellipse((200, 200, 380, 380), fill=(220, 30, 30))      # forme excentrée dans l'image
    path = _img(tmp_path, draw)
    pat = image_to_pattern(path, Settings(width_mm=60, n_colors=1))
    pts = np.array([(x, y) for x, y, c in pat.stitches if (c & pe.COMMAND_MASK) in (pe.STITCH, pe.JUMP)])
    assert abs((pts[:, 0].min() + pts[:, 0].max()) / 2) < 1 and abs((pts[:, 1].min() + pts[:, 1].max()) / 2) < 1
