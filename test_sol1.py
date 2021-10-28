import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

from sol1 import *
import skimage.color

RGB_REP = 2
GREY_REP = 1

path_to_images_dir = 'externals/presubmit_externals'  # todo: change to a path in your system that contain images


def _generate_gradient_image():
    x = np.hstack([np.repeat(np.arange(0, 50, 2), 10)[None, :], np.array([255] * 6)[None, :]])
    grad = np.tile(x, (256, 1))

    return grad


def test_read_image():
    images = Path(path_to_images_dir).glob('*.jpg')

    for image in images:
        img = read_image(image, GREY_REP)
        assert len(img.shape) == 2
        assert img.ndim == 2
        assert img.dtype == np.float64

        img = read_image(image, RGB_REP)
        assert len(img.shape) == 3
        assert img.ndim == 3
        assert img.dtype == np.float64


def test_plot_image():
    images = Path(path_to_images_dir).glob('*.jpg')

    for image in images:
        try:
            imdisplay(image, GREY_REP)
            imdisplay(image, RGB_REP)
            assert 'Great, check printed plot'

        except Exception as e:
            assert not f'Souldnt raise exception, captured this: {e}'


def test_rgb2yiq():
    images = Path(path_to_images_dir).glob('*.jpg')

    for image in images:
        img = read_image(image, RGB_REP)
        yiq_skimage = skimage.color.rgb2yiq(img)
        yiq = rgb2yiq(img)

        assert np.all(yiq.flatten() < 1)
        assert np.all(yiq.flatten() > -1)
        assert np.all(yiq[:, :, 0] >= 0)
        assert np.all(yiq[:, :, 0] <= 1)
        assert np.all(yiq[:, :, 1] >= -1)
        assert np.all(yiq[:, :, 1] <= 1)
        assert np.all(yiq[:, :, 2] >= -1)
        assert np.all(yiq[:, :, 2] <= 1)

        delta = 0.01
        assert np.all((yiq.flatten() - yiq_skimage.flatten()) < delta)


def test_yiq2rgb():
    images = Path(path_to_images_dir).glob('*.jpg')

    for image in images:
        img = read_image(image, RGB_REP)
        yiq_skimage = skimage.color.rgb2yiq(img)
        rgb_skimage = skimage.color.yiq2rgb(yiq_skimage)
        yiq = rgb2yiq(img)
        rgb = yiq2rgb(yiq)

        delta = 0.01

        assert np.all((img.flatten() - rgb.flatten()) < delta)
        assert np.all((img.flatten() - rgb_skimage.flatten()) < delta)


def test_equalizer():
    images = Path(path_to_images_dir).glob('*.jpg')

    for image in images:
        for color in [GREY_REP, RGB_REP]:
            img = read_image(image, color)
            im_eq, hist_orig, hist_eq = histogram_equalize(img)
            np.clip(a=im_eq, a_min=0, a_max=1, out=im_eq)
            assert hist_orig.size == 256 and hist_eq.size == 256
            _show_eq(img, color, im_eq, hist_orig, hist_eq)


def test_toy_eq():
    toy_example = _generate_gradient_image()
    im_eq, hist_orig, hist_eq = histogram_equalize((toy_example / 255))
    np.clip(a=im_eq, a_min=0, a_max=1, out=im_eq)
    _show_eq(toy_example, GREY_REP, im_eq, hist_orig, hist_eq)


def test_quant():
    images = Path(path_to_images_dir).glob('*.jpg')
    n_iters = 15
    for i, image in enumerate(images):
        img = read_image(image, GREY_REP)

        for n_q in range(1, 20, 3):
            im_quant, errors = quantize(img, n_quant=n_q, n_iter=n_iters)
            _show_quant(image, im_quant, errors, n_q)

            assert len(errors) <= n_iters
            assert np.all(im_quant <= 255)
            assert np.all(im_quant >= 0)
            assert np.issubdtype(im_quant.dtype, np.integer)


def test_quant_iter_halts():
    images = Path(path_to_images_dir).glob('*.jpg')
    n_iters = 2
    for i, image in enumerate(images):
        img = read_image(image, GREY_REP)

        for n_q in range(2, 20, 5):
            im_quant, errors = quantize(img, n_quant=n_q, n_iter=n_iters)
            _show_quant(image, im_quant, errors, n_q)

            assert len(errors) <= n_iters
            assert np.all(im_quant <= 255)
            assert np.all(im_quant >= 0)
            assert np.issubdtype(im_quant.dtype, np.integer)


def test_toy_quant():
    toy_example = _generate_gradient_image() / 255
    im_eq, hist_orig, hist_eq = histogram_equalize((toy_example))
    n_iters = 15
    for n_q in range(1, 25, 3):
        im_quant, errors = quantize(im_eq, n_quant=n_q, n_iter=n_iters)
        _show_quant(Path('toy example'), im_quant, errors, n_q)

        assert len(errors) <= n_iters
        assert np.all(im_quant <= 255)
        assert np.all(im_quant >= 0)
        assert np.issubdtype(im_quant.dtype, np.integer)


def test_from_pdf_example():
    toy_example = _generate_gradient_image() / 255
    n_iters = 10
    im_eq, hist_orig, hist_eq = histogram_equalize((toy_example))
    np.clip(a=im_eq, a_min=0, a_max=1, out=im_eq)
    _show_eq(toy_example, GREY_REP, im_eq, hist_orig, hist_eq)
    im_quant, errors = quantize(im_eq, n_quant=5, n_iter=10)
    assert len(errors) <= n_iters
    _show_quant(Path('toy example'), im_quant, errors, 5)

    assert len(errors) <= n_iters
    assert np.all(im_quant <= 255)
    assert np.all(im_quant >= 0)
    assert np.issubdtype(im_quant.dtype, np.integer)


def _show_quant(image, im_quant, errors, n_q):
    fix, axs = plt.subplots(1, 2)
    axs[0].imshow(im_quant, cmap='gray')
    axs[0].set_title(f'{image.name} - {n_q} quants')
    axs[1].plot(np.arange(1, len(errors) + 1), errors, 'o-')
    axs[1].set_title(f'# iterations = {len(errors)}')
    plt.show()
    plt.clf()


def _show_eq(img, color, im_eq, hist_orig, hist_eq):
    fix, axs = plt.subplots(2, 2)
    axs[0][0].bar(np.arange(256), hist_orig, width=1, label='hist_orig')
    axs[0][0].bar(np.arange(256), hist_eq, width=1, alpha=0.7, label='hist_eq')
    axs[0][0].legend()
    axs[0][0].set_title('hist')

    axs[0][1].bar(np.arange(256), np.cumsum(hist_orig), width=1, label='cumsum_hist_orig')
    axs[0][1].bar(np.arange(256), np.cumsum(hist_eq), width=1, alpha=0.7, label='cumsum_hist_eq')
    axs[0][1].legend()
    axs[0][1].set_title('cumsum_hist')

    cm = 'viridis' if color == RGB_REP else 'gray'
    axs[1][0].imshow(img, cmap=cm)
    axs[1][0].set_title('original_img')
    axs[1][1].imshow(im_eq, cmap=cm)
    axs[1][1].set_title('im_eq')

    plt.show()
    plt.clf()
