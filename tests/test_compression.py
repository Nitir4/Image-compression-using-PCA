import importlib
import threading
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

from Complete.compression import custom_pca, compress_image_full
from Complete import main


@pytest.mark.parametrize('shape', [(1,1),(1,12),(12,1),(8,11),(9,7,3)])
@pytest.mark.parametrize('method', ['PCA','SVD'])
def test_full_rank_preserves_pixels_and_shape(shape, method):
    original = np.random.default_rng(505).integers(0,256,shape,dtype=np.uint8)
    result, ratio = compress_image_full(original, 100, method)
    np.testing.assert_array_equal(result, original)
    assert ratio == pytest.approx(1)


@pytest.mark.parametrize('k', [0,-1,1.5,True])
def test_invalid_rank_is_rejected(k):
    with pytest.raises(ValueError):
        custom_pca(np.ones((3,4)),k)


def test_rank_one_recovers_affine_channel_and_constant_image():
    original = np.arange(6)[:,None]*np.array([1,2,3])[None,:]+80
    reconstructed,_ = custom_pca(original,1)
    np.testing.assert_array_equal(reconstructed,original)
    reconstructed, ratio = compress_image_full(np.full((4,5),127),1,'PCA')
    assert ratio == 1
    assert np.all(reconstructed==127)


def test_save_uses_full_resolution_while_preview_and_tk_stay_on_ui_thread(tmp_path, monkeypatch):
    callbacks = []
    owner = threading.get_ident()
    widget = SimpleNamespace(config=lambda **kwargs: None)
    original = np.arange(1200,dtype=np.uint16).reshape(30,40).astype(np.uint8)
    app = SimpleNamespace(_ui=callbacks.append, prog1=widget,comp_lbl1=widget,stats1=widget,comp_pil=None)
    app._show_compression=lambda image,stats:main.ImageCompressionApp._show_compression(app,image,stats)
    previews=[]
    def photo(image):
        assert threading.get_ident()==owner
        previews.append(image.size)
        return object()
    monkeypatch.setattr(main.ImageTk,'PhotoImage',photo)
    thread=threading.Thread(target=main.ImageCompressionApp._thread_compress,args=(app,original,30,'PCA',False))
    thread.start(); thread.join(timeout=10)
    assert not thread.is_alive() and app.comp_pil is None and not previews
    for callback in callbacks: callback()
    destination=tmp_path/'output.png'
    app.file_lbl=SimpleNamespace(cget=lambda key:'source.png')
    monkeypatch.setattr(main.filedialog,'asksaveasfilename',lambda **kwargs:str(destination))
    monkeypatch.setattr(main.messagebox,'showinfo',lambda *args:None)
    main.ImageCompressionApp.save_image(app)
    with Image.open(destination) as result:
        assert result.size==(40,30)
        np.testing.assert_array_equal(np.asarray(result),original)
    large=Image.new('RGB',(500,400),'red')
    main.ImageCompressionApp._show_compression(app,large,'stats')
    main.ImageCompressionApp.save_image(app)
    with Image.open(destination) as result:
        assert result.size==(500,400) and result.mode=='RGB'
    assert previews[-1]==(200,160)


def test_legacy_modules_import_and_reconstruct_normalized_pixels():
    for module in ['main','GUI','displayimage','imagecompression','imageprocessing','plot']:
        importlib.import_module('ENTERk.'+module)
    from ENTERk.PCA import PCA
    original=np.random.default_rng(5).random((9,1))
    recovered,_,_=PCA(original,1)
    np.testing.assert_allclose(recovered,original)


def test_small_image_ssim_does_not_crash():
    assert np.isnan(main.ssim_metric(np.zeros((1,1)),np.zeros((1,1))))
    assert main.ssim_metric(np.zeros((3,3)),np.zeros((3,3)))==pytest.approx(1)


def test_failed_worker_restores_controls_on_ui_thread(monkeypatch):
    queued=[]
    complete=threading.Event()
    states=[]
    button=SimpleNamespace(config=lambda **kwargs:states.append(kwargs.get('state')))
    label=SimpleNamespace(config=lambda **kwargs:None)
    app=SimpleNamespace(_busy=False,orig_pil=Image.new('L',(3,3)),orig_gray=np.zeros((3,3)),
        comp_pil=None,btn_load=button,btn_cmp=button,btn_cmp2=button,btn_an=button,btn_save=button)
    app._finish_job=lambda:main.ImageCompressionApp._finish_job(app)
    def queue_callback(callback):
        queued.append(callback)
        if callback is app._finish_job: complete.set()
    app._ui=queue_callback
    errors=[]
    monkeypatch.setattr(main.messagebox,'showerror',lambda *args:errors.append(args))
    def broken(*args): raise RuntimeError('Regression probe')
    main.ImageCompressionApp._start_job(app,broken,(1,False),label)
    assert complete.wait(10)
    assert app._busy and not errors
    for callback in queued: callback()
    assert not app._busy and errors[0][1]=='Regression probe'
    assert states[-1]==main.tk.DISABLED
