"""Verified local DPDFNet model management and streaming CPU inference."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import sys
import tempfile
import urllib.error
import urllib.request

MODEL_NAME = 'dpdfnet2_48khz_hr.onnx'
MODEL_URL = ('https://github.com/k2-fsa/sherpa-onnx/releases/download/'
             'speech-enhancement-models/' + MODEL_NAME)
# Fingerprint of the upstream release asset fetched over verified HTTPS.
MODEL_SHA256 = '0b399f8a58dc4d70d8cd97541f5c39869406145193b957d00a03b66070944928'
SAMPLE_RATE = 48000
MODEL_DELAY = 1920  # DPDFNet uses four 480-sample lookahead hops (40 ms).


def check_cancel(cancel):
    if cancel is not None and cancel.is_set():
        raise InterruptedError('Обработка отменена.')


def model_location(model_dir=None) -> Path:
    if model_dir is not None:
        return Path(model_dir).resolve() / MODEL_NAME
    if os.environ.get('AUDIO_CLEANER_MODELS'):
        return Path(os.environ['AUDIO_CLEANER_MODELS']).resolve() / MODEL_NAME
    bundled = Path(getattr(sys, '_MEIPASS', Path(__file__).parent)) / 'models' / MODEL_NAME
    if bundled.is_file():
        return bundled
    cache = Path(os.environ['LOCALAPPDATA']) if os.environ.get('LOCALAPPDATA') else Path.home() / '.cache'
    return cache / 'AudioCleaner' / 'models' / MODEL_NAME


def verified(path: Path) -> bool:
    if not path.is_file():
        return False
    with path.open('rb') as f:
        return _digest(f) == MODEL_SHA256


def _digest(file) -> str:
    digest = hashlib.sha256()
    for block in iter(lambda: file.read(1024 * 1024), b''):
        digest.update(block)
    return digest.hexdigest()


def require_model(model_dir=None) -> Path:
    path = model_location(model_dir)
    if not verified(path):
        raise RuntimeError(
            'Нейросетевая модель не установлена или повреждена.\n'
            'Запустите install_windows.bat для установки/обновления.\n'
            'Либо выберите режим «Быстрая очистка».\n'
            f'Ожидаемый файл: {path}')
    return path


def download_model(model_dir=None, cancel=None, progress=None) -> Path:
    path = model_location(model_dir)
    check_cancel(cancel)
    if verified(path):
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix='.download-', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as output:
            try:
                with urllib.request.urlopen(MODEL_URL, timeout=45) as response:
                    total = int(response.headers.get('Content-Length', 0))
                    size = 0
                    while True:
                        check_cancel(cancel)
                        block = response.read(256 * 1024)
                        if not block:
                            break
                        output.write(block)
                        size += len(block)
                        if progress:
                            progress(f'Загрузка модели: {size / 1048576:.1f}' +
                                     (f' / {total / 1048576:.1f} МБ' if total else ' МБ'))
            except (OSError, urllib.error.URLError) as exc:
                if isinstance(exc, InterruptedError):
                    raise
                raise RuntimeError('Не удалось скачать модель с GitHub. Проверьте Интернет и повторите установку.') from exc
        if not verified(Path(name)):
            raise RuntimeError('Контрольная сумма модели не совпала. Файл не будет использован; повторите загрузку.')
        check_cancel(cancel)
        os.replace(name, path)
        return path
    finally:
        Path(name).unlink(missing_ok=True)


def denoise_pcm(source: Path, target: Path, model: Path, strength: int,
                cancel=None, progress=None):
    """Process 100 ms blocks without resetting model state at block boundaries.

    The native API compensates STFT delay, but not the model lookahead. Discard
    that delay and feed zeros to recover the real tail before flush/dry mixing.
    """
    try:
        import numpy as np
        import sherpa_onnx as sherpa
    except ImportError as exc:
        raise RuntimeError('Не установлены компоненты AI. Запустите install_windows.bat.') from exc
    # Speech models can classify very quiet inputs as noise. Normalize their
    # operating level with one fixed gain, then undo it before dry mixing/output.
    # Scan in bounded blocks and honor cancellation for multi-hour recordings.
    peak, loudest_rms = 0.0, 0.0
    if progress:
        progress('Оценка уровня записи перед нейросетью…')
    with source.open('rb') as input_file:
        while True:
            check_cancel(cancel)
            block = input_file.read(4800 * 4)
            if not block:
                break
            samples = np.frombuffer(block, dtype='<f4').astype(np.float64)
            if not np.all(np.isfinite(samples)):
                raise RuntimeError('Исходная запись содержит некорректные значения аудио.')
            peak = max(peak, float(np.max(np.abs(samples))))
            loudest_rms = max(loudest_rms, float(np.sqrt(np.mean(samples**2))))
    input_gain = 1.0
    if peak > 0 and loudest_rms > 0:
        input_gain = max(1.0, min(100.0, .1 / loudest_rms, .9 / peak))
    config = sherpa.OnlineSpeechDenoiserConfig(
        model=sherpa.OfflineSpeechDenoiserModelConfig(
            dpdfnet=sherpa.OfflineSpeechDenoiserDpdfNetModelConfig(
                model=str(model), attenuation_limit_db=0.0),
            num_threads=2, provider='cpu'))
    if not config.validate():
        raise RuntimeError('Ошибка настройки нейросетевой модели. Повторите установку.')
    denoiser = sherpa.OnlineSpeechDenoiser(config)
    if denoiser.sample_rate != SAMPLE_RATE:
        raise RuntimeError('Неподдерживаемая частота модели.')
    total = source.stat().st_size
    processed = 0
    written = 0
    delay = MODEL_DELAY
    mix = strength / 100
    peak_rms = 0.0
    with source.open('rb') as input_file, source.open('rb') as dry, target.open('wb') as output:
        def write(samples):
            nonlocal written, delay, peak_rms
            data = np.asarray(samples, dtype='<f4')
            skip = min(delay, data.size)
            delay -= skip
            data = data[skip:]
            data = data[:(total - written) // 4] / input_gain
            if mix < 1:
                original = np.frombuffer(dry.read(data.size * 4), dtype='<f4')
                if original.size != data.size:
                    raise RuntimeError('Нарушена синхронизация аудио.')
                data = data * mix + original * (1 - mix)
            if not np.all(np.isfinite(data)):
                raise RuntimeError('Модель вернула некорректные значения аудио.')
            if data.size:
                peak_rms = max(peak_rms, float(np.sqrt(np.mean(data.astype(np.float64)**2))))
            output.write(data.astype('<f4', copy=False).tobytes())
            written += data.size * 4
        while True:
            check_cancel(cancel)
            block = input_file.read(4800 * 4)
            if not block:
                break
            write(denoiser(np.frombuffer(block, dtype='<f4') * input_gain, SAMPLE_RATE).samples)
            processed += len(block)
            if progress:
                progress(f'Нейросетевая очистка: {processed / total:.0%}')
        check_cancel(cancel)
        write(denoiser(np.zeros(MODEL_DELAY, dtype=np.float32), SAMPLE_RATE).samples)
        write(denoiser.flush().samples)
    if not total or written != total:
        raise RuntimeError('Не удалось сохранить полную длительность записи.')
    return peak_rms
