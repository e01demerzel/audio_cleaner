"""Real-model integration tests; install the model before running this suite."""
from pathlib import Path
import subprocess
import tempfile
import threading
import unittest
from unittest.mock import patch
import wave

import numpy as np

from audio_cleaner import clean_audio, ffmpeg_path, loudness_gain
from neural import denoise_pcm, download_model, require_model


def read_audio(path):
    return np.frombuffer(subprocess.check_output([
        ffmpeg_path(), '-v', 'error', '-i', str(path), '-ar', '48000',
        '-ac', '1', '-f', 'f32le', '-']), dtype='<f4').copy()


def si_sdr(estimated, reference):
    estimated = estimated.astype(np.float64)
    reference = reference.astype(np.float64)
    estimated -= estimated.mean()
    reference -= reference.mean()
    projection = np.dot(estimated, reference) / np.dot(reference, reference) * reference
    return 10 * np.log10(np.sum(projection**2) / (np.sum((estimated - projection)**2) + 1e-20))


class NeuralQualityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = require_model()
        cls.directory = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.directory.cleanup)
        cls.root = Path(cls.directory.name)
        speech = read_audio(Path(__file__).parent / 'fixtures' / 'jfk.wav')
        cls.clean = np.pad(speech, (48000, 48000))
        t = np.arange(cls.clean.size) / 48000
        rng = np.random.default_rng(23)
        noise = (rng.normal(0, .035, t.size) + .025 * np.sin(2 * np.pi * 120 * t))
        noise *= 1 + .4 * np.sin(2 * np.pi * .7 * t)
        cls.source = cls.root / 'шумная речь.wav'
        with wave.open(str(cls.source), 'wb') as w:
            w.setparams((1, 2, 48000, 0, 'NONE', 'not compressed'))
            w.writeframes((np.clip(cls.clean + noise, -1, 1) * 32767).astype('<i2').tobytes())
        cls.noisy = read_audio(cls.source)
        # The high-pass is common to both engines. Apply it to the reference
        # too: otherwise its expected phase response is counted as speech loss.
        def highpass(data):
            return np.frombuffer(subprocess.check_output([
                ffmpeg_path(), '-v', 'error', '-f', 'f32le', '-ar', '48000',
                '-ac', '1', '-i', '-', '-af', 'highpass=f=75', '-f', 'f32le', '-'],
                input=data.astype('<f4').tobytes()), dtype='<f4').copy()
        cls.reference = highpass(cls.clean)
        cls.baseline = highpass(cls.noisy)
        cls.ai = read_audio(clean_audio(cls.source, cls.root / 'ai.wav',
                                       enhance=False, clean_pauses=False))
        cls.fast = read_audio(clean_audio(cls.source, cls.root / 'fast.wav',
                                         engine='fast', enhance=False))
        cls.enhanced = read_audio(clean_audio(cls.source, cls.root / 'enhanced.wav'))

    def test_speech_preservation_and_quality_against_clean_reference(self):
        before = si_sdr(self.baseline, self.reference)
        after = si_sdr(self.ai, self.reference)
        # FFmpeg afftdn adds 25 ms latency. Compensate before quality scoring
        # so the old algorithm is compared fairly, not penalized for a delay.
        delay = 1200
        old = si_sdr(self.fast[delay:], self.reference[:-delay])
        print(f'\nReal speech SI-SDR: noisy={before:.1f}, fast={old:.1f}, AI={after:.1f} dB')
        self.assertGreater(after, before + 5)
        self.assertGreater(after, old + 2)

    def test_background_noise_reduction_and_normalized_pauses(self):
        pause = slice(12000, 36000)
        suppression = 20 * np.log10(np.std(self.noisy[pause]) / max(np.std(self.ai[pause]), 1e-10))
        print(f'\nNoise attenuation in test pause: {suppression:.1f} dB')
        self.assertGreater(suppression, 25)
        # The final loudness pass must not bring the background back.
        self.assertLess(np.sqrt(np.mean(self.enhanced[pause]**2)), 10 ** (-65 / 20))
        self.assertLess(np.max(np.abs(self.enhanced)), .85)
        self.assertGreater(np.std(self.enhanced[48000:-48000]), .025)

    def test_duration_and_alignment(self):
        self.assertEqual(len(self.ai), len(self.noisy))
        self.assertEqual(len(self.enhanced), len(self.noisy))
        # A model's 40 ms lookahead must be removed before mixing/encoding.
        fft_size = 1 << (2 * len(self.ai) - 1).bit_length()
        correlation = np.fft.irfft(np.fft.rfft(self.ai, fft_size) *
                                  np.conj(np.fft.rfft(self.clean, fft_size)), fft_size)
        lag = int(np.argmax(correlation))
        if lag > fft_size // 2:
            lag -= fft_size
        self.assertLessEqual(abs(lag), 2)

    def test_partial_strength_mixes_aligned_audio(self):
        mixed = read_audio(clean_audio(self.source, self.root / 'half.wav', 50,
                                       enhance=False, clean_pauses=False))
        dry = read_audio(clean_audio(self.source, self.root / 'dry.wav', 0, enhance=False))
        expected = .5 * self.ai + .5 * dry
        self.assertLess(np.sqrt(np.mean((mixed - expected)**2)), .0001)

    def test_short_files_recover_tail_and_keep_exact_length(self):
        for length in (7, 1001):
            with self.subTest(samples=length):
                source, target = self.root / 'short.f32', self.root / 'short-out.f32'
                np.linspace(-.1, .1, length, dtype='<f4').tofile(source)
                denoise_pcm(source, target, self.model, 100)
                self.assertEqual(target.stat().st_size, length * 4)

    def test_stereo_m4a_recorder_input_to_mono_flac(self):
        encoded = self.root / 'диктофон.m4a'
        subprocess.run([ffmpeg_path(), '-v', 'error', '-y', '-i', str(self.source),
                        '-ss', '3', '-t', '1.1', '-ac', '2', '-ar', '44100',
                        '-c:a', 'aac', str(encoded)], check=True)
        expected_length = len(read_audio(encoded))
        result = clean_audio(encoded, self.root / 'диктофон.flac', enhance=False)
        self.assertEqual(len(read_audio(result)), expected_length)
        # Decode without downmixing to inspect the actual output channel count.
        pcm = self.root / 'decoded.wav'
        subprocess.run([ffmpeg_path(), '-v', 'error', '-y', '-i', str(result),
                        '-c:a', 'pcm_s16le', str(pcm)], check=True)
        with wave.open(str(pcm), 'rb') as w:
            self.assertEqual(w.getnchannels(), 1)
            self.assertEqual(w.getframerate(), 48000)

    def test_pause_suppression_adapts_to_quiet_speech(self):
        quiet = self.root / 'quiet.wav'
        samples = self.clean[3 * 48000:5 * 48000] * .01
        with wave.open(str(quiet), 'wb') as w:
            w.setparams((1, 2, 48000, 0, 'NONE', 'not compressed'))
            w.writeframes((samples * 32767).astype('<i2').tobytes())
        plain = read_audio(clean_audio(quiet, self.root / 'quiet-plain.wav',
                                       enhance=False, clean_pauses=False))
        expanded = read_audio(clean_audio(quiet, self.root / 'quiet-expanded.wav', enhance=False))
        self.assertGreater(np.std(plain), .0001)
        self.assertGreater(np.std(expanded), .9 * np.std(plain))

    def test_cancellation_during_inference_preserves_previous_result(self):
        target = self.root / 'previous.wav'
        target.write_bytes(b'original result')
        cancel = threading.Event()
        def progress(message):
            if message.startswith('Нейросетевая очистка:'):
                cancel.set()
        with self.assertRaises(InterruptedError):
            clean_audio(self.source, target, cancel=cancel, progress=progress)
        self.assertEqual(target.read_bytes(), b'original result')
        self.assertFalse(list(self.root.glob('.audio-cleaner-*')))


class ModelAndLoudnessTests(unittest.TestCase):
    def test_corrupt_model_is_rejected_without_silent_fallback(self):
        from neural import MODEL_NAME
        with tempfile.TemporaryDirectory() as folder:
            Path(folder, MODEL_NAME).write_bytes(b'not an onnx model')
            with self.assertRaisesRegex(RuntimeError, 'повреждена'):
                require_model(folder)

    def test_cached_model_works_offline(self):
        model = require_model()
        with patch('urllib.request.urlopen', side_effect=AssertionError('unexpected network')):
            self.assertEqual(download_model(model.parent), model)

    def test_silence_not_amplified_and_peak_headroom_respected(self):
        self.assertEqual(loudness_gain('{"input_i":"-inf","input_tp":"-inf"}'), 0)
        self.assertAlmostEqual(loudness_gain('{"input_i":"-35","input_tp":"-8"}'), 6.5)
        self.assertEqual(loudness_gain('{"input_i":"-70","input_tp":"-50"}'), 40)


if __name__ == '__main__':
    unittest.main()
