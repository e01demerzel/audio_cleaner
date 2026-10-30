import array
import math
from pathlib import Path
import random
import subprocess
import tempfile
import threading
import unittest
import wave

from audio_cleaner import clean_audio, ffmpeg_path


class AudioTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.source = self.root / 'исходник.wav'
        rng = random.Random(42)
        samples = array.array('h')
        for i in range(48000 * 3):
            t = i / 48000
            # Speech-band signal in the middle, noise-only margins.
            voice = .13 * math.sin(2 * math.pi * 440 * t) if 1 <= t < 2 else 0
            samples.append(round(32767 * (voice + rng.uniform(-.025, .025))))
        with wave.open(str(self.source), 'wb') as w:
            w.setparams((1, 2, 48000, 0, 'NONE', 'not compressed'))
            w.writeframes(samples.tobytes())

    def read(self, path):
        with wave.open(str(path), 'rb') as w:
            self.assertEqual(w.getframerate(), 48000)
            self.assertEqual(w.getnframes(), 144000)
            return array.array('h', w.readframes(w.getnframes()))

    def test_denoise_reduces_stationary_noise(self):
        original = self.source.read_bytes()
        output = clean_audio(self.source, self.root / 'очищено.wav', 80, False)
        before, after = self.read(self.source), self.read(output)
        def rms(data):
            return math.sqrt(sum(x*x for x in data) / len(data))
        self.assertLess(rms(after[24000:44000]), rms(before[24000:44000]) * .8)
        self.assertGreater(rms(after[60000:80000]), rms(after[24000:44000]) * 3)
        self.assertEqual(self.source.read_bytes(), original)

    def test_enhancement_and_output_formats(self):
        for suffix in ('.wav', '.flac', '.mp3'):
            with self.subTest(suffix=suffix):
                target = clean_audio(self.source, self.root / ('result' + suffix))
                result = subprocess.run([ffmpeg_path(), '-v', 'error', '-i', str(target),
                                         '-f', 'null', '-'], capture_output=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertGreater(target.stat().st_size, 1000)
        processed = self.read(self.root / 'result.wav')
        original = self.read(self.source)
        self.assertGreater(max(map(abs, processed)), max(map(abs, original)))
        self.assertLess(max(map(abs, processed)), 32767)

    def test_same_path_rejected(self):
        original = self.source.read_bytes()
        with self.assertRaises(ValueError):
            clean_audio(self.source, self.source)
        self.assertEqual(self.source.read_bytes(), original)

    def test_bad_input_preserves_existing_result(self):
        bad = self.root / 'broken.mp3'
        bad.write_text('not audio')
        target = self.root / 'existing.wav'
        target.write_bytes(b'keep me')
        with self.assertRaises(RuntimeError):
            clean_audio(bad, target)
        self.assertEqual(target.read_bytes(), b'keep me')
        self.assertFalse(list(self.root.glob('.audio-cleaner-*')))

    def test_cancellation_preserves_existing_result(self):
        target = self.root / 'existing.wav'
        target.write_bytes(b'keep me')
        cancel = threading.Event()
        cancel.set()
        with self.assertRaises(InterruptedError):
            clean_audio(self.source, target, cancel=cancel)
        self.assertEqual(target.read_bytes(), b'keep me')
        self.assertFalse(list(self.root.glob('.audio-cleaner-*')))


if __name__ == '__main__':
    unittest.main()
