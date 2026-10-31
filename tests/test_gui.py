"""Exercise real Tk controls and exports (requires a desktop/X display)."""
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

from audio_cleaner import ffmpeg_path, launch_gui


@unittest.skipUnless(os.environ.get('DISPLAY') or sys.platform == 'win32',
                     'GUI checks require a desktop or X display')
class FormatSelectionTests(unittest.TestCase):
    def test_select_format_choose_path_and_export(self):
        import tkinter as tk
        from tkinter import ttk

        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / 'диктофон.wav'
            shutil.copy2(Path(__file__).parent / 'fixtures' / 'jfk.wav', source)
            original = source.read_bytes()
            root = tk.Tk()
            try:
                with patch('tkinter.Tk', return_value=root), patch.object(root, 'mainloop'):
                    launch_gui()
                root.update()
                def children(widget):
                    for child in widget.winfo_children():
                        yield child
                        yield from children(child)
                widgets = list(children(root))
                entries = [w for w in widgets if isinstance(w, ttk.Entry)]
                pickers = [w for w in widgets if isinstance(w, ttk.Button) and w.cget('text') == 'Выбрать…']
                start = next(w for w in widgets if isinstance(w, ttk.Button) and w.cget('text') == 'Очистить запись')
                combos = [w for w in widgets if isinstance(w, ttk.Combobox)]
                formats = next(w for w in combos if 'AAC' in w.cget('values'))
                mode = next(w for w in combos if w is not formats)
                mode.current(1)  # Export tests use the fast engine; no model/network needed.
                self.assertEqual(formats.get(), 'MP3')
                with patch('tkinter.filedialog.askopenfilename', return_value=str(source)):
                    pickers[0].invoke()
                self.assertEqual(Path(entries[1].get()).suffix, '.mp3')
                custom = Path(folder) / 'моё.имя.mp3'
                entries[1].delete(0, tk.END)
                entries[1].insert(0, str(custom))
                for selected in ('WAV', 'FLAC', 'AAC', 'MP3'):
                    with self.subTest(format=selected):
                        formats.set(selected)
                        formats.event_generate('<<ComboboxSelected>>')
                        root.update()
                        expected = custom.with_suffix('.' + selected.lower())
                        self.assertEqual(Path(entries[1].get()), expected)
                        # The save dialog keeps the format selected in the window.
                        with patch('tkinter.filedialog.asksaveasfilename', return_value=str(custom)):
                            pickers[1].invoke()
                        self.assertEqual(Path(entries[1].get()), expected)
                        with patch('tkinter.messagebox.showerror') as error:
                            start.invoke()
                            deadline = time.monotonic() + 20
                            while str(start.cget('state')) == 'disabled' and time.monotonic() < deadline:
                                root.update()
                                time.sleep(.02)
                            self.assertEqual(str(start.cget('state')), 'normal')
                            error.assert_not_called()
                        decoded = subprocess.run([ffmpeg_path(), '-v', 'error', '-f', selected.lower(),
                            '-i', str(expected), '-f', 'null', '-'], capture_output=True)
                        self.assertEqual(decoded.returncode, 0, decoded.stderr)
                # Choosing another source must retain the selected format.
                formats.set('AAC')
                formats.event_generate('<<ComboboxSelected>>')
                with patch('tkinter.filedialog.askopenfilename', return_value=str(source)):
                    pickers[0].invoke()
                self.assertEqual(Path(entries[1].get()).suffix, '.aac')
                self.assertEqual(source.read_bytes(), original)
            finally:
                root.destroy()


if __name__ == '__main__':
    unittest.main()
