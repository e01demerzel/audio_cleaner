"""Локальная очистка диктофонных записей, Python 3.10+."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import queue
import shutil
import subprocess
import tempfile
import threading


def ffmpeg_path() -> str:
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        binary = shutil.which('ffmpeg')
        if binary:
            return binary
        raise RuntimeError('FFmpeg не найден. Запустите install_windows.bat.')


def filters(strength: int, enhance: bool = True) -> str:
    if not 0 <= strength <= 100:
        raise ValueError('Сила очистки должна быть от 0 до 100.')
    chain = ['highpass=f=75']
    if strength:
        chain.append(f'afftdn=nr={1 + strength * .23:.2f}:nf=-30:tn=0:gs=8')
    if enhance:
        chain += ['equalizer=f=2500:t=q:w=1:g=2',
                  'acompressor=threshold=0.125:ratio=2:attack=15:release=180:makeup=1.5',
                  'loudnorm=I=-16:TP=-1.5:LRA=9']
    return ','.join(chain)


def clean_audio(source: str | Path, target: str | Path, strength: int = 60,
                enhance: bool = True, cancel: threading.Event | None = None,
                progress=None) -> Path:
    source, target = Path(source).resolve(), Path(target).resolve()
    if not source.is_file():
        raise ValueError('Исходный файл не найден.')
    if source == target:
        raise ValueError('Результат должен отличаться от исходного файла.')
    if target.suffix.lower() not in ('.wav', '.flac', '.mp3'):
        raise ValueError('Выберите формат WAV, FLAC или MP3.')
    chain = filters(strength, enhance)
    codec = {'.wav': ['-c:a', 'pcm_s16le'], '.flac': ['-c:a', 'flac'],
             '.mp3': ['-c:a', 'libmp3lame', '-b:a', '192k']}[target.suffix.lower()]
    fd, temp = tempfile.mkstemp(prefix='.audio-cleaner-', suffix=target.suffix, dir=target.parent)
    os.close(fd)
    flags = subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
    try:
        # File-backed stderr avoids pipe deadlocks and keeps memory bounded.
        with tempfile.TemporaryFile(mode='w+b') as log:
            process = subprocess.Popen(
                [ffmpeg_path(), '-hide_banner', '-nostdin', '-y', '-i', str(source),
                 '-map', '0:a:0', '-vn', '-af', chain, '-ar', '48000', *codec, temp],
                stdout=subprocess.DEVNULL, stderr=log, creationflags=flags)
            try:
                while True:
                    if cancel is not None and cancel.wait(.15):
                        raise InterruptedError('Обработка отменена.')
                    try:
                        code = process.wait(timeout=.15)
                        break
                    except subprocess.TimeoutExpired:
                        if progress:
                            progress('Обработка записи…')
                if cancel is not None and cancel.is_set():
                    raise InterruptedError('Обработка отменена.')
                if code:
                    log.seek(0, os.SEEK_END)
                    log.seek(max(0, log.tell() - 5000))
                    raise RuntimeError('Не удалось прочитать или обработать запись.\n' +
                                       log.read().decode('utf-8', errors='replace'))
            finally:
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=3)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
        os.replace(temp, target)
        return target
    finally:
        Path(temp).unlink(missing_ok=True)


def launch_gui():
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk

    root = tk.Tk()
    root.title('Чистый голос — очистка диктофонных записей')
    root.geometry('720x540')
    root.minsize(660, 520)
    panel = ttk.Frame(root, padding=22)
    panel.pack(fill='both', expand=True)
    source, target = tk.StringVar(), tk.StringVar()
    strength, enhance = tk.IntVar(value=60), tk.BooleanVar(value=True)
    status = tk.StringVar(value='Выберите запись. Обработка выполняется на вашем компьютере.')
    events = queue.Queue()
    cancel = threading.Event()
    busy = False
    result = None

    def choose_source():
        name = filedialog.askopenfilename(title='Открыть диктофонную запись', filetypes=[
            ('Аудиозаписи', '*.wav *.mp3 *.m4a *.aac *.flac *.ogg *.wma *.mp4'), ('Все файлы', '*.*')])
        if name:
            source.set(name)
            p = Path(name)
            target.set(str(p.with_name(p.stem + '_очищено.wav')))

    def choose_target():
        name = filedialog.asksaveasfilename(title='Сохранить результат', defaultextension='.wav',
            initialfile=Path(target.get()).name or 'очищено.wav',
            filetypes=[('WAV — без потерь', '*.wav'), ('FLAC — без потерь', '*.flac'), ('MP3', '*.mp3')])
        if name:
            target.set(name)

    ttk.Label(panel, text='Чистый голос', font=('Segoe UI', 20, 'bold')).pack(anchor='w')
    ttk.Label(panel, text='Уменьшение фонового шума и выравнивание громкости речи').pack(anchor='w', pady=(0, 16))
    for label, variable, command in [('Исходная запись', source, choose_source), ('Результат', target, choose_target)]:
        ttk.Label(panel, text=label).pack(anchor='w')
        row = ttk.Frame(panel)
        row.pack(fill='x', pady=(3, 10))
        ttk.Entry(row, textvariable=variable).pack(side='left', fill='x', expand=True)
        ttk.Button(row, text='Выбрать…', command=command).pack(side='right', padx=(8, 0))
    label = ttk.Label(panel)
    label.pack(anchor='w')
    def update_label(*_):
        label.configure(text=f'Сила подавления шума: {strength.get()}%')
    strength.trace_add('write', update_label)
    update_label()
    tk.Scale(panel, from_=0, to=100, orient='horizontal', variable=strength, showvalue=False).pack(fill='x')
    ttk.Checkbutton(panel, text='Усилить голос и выровнять громкость', variable=enhance).pack(anchor='w', pady=6)
    ttk.Label(panel, text='При металлическом звучании уменьшите силу очистки.').pack(anchor='w')
    bar = ttk.Progressbar(panel, mode='indeterminate')
    bar.pack(fill='x', pady=10)
    buttons = ttk.Frame(panel)
    buttons.pack(fill='x')

    def start():
        nonlocal busy, result
        if busy:
            return
        src, dst, amount, boost = source.get(), target.get(), strength.get(), enhance.get()
        if not src or not dst:
            messagebox.showwarning('Выберите файлы', 'Укажите исходную запись и путь результата.')
            return
        if Path(dst).exists() and not messagebox.askyesno('Заменить файл?', 'Файл результата уже существует. Заменить?'):
            return
        busy, result = True, None
        cancel.clear()
        start_button.configure(state='disabled')
        play_button.configure(state='disabled')
        cancel_button.configure(state='normal')
        bar.start()
        status.set('Очистка записи… Для длинных файлов потребуется время.')
        def work():
            try:
                events.put(('done', clean_audio(src, dst, amount, boost, cancel)))
            except Exception as exc:
                events.put(('error', exc))
        threading.Thread(target=work, daemon=True).start()

    def play():
        if result and os.name == 'nt':
            try:
                os.startfile(str(result))
            except OSError as exc:
                messagebox.showerror('Не удалось открыть проигрыватель', str(exc))

    start_button = ttk.Button(buttons, text='Очистить запись', command=start)
    start_button.pack(side='left')
    cancel_button = ttk.Button(buttons, text='Отмена', command=cancel.set, state='disabled')
    cancel_button.pack(side='left', padx=8)
    play_button = ttk.Button(buttons, text='Прослушать результат', command=play, state='disabled')
    play_button.pack(side='left')
    ttk.Label(panel, textvariable=status, wraplength=630).pack(anchor='w', pady=10)

    def poll():
        nonlocal busy, result
        try:
            kind, value = events.get_nowait()
        except queue.Empty:
            pass
        else:
            busy = False
            bar.stop()
            start_button.configure(state='normal')
            cancel_button.configure(state='disabled')
            if kind == 'done':
                result = value
                status.set('Готово! Результат: ' + str(value))
                if os.name == 'nt':
                    play_button.configure(state='normal')
            else:
                status.set(str(value) if isinstance(value, InterruptedError) else 'Ошибка обработки.')
                if not isinstance(value, InterruptedError):
                    messagebox.showerror('Ошибка', str(value))
        root.after(100, poll)

    def close():
        if busy:
            cancel.set()
            status.set('Отменяем обработку перед закрытием…')
            root.after(150, close)
        else:
            root.destroy()
    root.protocol('WM_DELETE_WINDOW', close)
    poll()
    root.mainloop()


def main():
    parser = argparse.ArgumentParser(description='Локальное подавление шума и усиление речи')
    parser.add_argument('source', nargs='?')
    parser.add_argument('target', nargs='?')
    parser.add_argument('--strength', type=int, default=60)
    parser.add_argument('--no-enhance', action='store_true')
    args = parser.parse_args()
    if args.source:
        if not args.target:
            parser.error('Укажите путь результата.')
        try:
            print(clean_audio(args.source, args.target, args.strength, not args.no_enhance))
        except Exception as exc:
            parser.exit(1, f'{exc}\n')
    else:
        launch_gui()


if __name__ == '__main__':
    main()
