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


def default_output_path(source: str | Path) -> Path:
    source = Path(source)
    return source.with_name(source.stem + '_очищено.mp3')


def ffmpeg_path() -> str:
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        binary = shutil.which('ffmpeg')
        if binary:
            return binary
        raise RuntimeError('FFmpeg не найден. Запустите install_windows.bat.')


def filters(strength: int, enhance: bool = True, engine: str = 'neural',
            clean_pauses: bool = True, pause_threshold: float = .0056) -> str:
    if not 0 <= strength <= 100:
        raise ValueError('Сила очистки должна быть от 0 до 100.')
    if engine not in ('neural', 'fast'):
        raise ValueError('Неизвестный режим очистки.')
    chain = ['highpass=f=75']
    if engine == 'fast' and strength:
        chain.append(f'afftdn=nr={1 + strength * .23:.2f}:nf=-30:tn=0:gs=8')
    if engine == 'neural' and strength and clean_pauses:
        # Downward expansion, BEFORE loudness gain, without cutting out time.
        chain.append(f'agate=threshold={pause_threshold:.9f}:ratio=3:range=0.01:attack=10:release=250:detection=rms')
    if enhance:
        chain += ['equalizer=f=2500:t=q:w=1:g=2',
                  'acompressor=threshold=0.125:ratio=2:attack=15:release=180:makeup=1']
    return ','.join(chain)


def run_ffmpeg(arguments, cancel=None) -> str:
    from neural import check_cancel
    check_cancel(cancel)
    flags = subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
    with tempfile.TemporaryFile(mode='w+b') as log:
        process = subprocess.Popen(
            [ffmpeg_path(), '-hide_banner', '-nostdin', '-y', *arguments],
            stdout=subprocess.DEVNULL, stderr=log, creationflags=flags)
        try:
            while True:
                check_cancel(cancel)
                try:
                    code = process.wait(timeout=.1)
                    break
                except subprocess.TimeoutExpired:
                    pass
            check_cancel(cancel)
            log.seek(0, os.SEEK_END)
            log.seek(max(0, log.tell() - 20000))
            summary = log.read().decode('utf-8', errors='replace')
            if code:
                raise RuntimeError('Не удалось прочитать или обработать запись.\n' + summary[-5000:])
            return summary
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()


def loudness_gain(summary: str) -> float:
    """Fixed gain: never amplify pauses separately, and cap gain at 40 dB."""
    import json
    import math
    import re
    matches = re.findall(r'\{\s*"input_i".*?\}', summary, re.S)
    if not matches:
        raise RuntimeError('Не удалось измерить громкость записи.')
    measured = json.loads(matches[-1])
    integrated, peak = float(measured['input_i']), float(measured['input_tp'])
    if not math.isfinite(integrated) or not math.isfinite(peak):
        return 0.0  # Silence stays silent.
    return min(-16 - integrated, -1.5 - peak, 40.0)


def clean_audio(source: str | Path, target: str | Path, strength: int = 100,
                enhance: bool = True, cancel: threading.Event | None = None,
                progress=None, *, engine: str = 'neural', clean_pauses: bool = True,
                model_dir=None) -> Path:
    from neural import check_cancel, denoise_pcm, require_model
    source, target = Path(source).resolve(), Path(target).resolve()
    if not source.is_file():
        raise ValueError('Исходный файл не найден.')
    if source == target:
        raise ValueError('Результат должен отличаться от исходного файла.')
    if target.suffix.lower() not in ('.wav', '.flac', '.mp3'):
        raise ValueError('Выберите формат WAV, FLAC или MP3.')
    chain = filters(strength, enhance, engine, clean_pauses)
    check_cancel(cancel)
    model = require_model(model_dir) if engine == 'neural' and strength else None
    codec = {'.wav': ['-c:a', 'pcm_s16le'], '.flac': ['-c:a', 'flac'],
             '.mp3': ['-c:a', 'libmp3lame', '-b:a', '192k']}[target.suffix.lower()]
    def report(message):
        if progress:
            progress(message)
    with tempfile.TemporaryDirectory(prefix='.audio-cleaner-', dir=target.parent) as folder:
        work = Path(folder)
        input_args = ['-i', str(source), '-map', '0:a:0', '-vn']
        if model is not None:
            report('Подготовка аудио для нейросети…')
            decoded, denoised = work / 'input.f32', work / 'clean.f32'
            run_ffmpeg([*input_args, '-ar', '48000', '-ac', '1', '-f', 'f32le', str(decoded)], cancel)
            peak_rms = denoise_pcm(decoded, denoised, model, strength, cancel, progress)
            # Keep the pause expander well below the voice even for quiet inputs.
            threshold = max(.000000001, min(.0056, peak_rms * .025))
            chain = filters(strength, enhance, engine, clean_pauses, threshold)
            input_args = ['-f', 'f32le', '-ar', '48000', '-ac', '1', '-i', str(denoised)]
        if enhance:
            report('Измерение громкости…')
            summary = run_ffmpeg([*input_args, '-af', chain +
                ',loudnorm=I=-16:TP=-1.5:LRA=50:print_format=json', '-f', 'null', '-'], cancel)
            gain = loudness_gain(summary)
            chain += f',volume={gain:.6f}dB,alimiter=limit=0.841395:level=false:latency=true'
        report('Сохранение результата…')
        rendered = work / ('result' + target.suffix.lower())
        run_ffmpeg([*input_args, '-af', chain, '-ar', '48000', *codec, str(rendered)], cancel)
        check_cancel(cancel)
        os.replace(rendered, target)
    return target


def launch_gui():
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk

    root = tk.Tk()
    root.title('Чистый голос — очистка диктофонных записей')
    root.geometry('760x650')
    root.minsize(700, 630)
    panel = ttk.Frame(root, padding=22)
    panel.pack(fill='both', expand=True)
    source, target = tk.StringVar(), tk.StringVar()
    strength, enhance = tk.IntVar(value=100), tk.BooleanVar(value=True)
    mode = tk.StringVar(value='AI — глубокая очистка речи')
    pauses = tk.BooleanVar(value=True)
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
            target.set(str(default_output_path(name)))

    def choose_target():
        name = filedialog.asksaveasfilename(title='Сохранить результат', defaultextension='.mp3',
            initialfile=Path(target.get()).name or 'очищено.mp3',
            filetypes=[('MP3', '*.mp3'), ('WAV — без потерь', '*.wav'), ('FLAC — без потерь', '*.flac')])
        if name:
            target.set(name)

    ttk.Label(panel, text='Чистый голос', font=('Segoe UI', 20, 'bold')).pack(anchor='w')
    ttk.Label(panel, text='Нейросетевая очистка голоса · локально, без отправки записи').pack(anchor='w', pady=(0, 16))
    for label, variable, command in [('Исходная запись', source, choose_source), ('Результат', target, choose_target)]:
        ttk.Label(panel, text=label).pack(anchor='w')
        row = ttk.Frame(panel)
        row.pack(fill='x', pady=(3, 10))
        ttk.Entry(row, textvariable=variable).pack(side='left', fill='x', expand=True)
        ttk.Button(row, text='Выбрать…', command=command).pack(side='right', padx=(8, 0))
    ttk.Label(panel, text='Режим обработки').pack(anchor='w')
    mode_box = ttk.Combobox(panel, textvariable=mode, state='readonly', values=[
        'AI — глубокая очистка речи', 'Быстрая очистка — без нейросети'])
    mode_box.pack(fill='x', pady=(3, 10))
    label = ttk.Label(panel)
    label.pack(anchor='w')
    def update_label(*_):
        label.configure(text=f'Сила подавления шума: {strength.get()}%')
    strength.trace_add('write', update_label)
    update_label()
    tk.Scale(panel, from_=0, to=100, orient='horizontal', variable=strength, showvalue=False).pack(fill='x')
    ttk.Checkbutton(panel, text='Усилить голос и выровнять громкость', variable=enhance).pack(anchor='w', pady=6)
    ttk.Label(panel, text='100% — полная очистка. Уменьшение силы возвращает часть оригинала и его шума.', wraplength=670).pack(anchor='w')
    ttk.Checkbutton(panel, text='Подавлять остаточный фон в паузах (AI)', variable=pauses).pack(anchor='w', pady=6)
    bar = ttk.Progressbar(panel, mode='indeterminate')
    bar.pack(fill='x', pady=10)
    buttons = ttk.Frame(panel)
    buttons.pack(fill='x')

    def start():
        nonlocal busy, result
        if busy:
            return
        src, dst, amount, boost = source.get(), target.get(), strength.get(), enhance.get()
        engine = 'neural' if mode.get().startswith('AI') else 'fast'
        remove_pauses = pauses.get()
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
                events.put(('done', clean_audio(src, dst, amount, boost, cancel,
                    lambda text: events.put(('progress', text)), engine=engine, clean_pauses=remove_pauses)))
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
        while True:
            try:
                kind, value = events.get_nowait()
            except queue.Empty:
                break
            if kind == 'progress':
                status.set(value)
                continue
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
    parser.add_argument('target', nargs='?', help='Путь результата (по умолчанию: имя_очищено.mp3)')
    parser.add_argument('--strength', type=int, default=100)
    parser.add_argument('--engine', choices=['neural', 'fast'], default='neural')
    parser.add_argument('--keep-pauses', action='store_true', help='Не подавлять остаточный фон в паузах')
    parser.add_argument('--model-dir', help='Каталог нейросетевой модели')
    parser.add_argument('--download-model', action='store_true', help='Установить/проверить модель и выйти')
    parser.add_argument('--no-enhance', action='store_true')
    args = parser.parse_args()
    if args.download_model:
        from neural import download_model
        try:
            print(download_model(args.model_dir, progress=print))
        except Exception as exc:
            parser.exit(1, f'{exc}\n')
        return
    if args.source:
        target = args.target or default_output_path(args.source)
        try:
            print(clean_audio(args.source, target, args.strength, not args.no_enhance,
                              engine=args.engine, clean_pauses=not args.keep_pauses, model_dir=args.model_dir))
        except Exception as exc:
            parser.exit(1, f'{exc}\n')
    else:
        launch_gui()


if __name__ == '__main__':
    main()
