# Third-party components

This application uses the following separately licensed components.

- **DPDFNet** by Ceva: https://github.com/ceva-ip/DPDFNet , Apache-2.0. The unmodified `dpdfnet2_48khz_hr.onnx` release asset is downloaded from https://github.com/k2-fsa/sherpa-onnx/releases/tag/speech-enhancement-models . SHA-256: `0b399f8a58dc4d70d8cd97541f5c39869406145193b957d00a03b66070944928`. License: `licenses/DPDFNet-LICENSE.txt`. Model is downloaded during installation, bundled into EXE builds, and not stored in Git.
- **sherpa-onnx** by k2-fsa/Xiaomi and contributors, Apache-2.0: https://github.com/k2-fsa/sherpa-onnx . License: `licenses/sherpa-onnx-LICENSE.txt`. Includes separately licensed runtime dependencies, including ONNX Runtime (MIT); see installed package license notices.
- **imageio-ffmpeg**, BSD-2-Clause: https://github.com/imageio/imageio-ffmpeg . Its bundled **FFmpeg** binary has its own licenses/build configuration, including GPL-enabled components. See the installed package's `binaries/README.md` and the binary's `-version`/`-L` output. Consult those terms before redistributing a built EXE.
- **NumPy**, BSD-3-Clause: https://numpy.org . See the installed package's license notices.
- Test fixture attribution is recorded in `tests/fixtures/README.md`.
