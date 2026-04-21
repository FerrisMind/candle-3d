<p align="left">
  <a href="README.md"><img src="https://img.shields.io/badge/English-232323" alt="English"></a>
  <a href="README.RU.md"><img src="https://img.shields.io/badge/Русский-D65C5C" alt="Русский"></a>
  <a href="README.PT_BR.md"><img src="https://img.shields.io/badge/Português_BR-232323" alt="Português"></a>
</p>

---

<p align="center">
  <b>Rust workspace для clean-room 3D-инференса, инспекции контрактов и каноникализации весов.</b><br>
  CUDA-first runtime для Pi3, Pi3X и TripoSR.
</p>

<p align="center">
  <img src="https://img.shields.io/badge/License-Apache--2.0-2ea44f" alt="Apache 2.0">
  <img src="https://img.shields.io/badge/Rust-1.85+-93450a?logo=rust" alt="Rust 1.85+">
  <img src="https://img.shields.io/badge/Runtime-CUDA%20first-76B900" alt="CUDA first runtime">
</p>

<h1 align="center">Lux3D</h1>

## Содержание

- [Что это?](#что-это)
- [Ключевые возможности](#ключевые-возможности)
- [Структура репозитория](#структура-репозитория)
- [Быстрый старт](#быстрый-старт)
- [Системные требования](#системные-требования)
- [Лицензия](#лицензия)

## Что это?

Lux3D — это Rust 2024 workspace, который предоставляет:

- Candle-first runtime для инференса
- инспекцию контрактов поддерживаемых семейств моделей
- нормализацию и валидацию canonical weights
- экспорт геометрии в point cloud и mesh форматы

Поддерживаемые семейства моделей:

- `pi3`
- `pi3x`
- `triposr`

## Ключевые возможности

- `lux3d-core` реализует контракты, загрузку runtime, экспорт и проверку весов.
- `lux3d-cli` предоставляет команды `inspect`, `weights normalize` и `run`.
- Python baseline tooling описан в [`tools/python_baseline/README.md`](tools/python_baseline/README.md).
- Лицензирование по каждому семейству моделей можно проверить через CLI перед распространением артефактов.

## Структура Репозитория

| Путь | Назначение |
|------|------------|
| `crates/lux3d-core` | Основной runtime, контракты, экспорт и валидация весов |
| `crates/lux3d-cli` | CLI-оболочка для инспекции, нормализации и инференса |
| `tools/python_baseline` | Python-скрипты для parity и canonical-weight normalization |
| `.github/assets` | Документационные ассеты проекта |

## Быстрый Старт

### Проверка Workspace

```powershell
cargo metadata --no-deps
cargo run -p lux3d-cli -- --help
```

### Инспекция Контрактов

```powershell
cargo run -p lux3d-cli -- inspect --repo-root <runtime-root> pi3
cargo run -p lux3d-cli -- inspect --repo-root <runtime-root> pi3x
cargo run -p lux3d-cli -- inspect --repo-root <runtime-root> triposr
```

### Нормализация Canonical Weights

```powershell
cargo run -p lux3d-cli -- weights normalize --repo-root <runtime-root> --raw-model-dir <raw-pi3-dir> --output-dir <canonical-pi3-dir> pi3
cargo run -p lux3d-cli -- weights normalize --repo-root <runtime-root> --raw-model-dir <raw-pi3x-dir> --output-dir <canonical-pi3x-dir> pi3x
cargo run -p lux3d-cli -- weights normalize --repo-root <runtime-root> --raw-model-dir <raw-triposr-dir> --output-dir <canonical-triposr-dir> triposr
```

### Запуск Инференса

```powershell
# Pi3 -> PLY с явной директорией canonical package в layout как 3d/canonical-weights/pi3
cargo run -p lux3d-cli -- run pi3 --model-path <canonical-pi3-dir> --source <input-sequence> --output <output-file.ply>

# Pi3X core -> PLY с явной директорией canonical package в layout как 3d/canonical-weights/pi3x
cargo run -p lux3d-cli -- run pi3x --model-path <canonical-pi3x-dir> --source <input-sequence> --conditions <conditions-file> --output <output-file.ply>

# Pi3X VO -> PLY с явной директорией canonical package в layout как 3d/canonical-weights/pi3x
cargo run -p lux3d-cli -- run pi3x --model-path <canonical-pi3x-dir> --source <input-video> --vo --chunk-size 8 --overlap 4 --conf-threshold 0.05 --inject-condition pose,depth,ray --output <output-file.ply>

# TripoSR -> OBJ с явной директорией canonical package в layout как 3d/canonical-weights/triposr
cargo run -p lux3d-cli -- run triposr --model-path <canonical-triposr-dir> --source <input-image> --mc-resolution 256 --mc-threshold 25.0 --output <output-file.obj>

# Автоскачивание canonical packages из Hugging Face в пользовательский cache
cargo run -p lux3d-cli -- run pi3 --source <input-sequence> --output <output-file.ply>
```

## Системные Требования

- Rust 1.85 или новее
- Cargo с поддержкой `edition = "2024"`
- Python 3.x для baseline tooling и нормализации
- NVIDIA GPU с CUDA для проверенных инференс-запусков
- Директории canonical model package, переданные через `--model-path`, либо автоматически скачанные из Hugging Face в пользовательский cache

## Лицензия

Код и документация этого репозитория лицензированы по Apache 2.0. См. [LICENSE](LICENSE).

Upstream-модельные артефакты и canonical weights сохраняют свои исходные лицензии и ограничения на использование. Перед распространением проверяй условия для конкретного семейства модели.

Автор проекта: FerrisMind
