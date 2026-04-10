<p align="left">
  <a href="README.md"><img src="https://img.shields.io/badge/English-232323" alt="English"></a>
  <a href="README.RU.md"><img src="https://img.shields.io/badge/Русский-D65C5C" alt="Русский"></a>
  <a href="README.PT_BR.md"><img src="https://img.shields.io/badge/Português_BR-232323" alt="Português"></a>
</p>

---

<p align="center">
  <img src=".github/assets/lux3d-logo.svg" alt="Логотип Lux3D" width="512" height="512">
</p>

<p align="center">
  <b>Rust workspace для clean-room 3D-инференса, инспекции контрактов и каноникализации весов.</b><br>
  CUDA-first runtime от FerrisMind для Pi3, Pi3X и TripoSR внутри LuxRT.
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Author-FerrisMind-232323" alt="Author FerrisMind">
  <img src="https://img.shields.io/badge/License-Apache--2.0-2ea44f" alt="Apache 2.0">
  <img src="https://img.shields.io/badge/Rust-1.85+-93450a?logo=rust" alt="Rust 1.85+">
  <img src="https://img.shields.io/badge/Runtime-CUDA%20first-76B900" alt="CUDA first runtime">
</p>

<h1 align="center">Lux3D</h1>

<p align="center">
  <img src=".github/assets/lux3d-overview.svg" alt="Обзор workspace Lux3D" width="900">
</p>

## 📚 Содержание

- [Что это?](#-что-это)
- [Демо](#-демо)
- [Ключевые возможности](#-ключевые-возможности)
- [Структура репозитория](#️-структура-репозитория)
- [Установка и настройка](#️-установка-и-настройка)
- [Как начать использовать](#-как-начать-использовать)
- [Системные требования](#️-системные-требования)
- [Благодарности](#-благодарности)
- [Лицензия](#-лицензия)

## ✨ Что это?

Lux3D — это Rust 2024 workspace в каталоге `3d-rs`, который реализует Candle-first runtime, инспекцию контрактов и экспорт геометрии для Pi3, Pi3X и TripoSR. Автор проекта — FerrisMind. Workspace рассчитан на запуск внутри общего репозитория `LuxRT`, где рядом находятся vendor-исходники, сырые веса моделей и каноникализация весов.

## 🎬 Демо

Публичный демо-ассет в репозиторий пока не добавлен. Проверенные локальные входные точки — это команды CLI ниже: они генерируют `.ply` для Pi3/Pi3X и `.obj` для TripoSR.

## 🚀 Ключевые возможности

- `lux3d-core` реализует контракты, загрузку runtime, экспорт геометрии и валидацию canonical weights.
- `lux3d-cli` предоставляет команды `inspect`, `weights normalize` и `run`.
- Поддерживаются три семейства моделей: `pi3`, `pi3x` и `triposr`.
- Canonical safetensors хранятся в `3d/canonical-weights/<family>`, без изменения vendor-деревьев.
- Есть Python-инструменты в [`tools/python_baseline/README.md`](tools/python_baseline/README.md) для фиксации parity baseline и нормализации весов.
- Лицензия репозитория отделена от лицензий upstream-кода и весов; политика по каждому семейству видна через `inspect`.

## 🗂️ Структура Репозитория

| Путь | Назначение |
|------|------------|
| `crates/lux3d-core` | Основной runtime, контракты, экспорт и проверка весов |
| `crates/lux3d-cli` | CLI-оболочка для инспекции, нормализации и инференса |
| `tools/python_baseline` | Python-скрипты для parity и canonical-weight normalization |
| `../3d/models` | Сырые upstream-веса, которые ожидает runtime |
| `../3d/canonical-weights` | Нормализованные safetensors, manifest и checksums |
| `../tp/3d` | Vendor-репозитории, используемые как source of truth |

`--repo-root` всегда должен указывать на родительский каталог LuxRT, в котором находятся `3d-rs`, `3d/` и `tp/`.

## 🛠️ Установка И Настройка

### Предварительные Требования

- Rust 1.85 или новее
- Cargo с поддержкой `edition = "2024"`
- Python 3.x для нормализации и baseline tooling
- NVIDIA GPU и CUDA runtime для проверенных команд `run`
- Корень LuxRT, в котором есть `3d-rs`, `3d/` и `tp/`

### Ожидаемая Структура Родителя

```text
<repo-root>/
  3d-rs/
  3d/
    models/
    canonical-weights/
    _generated/
  tp/
    3d/
      Pi3/
      TripoSR/
```

### Проверка Workspace

```powershell
cargo metadata --no-deps
cargo run -p lux3d-cli -- --help
```

### Python Baseline Tooling

Инструкции по окружениям и golden/smoke-командам находятся в [`tools/python_baseline/README.md`](tools/python_baseline/README.md).

## 📖 Как Начать Использовать

### Инспекция Контрактов И Лицензионных Примечаний

```powershell
cargo run -p lux3d-cli -- inspect --repo-root H:\GitHub\LuxRT pi3
cargo run -p lux3d-cli -- inspect --repo-root H:\GitHub\LuxRT pi3x
cargo run -p lux3d-cli -- inspect --repo-root H:\GitHub\LuxRT triposr
```

### Нормализация Canonical Weights

```powershell
cargo run -p lux3d-cli -- weights normalize --repo-root H:\GitHub\LuxRT pi3
cargo run -p lux3d-cli -- weights normalize --repo-root H:\GitHub\LuxRT pi3x
cargo run -p lux3d-cli -- weights normalize --repo-root H:\GitHub\LuxRT triposr
```

### Запуск Экспорта Инференса

```powershell
# Pi3 -> PLY
cargo run -p lux3d-cli -- run --repo-root H:\GitHub\LuxRT pi3 --source H:\GitHub\LuxRT\tp\3d\Pi3\examples\house --output H:\GitHub\LuxRT\3d\_generated\pi3.ply

# Pi3X core -> PLY
cargo run -p lux3d-cli -- run --repo-root H:\GitHub\LuxRT pi3x --source H:\GitHub\LuxRT\tp\3d\Pi3\examples\room\rgb --conditions H:\GitHub\LuxRT\tp\3d\Pi3\examples\room\condition.npz --output H:\GitHub\LuxRT\3d\_generated\pi3x.ply

# Pi3X VO -> PLY
cargo run -p lux3d-cli -- run --repo-root H:\GitHub\LuxRT pi3x --source H:\GitHub\LuxRT\tp\3d\Pi3\examples\skating.mp4 --vo --chunk-size 8 --overlap 4 --conf-threshold 0.05 --inject-condition pose,depth,ray --output H:\GitHub\LuxRT\3d\_generated\pi3x-vo.ply

# TripoSR -> OBJ
cargo run -p lux3d-cli -- run --repo-root H:\GitHub\LuxRT triposr --source H:\GitHub\LuxRT\tp\3d\TripoSR\examples\horse.png --mc-resolution 256 --mc-threshold 25.0 --output H:\GitHub\LuxRT\3d\_generated\triposr.obj
```

## 🖥️ Системные Требования

- Референсный рабочий процесс в этом репозитории оформлен под Windows PowerShell.
- `inspect` и `cargo metadata` не требуют GPU-инференса.
- `run` сейчас инициализирует `candle_core::Device::new_cuda(0)`, поэтому проверенные инференс-запуски требуют NVIDIA GPU с CUDA.
- Сырые веса должны существовать по путям:
  - `3d/models/yyfz233-Pi3/`
  - `3d/models/yyfz233-Pi3X/`
  - `3d/models/stabilityai-TripoSR/`
- Canonical outputs ожидаются в `3d/canonical-weights/<family>/`.

## 🙏 Благодарности

Этот workspace опирается на:

- upstream-исходники Pi3 и Pi3X в `tp/3d/Pi3`
- TripoSR в `tp/3d/TripoSR`
- Candle, Clap, nalgebra, mcubes и safetensors
- FerrisMind как автора Rust workspace и упаковки репозитория

## 📄 Лицензия

Код и документация внутри `3d-rs` лицензированы по Apache 2.0. См. [LICENSE](LICENSE).

Vendor-деревья, сырые веса моделей и каноникалированные артефакты моделей сохраняют свои upstream-лицензии и ограничения на использование. Перед распространением артефактов используйте `cargo run -p lux3d-cli -- inspect --repo-root <repo-root> <family>`, чтобы проверить политику лицензирования для конкретного семейства.

Copyright (c) 2026 FerrisMind
