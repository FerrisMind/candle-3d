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

<h1 align="center">Lux3D / candle-3d</h1>

## Содержание

- [Что это?](#что-это)
- [Ключевые возможности](#ключевые-возможности)
- [Структура репозитория](#структура-репозитория)
- [Быстрый старт](#быстрый-старт)
- [Производительность](#производительность)
- [Системные требования](#системные-требования)
- [Лицензия](#лицензия)

## Что это?

Lux3D / candle-3d — это Rust 2024 workspace, который предоставляет:

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
- `lux3d-server-core` предоставляет OpenAI-style async HTTP API для Pi3, Pi3X и TripoSR.
- Python baseline tooling описан в [`tools/python_baseline/README.md`](https://github.com/oxide-lab/Lux3d/blob/main/tools/python_baseline/README.md).
- Лицензирование по каждому семейству моделей можно проверить через CLI перед распространением артефактов.
- **Экспериментально:** инференс на Vulkan и WGPU через `--device vulkan` / `--device wgpu` (сборка с `--features vulkan` / `--features wgpu`). Не готово для продакшена — может работать нестабильно, давать неверный результат или не запускаться в зависимости от железа и драйверов. Единственный проверенный бэкенд — CUDA.

## Структура Репозитория

| Путь | Назначение |
|------|------------|
| `crates/lux3d-core` | Основной runtime, контракты, экспорт и валидация весов |
| `crates/lux3d-cli` | CLI-оболочка для инспекции, нормализации и инференса |
| `crates/lux3d-server-core` | Встраиваемый Axum router и HTTP API |
| `crates/lux3d-server` | Standalone HTTP server binary |
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

> **Экспериментальные бэкенды (Vulkan / WGPU):** выбор устройства через `--device`:
>
> ```powershell
> cargo run -p lux3d-cli --features vulkan -- run triposr --device vulkan --source <input-image> --output <output-file.obj>
> cargo run -p lux3d-cli --features wgpu -- run pi3 --device wgpu --source <input-sequence> --output <output-file.ply>
> ```
>
> Эти пути экспериментальные и не проверяются в CI. Инференс может работать нестабильно, выдавать некорректный результат или вовсе не запускаться в зависимости от GPU, драйвера и ОС. Для продакшена используй CUDA.

## Производительность

Замеры инференса (`[stage] infer`: препроцессинг + нейросетевой проход, одна итерация на процесс, без учёта загрузки весов) на NVIDIA GeForce RTX 3060 12 GiB, Windows (WDDM), форк candle `wgpu/vulkan` rev `6cdde81d`, candle-3d `4a17a8c` (2026-09-06):

| модель | CUDA | Vulkan | WGPU | Vulkan/CUDA | WGPU/CUDA |
|---|---:|---:|---:|---:|---:|
| pi3 (5 кадров, 518×518) | **4.82 с** | 7.66 с | 77.7 с | 1.59× | 16.1× |
| pi3x (6 кадров, 518×518) | **7.00 с** | 11.43 с | 142.4 с | 1.63× | 20.3× |
| triposr (одно изображение) | **1.23 с** | 3.19 с | 8.10 с | 2.59× | 6.6× |

Корректность: выходы всех комбинаций бэкенд × модель сверены с эталонными мешами CUDA (дельты bbox/центра/среднего в пределах 1% от max extent, число вершин/граней в пределах 0.5% — все PASS).

Поведение памяти при повторном инференсе (цикл из 10 итераций в одном процессе):

- **CUDA** — потребление плоское; буфер attention scores декодера ограничен 768 MiB чанкингом по query-оси (`LUX3D_MAX_SDPA_SCORES_BYTES`) — убран ~4 GiB транзиент, который раньше фрагментировал пул CUDA и замедлял 12-гигабайтные карты ~в 100 раз.
- **Vulkan** — плоско ~7.5 GiB; удержание памяти ограничено inflight-бюджетом (256 MiB × grace-полоса 8) и пулом GPU-буферов на 2 GiB (`CANDLE_VK_INFLIGHT_GRACE`, `CANDLE_VK_POOL_MAX_BYTES`). Капы батчей настроены по flush-reason профилированию: transfer-байты 512 MiB (`CANDLE_VK_MAX_BATCH_TRANSFER_BYTES`), descriptor sets 8× от диспатчей — закрытие батча на каждой большой копии активаций стоило ~4-9 ms WDDM fence-signal латентности на каждый сабмит.
- **WGPU** — 10-итерационный бенчмарк pi3 проходит с пиком VRAM 9.65 GiB и нулём ошибок (раньше OOM на ~3-й итерации); free pool / recycle backlog / in-flight удержание ограничены по байтам (`CANDLE_WGPU_POOL_MAX_BYTES`, `CANDLE_WGPU_INFLIGHT_MAX_BYTES`).

Известные разрывы: Vulkan отстаёт от CUDA на 1.6–2.6× из-за CPU-оверхеда на каждый диспатч под WDDM (GPU-кернелы занимают ~0.5 s стены); закрытие требует op-fusion. WGPU отстаёт на 6.6–20× — упирается в качество WGSL GEMM-кернелов; запланирован tiled register-blocked кернел (по образцу `mul_mm.comp` из llama.cpp). Criterion-харнесс для бенчмарков целых моделей — в `crates/lux3d-core/benches/` (одна пара (бэкенд, модель) на процесс: `LUX3D_BENCH_DEVICE=… LUX3D_BENCH_MODEL=… cargo bench -p lux3d-core --bench bench_main --features vulkan,wgpu`).

## Системные Требования

- Rust 1.85 или новее
- Cargo с поддержкой `edition = "2024"`
- Python 3.x для baseline tooling и нормализации
- NVIDIA GPU с CUDA для проверенных инференс-запусков
- **Бэкенды Vulkan и WGPU** (через форк [FerrisMind/candle](https://github.com/FerrisMind/candle) `wgpu/vulkan`): корректность и стабильность памяти проверены на RTX 3060 (см. [Производительность](#производительность)); медленнее CUDA, WGPU — существенно.
- Бэкенд Metal на macOS в этом репозитории не тестировался, но теоретически может поддерживаться.
- Директории canonical model package, переданные через `--model-path`, либо автоматически скачанные из Hugging Face в пользовательский cache

## Лицензия

Код и документация этого репозитория лицензированы по Apache 2.0. См. [LICENSE](https://github.com/oxide-lab/Lux3d/blob/main/LICENSE).

Upstream-модельные артефакты и canonical weights сохраняют свои исходные лицензии и ограничения на использование. Перед распространением проверяй условия для конкретного семейства модели.
