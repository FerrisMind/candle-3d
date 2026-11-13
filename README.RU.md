<p align="left">
  <a href="README.md"><img src="https://img.shields.io/badge/English-232323" alt="English"></a>
  <a href="README.RU.md"><img src="https://img.shields.io/badge/Русский-D65C5C" alt="Русский"></a>
  <a href="README.PT_BR.md"><img src="https://img.shields.io/badge/Português_BR-232323" alt="Português"></a>
</p>

---

<p align="center">
  <b>Rust workspace для clean-room 3D-инференса, инспекции контрактов и каноникализации весов.</b><br>
  CUDA-first runtime для Pi3, Pi3X, TripoSR, VGGT и TripoSG.
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
- [Демо](#демо)
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
- `vggt` — VGGT-1B: multi-view point cloud, depth и camera. Оригинальные веса CC-BY-NC-4.0
  (non-commercial); коммерческий checkpoint распространяется отдельно.
- `triposg` — TripoSG: image-to-mesh инференс (DINOv2-large encoder, 21-блочный rectified-flow
  DiT, SDF VAE decoder/encoder, flash decoder + clean-room DiffDMC по умолчанию). Лицензия MIT;
  BriaRMBG 1.4 для RGB-входа поставляется отдельным пакетом `rmbg-1.4` под лицензией BRIA.

## Ключевые возможности

- `lux3d-core` реализует контракты, загрузку runtime, экспорт и проверку весов.
- `lux3d-cli` предоставляет команды `inspect`, `weights normalize` и `run`.
- `lux3d-server-core` предоставляет OpenAI-style async HTTP API для Pi3, Pi3X, TripoSR, VGGT и TripoSG.
- Python baseline tooling описан в [`tools/python_baseline/README.md`](https://github.com/oxide-lab/Lux3d/blob/main/tools/python_baseline/README.md).
- Лицензирование по каждому семейству моделей можно проверить через CLI перед распространением артефактов.
- **Экспериментально:** инференс на Vulkan и WGPU через `--device vulkan` / `--device wgpu` (сборка с `--features vulkan` / `--features wgpu`). CUDA — основной бэкенд; для путей Pi3/Pi3X/TripoSR и VGGT есть проверки корректности на Vulkan, а для TripoSG — smoke-проверки Vulkan/WGPU (см. [Производительность](#производительность)). WGPU остаётся экспериментальным — может работать нестабильно, давать неверный результат или не запускаться в зависимости от железа и драйверов.

## Демо

https://github.com/user-attachments/assets/20c5ce6c-7447-41ea-a79c-4b611e74c552

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
cargo run -p lux3d-cli -- inspect --repo-root <runtime-root> vggt
cargo run -p lux3d-cli -- inspect --repo-root <runtime-root> triposg
```

### Нормализация Canonical Weights

```powershell
cargo run -p lux3d-cli -- weights normalize --repo-root <runtime-root> --raw-model-dir <raw-pi3-dir> --output-dir <canonical-pi3-dir> pi3
cargo run -p lux3d-cli -- weights normalize --repo-root <runtime-root> --raw-model-dir <raw-pi3x-dir> --output-dir <canonical-pi3x-dir> pi3x
cargo run -p lux3d-cli -- weights normalize --repo-root <runtime-root> --raw-model-dir <raw-triposr-dir> --output-dir <canonical-triposr-dir> triposr
cargo run -p lux3d-cli -- weights normalize --repo-root <runtime-root> --raw-model-dir <raw-vggt-dir> --output-dir <canonical-vggt-dir> vggt
cargo run -p lux3d-cli -- weights normalize --repo-root <runtime-root> --raw-model-dir <raw-triposg-dir> --output-dir <canonical-triposg-dir> triposg
```

### Квантование Весов в GGUF

`weights quantize` упаковывает canonical `model.safetensors` в `model.gguf`.
Квантуются только rank-2 матрицы `*.weight`; нормы, bias, эмбеддинги, свёртки
и строки, не кратные блоку dtype, остаются в fallback dtype. Квантованные
linear-слои идут через `QMatMul` (integer dot-product ядра), остальное
де-квантуется по требованию.

Конвертер встраивает соседний `resolved_config.json` в metadata GGUF, поэтому
`model.gguf` может быть **самостоятельным пакетом**: положите его одного в
каталог и передайте этот каталог в `--model-path`. Загрузчик синтезирует
manifest/checksums и читает конфиг из GGUF. Если `model.gguf` лежит рядом с
полным canonical-пакетом, файлы пакета по-прежнему проверяются, а веса берутся
из GGUF. Rank-0 тензоры (тренировочные буферы вроде BatchNorm
`num_batches_tracked`) опускаются — при инференсе они не используются.
`--model-path` принимает и сам `.gguf`-файл, поэтому кванты можно раскладывать
плоско (`TripoSG-1.5B-Q8_0.gguf`, `VGGT-1B-Q8_0.gguf`, ...).

`general.architecture` берётся из resolved config: сначала top-level
`architecture`, затем `family`, затем имя модели (`triposg`, `vggt`, `rmbg`).
Например, RMBG пишет `BriaRMBG`, TripoSG — `triposg`, VGGT (в конфиге нет
обоих полей) — `vggt`. `lux3d.family` всегда хранит family рантайма.

```powershell
# Q8_0 веса, F16 fallback (эталонное качество по умолчанию)
cargo run -p lux3d-cli -- weights quantize triposg `
  --source <canonical-triposg-dir>/model.safetensors `
  --output <gguf-triposg-dir>/model.gguf --dtype q8_0 --fallback-dtype f16

# K-кванты: меньше файл, блоки по 256
cargo run -p lux3d-cli -- weights quantize vggt `
  --source <canonical-vggt-dir>/model.safetensors `
  --output <gguf-vggt-dir>/model.gguf --dtype q4_k --fallback-dtype f16

# RMBG-1.4 — вспомогательный пакет: отдельный GGUF, F16, те же правила standalone.
cargo run -p lux3d-cli -- weights quantize --aux-family rmbg `
  --source <canonical-rmbg-dir>/model.safetensors `
  --output <gguf-rmbg-dir>/model.gguf --dtype f16 --fallback-dtype f16

# Запуск только по GGUF-каталогам; safetensors/manifest/checksums не нужны.
cargo run -p lux3d-cli -- run triposg --device cuda `
  --model-path <gguf-triposg-dir> --rmbg-model-path <gguf-rmbg-dir> `
  --source <image> --output out.obj

# `--all-2d` квантует и rank-2 тензоры без суффикса `.weight`;
# `--prefix transformer.` (можно повторять) ограничивает набор для отладки.
```

Поддерживаемые `--dtype`: `f16`, `f32`, `bf16`, `q8_0`, `q4_0`, `q4_1`,
`q5_0`, `q5_1`, `q2_k`, `q3_k`, `q4_k`, `q5_k`, `q6_k`. `q8_1`/`q8_k` —
активационные GGML-типы, как веса они отклоняются. GGUF без встроенного
конфига (при конвертации не было соседнего `resolved_config.json`) годится
только как дополнение к полному canonical-пакету; standalone-загрузка его
отклоняет.

Замеры на canonical-пакетах (`RTX 3060 12 GiB`; TripoSG
`--steps 2 --num-tokens 1024 --flash-octree-depth 5`, VGGT 3 кадра):

| модель | вариант | размер GGUF | квантовано матриц |
|---|---|---|---|
| TripoSG | canonical F32 | 7.40 GiB | — |
| TripoSG | q8_0 | 1.97 GiB | 527 |
| TripoSG | q6_k | 1.52 GiB | 525 |
| TripoSG | q5_k | 1.28 GiB | 525 |
| TripoSG | q4_k | 1.05 GiB | 525 |
| VGGT | canonical F32 | 4.68 GiB | — |
| VGGT | q8_0 | 1.33 GiB | 384 |
| VGGT | q6_k | 1.09 GiB | 331 |
| VGGT | q5_k | 0.95 GiB | 331 |
| VGGT | q4_k | 0.82 GiB | 331 |

RMBG-1.4 (вспомогательный): F32 0.16 GiB → F16 0.08 GiB; записано 686
тензоров, 112 rank-0 тренировочных буферов опущено.

### Метрики Качества Квантования

`scripts/compare_geometry.py` считает отклонение от float-эталона:
симметричный Chamfer-L1 (относительно диагонали bbox эталона), F-score@tau,
voxel IoU по поверхности, normal consistency для мешей; pointmap/depth PSNR,
AbsRel, Chamfer и camera-pose AUC для тензорных дампов VGGT.
`scripts/eval_quantization.py` прогоняет эталон и все квантованные пакеты и
печатает таблицу.

```powershell
# Дрейф меша (TripoSG/TripoSR)
python scripts/compare_geometry.py mesh <reference.obj> <quantized.obj>

# VGGT: сохранить сырые тензоры рядом с PLY и сравнить
cargo run -p lux3d-cli -- run vggt --model-path <pkg> --source <frames> --output <out.ply> --dump-npz <out.npz>
python scripts/compare_geometry.py vggt <reference.npz> <quantized.npz>

# Серия прогонов по нескольким квантованным пакетам
python scripts/eval_quantization.py triposg --binary target/release/lux3d-cli.exe `
  --reference <canonical-triposg-dir> --variant q8_0=<dir-q8_0> --variant q4_k=<dir-q4_k> `
  --source <input-image> --device cuda --output-dir out/eval -- --steps 2 --num-tokens 1024 --flash-octree-depth 5
```

Дрейф меша TripoSG (2 шага, 1024 токена, depth 5; эталон — F32 той же
подложки):

| вариант | backend | CD rel | F@1% | voxel IoU | VRAM (дельта) | время |
|---|---|---|---|---|---|---|
| q8_0 | CUDA | 0.026% | 0.996 | 0.994 | 3.1 GiB | 10.4 s |
| q8_0 | Vulkan | 0.184% | 0.978 | 0.970 | 3.9 GiB | 13.3 s |
| q6_k | CUDA | 0.135% | 0.988 | 0.988 | — | — |
| q6_k | Vulkan | 0.134% | 0.995 | 0.995 | — | — |
| q5_k | CUDA | 0.155% | 0.993 | 0.994 | — | — |
| q5_k | Vulkan | 0.224% | 0.980 | 0.984 | — | — |
| q4_k | CUDA | 0.253% | 0.973 | 0.953 | 2.2 GiB | 10.0 s |
| q4_k | Vulkan | 0.180% | 0.982 | 0.972 | 2.3 GiB | 12.7 s |
| F32 | CUDA | — | — | — | 8.6 GiB | 14.9 s |
| F32 | Vulkan | — | — | — | 8.8 GiB | 17.1 s |

Двухшаговый сэмплинг чувствителен к входу. На жёстком RGB-входе
(`mesh-input-rgb.png`, с путём через RMBG) Q5_K и Q4_0 ушли до ~1.4% Chamfer,
тогда как Q4_K остался на 0.23%, а Q8_0 — на 0.06%. Эти выбросы повторяемы
на данном входе, но не воспроизводятся на alpha-входе из таблицы, поэтому
Q5_K/Q4_0 считаем входозависимыми; для деплоя — Q8_0/Q6_K/Q4_K.

VGGT (3 кадра; эталон — F32 тензоры той же подложки; pose AUC по 2
относительным парам кадров):

| вариант | backend | pointmap PSNR | depth PSNR | depth AbsRel | CD rel | pose AUC@5 | VRAM (дельта) |
|---|---|---|---|---|---|---|---|
| q8_0 | CUDA | 53.1 dB | 48.1 dB | 0.0038 | 0.274% | 1.0 | 4.7 GiB |
| q8_0 | Vulkan | 55.9 dB | 53.5 dB | 0.0020 | 0.182% | 1.0 | 4.4 GiB |
| q8_0 | WGPU | 55.4 dB | 51.0 dB | 0.0027 | 0.193% | 1.0 | — |
| q6_k | CUDA | 45.7 dB | 44.8 dB | 0.0053 | 0.571% | 1.0 | — |
| q6_k | Vulkan | 46.5 dB | 49.5 dB | 0.0026 | 0.413% | 0.5 | — |
| q5_k | CUDA | 34.7 dB | 35.9 dB | 0.0144 | 1.54% | 0.0 | — |
| q5_k | Vulkan | 36.4 dB | 35.0 dB | 0.0180 | 1.39% | 1.0 | — |
| q4_k | CUDA | 24.7 dB | 23.5 dB | 0.0683 | 3.50% | 0.0 | 4.2 GiB |
| q4_k | Vulkan | 26.9 dB | 25.8 dB | 0.0511 | 2.84% | 0.0 | 3.7 GiB |
| F32 | CUDA | — | — | — | — | — | 6.7 GiB |
| F32 | Vulkan | — | — | — | — | — | 7.3 GiB |

Рекомендации: TripoSG укладывается в ~0.25% Chamfer даже на Q4_K, поэтому
Q4_K/Q5_K для него практичны. VGGT чувствителен к позе: Q8_0 почти без
потерь, Q6_K — приемлемый запасной вариант, Q5_K — на границе, Q4_K
непригоден (pointmap и поза камеры разваливаются). Это совпадает с
публикациями по QVGGT: наивное 4-битное квантование разрушает
реконструкцию такого типа. Перед выкаткой младшего бита прогоните
`--dump-npz` и скрипты выше на своём валидационном наборе. RMBG-1.4 — маленькая
matting-сеть, держим её отдельным F16 GGUF: размер вдвое меньше, дрейф маски
не измеряется.

### Запуск Инференса

```powershell
# Pi3 -> PLY с явной директорией canonical package в layout как 3d/canonical-weights/pi3
cargo run -p lux3d-cli -- run pi3 --model-path <canonical-pi3-dir> --source <input-sequence> --output <output-file.ply>

# Pi3X core -> PLY с явной директорией canonical package в layout как 3d/canonical-weights/pi3x
cargo run -p lux3d-cli -- run pi3x --model-path <canonical-pi3x-dir> --source <input-sequence> --conditions <conditions-file> --output <output-file.ply>

# Pi3X VO -> PLY с явной директорией canonical package в layout как 3d/canonical-weights/pi3x
cargo run -p lux3d-cli -- run pi3x --model-path <canonical-pi3x-dir> --source <input-video> --vo --chunk-size 8 --overlap 4 --conf-threshold 0.05 --inject-condition pose,depth,ray --output <output-file.ply>

# TripoSR -> GLB с явной директорией canonical package в layout как 3d/canonical-weights/triposr
cargo run -p lux3d-cli -- run triposr --model-path <canonical-triposr-dir> --source <input-image> --mc-resolution 256 --mc-threshold 25.0 --output <output-file.glb>

# VGGT -> PLY (world points фильтруются по confidence > 1.0 по умолчанию)
cargo run -p lux3d-cli -- run vggt --model-path <canonical-vggt-dir> --source <input-sequence> --output <output-file.ply>

# Автоскачивание canonical package VGGT (oxide-lab/VGGT-1B)
cargo run -p lux3d-cli -- run vggt --source <input-sequence> --output <output-file.ply>

# TripoSG -> GLB (по умолчанию: 50 rectified-flow шагов, CFG 7.0, seed 42, flash decoder depth 9)
cargo run -p lux3d-cli --features cuda -- run triposg --rmbg-model-path <canonical-rmbg-dir> --model-path <canonical-triposg-dir> --source <input-image> --output <output-file.glb>
# Flash decoder включён по умолчанию; отключить и вернуться к dense/hierarchical MC:
cargo run -p lux3d-cli --features cuda -- run triposg --model-path <canonical-triposg-dir> --source <input-image> --flash-decoder false --dense-octree-depth 8 --hierarchical-octree-depth 9 --output <output-file.glb>
# Быстрый preview: 2 шага, flash depth 5
cargo run -p lux3d-cli --features cuda -- run triposg --model-path <canonical-triposg-dir> --source <input-image> --steps 2 --flash-decoder true --flash-octree-depth 5 --output <preview.glb>

# Автоскачивание canonical package TripoSG (oxide-lab/TripoSG); RMBG 1.4 — отдельный пакет
cargo run -p lux3d-cli --features cuda -- run triposg --source <input-image> --output <output-file.glb>

# Автоскачивание canonical packages из Hugging Face в пользовательский cache
cargo run -p lux3d-cli -- run pi3 --source <input-sequence> --output <output-file.ply>
```

**Форматы вывода:** mesh-команды (`triposr`, `triposg`) выбирают writer по расширению: `.glb` (основной/по умолчанию), `.obj`, `.ply`, `.stl`. Point-cloud команды (`pi3`, `pi3x`, `vggt`) — `.ply` (основной) или `.xyz`. HTTP mesh API по умолчанию отдаёт GLB, point-cloud API — PLY; формат задаётся `options.output_format` (`glb`/`obj`/`ply`/`stl` и `ply`/`xyz` соответственно).

> **Экспериментальные бэкенды (Vulkan / WGPU):** выбор устройства через `--device`:
>
> ```powershell
> cargo run -p lux3d-cli --features vulkan -- run triposr --device vulkan --source <input-image> --output <output-file.glb>
> cargo run -p lux3d-cli --features wgpu -- run pi3 --device wgpu --source <input-sequence> --output <output-file.ply>
> cargo run -p lux3d-cli --features vulkan -- run triposg --device vulkan --source <input-image> --steps 2 --num-tokens 1024 --flash-octree-depth 5 --output <preview.glb>
> ```
>
> Эти пути экспериментальные и не проверяются в CI. Инференс может работать нестабильно, выдавать некорректный результат или вовсе не запускаться в зависимости от GPU, драйвера и ОС. Для продакшена используй CUDA.

## Производительность

Замеры инференса (`[stage] infer`: препроцессинг + нейросетевой проход, одна итерация на процесс, без учёта загрузки весов) на NVIDIA GeForce RTX 3060 12 GiB, Windows (WDDM), форк candle `wgpu/vulkan` rev `efb506c9`, candle-3d (2026-09-06):

| модель | CUDA | Vulkan | WGPU | Vulkan/CUDA | WGPU/CUDA |
|---|---:|---:|---:|---:|---:|
| pi3 (5 кадров, 518×518) | **4.82 с** | 7.83 с | 13.96 с | 1.62× | 2.90× |
| pi3x (6 кадров, 518×518) | **7.00 с** | 11.10 с | 31.84 с | 1.59× | 4.55× |
| triposr (одно изображение) | **1.23 с** | 3.36 с | 4.89 с | 2.73× | 3.98× |

VGGT-1B добавлен позже и измерен отдельно (форк rev `f94d26eb`, 2026-11-11): 3 кадра 518×518,
camera + DPT depth + DPT point heads. RTX 3060 12 GiB: CUDA **6.76 с**, Vulkan 10.96 с (1.62×),
WGPU 19.34 с, CPU 107.66 с. Пик CUDA VRAM 6.52 GiB на 3 кадрах (Vulkan 6.1–7.4 GiB, WGPU
6.8 GiB по повторным прогонам); на NVIDIA L4 23 GiB — 7.98 GiB на 3 кадрах
(4.19 с infer) и 8.30 GiB на 12 кадрах (26.21 с infer).

TripoSG измерен по стадиям на RTX 3060 12 GiB (Windows/WDDM, release build, одно изображение
`test-assets/mesh-input.png`):

| Стадия | CUDA | Vulkan | WGPU |
|---|---:|---:|---:|
| load | 7.12 с | 6.68 с | 6.60 с |
| infer (preprocess + DINO + denoise) | 6.84 с | 8.56 с | 13.13 с |
| flash SDF grid + DiffDMC + trimesh processing | 0.73 с | 0.26 с | 0.45 с |
| export | 0.00 с | 0.00 с | 0.01 с |
| пик VRAM | 8.29 GiB | 8.36 GiB | 8.65 GiB |

Все три колонки — одна команда (`--steps 2 --num-tokens 1024 --flash-octree-depth 5`).
TripoSG держит все три подсети (DINOv2, DiT, VAE) в памяти — 7.4 GiB весов, — поэтому память
определяется чекпоинтом. После исправления дефолтов памяти (Vulkan: transient pool 0 / inflight
grace 2; WGPU: блоки аллокатора 2 MiB) Vulkan укладывается в 0.1 GiB от CUDA, WGPU — в ~0.4 GiB
(бывают колебания ±0.5 GiB из-за асинхронного staging).
12 GiB хватает для preview с 2048 токенами (Vulkan: load 6.56 с, infer 12.86 с, пик 9.13 GiB),
но для эталонного прогона на 50 шагов / depth 9 по-прежнему рекомендуется 16 GiB.

Тёплый criterion (10 сэмплов, установившийся режим — протокол candle-bench, одно сквозное окно измерений): vulkan 3.14 с / 4.90 с на итерацию (1.16×/1.20× от CUDA 2.72 с / 4.09 с), wgpu 8.21 с / 24.0 с (3.0×/5.9×). Самый крупный одиночный рычаг — слитый кернел LayerNorm+RoPE (`rope_layernorm.comp`/`.wgsl` в форке): один диспатч на q/k вместо медленного пути LayerNorm, обе окружающие копии и цепочка rope из 8 опов на каждом rope-вызове — wgpu pi3 в тёплом цикле 54.9 с → 8.21 с.

Корректность: выходы всех комбинаций бэкенд × модель сверены с эталонными мешами CUDA (дельты bbox/центра/среднего в пределах 1% от max extent, число вершин/граней в пределах 0.5% — все PASS); слитый кернел дополнительно покрыт юнит-тестом с точным совпадением (разница 0.0) против составной формы на обоих бэкендах.

Поведение памяти при повторном инференсе (цикл из 10 итераций в одном процессе):

- **CUDA** — потребление плоское; буфер attention scores декодера ограничен 768 MiB чанкингом по query-оси (`LUX3D_MAX_SDPA_SCORES_BYTES`) — убран ~4 GiB транзиент, который раньше фрагментировал пул CUDA и замедлял 12-гигабайтные карты ~в 100 раз.
- **Vulkan** — плоско; удержание памяти ограничено inflight-бюджетом (256 MiB × grace-полоса 2), транзиенты по умолчанию не пулятся (`CANDLE_VK_INFLIGHT_GRACE`, `CANDLE_VK_POOL_MAX_BYTES=0`); пул можно включить для latency-first прогонов. Капы батчей настроены по flush-reason профилированию: transfer-байты 512 MiB (`CANDLE_VK_MAX_BATCH_TRANSFER_BYTES`), descriptor sets 8× от диспатчей — закрытие батча на каждой большой копии активаций стоило ~4-9 ms WDDM fence-signal латентности на каждый сабмит.
- **WGPU** — 10-итерационный бенчмарк pi3x проходит с пиком VRAM 10.4 GiB и нулём ошибок (раньше OOM на ~3-й итерации); free pool / recycle backlog / in-flight удержание ограничены по байтам (`CANDLE_WGPU_POOL_MAX_BYTES`, `CANDLE_WGPU_INFLIGHT_MAX_BYTES`), а блоки аллокатора 2 MiB убирают большую часть слака чекпоинта (`CANDLE_WGPU_BLOCK_BYTES`).

Известные разрывы: оставшийся тёплый разрыв Vulkan/CUDA (1.16–1.20×) — структурный WDDM-налог на сабмит (~4-9 ms на сабмит при строго зависимой цепи операций) — свойство платформы, кодом больше не достижимо; Linux без WDDM закрыл бы его. WGPU отстаёт на 3.0–5.9× в тёплом цикле (было 13.9–19.5×) — качество WGSL GEMM плюс те же межpass-зазоры WDDM. Criterion-харнесс для бенчмарков целых моделей — в `crates/lux3d-core/benches/` (одна пара (бэкенд, модель) на процесс: `LUX3D_BENCH_DEVICE=… LUX3D_BENCH_MODEL=… cargo bench -p lux3d-core --bench bench_main --features vulkan,wgpu`).

## Системные Требования

- Rust 1.85 или новее
- Cargo с поддержкой `edition = "2024"`
- Python 3.x для baseline tooling и нормализации
- NVIDIA GPU с CUDA для проверенных инференс-запусков
- **Бэкенды Vulkan и WGPU** (через форк [FerrisMind/candle](https://github.com/FerrisMind/candle) `wgpu/vulkan`): корректность и стабильность памяти проверены на RTX 3060 (см. [Производительность](#производительность)); медленнее CUDA, WGPU — существенно. Пик TripoSG на обоих бэкендах в пределах ~0.4 GiB от CUDA, так что набор весов 7.9 GB оставляет ~3 GiB запаса на карте 12 GiB.
- Бэкенд Metal на macOS в этом репозитории не тестировался, но теоретически может поддерживаться.
- Директории canonical model package, переданные через `--model-path`, либо автоматически скачанные из Hugging Face в пользовательский cache
- TripoSG требует примерно 8.3–8.7 GiB на preview с 1024 токенами (9.1 GiB с 2048 токенами) на всех трёх бэкендах; для эталонного прогона на 50 шагов рекомендуется 16 GiB+.

## Лицензия

Код и документация этого репозитория лицензированы по Apache 2.0. См. [LICENSE](https://github.com/oxide-lab/Lux3d/blob/main/LICENSE).

Upstream-модельные артефакты и canonical weights сохраняют свои исходные лицензии и ограничения на использование. Перед распространением проверяй условия для конкретного семейства модели.

**VGGT-1B:** оригинальный код и checkpoint `facebook/VGGT-1B` — CC-BY-NC-4.0 (non-commercial). Для коммерческого использования нужен отдельно распространяемый `facebook/VGGT-1B-Commercial`. Canonical package для `--model-path`/автоскачивания публикуется как `oxide-lab/VGGT-1B`; сырой upstream checkpoint остаётся в `facebook/VGGT-1B`. Нативный track head в этом inference slice намеренно не реализован; исключение зафиксировано в контракте.

**TripoSG:** upstream-код и веса — MIT. Сырой checkpoint лежит в `VAST-AI/TripoSG`; canonical package для `--model-path`/автоскачивания публикуется как `oxide-lab/TripoSG`. Upstream `load_image` использует BriaRMBG 1.4, когда нет валидной alpha-маски; Lux3D реализует этот fallback полностью. Веса RMBG не входят в MIT-пакет TripoSG: они нормализованы в отдельный canonical package `rmbg-1.4` (лицензия BRIA, source-available/non-commercial; для коммерческого использования нужна лицензия BRIA) и подключаются через `--rmbg-model-path` или соседнюю папку `rmbg-1.4`. Upstream pipeline по умолчанию использует flash decoder (`use_flash_decoder=True`, `flash_octree_depth=9`); Lux3D реализует тот же flash SDF-планировщик и clean-room Rust DiffDMC. Код и таблицы Diso (CC BY-NC 4.0) не копируются, поэтому на редких ambiguous-ячейках топология может отличаться; non-flash путь остаётся доступен через `--flash-decoder false`. Также реализован полный VAE encode API (FPS-сэмплирование, `vae.quant`, распределение mean/logvar/std), хотя image-to-3D pipeline вызывает только `vae.decode`.
