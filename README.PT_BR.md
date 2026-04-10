<p align="left">
  <a href="README.md"><img src="https://img.shields.io/badge/English-232323" alt="English"></a>
  <a href="README.RU.md"><img src="https://img.shields.io/badge/Русский-232323" alt="Русский"></a>
  <a href="README.PT_BR.md"><img src="https://img.shields.io/badge/Português_BR-3ABF7A" alt="Português"></a>
</p>

---

<p align="center">
  <img src=".github/assets/lux3d-logo.svg" alt="Logotipo Lux3D" width="512" height="512">
</p>

<p align="center">
  <b>Workspace Rust para inferência 3D clean-room, inspeção de contratos e canonicalização de pesos.</b><br>
  Runtime CUDA-first de FerrisMind para Pi3, Pi3X e TripoSR dentro do LuxRT.
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Author-FerrisMind-232323" alt="Author FerrisMind">
  <img src="https://img.shields.io/badge/License-Apache--2.0-2ea44f" alt="Apache 2.0">
  <img src="https://img.shields.io/badge/Rust-1.85+-93450a?logo=rust" alt="Rust 1.85+">
  <img src="https://img.shields.io/badge/Runtime-CUDA%20first-76B900" alt="CUDA first runtime">
</p>

<h1 align="center">Lux3D</h1>

<p align="center">
  <img src=".github/assets/lux3d-overview.svg" alt="Visão geral do workspace Lux3D" width="900">
</p>

## 📚 Índice

- [O que é isso?](#-o-que-é-isso)
- [Demo](#-demo)
- [Principais Recursos](#-principais-recursos)
- [Estrutura do Repositório](#️-estrutura-do-repositório)
- [Instalação e Configuração](#️-instalação-e-configuração)
- [Como Começar a Usar](#-como-começar-a-usar)
- [Requisitos do Sistema](#️-requisitos-do-sistema)
- [Agradecimentos](#-agradecimentos)
- [Licença](#-licença)

## ✨ O que é isso?

Lux3D é o workspace Rust 2024 em `3d-rs` que implementa runtime Candle-first, inspeção de contratos e exportação de geometria para Pi3, Pi3X e TripoSR. O projeto é de autoria de FerrisMind e foi feito para rodar dentro do repositório maior `LuxRT`, onde vivem os fontes vendor, os pesos brutos dos modelos e os pesos canônicos.

## 🎬 Demo

Ainda não existe um asset público de demo no repositório. Os pontos de entrada locais já verificados são os comandos CLI abaixo, que geram saídas `.ply` para Pi3/Pi3X e `.obj` para TripoSR.

## 🚀 Principais Recursos

- `lux3d-core` implementa contratos, carregamento de runtime, exportação geométrica e validação de canonical weights.
- `lux3d-cli` expõe `inspect`, `weights normalize` e `run`.
- Suporta três famílias de modelos: `pi3`, `pi3x` e `triposr`.
- Mantém os safetensors canônicos em `3d/canonical-weights/<family>` sem modificar as árvores vendor.
- Inclui tooling Python em [`tools/python_baseline/README.md`](tools/python_baseline/README.md) para captura de parity baseline e normalização de pesos.
- Separa a licença do repositório das licenças upstream de código vendor e pesos, exibindo a política por família via `inspect`.

## 🗂️ Estrutura Do Repositório

| Caminho | Finalidade |
|---------|------------|
| `crates/lux3d-core` | Runtime principal, contratos, exportação e validação de pesos |
| `crates/lux3d-cli` | Interface CLI para inspeção, normalização e execuções |
| `tools/python_baseline` | Scripts Python para parity e canonical-weight normalization |
| `../3d/models` | Pesos upstream brutos esperados pelo runtime |
| `../3d/canonical-weights` | Safetensors normalizados, manifestos e checksums |
| `../tp/3d` | Repositórios vendor usados como source of truth |

`--repo-root` sempre deve apontar para o diretório pai do LuxRT que contém `3d-rs`, `3d/` e `tp/`.

## 🛠️ Instalação E Configuração

### Pré-requisitos

- Rust 1.85 ou mais recente
- Cargo com suporte a `edition = "2024"`
- Python 3.x para normalização e baseline tooling
- GPU NVIDIA com CUDA runtime para comandos `run` verificados
- Um repo root do LuxRT contendo `3d-rs`, `3d/` e `tp/`

### Layout Esperado Do Diretório Pai

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

### Verificações Do Workspace

```powershell
cargo metadata --no-deps
cargo run -p lux3d-cli -- --help
```

### Tooling Python De Baseline

Veja [`tools/python_baseline/README.md`](tools/python_baseline/README.md) para criação de ambientes e comandos golden/smoke.

## 📖 Como Começar A Usar

### Inspecionar Contratos E Notas De Licença

```powershell
cargo run -p lux3d-cli -- inspect --repo-root H:\GitHub\LuxRT pi3
cargo run -p lux3d-cli -- inspect --repo-root H:\GitHub\LuxRT pi3x
cargo run -p lux3d-cli -- inspect --repo-root H:\GitHub\LuxRT triposr
```

### Normalizar Pesos Canônicos

```powershell
cargo run -p lux3d-cli -- weights normalize --repo-root H:\GitHub\LuxRT pi3
cargo run -p lux3d-cli -- weights normalize --repo-root H:\GitHub\LuxRT pi3x
cargo run -p lux3d-cli -- weights normalize --repo-root H:\GitHub\LuxRT triposr
```

### Executar Exportações De Inferência

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

## 🖥️ Requisitos Do Sistema

- Os exemplos de referência deste repositório usam Windows PowerShell.
- `inspect` e `cargo metadata` não exigem suporte a inferência por GPU.
- `run` atualmente inicializa `candle_core::Device::new_cuda(0)`, então execuções de inferência verificadas exigem hardware NVIDIA com CUDA.
- Os pesos brutos precisam existir em:
  - `3d/models/yyfz233-Pi3/`
  - `3d/models/yyfz233-Pi3X/`
  - `3d/models/stabilityai-TripoSR/`
- As saídas canônicas são esperadas em `3d/canonical-weights/<family>/`.

## 🙏 Agradecimentos

Este workspace depende do excelente trabalho de:

- árvores de código upstream de Pi3 e Pi3X em `tp/3d/Pi3`
- TripoSR em `tp/3d/TripoSR`
- Candle, Clap, nalgebra, mcubes e safetensors
- FerrisMind como autor do workspace Rust e da organização deste repositório

## 📄 Licença

O código e a documentação dentro de `3d-rs` estão sob Apache 2.0. Veja [LICENSE](LICENSE).

As árvores vendor, os pesos brutos dos modelos e os artefatos canônicos mantêm suas licenças upstream e respectivas restrições de uso. Antes de redistribuir artefatos, use `cargo run -p lux3d-cli -- inspect --repo-root <repo-root> <family>` para revisar a política de licença da família correspondente.

Copyright (c) 2026 FerrisMind
