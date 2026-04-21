<p align="left">
  <a href="README.md"><img src="https://img.shields.io/badge/English-232323" alt="English"></a>
  <a href="README.RU.md"><img src="https://img.shields.io/badge/Русский-232323" alt="Русский"></a>
  <a href="README.PT_BR.md"><img src="https://img.shields.io/badge/Português_BR-3ABF7A" alt="Português"></a>
</p>

---

<p align="center">
  <b>Workspace Rust para inferência 3D clean-room, inspeção de contratos e canonicalização de pesos.</b><br>
  Runtime CUDA-first para Pi3, Pi3X e TripoSR.
</p>

<p align="center">
  <img src="https://img.shields.io/badge/License-Apache--2.0-2ea44f" alt="Apache 2.0">
  <img src="https://img.shields.io/badge/Rust-1.85+-93450a?logo=rust" alt="Rust 1.85+">
  <img src="https://img.shields.io/badge/Runtime-CUDA%20first-76B900" alt="CUDA first runtime">
</p>

<h1 align="center">Lux3D</h1>

## Índice

- [O que é isso?](#o-que-é-isso)
- [Principais Recursos](#principais-recursos)
- [Estrutura do Repositório](#estrutura-do-repositório)
- [Início Rápido](#início-rápido)
- [Requisitos do Sistema](#requisitos-do-sistema)
- [Licença](#licença)

## O que é isso?

Lux3D é um workspace Rust 2024 que fornece:

- runtime Candle-first para inferência
- inspeção de contratos para famílias de modelos suportadas
- normalização e validação de canonical weights
- exportação de geometria para point clouds e meshes

Famílias de modelos suportadas:

- `pi3`
- `pi3x`
- `triposr`

## Principais Recursos

- `lux3d-core` implementa contratos, carregamento de runtime, exportação e validação de pesos.
- `lux3d-cli` expõe `inspect`, `weights normalize` e `run`.
- O tooling Python de baseline está descrito em [`tools/python_baseline/README.md`](https://github.com/oxide-lab/Lux3d/blob/main/tools/python_baseline/README.md).
- O licenciamento por família de modelo pode ser inspecionado pelo CLI antes da redistribuição.

## Estrutura Do Repositório

| Caminho | Finalidade |
|---------|------------|
| `crates/lux3d-core` | Runtime principal, contratos, exportação e validação de pesos |
| `crates/lux3d-cli` | Interface CLI para inspeção, normalização e execuções |
| `tools/python_baseline` | Scripts Python para parity e canonical-weight normalization |
| `.github/assets` | Assets de documentação do projeto |

## Início Rápido

### Verificações Do Workspace

```powershell
cargo metadata --no-deps
cargo run -p lux3d-cli -- --help
```

### Inspecionar Contratos

```powershell
cargo run -p lux3d-cli -- inspect --repo-root <runtime-root> pi3
cargo run -p lux3d-cli -- inspect --repo-root <runtime-root> pi3x
cargo run -p lux3d-cli -- inspect --repo-root <runtime-root> triposr
```

### Normalizar Pesos Canônicos

```powershell
cargo run -p lux3d-cli -- weights normalize --repo-root <runtime-root> --raw-model-dir <raw-pi3-dir> --output-dir <canonical-pi3-dir> pi3
cargo run -p lux3d-cli -- weights normalize --repo-root <runtime-root> --raw-model-dir <raw-pi3x-dir> --output-dir <canonical-pi3x-dir> pi3x
cargo run -p lux3d-cli -- weights normalize --repo-root <runtime-root> --raw-model-dir <raw-triposr-dir> --output-dir <canonical-triposr-dir> triposr
```

### Executar Inferência

```powershell
# Pi3 -> PLY com diretório de canonical package explícito no layout de 3d/canonical-weights/pi3
cargo run -p lux3d-cli -- run pi3 --model-path <canonical-pi3-dir> --source <input-sequence> --output <output-file.ply>

# Pi3X core -> PLY com diretório de canonical package explícito no layout de 3d/canonical-weights/pi3x
cargo run -p lux3d-cli -- run pi3x --model-path <canonical-pi3x-dir> --source <input-sequence> --conditions <conditions-file> --output <output-file.ply>

# Pi3X VO -> PLY com diretório de canonical package explícito no layout de 3d/canonical-weights/pi3x
cargo run -p lux3d-cli -- run pi3x --model-path <canonical-pi3x-dir> --source <input-video> --vo --chunk-size 8 --overlap 4 --conf-threshold 0.05 --inject-condition pose,depth,ray --output <output-file.ply>

# TripoSR -> OBJ com diretório de canonical package explícito no layout de 3d/canonical-weights/triposr
cargo run -p lux3d-cli -- run triposr --model-path <canonical-triposr-dir> --source <input-image> --mc-resolution 256 --mc-threshold 25.0 --output <output-file.obj>

# Download automático de canonical packages do Hugging Face para o cache do usuário
cargo run -p lux3d-cli -- run pi3 --source <input-sequence> --output <output-file.ply>
```

## Requisitos Do Sistema

- Rust 1.85 ou mais recente
- Cargo com suporte a `edition = "2024"`
- Python 3.x para baseline tooling e normalização
- GPU NVIDIA com CUDA para execuções de inferência verificadas
- Diretórios de canonical model package fornecidos por `--model-path` ou baixados automaticamente do Hugging Face para o cache do usuário

## Licença

O código e a documentação deste repositório estão sob Apache 2.0. Veja [LICENSE](https://github.com/oxide-lab/Lux3d/blob/main/LICENSE).

Os artefatos de modelo upstream e os canonical weights mantêm suas licenças e restrições originais de uso. Antes de redistribuir, verifique os termos da família de modelo correspondente.

Autor do projeto: FerrisMind
