<p align="left">
  <a href="README.md"><img src="https://img.shields.io/badge/English-232323" alt="English"></a>
  <a href="README.RU.md"><img src="https://img.shields.io/badge/Русский-232323" alt="Русский"></a>
  <a href="README.PT_BR.md"><img src="https://img.shields.io/badge/Português_BR-3ABF7A" alt="Português"></a>
</p>

---

<p align="center">
  <b>Workspace Rust para inferência 3D clean-room, inspeção de contratos e canonicalização de pesos.</b><br>
  Runtime CUDA-first para Pi3, Pi3X, TripoSR, VGGT e TripoSG.
</p>

<p align="center">
  <img src="https://img.shields.io/badge/License-Apache--2.0-2ea44f" alt="Apache 2.0">
  <img src="https://img.shields.io/badge/Rust-1.85+-93450a?logo=rust" alt="Rust 1.85+">
  <img src="https://img.shields.io/badge/Runtime-CUDA%20first-76B900" alt="CUDA first runtime">
</p>

<h1 align="center">Lux3D / candle-3d</h1>

## Índice

- [O que é isso?](#o-que-é-isso)
- [Principais Recursos](#principais-recursos)
- [Demo](#demo)
- [Estrutura do Repositório](#estrutura-do-repositório)
- [Início Rápido](#início-rápido)
- [Desempenho](#desempenho)
- [Requisitos do Sistema](#requisitos-do-sistema)
- [Licença](#licença)

## O que é isso?

Lux3D / candle-3d é um workspace Rust 2024 que fornece:

- runtime Candle-first para inferência
- inspeção de contratos para famílias de modelos suportadas
- normalização e validação de canonical weights
- exportação de geometria para point clouds e meshes

Famílias de modelos suportadas:

- `pi3`
- `pi3x`
- `triposr`
- `vggt` — VGGT-1B: point cloud, depth e câmera multi-view. Pesos originais CC-BY-NC-4.0
  (não comercial); o checkpoint comercial é distribuído separadamente.
- `triposg` — TripoSG: inferência image-to-mesh (encoder DINOv2-large, DiT rectified-flow de
  21 blocos, decoder/encoder VAE de SDF, flash decoder + DiffDMC clean-room por padrão).
  Licença MIT; o BriaRMBG 1.4 para entrada RGB é distribuído no pacote separado `rmbg-1.4`
  sob a licença BRIA.

## Principais Recursos

- `lux3d-core` implementa contratos, carregamento de runtime, exportação e validação de pesos.
- `lux3d-cli` expõe `inspect`, `weights normalize` e `run`.
- O tooling Python de baseline está descrito em [`tools/python_baseline/README.md`](https://github.com/oxide-lab/Lux3d/blob/main/tools/python_baseline/README.md).
- O licenciamento por família de modelo pode ser inspecionado pelo CLI antes da redistribuição.
- **Experimental:** inferência em Vulkan e WGPU via `--device vulkan` / `--device wgpu` (build com `--features vulkan` / `--features wgpu`). CUDA é o backend principal; os caminhos Pi3/Pi3X/TripoSR e VGGT têm verificações de correção em Vulkan, e o TripoSG tem cobertura de smoke em Vulkan/WGPU (veja [Desempenho](#desempenho)). WGPU continua experimental — pode ser instável, gerar resultados incorretos ou simplesmente não funcionar, dependendo do hardware e dos drivers.

## Demo

https://github.com/user-attachments/assets/f972c1e5-15b0-4206-aef4-8975e9eb6df4

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
cargo run -p lux3d-cli -- inspect --repo-root <runtime-root> vggt
cargo run -p lux3d-cli -- inspect --repo-root <runtime-root> triposg
```

### Normalizar Pesos Canônicos

```powershell
cargo run -p lux3d-cli -- weights normalize --repo-root <runtime-root> --raw-model-dir <raw-pi3-dir> --output-dir <canonical-pi3-dir> pi3
cargo run -p lux3d-cli -- weights normalize --repo-root <runtime-root> --raw-model-dir <raw-pi3x-dir> --output-dir <canonical-pi3x-dir> pi3x
cargo run -p lux3d-cli -- weights normalize --repo-root <runtime-root> --raw-model-dir <raw-triposr-dir> --output-dir <canonical-triposr-dir> triposr
cargo run -p lux3d-cli -- weights normalize --repo-root <runtime-root> --raw-model-dir <raw-vggt-dir> --output-dir <canonical-vggt-dir> vggt
cargo run -p lux3d-cli -- weights normalize --repo-root <runtime-root> --raw-model-dir <raw-triposg-dir> --output-dir <canonical-triposg-dir> triposg
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

# VGGT -> PLY (world points filtrados por confidence > 1.0 por padrão)
cargo run -p lux3d-cli -- run vggt --model-path <canonical-vggt-dir> --source <input-sequence> --output <output-file.ply>

# Download automático do canonical package VGGT (oxide-lab/VGGT-1B)
cargo run -p lux3d-cli -- run vggt --source <input-sequence> --output <output-file.ply>

# TripoSG -> OBJ (padrões: 50 passos rectified-flow, CFG 7.0, seed 42, flash decoder depth 9)
cargo run -p lux3d-cli --features cuda -- run triposg --rmbg-model-path <canonical-rmbg-dir> --model-path <canonical-triposg-dir> --source <input-image> --output <output-file.obj>
# O flash decoder é o padrão; desative para usar marching cubes denso/hierárquico:
cargo run -p lux3d-cli --features cuda -- run triposg --model-path <canonical-triposg-dir> --source <input-image> --flash-decoder false --dense-octree-depth 8 --hierarchical-octree-depth 9 --output <output-file.obj>
# Preview rápido: 2 passos, flash depth 5
cargo run -p lux3d-cli --features cuda -- run triposg --model-path <canonical-triposg-dir> --source <input-image> --steps 2 --flash-decoder true --flash-octree-depth 5 --output <preview.obj>

# Download automático do canonical package TripoSG (oxide-lab/TripoSG); RMBG 1.4 é um pacote separado
cargo run -p lux3d-cli --features cuda -- run triposg --source <input-image> --output <output-file.obj>

# Download automático de canonical packages do Hugging Face para o cache do usuário
cargo run -p lux3d-cli -- run pi3 --source <input-sequence> --output <output-file.ply>
```

> **Backends experimentais (Vulkan / WGPU):** selecione um dispositivo não-CUDA com `--device`:
>
> ```powershell
> cargo run -p lux3d-cli --features vulkan -- run triposr --device vulkan --source <input-image> --output <output-file.obj>
> cargo run -p lux3d-cli --features wgpu -- run pi3 --device wgpu --source <input-sequence> --output <output-file.ply>
> cargo run -p lux3d-cli --features vulkan -- run triposg --device vulkan --source <input-image> --steps 2 --num-tokens 1024 --flash-octree-depth 5 --output <preview.obj>
> ```
>
> Esses caminhos são experimentais e não são verificados no CI. A inferência pode ser instável, produzir saída incorreta ou falhar completamente, dependendo da GPU, do driver e do SO. Use CUDA para cargas de produção.

## Desempenho

Medições por estágio do TripoSG em uma NVIDIA GeForce RTX 3060 12 GiB (Windows/WDDM, build
release, uma imagem `test-assets/mesh-input.png`):

| Estágio | CUDA | Vulkan | WGPU |
|---|---:|---:|---:|
| load | 7,12 s | 6,68 s | 6,60 s |
| infer (preprocess + DINO + denoise) | 6,84 s | 8,56 s | 13,13 s |
| flash SDF grid + DiffDMC + trimesh processing | 0,73 s | 0,26 s | 0,45 s |
| export | 0,00 s | 0,00 s | 0,01 s |
| pico de VRAM | 8,29 GiB | 8,36 GiB | 8,65 GiB |

As três colunas usam o mesmo comando (`--steps 2 --num-tokens 1024 --flash-octree-depth 5`).
O TripoSG mantém as três sub-redes (DINOv2, DiT, VAE) residentes — 7,4 GiB de pesos —, então a
memória é dominada pelo checkpoint. Após as correções dos padrões de memória (Vulkan: pool
transiente 0 / inflight grace 2; WGPU: blocos de alocador de 2 MiB), o Vulkan fica dentro de
0,1 GiB do CUDA e o WGPU dentro de ~0,4 GiB (com variação ocasional de ±0,5 GiB do staging
assíncrono). 12 GiB bastam para o preview com 2048 tokens (Vulkan: load 6,56 s,
infer 12,86 s, pico 9,13 GiB), mas 16 GiB ainda são recomendados para a execução de referência
com 50 passos / depth 9.

## Requisitos Do Sistema

- Rust 1.85 ou mais recente
- Cargo com suporte a `edition = "2024"`
- Python 3.x para baseline tooling e normalização
- GPU NVIDIA com CUDA para execuções de inferência verificadas
- **Experimental:** backends Vulkan e WGPU (via fork [FerrisMind/candle](https://github.com/FerrisMind/candle) `wgpu/vulkan`). Podem ser instáveis ou não funcionar em parte do hardware; não são testados no mesmo nível que CUDA. O pico do TripoSG fica dentro de ~0,4 GiB do CUDA nos dois backends, então o conjunto de pesos de 7,9 GB deixa ~3 GiB de margem em uma placa de 12 GiB.
- O backend Metal no macOS não foi testado neste repositório, mas pode ser teoricamente suportado.
- Diretórios de canonical model package fornecidos por `--model-path` ou baixados automaticamente do Hugging Face para o cache do usuário
- O TripoSG precisa de cerca de 8,3–8,7 GiB no preview com 1024 tokens (9,1 GiB com 2048 tokens) nos três backends; 16 GiB+ são recomendados para a execução de referência com 50 passos.

## Licença

O código e a documentação deste repositório estão sob Apache 2.0. Veja [LICENSE](https://github.com/oxide-lab/Lux3d/blob/main/LICENSE).

Os artefatos de modelo upstream e os canonical weights mantêm suas licenças e restrições originais de uso. Antes de redistribuir, verifique os termos da família de modelo correspondente.

**VGGT-1B:** o código original e o checkpoint `facebook/VGGT-1B` são CC-BY-NC-4.0 (não comercial). O uso comercial exige o checkpoint `facebook/VGGT-1B-Commercial`, distribuído separadamente. O canonical package usado por `--model-path`/download automático é publicado como `oxide-lab/VGGT-1B`; o checkpoint upstream bruto permanece em `facebook/VGGT-1B`. O track head nativo não é implementado nesta fatia de inferência; o contrato registra essa exclusão.

**TripoSG:** o código e os pesos upstream são MIT. O checkpoint bruto fica em `VAST-AI/TripoSG`; o canonical package usado por `--model-path`/download automático é publicado como `oxide-lab/TripoSG`. O `load_image` upstream usa BriaRMBG 1.4 quando não há alpha válido; o Lux3D implementa esse fallback por completo. Os pesos RMBG não entram no pacote MIT do TripoSG: são normalizados no pacote separado `rmbg-1.4`, sob a licença BRIA source-available/não comercial (uso comercial exige licença BRIA), resolvido por `--rmbg-model-path` ou por um diretório irmão `rmbg-1.4`. O pipeline upstream usa o flash decoder por padrão (`use_flash_decoder=True`, `flash_octree_depth=9`); o Lux3D implementa o mesmo schedule flash de SDF e um DiffDMC Rust clean-room. Código e tabelas do Diso (CC BY-NC 4.0) não são copiados, então células checkerboard ambíguas podem resolver de forma diferente; o caminho denso/hierárquico non-flash continua disponível com `--flash-decoder false`. A API completa de encode da VAE (FPS, `vae.quant`, distribuição mean/logvar/std) também está implementada, embora o pipeline image-to-3D use apenas `vae.decode`.
