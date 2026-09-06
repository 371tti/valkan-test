# Compact design

この文書だけで、`rebuild1` の現在地と次に守ることを把握できる状態にします。詳細が必要なときだけ reference docs を読んでください。

## Core

`rebuild1` は app / user code / renderer を async message protocol で分離する renderer です。

最重要ルール:

- renderer 境界の外へ Vulkan 都合を漏らさない。
- Vulkan object は renderer thread だけが所有する。
- renderer は ECS world / user state / file importer を直接読まない。
- user/ECS 側で render extraction し、owned な `FrameSnapshot` を renderer に送る。
- pass dependency と image layout transition は render graph に集める。
- 暗黙 fallback はしない。

## Current State

完了:

- headless app path
- winit window path
- async bounded renderer transport
- dedicated renderer thread
- Vulkan instance / validation callback / device / surface / swapchain
- swapchain image view / scene-post render passes / framebuffers
- one fixed four-layer D16 array for Stable CSM cascades, plus transmittance targets
- frames-in-flight / acquire / submit / present
- per-swapchain-image present semaphore
- render graph compiler: pass/resource declaration, dependency sort, lifetime, barrier plan
- explicit barrier plan generated from resource usage
- executable graph path: Stable CSM + PCSS -> scene -> optional TAA -> post composition (`PostColor`) -> SMAA edge detection -> blending weights -> neighbourhood blending -> optional readback -> present
- build-time Slang -> SPIR-V with generated Vulkan shader asset registry
- temporary debug triangle pipeline was removed; startup without assets presents a clear scene frame
- window capture で triangle 表示確認
- `SurfaceId` / `SurfaceGeneration`
- stale frame drop: `FrameDropped` + `DropReason::StaleSurfaceGeneration`
- deferred destroy
- `.r1scene` importer skeleton
- `LoadAsset` / `AssetLoaded` / `AssetLoadFailed`
- GPU asset handle / store skeleton
- asset file import runs on worker task, not renderer thread
- `FrameSnapshot.render_items` is the scene draw input
- named material texture slots
- `TextureDescriptor` validated payloads
- `MaterialDescriptor` with alpha mode/cutoff
- material slot -> shader binding constants in `shader_interface`
- asset store split into material / texture / store modules
- imported `mesh plane` becomes renderer-owned vertex/index geometry
- Vulkan mesh upload owns backend-local vertex/index buffers
- Vulkan buffer helper extracted from temporary triangle path
- GLB triangle primitive import for explicit app model loading
- window app loads `assets/model.glb` when the app-level sample asset exists
- each render item records an indexed Vulkan mesh draw in the scene pass
- `FrameSnapshot` carries owned `CameraSnapshot` data per view
- window app has old-style free camera controls
- GLB scene bounds frame the app-side camera after `AssetLoaded`
- window extraction submits every loaded mesh/material pair, not only the first mesh
- scene framebuffer has a depth target and mesh pipeline uses depth test/write
- pass cadence is visible through `pass_schedule.rs`
- post composition samples resolved scene color and writes `PostColor`; the dedicated three-pass SMAA chain writes `edgesTex`, then `blendTex`, then samples `PostColor` and writes the acquired swapchain image
- directional shadow passes address one four-layer D16 array as `cascade`
- translucent shadow uses only the reference direction and keeps the nearest visible transmittance layer
- Stable CSM keeps each cascade center snapped to its light-space texel grid; PCSS uses raw blocker taps followed by a hardware-comparison PCF filter
- Vulkan material module uploads imported texture payloads into sampled images
- Vulkan material module uploads material parameter buffers and descriptor sets
- `--window-smoke` requires an existing asset, then verifies Vulkan startup, asset load, mesh draw, post, present, and shutdown. `REBUILD1_WINDOW_QUALITY=high` additionally exercises the dedicated GodRay volume; `-QualitySequence '1,4,1,4'` repeats the asset-backed CSM/GodRay quality transition (quote the sequence in PowerShell).

残りの gate:

- high-quality dedicated GodRay smoke を検証スクリプトの標準手順に含める
- SMAA の edge/resolve 負荷を profiler 実測後に最適化する
- screenshot/RenderDoc による視覚 acceptance を必要な scene ごとに実施する

## Thread And Ownership

```text
main thread
  owns winit window
  sends surface/window events

user/ECS task
  owns game state
  extracts FrameSnapshot
  sends RendererCommand

renderer thread
  owns Vulkan objects
  receives RendererCommand
  records/submits/presents frames
  sends RendererEvent

worker tasks
  import files
  decode images
  compile shaders
```

No Vulkan handle crosses the renderer boundary.

## Protocol Shape

Allowed message scale:

```text
ConfigureSurface
ResizeSurface
SubmitFrame(FrameSnapshot)
CreateMesh / DestroyMesh
CreateTexture / DestroyTexture
Shutdown
```

Forbidden message scale:

```text
BindPipeline
BindDescriptorSet
SetViewport
DrawIndexed
PipelineBarrier
```

The protocol sends intent and extracted data, not Vulkan command streams.

## FrameSnapshot Contract

`FrameSnapshot` must be owned data.

Allowed:

- `FrameId`
- `SurfaceId`
- `SurfaceGeneration`
- camera/light/debug snapshots
- `MeshHandle`
- `MaterialHandle`
- `TextureHandle`
- `Vec<RenderItemPacket>`

Forbidden:

- `&World`
- ECS entity/component references
- `&Mesh`
- renderer-local Vulkan handles
- asset importer objects

## Stage 7 Gate

Stage 6 fixed the material/texture data path. Stage 7 is turning that data into real mesh rendering without breaking the renderer boundary:

1. asset load failure returns `AssetLoadFailed`, not fallback geometry.
2. `UnloadAsset` invalidates handles before deferred destroy.
3. renderer still receives owned `FrameSnapshot` only.
4. real mesh/texture upload must stay behind renderer asset modules and Vulkan backend-local owners.
5. sampled image / sampler / descriptor set objects must stay inside the Vulkan backend.
6. no-asset startup presents a clear frame; diagnostics should use explicit mesh assets, not a dedicated temporary pipeline.
7. camera/view state belongs to app/user code and crosses the boundary only as `CameraSnapshot`.

This keeps asset lifetime, resize ordering, and frame submission from tangling together.

## Shadow Gate

The current graph treats shadows as real resources, not a fake pre-pass:

1. `shadow_direction_0..3/cascade_0..3` write hard comparison depth into the four stable CSM layers.
2. `translucent_shadow_0..3` uses the CSM layer and keeps the nearest colored transmittance layer.
3. `scene` evaluates PCSS using raw blocker taps plus a LINEAR comparison sampler, blends adjacent cascades, and writes HDR plus scene metadata.
4. `god_ray_*` is inserted for the high-quality volumetric route; it evaluates analytic height density along 48 camera-ray strata, applies CSM/PCSS plus translucent transmittance, integrates Beer-Lambert scattering, and resolves through a half-resolution bilateral upscale.
5. `TAA` is an optional history route. It samples scene HDR directly, removes current/previous jitter phase from history UVs, selects a matching 2x2 depth/normal candidate, and uses reactive history only when transparent coverage matches. The production default keeps it dormant and uses the dedicated three-pass SMAA chain.
6. `post` samples resolved scene output, maps stable output UVs to the current jittered metadata grid for SSAO/SSR, runs bloom/GodRay/camera effects/tone mapping, and writes the complete composition to `PostColor`.
7. `smaa_edges` reads `PostColor` and writes the luma edge field; `smaa_weights` reads that field and writes AreaTex blending weights; `smaa` reads `PostColor` plus `blendTex` and writes the final anti-aliased image to the acquired swapchain image.
8. `framebuffer_readback` runs only for requested final-frame summaries.
9. `present` is the external side-effect pass.

Directional depth is device-owned and does not follow swapchain extent. The four cascade layers and split layout are fixed; continuous shadow resolution, tap counts, and bias scales are sent through `SetQualitySettings`, while `SetQualityFeatures(RenderFeatureToggles)` changes only feature ON/OFF state. A fixed frustum-bounding sphere, symmetric receiver overlap at each split, square projection, and one-texel light-space snap keep the four cascade projections stable. PCSS uses receiver-plane depth gradients for per-tap comparisons. Optional TAA history and dedicated GodRay targets follow swapchain extent and are recreated with the swapchain; PCSS and GodRay histories are kept as separate ping-pong resources.

## Module Shape

```text
app/
  camera.rs      # app-side free camera state and extraction
  window.rs      # winit/window owner
  headless.rs    # null renderer path

protocol/
  command.rs     # RendererCommand / RendererEvent
  envelope.rs    # protocol metadata
  ids.rs         # typed ids and handles
  material.rs    # material slots, alpha mode, texture descriptors
  snapshot.rs    # FrameSnapshot and packets
  surface.rs     # surface descriptors and extents
  transport.rs   # async bounded channel

import.rs        # .r1scene skeleton and GLB geometry importer

renderer/
  mod.rs         # renderer thread and backend trait
  assets/
    store.rs     # protocol handles and scene ownership
    material.rs  # renderer material descriptors
    mesh.rs      # renderer-owned mesh geometry records
    texture.rs   # validated texture payload records
    garbage.rs   # deferred destroy queue
  graph.rs       # pass/resource declarations and frame graph compiler
  pass_schedule.rs # frame pass cadence outside Vulkan recording
  pipeline/      # shader interface constants
  surface.rs     # null renderer surface registry
  vulkan.rs      # Vulkan orchestration and command handling
  vulkan/
    debug.rs     # validation callback
    buffer.rs    # host-visible typed buffer upload helper
    frame.rs     # frames-in-flight and command recording
    material.rs  # backend-local sampled images and material descriptors
    mesh.rs      # backend-local mesh buffers and mesh pipeline
    post.rs      # scene color samplers and post pipeline
    taa.rs       # optional temporal history route (dormant by default)
    swapchain.rs # swapchain-owned resources
```

## Implementation Rules

- Keep functions short and one-purpose.
- Put validation at boundaries.
- Use constrained types instead of repeated guards.
- Add a specific doc comment to every function.
- Use `trace` for frame/resource detail.
- Use `info` only for lifecycle milestones.
- Do not add visual-claim unit tests.
- Rendering correctness is checked by validation layer, screenshot/manual capture, or future golden images.

## Reading Order

Default:

1. `compact_design.md`
2. `roadmap.md`
3. `agent_implementation_notes.md`

Reference only:

- `architecture.md`
- `messaging.md`
- `async_runtime.md`
- `ecs_integration.md`
- `code_generation_policy.md`
- `code_quality.md`
- `render_graph.md`
- `assets.md`
- `shader_pipeline.md`
- `testing.md`
- `aaa_traits.md`
