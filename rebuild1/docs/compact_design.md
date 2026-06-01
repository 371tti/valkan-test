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
- frames-in-flight / acquire / submit / present
- per-swapchain-image present semaphore
- render graph compiler: pass/resource declaration, dependency sort, lifetime, barrier plan
- explicit barrier plan generated from resource usage
- executable graph path: scene color + scene depth -> post -> swapchain present
- executable graph path: shadow map -> reflection target -> scene target -> post -> present
- build-time GLSL -> SPIR-V
- debug triangle vertex buffer / frame uniform / basic graphics pipeline
- window capture で triangle 表示確認
- `SurfaceId` / `SurfaceGeneration`
- stale frame drop: `FrameDropped` + `DropReason::StaleSurfaceGeneration`
- deferred destroy
- `.r1scene` importer skeleton
- `LoadAsset` / `AssetLoaded` / `AssetLoadFailed`
- GPU asset handle / store skeleton
- asset file import runs on worker task, not renderer thread
- `DrawPacket::DebugTriangle` drives the temporary debug triangle
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
- `DrawPacket::Mesh` records indexed Vulkan mesh draws in the swapchain pass
- `FrameSnapshot` carries owned `CameraSnapshot` data per view
- window app has old-style free camera controls
- GLB scene bounds frame the app-side camera after `AssetLoaded`
- window extraction submits every loaded mesh/material pair, not only the first mesh
- scene framebuffer has a depth target and mesh pipeline uses depth test/write
- pass cadence is visible through `pass_schedule.rs`
- post pipeline samples scene color and writes the acquired swapchain image
- shadow pass owns a real depth target and depth-only mesh pipeline
- reflection pass owns real color/depth targets and mesh pipeline
- Vulkan material module uploads imported texture payloads into sampled images
- Vulkan material module uploads material parameter buffers and descriptor sets
- `--window-smoke` verifies Vulkan startup, `assets/model.glb` load, mesh draw, post, present, and shutdown

未着手または次の gate:

- shader interface validation
- sampled material texture use in the mesh shader
- visual verification scenes for texture, alpha cutout, shadow, reflection, and camera effects

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
- `Vec<DrawPacket>`

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
6. debug triangle stays a `DrawPacket` for no-asset startup and diagnostics.
7. camera/view state belongs to app/user code and crosses the boundary only as `CameraSnapshot`.

This keeps asset lifetime, resize ordering, and frame submission from tangling together.

## Stage 7.5 Gate

Stage 7.5 moved the renderer from a scene/post-only graph to the standard frame graph:

1. `shadow` writes a real shadow depth target.
2. `reflection` reads shadow and writes real reflection color/depth targets.
3. `scene` reads shadow/reflection graph resources and writes scene color/depth.
4. `post` samples scene color and writes the acquired swapchain image.
5. `present` is the external side-effect pass.

Texture and material upload are backend-local Vulkan owners now. The next visual slice is not another graph rewrite; it is binding sampled material data into the mesh shader and verifying the result with fixed scenes.

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
    post.rs      # scene_color sampler and post pipeline
    swapchain.rs # swapchain-owned resources
    triangle.rs  # temporary debug triangle resources
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
