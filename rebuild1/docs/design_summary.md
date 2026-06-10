# Design summary

## One sentence

`rebuild1` は、app / user code / renderer を async message protocol で分離し、renderer 専用 thread に Vulkan を閉じ込める、小さく読める renderer 設計です。

## Non-negotiable decisions

1. app / user code / renderer は直接結合しない。
2. renderer は `RendererCommand` を受け取り、`RendererEvent` を返す。
3. Vulkan object と Vulkan 都合は renderer 境界の外へ漏らさない。
4. winit window は main thread が所有し、renderer には `SurfaceDescriptor` だけを渡す。
5. ECS は renderer の上に載せる。renderer と protocol は ECS crate に依存しない。
6. ECS world から renderer へ渡すのは render extraction 済みの `FrameSnapshot` だけにする。
7. renderer core は暗黙 fallback をしない。fallback は user/app policy。
8. asset import、GPU upload、render target、render graph、pipeline は別の責務として分ける。
9. pass の依存、resource、layout transition は render graph に集める。
10. shader binding は Rust と shader で暗黙同期しない。名前付き interface に集約する。
11. safety は過剰な guard ではなく、boundary validation と constrained type で作る。
12. すべての関数に、具体的に何をする関数かを書く。

## Runtime shape

```text
main thread
  owns winit window
  sends AppEvent by try_send

user task
  owns UserApp / 将来の ECS World
  runs simulation and render extraction
  sends RendererCommand
  receives RendererEvent

renderer thread
  owns Vulkan objects
  receives RendererCommand
  updates SceneCache / AssetStore / RenderGraph
  records command buffers and presents
  sends RendererEvent

io workers
  read files
  decode images
  compile shaders
  import assets to intermediate data
```

## Frame flow

```text
winit event
  -> AppEvent
  -> UserApp / ECS schedule
  -> render extraction
  -> FrameSnapshot
  -> RendererCommand::SubmitFrame
  -> RendererTask
  -> SceneCache
  -> RenderGraph
  -> queue submit / present
  -> RendererEvent::FramePresented
```

`FrameSnapshot` は owned data です。参照、raw pointer、Vulkan handle、ECS entity、component reference を含めません。

## Asset flow

```text
UserApp / ECS asset system
  -> RendererCommand::LoadAsset
  -> importer
  -> ImportedScene
  -> AssetStore upload
  -> RendererEvent::AssetLoaded
  -> user/ECS stores protocol handles
  -> render extraction writes handles into FrameSnapshot
```

読み込みに失敗したとき、renderer は隠れた cube や placeholder を選びません。失敗は `AssetLoadFailed` として返します。

## Module map

```text
app/
  winit integration, event loop, window owner

protocol/
  commands, events, envelopes, ids, handles, material descriptors, transport

user/
  UserApp, 将来の ECS world owner, render extraction

renderer/
  mod.rs         # renderer thread loop, backend trait, null backend
  surface.rs     # backend-neutral surface registry for null/testing
  vulkan.rs      # Vulkan backend orchestration and protocol command handling
  vulkan/
    debug.rs     # validation layer, debug utils messenger, callback logging
    swapchain.rs # surface support query, swapchain, views, render pass, framebuffers
    frame.rs     # frames-in-flight, command pools, sync
  graph/         # pass declarations, resources, barriers
  targets/       # depth, scene, fixed shadow targets
  assets/        # handle store, material records, texture payloads, deferred destroy
  import.rs      # file import to intermediate data
  pipeline/      # shader interface, shader modules, layouts, pipelines
  scene/         # renderer-local SceneCache and draw packets
```

## Naming

- `RendererTask`: concrete renderer loop running on the dedicated renderer thread.
- `RendererBackend`: trait for a renderer implementation that consumes protocol channels.
- `RendererTransport`: channel/log/remote bridge that only moves messages.
- `FrameSnapshot`: extracted per-frame render request from user/ECS.
- `SceneCache`: renderer-local copy/cache built from snapshots.
- `RenderItemPacket`: renderer-local resolved mesh/material item used by graph passes.
- `ExternalObjectId`: stable debug/picking id; not an ECS entity.

## Reading order

1. `design_summary.md`
2. `compact_design.md`
3. `roadmap.md`
4. `agent_implementation_notes.md`

Reference docs:

5. `code_generation_policy.md`
6. `architecture.md`
7. `messaging.md`
8. `async_runtime.md`
9. `ecs_integration.md`
10. `code_quality.md`
11. `render_graph.md`
12. `assets.md`
13. `shader_pipeline.md`
14. `testing.md`

## Current correction

実装を進めた結果、Vulkan backend が早い段階で大きくなり始めました。これは前作の「巨大な `Renderer`」に戻る兆候なので、以下を現時点の軌道修正として固定します。

- `vulkan.rs` は protocol command handling と大きな owner の orchestration に絞る。
- validation/debug utils は `vulkan/debug.rs` に隔離する。
- resize で再作成される swapchain resources は `vulkan/swapchain.rs` に隔離する。
- 次に command pool / command buffer / semaphore / fence を追加するときは `vulkan/frame.rs` か `renderer/frame/` を先に作り、`vulkan.rs` に足し続けない。
- temporary debug triangle は削除済み。asset 未ロード時は scene clear frame を present し、描画は `FrameSnapshot.render_items` に集約する。
- material/texture の data path は `protocol/material.rs` と `renderer/assets/{store,material,texture}.rs` に分ける。
- 次の実装 slice は `code_generation_policy.md` の module policy と stop signs を先に確認する。
