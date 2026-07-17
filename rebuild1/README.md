# rebuild1 design

`rebuild1/` は次の renderer を設計しながら実装していく作り直し用 crate です。前回のコードは `old/` に退避済みで、ここでは設計ドキュメントと実装を同時に保守します。

## 設計の芯

`rebuild1` は、app / user code / renderer を async message protocol で分離し、renderer 専用 thread に Vulkan を閉じ込める設計です。

将来 ECS を上に載せても、renderer は ECS world を読みません。user/ECS 側で render extraction を行い、owned な `FrameSnapshot` を renderer に送ります。

実装時は読みやすさを最適化対象にします。短い関数、明示された制約、boundary validation、constrained type を優先し、すべての関数に「具体的に何をする関数か」を書きます。

## 前作から持ち越さないもの

- 読み込み失敗時の暗黙 cube fallback
- 何でも持つ巨大な `Renderer`
- `assets/gpu.rs` に render target まで入れる構造
- renderer 内にファイル形式 importer を抱え込む構造
- pass 順序と image layout 遷移の手管理
- Rust 側 descriptor binding と shader layout の暗黙同期
- 見た目を保証しない unit test の増殖

## 設計ドキュメント

通常はこの 3 つだけ読めば現在地が分かるようにします。

- [compact_design.md](docs/compact_design.md): まず読む圧縮版。現在地、境界、次の gate
- [roadmap.md](docs/roadmap.md): 実装順、完了条件、未決事項
- [agent_implementation_notes.md](docs/agent_implementation_notes.md): 実装 agent に渡す境界、禁止事項、実装順の固定メモ

以下は迷ったときに読む reference です。

- [design_summary.md](docs/design_summary.md): 全体設計の短いまとめ、最終決定、読む順番
- [code_generation_policy.md](docs/code_generation_policy.md): 直近の実装で守る短期ポリシー、module 境界、完了チェック
- [architecture.md](docs/architecture.md): 全体構成、責務境界、初期化と frame flow
- [messaging.md](docs/messaging.md): app / user code / renderer をつなぐ command、event、snapshot protocol
- [async_runtime.md](docs/async_runtime.md): 最初から thread 分割する async runtime、task、channel 方針
- [aaa_traits.md](docs/aaa_traits.md): AAA 指向へ拡張するための境界、data flow、trait 方針
- [ecs_integration.md](docs/ecs_integration.md): 将来 ECS を載せるための render extraction 境界
- [code_quality.md](docs/code_quality.md): 短く、安全で、読みやすいコードを書くための制約
- [render_graph.md](docs/render_graph.md): pass、resource、barrier、resize の設計
- [assets.md](docs/assets.md): importer、GPU asset store、handle、fallback 方針
- [shader_pipeline.md](docs/shader_pipeline.md): shader interface、pipeline、hot reload 方針
- [testing.md](docs/testing.md): unit test と rendering test の線引き

## 現在の実装メモ

- app は headless path と winit window path を持つ。
- protocol は command/event/envelope/id/snapshot/surface/transport に分かれている。
- renderer 品質は `SetQualitySettings(RenderQualitySettings)` で送る。window app は通常操作用に `performance()` を送る。SSR / SSAO / AA / lighting / post look は renderer の実行ポリシーであり、ECS owned な `FrameSnapshot` には混ぜない。
- renderer は dedicated thread 上で `RendererBackend` を動かす。
- Vulkan backend は instance / validation debug callback / device / surface / swapchain / image view / fixed cascade shadow resources / scene/post render pass / framebuffer / frame resources / mesh / material / post pipeline まで持つ。
- `SubmitFrame` は target surface に対して acquire、graph compile、barrier record、opaque cascade shadow pass、translucent shadow pass、scene pass、post pass、readback pass、submit、present を行う。
- present 待ちの semaphore は swapchain image ごとに持ち、frame slot ごとに再利用しない。
- window path は surface configure 後に redraw-driven の最小 `SubmitFrame` loop を走らせる。
- renderer graph は `shadow_cascade[0..2] -> translucent_shadow[0..2] -> scene_color/depth -> post -> optional readback -> swapchain present` を実行計画として持ち、resource state から explicit barrier plan を生成する。
- shader source は機能別に分けた Slang (`crates/gr-render/shaders/{shared,scene,shadow,post}/`) を `build.rs` で SPIR-V にする。`slangc` と `spirv-val` を使い、生成 asset registry は `renderer::vulkan::shader` に集約する。temporary debug triangle pipeline は削除済みで、asset 未ロード時は scene clear frame を present する。
- Stage 4 first draw は window capture で triangle 表示確認済み。
- winit resize は command channel を詰まらせないよう、in-flight 1 件と pending 最新 1 件に coalesce する。
- `FrameSnapshot` は `SurfaceId` / `SurfaceGeneration` を持ち、古い generation の frame は `FrameDropped` で落とす。
- Stage 5 asset path は `.r1scene` importer skeleton、worker import、`LoadAsset` / `AssetLoaded` / `AssetLoadFailed`、`GpuAssetStore`、deferred destroy queue まで実装済み。
- `FrameSnapshot` は正式な mesh draw 入力として `render_items` を持つ。temporary `DrawPacket` / debug triangle 経路は削除済み。
- Stage 6 material/texture path は named slot、alpha mode、validated `TextureDescriptor`、`MaterialDescriptor`、shader binding constants まで実装済み。
- renderer assets は `store.rs` / `mesh.rs` / `material.rs` / `texture.rs` / `garbage.rs` に分割済み。
- Stage 7 の mesh rendering slice は完了済みで、`.r1scene` / GLB geometry は renderer-owned vertex/index geometry として store に残り、Vulkan backend で vertex/index buffer へ upload される。
- `old/assets/model.glb` は app-level sample として `rebuild1/assets/model.glb` にコピー済み。window path はこのファイルが存在するときだけ `LoadAsset` を送り、全 mesh/material pair を `render_items` として submit する。
- `render_items` は `vulkan/mesh.rs` の mesh pipeline で indexed draw を記録する。`FrameSnapshot` は owned `CameraSnapshot` を持ち、mesh shader は app-side camera の view-projection で world-space GLB を描画する。
- window path は old-style free camera controls を持つ。left click で cursor capture、Escape で release、WASD/arrow、Space/E、Shift/Q、Ctrl、mouse wheel で移動する。
- shadow resource は device-owned fixed cascade で、swapchain resize では作り直さない。near cascade は高解像度、mid/far cascade は段階的に軽くし、scene size ではなく camera frustum から shadow projection を作る。shadow map 自体は固定サイズに保ち、PCSS の粒状感は screen-space の深度/法線つき高解像度 blur でならす。必要な visual inspection 時だけ `REBUILD1_SHADOW_MAP_SIZE` で固定リソースサイズを変える。
- shadow sampling は通常 1 cascade だけを評価し、cascade 境界の blend 範囲だけ 2 cascade を読む。PCF tap は near/mid/far で 16/10/6 に分け、far shadow pass は low LOD geometry を使う。
- mesh LOD は三角形を単純に間引かない。meshoptimizer の border-locked simplification と vertex-cache reorder で index LOD を作り、同一 LOD stream は upload しない。
- translucent shadow は cascade ごとの独立 pass で opaque depth を shader sample し、transparent caster だけを transmittance color target へ multiplicative blend で記録する。複数透明 caster は色付き透過として合成し、scene shader が opaque shadow と transmittance をまとめて読む。
- swapchain dependent resource は scene color/depth、normal/material target、post framebuffer を持つ。post pipeline は scene color/depth と normal/material を読み、実投影係数で復元する material reflectance aware SSR、SSAO、高品質 FXAA、camera effects、tone mapping を fullscreen triangle で書く。
- `vulkan/material.rs` は imported texture payload を sampled image へ upload し、material parameter buffer と descriptor set を作る。暗黙 fallback texture は作らない。
- Stage 8 で GLB base-color texture、vertex normal、material texture sampling、alpha cutout shadow、scene shadow sampling、post camera effects を通した。
- Stage 9 shadow slice で fixed cascade shadow、translucent shadow pass、cascade descriptor indexing、sampled opaque-depth rejection、transparent shadow smoke scene を通した。
- camera effects は `FrameSnapshot::camera_effects` として app/user 側で抽出し、renderer は露出/ホワイトバランスの owned 値だけを post pass に適用する。contrast/saturation の renderer-wide 補正は quality command 側で扱う。
- lighting quality は low ambient、directional wrap、indirect strength を分ける。完全に光がない場所は暗く残しつつ、画面外や背面側の key light で normal terminator が急に黒く落ちる問題を抑える。
- mesh pipeline は scene / opaque shadow / translucent shadow ごとに untextured/textured variant を持ち、texture descriptor がある material だけ texture sampling shader を使う。
- `REBUILD1_WINDOW_ASSET=assets/stage8_textured_cutout.r1scene` で fixed texture/cutout verification scene を window smoke に流せる。
- `REBUILD1_WINDOW_ASSET=assets/stage9_translucent_shadow.r1scene` で transparent shadow verification scene を window smoke に流せる。
- `--window-smoke` は `assets/model.glb` がある場合に asset load 後の mesh frame まで待って自動終了する。
- asset load 失敗時に renderer が cube や placeholder を作る経路はない。
- まだ独立した user task、reflection JSON を使った descriptor codegen、screenshot golden image は未実装。

## 最終方針

- app / user code / renderer は直接結合しない。
- renderer は `RendererCommand` を受け取り、`RendererEvent` を返す。
- Vulkan object は renderer thread だけが所有する。
- ECS world から renderer へ渡すのは render extraction 済みの `FrameSnapshot` だけ。
- renderer core は暗黙 fallback をしない。
- asset import、GPU upload、render target、render graph、pipeline を分ける。
- pass dependency と image layout transition は render graph に集める。
- shader binding は名前付き interface に集約する。
- safety は boundary validation と constrained type で作る。
- すべての関数に具体的な説明を書く。
- trace log を主に使い、info は重要な lifecycle event に絞る。

## 目標構成

```text
app/
  window         # winit event loop and window owner

protocol/
  messages/      # command, event, request id, handle
  material       # material slots, alpha mode, texture descriptors
  transport/     # async bounded channel, record/replay bridge

user/
  app trait      # async user code entry point
  ecs            # 将来の ECS world owner
  extract        # ECS/user state -> FrameSnapshot

renderer/
  mod.rs         # renderer thread, backend trait, null backend
  assets/        # GPU asset handles, mesh/material/texture records, deferred destroy queue
  surface.rs     # null/backend-neutral surface registry
  vulkan.rs      # Vulkan backend orchestration
  pass_schedule.rs # frame pass cadence outside backend recording
  vulkan/
    buffer.rs    # host-visible typed buffer upload helper
    debug.rs     # validation layer and debug messenger
    swapchain.rs # swapchain image views, fixed shadows, scene/post render passes, framebuffers
    frame.rs     # frames in flight, command pools, sync, frame execution
    material.rs  # sampled images, sampler, material parameter buffers, descriptor sets
    mesh.rs      # backend-local mesh vertex/index buffers and mesh pipeline
    post.rs      # scene_color sampling and swapchain post pipeline
  graph/         # pass declarations, resources, barriers
  targets/       # depth, shadow, scene color
  pipeline/      # shader interface, shader modules, layouts, pipeline cache
  scene/         # renderable scene data

import.rs        # .r1scene skeleton and GLB geometry importer
```
