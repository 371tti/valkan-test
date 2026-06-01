# Agent implementation notes

この文書は、`rebuild1` を実装する agent にそのまま渡すための注意点です。

## 最重要方針

`rebuild1` は app / user code / renderer を async message protocol で分離する設計です。

renderer は dedicated thread 上で動作し、Vulkan object は renderer thread の外へ出してはいけません。重要なのは「Vulkan を別 thread に置く」ことだけではなく、renderer 境界の外へ Vulkan 都合を漏らさないことです。

renderer は ECS world や user state を直接読みません。user/ECS 側で render extraction を行い、renderer には owned な `FrameSnapshot` だけを送ります。

## 守るべき境界

- app は window / event loop / surface event を扱う。
- user code / ECS は game state を持つ。
- extraction は game state から `FrameSnapshot` を作る。
- protocol は `RendererCommand` / `RendererEvent` / handle / id / envelope を定義する。
- renderer は command を受け取り、GPU resource と frame execution だけを扱う。
- Vulkan backend は Vulkan object を所有する唯一の場所にする。

renderer 側から ECS world、asset importer、file format parser を読みに行きません。

## メッセージ粒度

protocol に Vulkan の低レベル命令を出しません。

禁止例:

```rust
BindPipeline
BindDescriptorSet
SetViewport
DrawIndexed
PipelineBarrier
```

許可する粒度:

```rust
ConfigureSurface
ResizeSurface
SubmitFrame(FrameSnapshot)
CreateMesh
DestroyMesh
CreateTexture
DestroyTexture
Shutdown
```

描画命令列を送るのではなく、renderer が解釈できる frame snapshot / draw packet を送ります。

## FrameSnapshot の制約

`FrameSnapshot` は完全に owned なデータにします。

避けるもの:

```rust
&World
EntityRef
&Mesh
Arc<MaterialWithEcsState>
renderer internal vk object
```

入れてよいもの:

```rust
FrameId
SurfaceId
SurfaceGeneration
CameraSnapshot
Vec<DrawPacket>
Vec<LightPacket>
RenderDebugSettings
MeshHandle
MaterialHandle
TextureHandle
Mat4
Bounds
```

`FrameSnapshot` は renderer thread に渡したあと、user/ECS 側の状態に依存してはいけません。

## SurfaceGeneration は必須

resize と submit frame の順序競合を安全に扱うため、surface には generation を持たせます。

```rust
SurfaceId
SurfaceGeneration
Extent2D
```

`FrameSnapshot` にも `surface_generation` を含めます。renderer 側で現在の surface generation と一致しない frame は描画せず drop します。

```rust
DropReason::StaleSurfaceGeneration
```

古い frame を無理に描画しません。現在の実装では `SurfaceId` / `SurfaceGeneration` が protocol と `FrameSnapshot` に入り、stale frame は `FrameDropped` で返します。

## Resize の扱い

winit resize は大量に発生するため、command queue を詰まらせてはいけません。

- in-flight resize は最大 1 件
- pending resize は最新 1 件だけ保持
- 古い resize は捨てる
- renderer 側で swapchain recreate する
- app thread で Vulkan 処理をしない

## SubmitFrame の扱い

`SubmitFrame` は同期 RPC にしません。app/user 側が描画完了を待つ構造にしません。

描画結果は `RendererEvent` で返します。

```rust
FramePresented
FrameDropped
RendererError
```

renderer が遅い場合に備えて、queued frame 数を制限します。無制限 queue にしません。

## Shutdown の扱い

runtime 内で `blocking_send` / `block_on` しません。特に Tokio runtime 内で blocking wait しません。

renderer shutdown は command と event で扱います。

```rust
RendererCommand::Shutdown
RendererEvent::RendererStopped
```

現在の event 名は `RendererStopped` です。`ShutdownComplete` に変える場合は protocol、docs、app 側の待機処理を同時に更新します。

必要な待機は renderer thread join 側に閉じ込めます。

## RenderGraph 方針

最初から巨大な render graph compiler を作りません。

最初の最小構成:

```text
PassDecl
ResourceDecl
Read/Write dependency
BarrierPlan
Execute callback
```

最初の実装では swapchain main pass だけで始めましたが、現在は scene/post/present graph に進んでいます。

```text
undefined/present -> color_attachment
color_attachment -> present
```

現在は compiled graph が scene color、scene depth、swapchain image の resource state と barrier を管理し、Vulkan executor は scene pass -> post pass -> present side effect の順で記録します。shadow/reflection は fake graph で先に置かず、実 target と executor を作るタイミングで追加します。

pass 順序と image layout transition を手作業で各 pass に分散させません。layout transition / barrier は render graph 側に集めます。

## Vulkan sync 注意点

present 待ちの semaphore は swapchain image ごとに持ちます。frame slot ごとに present semaphore を単純再利用しません。

frame resource と swapchain image resource の寿命を混同しません。

GPU resource destroy は即時破棄しません。asset resource を追加する前に deferred destroy を入れます。

```text
DestroyMesh
-> renderer 側で handle を無効化
-> GPU fence 完了後に vkDestroyBuffer
```

## Asset 方針

`import/` と `assets/` を混ぜません。

```text
import/
  file format -> CPU intermediate data

assets/
  CPU intermediate data -> GPU resource
```

renderer thread に glTF / OBJ / image decoder などの importer を抱え込ませません。

`assets/gpu.rs` のような巨大ファイルを作りません。最初から分割します。

```text
renderer/assets/
  mesh.rs
  texture.rs
  material.rs
  store.rs
  garbage.rs
```

## Fallback 方針

暗黙 fallback をしません。

禁止:

- 読み込み失敗時に勝手に cube を出す
- texture 不足時に無言で white texture にする
- shader interface 不一致を握りつぶす

必要なら明示的な debug/fallback asset として登録します。失敗は `RendererError` か validation error として返します。

## Shader / Pipeline 方針

Rust 側 descriptor binding と shader layout を暗黙同期しません。名前付き shader interface に集約します。

目標:

```text
ShaderInterface
  set
  binding
  name
  type
  stage
```

pipeline 作成時に Rust 側 layout と shader 側 interface を検証します。binding 番号の手書き同期を増やしません。

## Code quality

読みやすさを最優先します。

- 短い関数
- 明示的な責務
- boundary validation
- constrained type
- 巨大な `Renderer` を作らない
- 何でも持つ manager を作らない
- すべての関数に具体的な説明を書く

doc comment は単なる言い換えにしません。

悪い例:

```rust
/// Submits a frame.
fn submit_frame(...) {}
```

良い例:

```rust
/// Records and submits one frame for a configured surface.
///
/// Stale surface generations are dropped before acquiring a swapchain image.
/// This function owns acquire -> record -> submit -> present for one frame.
fn submit_frame(...) {}
```

## Log 方針

trace を主に使います。info は lifecycle event に絞ります。

info にしてよい例:

- renderer thread started
- Vulkan backend initialized
- surface configured
- swapchain recreated
- renderer shutdown complete

frame ごとの詳細は trace にします。

## Test 方針

見た目を保証しない unit test を増やしすぎません。

unit test するもの:

- protocol validation
- id / generation / handle
- render graph dependency
- barrier plan
- snapshot validation
- resize coalesce

rendering correctness は screenshot / golden image / manual visual test 側に寄せます。

## 実装順の優先

現在は GLB geometry load、all mesh indexed draw、camera snapshot、old-style free camera、scene color/depth、post pass、compiled graph execution、window smoke まで固まっています。次は次の順で進めます。

1. imported texture の Vulkan image / sampler upload
2. material descriptor set と mesh shader の sampled texture path
3. shader interface validation
4. actual shadow/reflection targets and passes
5. graph executor optimization: async compute scheduling、resource aliasing、render pass merge

graph は fake pass を置かず、resource lifetime と Vulkan executor が同時に説明できる範囲で広げます。
