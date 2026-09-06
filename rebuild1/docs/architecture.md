# Architecture

## 設計の芯

前作の一番大きな失敗は、Vulkan の lifetime を追いやすくするために `Renderer` に全部集めたことです。最初は分かりやすいですが、機能が増えるほど「どの変更も全部を触る」状態になります。

`rebuild1` では、所有権、初期化、frame 更新、asset、pass を分けます。さらに app / user code / renderer の接続を message protocol に限定し、最初から async task / renderer thread で分割します。

分割の基準はファイル数ではなく、次の問いに答えられるかです。

- この Vulkan object は誰が作り、誰が破棄するか
- resize で作り直すものか、device lifetime で生きるものか
- frame ごとに変わるものか、asset として長く生きるものか
- user/app policy か、renderer core の仕様か
- protocol をまたぐ data か、renderer 内部の data か
- ECS world 内の data か、render extraction 済みの data か

## 接続の原則

renderer は app/user code から直接好きな順番で呼ばれる object にしません。renderer は async channel から `RendererCommand` を受け取り、`RendererEvent` を返す service として扱います。

```text
AppHost -> UserApp task -> RendererCommand channel -> RendererTask -> RendererEvent channel
```

この境界を network protocol のように考えます。message は version、request id、frame id、payload を持ち、raw pointer や Vulkan handle を外へ出しません。

詳しくは [messaging.md](messaging.md) と [async_runtime.md](async_runtime.md) に置きます。

## Thread / task 境界

```text
main thread:
  winit event loop
  Window owner
  AppEvent producer

user task:
  UserApp state / 将来の ECS World
  input/update
  render extraction
  RendererCommand producer
  RendererEvent consumer

renderer thread:
  Vulkan owner
  command drain
  asset upload
  graph execute
  present

io workers:
  asset import
  image decode
  shader compile
```

message で分けた境界は最初から task/thread 境界にします。Vulkan object は renderer thread に閉じ込め、user task には protocol handle だけを返します。

## モジュール境界

```text
app/
  window と event loop。renderer の中身を知らない。

user/
  UserApp trait 実装。将来は ECS world を所有する。
  render extraction で ECS/user state を FrameSnapshot へ変換する。

protocol/
  command/event/snapshot/handle。app と renderer の接続面。
  transport は async bounded channel。record/replay は同じ protocol にぶら下げる。

renderer/
  task/
    RendererCommand を消費し、RendererEvent を返す renderer thread loop。
  device/
    Vulkan instance, device, queue, debug utils。
  swapchain/
    surface, swapchain image, image view, render pass, framebuffer, resize。
  frame/
    command pool, command buffer, fence, semaphore, per-frame buffer。
  graph/
    pass 宣言、resource 宣言、barrier、実行順。
  targets/
    depth, scene color, PCSS ping-pong history, optional TAA history, fixed cascade shadow などの render target。
  assets/
    GPU mesh, GPU texture, material buffer, descriptor。
  import/
    glTF, OBJ, image file などを中間表現へ読む。
  pipeline/
    shader module, pipeline layout, graphics pipeline, reload。
  scene/
    renderer 内部の scene cache。user object は持たない。
```

## 所有権テーブル

| Component | Owns | Recreated on resize | Notes |
| --- | --- | --- | --- |
| `AppHost` | window, event loop, input collection | no | renderer 内部を知らない。 |
| `UserTask` | gameplay state, 将来の ECS world | no | async に `RendererCommand` を作る。 |
| `Protocol` | command/event/snapshot type | no | raw pointer と Vulkan handle を入れない。 |
| `RendererTask` | renderer subsystem orchestration | no | dedicated thread で command を順に処理する境界。 |
| `DeviceContext` | instance, device, queues, debug utils | no | 最長 lifetime。ほかのほぼ全ての親。 |
| `SwapchainContext` | surface, swapchain, images, image views, render pass, framebuffers, PCSS history, optional TAA resources | yes | window size と surface format に依存。破棄順は child framebuffer/target -> pipeline/render pass -> image view -> swapchain。履歴は swapchain extent と常に同期する。 |
| `FrameResources` | command buffers, fences, semaphores, upload scratch | maybe | frames-in-flight 単位。 |
| `RenderTargets` | depth, scene color, PCSS ping-pong history, optional TAA, fixed cascade shadow | partly | scene/post と history は resize で作り直す。shadow cascade は device lifetime 側に置く。 |
| `RenderGraph` | pass list, resource state plan | yes | resource が変わったら再構築。 |
| `AssetStore` | GPU mesh, texture, material buffers | no | resize では壊さない。 |
| `PipelineLibrary` | shader modules, pipeline layouts, pipelines | maybe | swapchain format や render target format に依存。 |
| `SceneCache` | render items, camera, lights copied from snapshot | no | renderer 内部表現。user object は持たない。 |

## 初期化順

1. `app` が window と `AppHost` を作る
2. `AppHost` が async runtime と channel を作る
3. `AppHost` が `UserApp` task を spawn する
4. `AppHost` が renderer thread を起動し、`RendererTask` を spawn する
5. `RendererTask` が Vulkan instance/device/queue を作る
6. `RendererTask` が surface/swapchain を作る
7. `RendererTask` が frames-in-flight resource を作る
8. `RendererTask` が target、pipeline、graph を作る
9. `UserApp` が `LoadAsset` や `CreateScene` command を送る
10. renderer は結果を `AssetLoaded` / `AssetLoadFailed` event で返す

## Frame flow

1. `app` が winit callback で `AppEvent` を user task に `try_send` する
2. `UserApp` task が入力と時間を処理し、将来は ECS schedule を回す
3. ECS/user state から render extraction で `FrameSnapshot` を作る
4. `UserApp` task が必要な `RendererCommand` を async channel に積む
5. per-frame の描画要求は `FrameSnapshot` として `SubmitFrame` で送る
6. `RendererTask` が command queue を drain する
7. `RendererTask` が swapchain image を acquire する
8. `RendererTask` が `FrameSnapshot` を renderer 内部の `SceneCache` に反映する
9. `graph` が shadow -> scene/PCSS history -> dedicated GodRay volume -> optional TAA -> post の pass 順に command buffer を記録する
10. queue submit
11. present
12. renderer が `FramePresented` や error event を返す

## Resize flow

winit は resize event を大量に出します。すべてを bounded channel に積むと、swapchain 再作成より速く command queue が埋まります。

現在の方針:

- app 側は `ResizeSurface` を in-flight 1 件だけ送る。
- in-flight 中に来た resize は pending 最新 1 件だけ保持する。
- `SurfaceResized` event を受けたら pending 最新を送る。
- channel full は resize では fatal にしない。最新値を保持して次の drain で再送する。
- `Shutdown` は落としてはいけない lifecycle command なので、同期境界から capacity を待って送る。

## Trait 方針

trait は境界にだけ使います。`UserApp`、`RendererBackend`、`RendererTransport`、`AssetImporter`、`RenderPass` のような差し替え点には使います。async trait は境界でだけ使い、math、handle store、material data、command payload のような data は plain struct / enum を優先します。

詳しくは [aaa_traits.md](aaa_traits.md) に置きます。

## 読みやすさの方針

実装の最適化対象は、まず読者の認知負荷です。

短い関数、小さな owner、明示された制約、重複しない guard を優先します。安全性は「どこでも念のため check する」ではなく、boundary で validate して constrained type に変換し、内部関数はその制約を前提に読む形にします。

すべての関数には具体的な説明を付けます。説明なしの関数は、短くても未完成と扱います。

詳しくは [code_quality.md](code_quality.md) に置きます。

## ECS 方針

ECS は renderer の上に載せます。renderer と protocol は ECS crate に依存しません。

`UserApp` は ECS world を所有できますが、renderer thread は ECS world を読まず、lock も取りません。ECS schedule の最後に render extraction を行い、owned な `FrameSnapshot` を renderer に送ります。

詳しくは [ecs_integration.md](ecs_integration.md) に置きます。

## Error policy

renderer core は暗黙の代替描画をしません。モデルが読めない、texture がない、shader が壊れている、といった状態は error または warning として表に出します。

placeholder が必要な場合は user code 側で明示します。例えば `UserApp` が「ロード失敗時は checker material の plane を出す」と決めるのは許可します。renderer core が勝手に cube を差し込むのは禁止です。

error は原則 `RendererEvent` として返します。panic か event かを曖昧にしません。

## Drop policy

`unsafe` な破棄順序は resource wrapper に閉じ込めます。大きな `Drop for Renderer` に破棄順を全部書かない方針です。

各 wrapper は「親 device より短く生きる」「swapchain より短く生きる」などの lifetime ルールをドキュメントで持ちます。Rust の lifetime だけで表せない Vulkan の親子関係は、型名と module 境界で補います。
