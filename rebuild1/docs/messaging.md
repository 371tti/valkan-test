# Messaging protocol

## 目的

app、user code、renderer は直接相手の内部構造を触りません。接続面は message protocol に限定します。

message protocol の主目的はマルチスレッド化です。message で分けられるなら、最初から task/thread も分けます。

目標は network protocol みたいな分かりやすさです。すべてのやり取りは「誰が」「いつ」「何を要求したか」「結果は何か」を message として読める形にします。最初から async channel、renderer 専用 thread、record/replay、remote debug、tool 接続に耐える形にします。

## 境界

```text
main thread / winit
  -> AppEvent channel
  -> UserApp async task
  -> RendererCommand channel
  -> RendererTask on renderer thread
  -> RendererEvent channel
  -> UserApp async task / tools
```

`UserApp` は Vulkan object を持ちません。`RendererTask` は gameplay object を知りません。両者は handle と message だけで会話します。

将来 ECS を載せる場合も同じです。ECS world は `UserApp` 側にあり、renderer へは ECS から抽出した `FrameSnapshot` だけを送ります。

## Message categories

### Command

app/user code から renderer へ送る命令です。基本は fire-and-forget ですが、結果が必要な command は `request_id` を持ちます。

```text
RendererCommand
  ConfigureSurface
  ResizeSurface
  LoadAsset
  UnloadAsset
  CreateScene
  DestroyScene
  SetDebugOptions
  SetFramebufferReadback
  SetQualitySettings
  SetQualityFeatures
  SubmitFrame
  CaptureScreenshot
  Shutdown
```

### Event

renderer から外側へ返す通知です。

```text
RendererEvent
  RendererReady
  AssetLoaded
  AssetLoadFailed
  FramePresented
  ScreenshotReady
  ShaderReloaded
  ShaderReloadFailed
  ValidationWarning
  DeviceLost
  RendererStopped
```

### Snapshot

1 frame 分の描画要求です。細かい setter を何十回も呼ぶのではなく、その frame に必要な状態をまとめて渡します。

```text
FrameSnapshot
  frame_id
  scene_id
  views
  lights
  render_items
  camera_effects
  debug_draw
```

`FrameSnapshot` は renderer 側で即座に GPU command に変換できるように、user code の参照や借用を含みません。ECS `Entity`、component reference、ECS world pointer も含めません。

`camera_effects` は app/user/ECS 側で決めた露出、white balance、contrast、saturation の owned snapshot です。renderer は framebuffer 外側の world や camera controller を読みに行かず、post pass でこの値だけを使います。暗所保護の方針として、メータリングがほぼ黒のときは exposure を強く上げず、光がない領域を黒いまま残します。

連続値の renderer 設定（SSR/SSAO/AA の強度・半径・sample budget、PCSSのtap数・解像度・光源角半径・bias、lighting wrap、renderer-wide contrast など）は `SetQualitySettings(RenderQualitySettings)` で送ります。機能をON/OFFするだけの変更は `SetQualityFeatures(RenderFeatureToggles)` で送り、既存の連続値を変更しません。window の1〜4はこの2種類を組み合わせ、プロファイルの連続値と機能集合を同時に切り替えます。どちらも ECS world の snapshot ではなく renderer 状態として扱います。

`render_items` は renderer protocol 用の packet です。

```text
RenderItemPacket
  object_id optional ExternalObjectId
  transform
  mesh MeshHandle
  material MaterialHandle
  flags
  layer
```

`ExternalObjectId` は debug/picking/log 用の安定 ID です。ECS `Entity` そのものではありません。

## Envelope

message は payload だけでなく envelope を持ちます。

```text
MessageEnvelope
  protocol_version
  request_id
  frame_id
  payload
```

`request_id` は response と log を追うために使います。`frame_id` は replay と frame capture のために必ず持ちます。

## Handle policy

protocol をまたぐ参照は typed handle だけです。

```text
SceneHandle
MeshHandle
TextureHandle
MaterialHandle
PipelineHandle
```

raw pointer、Rust reference、Vulkan handle は protocol payload に入れません。renderer 内部の Vulkan handle は renderer の外へ漏らしません。

ECS の `Entity` も protocol payload に入れません。protocol をまたぐ identity は renderer handle か `ExternalObjectId` だけです。

## Surface descriptor

winit `Window` は main thread が所有します。renderer thread に window object を渡しません。

surface 作成に必要な raw display/window handle は `ConfigureSurface` command の payload として `SurfaceDescriptor` に詰めます。`SurfaceDescriptor` は window を所有せず、window が renderer より長生きすることは `AppHost` の責務として明記します。

```text
SurfaceDescriptor
  window_id
  size
  raw_display_handle
  raw_window_handle
```

platform によって surface 作成を main thread で行う必要が出た場合でも、Vulkan surface handle の所有者は renderer task に寄せます。その場合は `CreateSurface` だけを main thread helper に逃がし、swapchain/device/queue/present は renderer thread に残します。

## Simplicity rules

- command の種類を増やしすぎない。
- 1 field ずつ更新する chatty protocol にしない。
- per-frame 変更は `FrameSnapshot` に寄せる。
- ECS component 更新を細かい renderer command にしない。
- asset のように長く生きるものは `LoadAsset` / `UnloadAsset` に分ける。
- command はできるだけ idempotent にする。
- failure は event として返す。暗黙 fallback で隠さない。
- message は clone/debug/serialize しやすい形に寄せる。

## Transport

最初の transport は async bounded channel です。same-thread direct call は使いません。message 境界を設計した場所は、最初から thread/task 境界として扱います。

```text
RendererEndpoint
  async send(command)
  async recv_event()
  try_send_for_winit_callback(command)
  blocking_send_for_lifecycle(command)
```

transport variants:

- async renderer thread queue
- record/replay log
- remote tool bridge
- test fake endpoint

transport は protocol payload を理解しません。message を運ぶだけです。

## Async rules

- winit callback では `await` しない。
- winit callback は event を `try_send` するだけにする。
- user code の更新は async task で行う。
- renderer は専用 thread 上の async loop として動く。
- Vulkan object は renderer thread から出さない。
- GPU command recording、queue submit、present は renderer task 内で同期的に直列実行する。
- asset import、shader compile、screenshot write は async/blocking worker に逃がす。
- `SubmitFrame` は古い snapshot を溜めず、最新を優先する。
- `ResizeSurface` は state update として扱い、古い resize を溜めない。app 側で in-flight 1 件と pending 最新 1 件へ coalesce する。
- `Shutdown` は lifecycle command として扱い、bounded channel が満杯でも drop しない。

## User code trait

user code は trait で app host に差し込みます。

```text
trait UserApp {
    async fn init(&mut self, out: &mut CommandSink);
    async fn handle_event(&mut self, event: AppEvent, out: &mut CommandSink);
    async fn update(&mut self, dt: FrameTime, out: &mut CommandSink);
    async fn handle_renderer_event(&mut self, event: RendererEvent, out: &mut CommandSink);
}
```

この trait は renderer の trait ではありません。user code と app host の境界です。renderer に渡るのは `RendererCommand` だけです。

object-safe な trait object が必要なら `async_trait` か boxed future を使います。内部 data まで trait object にしない方針は維持します。

## Renderer backend trait

renderer backend は protocol を消費する実装境界です。実際に dedicated thread 上で動く concrete loop は `RendererTask` と呼びます。

```text
trait RendererBackend {
    async fn run(
        self,
        commands: CommandReceiver<RendererCommand>,
        events: EventSender<RendererEvent>,
    ) -> RendererResult;
}
```

実装は最初は Vulkan だけでよいです。将来、mock renderer、headless renderer、capture renderer を同じ protocol で差し替えられます。

## Record and replay

message protocol にすると、frame の再現がしやすくなります。

```text
record:
  AppInput + RendererCommand + RendererEvent

replay:
  RendererCommand log -> RendererBackend
```

AAA 指向で考えると、これは tooling の入口になります。render bug を frame log と asset bundle で再現できる設計に寄せます。
