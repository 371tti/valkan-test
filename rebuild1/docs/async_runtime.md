# Async runtime

## 目的

message protocol の主目的はマルチスレッド化です。`rebuild1` では「後で thread 分割する」のではなく、最初から async channel で app/user/renderer を分けます。

async は renderer の CPU/GPU work を魔法のように速くするためではありません。責務境界、backpressure、asset loading、shader compile、tooling、record/replay を整理するための土台です。

## 初期 task topology

```text
main thread
  winit event loop
  Window owner
  AppEvent producer

user task
  UserApp state / 将来の ECS World
  input/update
  render extraction
  RendererCommand producer
  RendererEvent consumer

renderer thread
  RendererTask
  Vulkan owner
  swapchain / device / graph / asset upload
  RendererEvent producer

io worker tasks
  asset file read
  image decode
  shader compile
  import to intermediate data

tool tasks
  log
  capture
  replay
```

Vulkan object は renderer thread に閉じ込めます。`ash::Device` や raw Vulkan handle を user task、io task、tool task に渡しません。

ECS world は user task に閉じ込めます。renderer task と io worker は ECS world を借用しません。

## Runtime policy

最初は `tokio` を候補にします。理由は channel、task、file IO、blocking worker まわりが揃っていて、tooling へ伸ばしやすいからです。

ただし winit の event loop は同期 callback です。main thread では `await` しません。winit callback は `try_send` で `AppEvent` を user task に流し、重い処理は async task 側で行います。

```text
winit callback
  -> try_send(AppEvent)
  -> request_redraw if needed

user task
  -> await AppEvent
  -> update state / ECS schedule
  -> render extraction to FrameSnapshot
  -> await RendererCommand send

renderer task
  -> await RendererCommand
  -> drain pending commands
  -> render/present
  -> send RendererEvent
```

## Renderer task

renderer task は専用 OS thread 上の single-owner loop とします。Vulkan command recording、queue submit、present はこの task 内で同期的に実行します。

これは「async 関数の中で Vulkan を細かく await する」設計ではありません。async boundary は command/event の受け渡し、asset import、shader compile、screenshot 保存などの外側に置きます。

## Channels

channel は bounded を基本にします。

| Channel | Direction | Policy |
| --- | --- | --- |
| `app_events` | main thread -> user task | input は短時間で drain。溢れた resize/mouse move は coalesce。 |
| `renderer_commands` | user task -> renderer task | bounded。`SubmitFrame` は最新を優先。 |
| `renderer_events` | renderer task -> user/tool task | bounded。重大 error は落とさない。 |
| `asset_jobs` | renderer/user -> io workers | bounded。結果は renderer command/event に戻す。 |
| `tool_events` | renderer/user -> tools | drop 可能な telemetry と保持必須 event を分ける。 |

unbounded channel は最初から使わない方針です。frame が詰まったときに memory が膨らむ設計を避けます。

## Backpressure

`SubmitFrame` は queue に何個も溜めません。renderer が遅い場合は古い frame snapshot を捨て、最新 snapshot を残します。

asset load は request id で追跡します。user task は完了を `AssetLoaded` / `AssetLoadFailed` event で受けます。blocking upload が必要でも user task は renderer 内部 lock を握りません。

ECS を使う場合、asset 完了 event は ECS event/resource に変換して次の schedule で component に反映します。renderer が ECS component を直接書き換えることはありません。

## Shutdown

終了は message で行います。

1. main thread が close request を受ける
2. user task に shutdown event を送る
3. user task が `Shutdown` command を renderer に送る
4. renderer task が GPU idle / resource drop を行う
5. renderer task が `RendererStopped` event を返す
6. runtime を止める

drop 順序が async task の自然終了に埋もれないように、shutdown command を明示します。

## Testing

async boundary は fake channel で test します。

- user task が renderer command を出す
- renderer fake が event を返す
- `SubmitFrame` coalescing が効く
- shutdown sequence が完了する
- record/replay log が同じ event sequence を返す
