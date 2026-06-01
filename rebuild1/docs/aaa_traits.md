# AAA direction and traits

## 方針

ここで言う AAA 指向は、最初から巨大 engine を作る意味ではありません。後で機能を足しても壊れにくい境界を先に置く、という意味です。

優先するもの:

- data flow が追える
- frame capture / replay ができる
- renderer thread が最初から分かれている
- asset streaming に伸ばせる
- pass と backend を差し替えられる
- debug tool が接続できる

優先しないもの:

- 最初から bindless 前提
- 最初から ECS/editor 全部入り
- 最初から render graph optimizer
- trait object だらけの抽象化

## Data-oriented boundary

user code は object graph や ECS world を renderer に渡しません。renderer に渡すのは packed data です。

```text
ECS World / game objects
  -> render extraction
  -> visible render items
  -> FrameSnapshot
  -> renderer internal scene cache
  -> graph draw packets
```

`FrameSnapshot` は network packet のように、参照を持たない plain data に寄せます。これにより、別スレッド化、record/replay、tool 表示が簡単になります。

## AAA-ready extension points

| Area | First version | AAA-ready extension |
| --- | --- | --- |
| transport | async bounded channel + renderer thread | remote capture, tool bridge, replay |
| assets | explicit load/unload | streaming, residency, background import |
| graph | fixed pass list | pass plugin, async compute, transient aliasing |
| scene | frame snapshot | ECS render extraction, visibility system, LOD, draw packet cache |
| pipeline | named pipeline library | pipeline cache, shader permutation DB |
| testing | manual capture | automated screenshot, replay-based regression |

## Trait usage policy

trait は境界に使います。内部の細かい処理を全部 trait にすると、読みやすさより追いにくさが勝ちます。

使う場所:

- user app entry point
- renderer service backend
- transport
- asset importer
- render pass plugin
- render extraction
- shader compiler backend
- screenshot/capture sink

避ける場所:

- math
- handle store
- small resource wrapper
- command payload
- graph resource description
- material data

## Candidate traits

### `UserApp`

app host が user code を呼ぶための境界です。

```text
trait UserApp {
    async fn init(&mut self, out: &mut CommandSink);
    async fn handle_event(&mut self, event: AppEvent, out: &mut CommandSink);
    async fn update(&mut self, dt: FrameTime, out: &mut CommandSink);
    async fn handle_renderer_event(&mut self, event: RendererEvent, out: &mut CommandSink);
}
```

### `RendererBackend`

protocol を消費する renderer backend です。

```text
trait RendererBackend {
    async fn run(
        self,
        commands: CommandReceiver<RendererCommand>,
        events: EventSender<RendererEvent>,
    ) -> RendererResult;
}
```

### `RendererTransport`

message を運ぶ層です。payload の意味は知らないようにします。

```text
trait RendererTransport {
    async fn send(&self, command: RendererCommand) -> TransportResult;
    async fn recv(&self) -> Option<RendererEvent>;
    fn try_send(&self, command: RendererCommand) -> TransportResult;
}
```

### `AssetImporter`

ファイル形式ごとの差し替え境界です。

```text
trait AssetImporter {
    fn import(&self, request: ImportRequest) -> ImportResult;
}
```

### `RenderPass`

graph に登録する pass の境界です。

```text
trait RenderPass {
    fn name(&self) -> PassName;
    fn declare(&self, builder: &mut PassBuilder);
    fn record(&self, context: &mut PassContext);
}
```

`declare` と `record` を分けます。resource 依存と command 記録を混ぜないためです。

### `RenderExtract`

将来 ECS を載せるときの抽出境界です。

```text
trait RenderExtract {
    fn extract(&mut self, world: &mut EcsWorld, out: &mut FrameSnapshotBuilder);
}
```

extract 中に `await` しません。ECS world を borrow している間は同期処理だけにし、async IO や renderer command は外へ出します。

## Static vs dynamic dispatch

最初は static dispatch を優先します。`Box<dyn Trait>` は plugin 的に差し替える必要がある場所だけにします。

```text
static:
  math, graph builder, resource store, pipeline description, render extraction

dynamic:
  importer registry, renderer backend, transport, optional render pass plugin
```

## Threading plan

最初から renderer thread を分けます。message boundary は thread 化の準備ではなく、thread 間通信そのものです。

```text
main thread:
  window event
  AppEvent try_send

user task:
  user app update
  RendererCommand send

renderer thread:
  RendererCommand recv
  asset upload
  graph execute
  RendererEvent send

io workers:
  import/decode/compile
```

blocking upload が必要な場面でも、protocol では `AssetLoaded` event で完了を返します。user code が renderer の内部 lock を直接握る設計にはしません。

## Tooling

AAA 指向では tool が重要です。message protocol を基盤にすると、次を後から足せます。

- frame command log
- render graph viewer
- asset residency viewer
- shader reload error panel
- screenshot capture
- replay runner

tool は renderer 内部に侵入せず、command/event/snapshot を読む形を基本にします。
