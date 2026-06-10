# Code quality

## 目的

このプロジェクトの最適化対象は、まず「読む時間」です。

速いコードを書く前に、短く、制約が見え、安全性の理由が局所的に分かるコードにします。Vulkan や async や ECS はただでさえ認知負荷が高いので、コード側で余計な迷路を作らない方針です。

## 基本方針

- コードは短いほどよい。
- 短さは行数だけでなく、読者が保持する状態の少なさで判断する。
- 安全性は大量の defensive guard ではなく、型、関数制約、module 境界で作る。
- すべての関数に、具体的に何をするものか説明を付ける。
- 関数名、引数型、戻り値、説明が同じことを言っている状態にする。
- 不要な抽象化、不要な wrapper、不要な test fixture を増やさない。

直近の実装では `code_generation_policy.md` を併読します。特に `vulkan.rs` に新しい resource family を足し続けないこと、protocol payload に renderer 内部の handle を漏らさないこと、保証できない test を増やさないことを短期の gate にします。

## Function contract

すべての関数は、直前に「何をするか」を書きます。

public API と trait method は `///` を使います。private function は `///` か直前コメントを使います。どちらでもよいですが、説明は必須です。

説明は抽象的な言い換えではなく、具体的な操作を書きます。

悪い例:

```text
// Handles frame.
```

良い例:

```text
// Drains pending renderer commands and records one frame into the acquired swapchain image.
```

必要な場合は次の項目を短く書きます。

```text
Does: この関数が実際に行うこと。
Requires: 呼び出し前に満たすべき条件。
Ensures: 呼び出し後に保証する状態。
Effects: 変更する外部状態、GPU resource、channel。
Errors: どんな失敗を返すか。
```

毎回全部を書く必要はありません。重要なのは、読者が関数本文を読む前に「どの制約の中で読むべきか」が分かることです。

## Example

```rust
/// Records draw commands for one already-extracted frame snapshot.
///
/// Requires: `snapshot` contains only live renderer handles resolved by `SceneCache`.
/// Effects: writes the current frame command buffer; does not submit the queue.
fn record_frame(snapshot: &FrameSnapshot, frame: &mut ActiveFrame) -> RendererResult {
    // ...
}
```

```rust
// Converts a validated ECS/user snapshot into renderer-local draw packets.
// Requires: all asset handles were validated by AssetStore before this call.
fn build_render_items(snapshot: FrameSnapshot, assets: &AssetStore) -> Vec<RenderItemPacket> {
    // ...
}
```

## Guard policy

guard は境界に置きます。

guard を置く場所:

- message decode / receive
- user input
- file IO
- asset import
- Vulkan / OS API result
- raw handle を typed handle に変換する場所
- public API

guard を重ねない場所:

- validated type を受け取る private function
- renderer thread 内で owner が明確な resource wrapper
- graph compile 後の pass record
- `FrameSnapshot` validation 後の draw packet 生成

内部関数が validated type を受け取るなら、同じ check を何度も書きません。必要なら `debug_assert!` で前提を確認し、release path の分岐を増やさないようにします。

## Constrained types

安全性は値の形で表します。

例:

```text
NonZeroExtent
LiveMeshHandle
ResolvedMaterial
CompiledGraph
AcquiredSwapchainImage
RecordingFrame
```

外部入力はまず validation して constrained type に変換します。内部関数は `u32` や `usize` の生値ではなく、意味のある型を受け取ります。

悪い流れ:

```text
fn record(width: u32, height: u32, mesh_index: usize)
  -> 毎回 width > 0, height > 0, mesh exists を確認する
```

良い流れ:

```text
fn record(extent: NonZeroExtent, mesh: LiveMeshHandle)
  -> record は描画だけを読む
```

## Function size

目安:

- 1 function は 1 action。
- 20 行以内ならかなり良い。
- 40 行を超えたら分割を検討する。
- 60 行を超えたら、Vulkan setup など明確な理由がない限り設計を疑う。

ただし、短くするためだけに処理を読みにくい順序へ散らしません。分割する単位は「手順」ではなく「責務」です。

良い分割:

- validate message
- create swapchain images
- compile graph
- build draw packets
- record pass

悪い分割:

- `do_part_1`
- `continue_setup`
- `handle_stuff`
- 引数を 10 個受け取る小関数の連鎖

## Module size

目安:

- 1 file は 300 行以内を目指す。
- 500 行を超えたら module split を検討する。
- split の理由は機能境界か lifetime 境界にする。

小さすぎる file を大量に作って navigation を難しくしません。読む順番が自然に分かることを優先します。

## Error handling

error は隠しません。

- boundary では `Result` を返す。
- renderer 内部 policy で勝手に fallback しない。
- recover する場合も event/log に出す。
- `unwrap` は test、prototype、明確に到達不能な invariant に限定する。
- 到達不能な invariant は `expect("具体的な不変条件")` にする。

エラー型は巨大な共通 enum にしすぎません。module boundary ごとに意味のある error を持ち、protocol へ出すときだけ `RendererEvent` に変換します。

## Logging policy

ログは後付けの printf debug ではなく、thread / async / protocol 境界を読むための設計要素として扱います。

- `trace` を主に使う。
- `trace` は command/event の送受信、resource state の変化、skip した分岐、validation 済み値の流れに置く。
- `info` は起動、停止、window 作成、renderer ready、手動で読む価値がある lifecycle event に限定する。
- 高頻度 polling や毎 frame の細部に無制限で log を置かない。
- log field は文章に埋め込まず、`window_id`, `frame_id`, `width`, `height`, `platform` のような structured field にする。
- `println!` は runnable sample の一時出力に使わず、通常の状態報告は tracing に寄せる。
- recover した失敗や fallback は event と log の両方で見えるようにする。

## Unsafe policy

`unsafe` は薄い wrapper に閉じ込めます。

crate 全体で `unsafe` を禁止しません。代わりに `unsafe_op_in_unsafe_fn` を deny し、Vulkan / OS handle など避けられない箇所だけを小さい module に隔離します。

`unsafe` block には必ず次を書きます。

```text
Safety: なぜこの unsafe が成立するか。
Requires: 呼び出し側が守る条件。
Owner: resource を誰が破棄するか。
```

`unsafe` が増えたら、まず wrapper の境界が間違っていないかを疑います。

## Naming

名前は具体的な動詞にします。

良い:

- `validate_surface_descriptor`
- `compile_graph`
- `build_draw_packets`
- `record_shadow_pass`
- `upload_texture`
- `resolve_material_handles`

避ける:

- `handle`
- `process`
- `update_all`
- `do_render`
- `prepare_stuff`

名前が長くなりすぎる場合、関数が複数の責務を持っている可能性を疑います。

## Review checklist

実装時は次を満たさない code を通しません。

- すべての関数に具体的な説明がある。
- 関数の `Requires` が型で表せるなら型にしている。
- boundary 以外に同じ guard が重複していない。
- `FrameSnapshot` や protocol payload に参照、raw pointer、Vulkan handle、ECS entity が入っていない。
- function が 1 action になっている。
- module split が責務境界に沿っている。
- `unsafe` の理由と owner が読める。
- fallback が暗黙に入っていない。
- protocol 境界、thread 境界、resource lifecycle に trace log がある。
- info log が lifecycle event に絞られている。
- コメントが「何をしているか」ではなく「なぜその制約なのか」を説明している。
