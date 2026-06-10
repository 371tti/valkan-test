# Testing

## 前作の問題

前作では OBJ/GLTF の読み込み値を確認する test が増えましたが、多くはレンダリング品質を保証していませんでした。field が埋まっていることと、画面上で正しく見えることは別です。

`rebuild1` では unit test と rendering test の役割を分けます。

## Unit test するもの

純粋ロジックに限定します。

- math
- camera metering
- pass schedule
- render graph の依存解決
- resource state transition plan
- message protocol の envelope / request id / frame id
- command log replay
- fake transport
- async channel backpressure
- shutdown sequence
- render extraction が ECS entity/component reference を snapshot に漏らさないこと
- constrained type の constructor validation
- handle generation
- path resolution
- import warning classification

これらは GPU を使わずに正誤を判定できます。

## Unit test しないもの

次のような test は増やしません。

- importer が作った material field を細かく眺めるだけの test
- shader と descriptor を通さない texture slot test
- 実際の draw をしない shadow の見た目 test
- fixture の期待値を大量に固定するだけの test
- renderer 内部状態に依存しすぎる protocol test

もちろん parser の仕様として必要な小さい test は残せます。ただし「レンダリングできること」の代用にしません。

## Protocol test

message protocol は unit test します。renderer を実際に動かさなくても、次は判定できます。

- `RendererCommand` が envelope を持つ
- `request_id` と response event が対応する
- `FrameSnapshot` に参照や Vulkan handle が入らない
- `FrameSnapshot` に ECS Entity や component reference が入らない
- command log を replay runner に食わせられる
- fake transport で `UserApp` を動かせる

これは描画品質の test ではなく、app/user code/renderer の境界が壊れていないことの test です。

## Code quality check

読みやすさの rule は review で確認します。

- すべての関数に具体的な説明がある
- guard が boundary に寄っている
- internal function が validated/constrained type を受け取る
- 同じ check が複数 module に散らばっていない
- `unsafe` の safety comment が owner と precondition を説明している

これは unit test で全部を保証するものではありません。lint、review、短い module 構成で守ります。

## Rendering test

見た目は rendering test で見ます。

候補:

- headless/offscreen render
- fixed scene の screenshot 比較
- depth/shadow map の small image dump
- GPU validation layer を CI または手元チェックで有効化
- RenderDoc capture を定期的に取る

最初は自動 screenshot 比較まで作り込まず、固定 scene と手動 capture の手順を整えるだけでもよいです。

## Test scenes

最低限の確認 scene は設計段階で決めておきます。

| Scene | Purpose |
| --- | --- |
| triangle | swapchain, pipeline, clear, present |
| textured plane | texture upload, sampler, UV |
| alpha cutout card | alpha mode, shadow cutout |
| normal mapped sphere | tangent/normal/material |
| shadow receivers | cascade shadow map and bias |
| translucent blockers | transparent shadow transmittance |
| post camera effects | tone mapping, exposure, white balance, dark preservation |

## Current manual checks

Stage 8 の手元確認は次を使います。

```powershell
cargo run -- --headless
$env:RUST_LOG='rebuild1=info,winit=info'; cargo run -- --window-smoke
$env:REBUILD1_WINDOW_ASSET='assets/stage8_textured_cutout.r1scene'; $env:RUST_LOG='rebuild1=info,winit=info'; cargo run -- --window-smoke
$env:REBUILD1_WINDOW_ASSET='assets/stage9_translucent_shadow.r1scene'; $env:RUST_LOG='rebuild1=info,winit=info'; cargo run -- --window-smoke
```

確認対象:

- default `assets/model.glb`: GLB geometry, base-color texture import, normals, material draw path
- `stage8_textured_cutout.r1scene`: explicit checker texture, alpha cutout, shadow cutout, post camera effects
- `stage9_translucent_shadow.r1scene`: transparent materials, translucent shadow pass, transmittance sampling
- validation callback: Vulkan error / warning が出ていないこと

## Acceptance rule

「test があるから正しい」ではなく、次のどれで保証するかを明示します。

- unit test
- graph validation
- validation layer
- screenshot comparison
- manual RenderDoc capture
- manual visual check

保証方法が言えない test は書きません。
