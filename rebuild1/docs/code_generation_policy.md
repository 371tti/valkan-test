# Code generation policy

## 目的

この文書は、これから数ステージの実装で守る短期ポリシーです。

設計思想は他の docs にあります。ここでは、コードを書く直前に確認する判断基準だけを固定します。特に今は Vulkan backend が大きくなりやすい時期なので、「どこに足すか」「何を足さないか」を明確にします。

## 現在の短期目標

Stage 7.5 の graph compiler foundation と real target slice は完了済みです。現在は `.r1scene` / GLB geometry を renderer-owned mesh record に変換し、asset load 時に Vulkan vertex/index buffer と texture/material descriptor を backend-local resource へ upload し、compiled graph が shadow -> reflection -> scene -> post -> present を実行します。

順序:

1. mesh shader を base-color texture sampling へ進め、暗黙 white texture は作らない。
2. shader interface validation の前に binding の散在を増やさない。
3. texture / alpha cutout / shadow / reflection / camera effects の fixed visual scenes を用意する。
4. render graph は pass/resource/barrier の単一 owner として維持し、pass 内に layout transition を戻さない。
5. graph optimizer は metadata から実 allocation / merge へ進めるときだけ拡張する。

この段階では、ECS integration、async compute graph、複雑な importer registry を先に作り込まない。

## 実装前チェック

コードを書く前に次を決める。

- 今回の owner は何か。
- 今回の lifetime boundary は何か。
- 触る message は何か。
- 追加する file/module はどこか。
- 今回は触らない責務は何か。

答えが曖昧なら、先に docs か module 境界を直す。

## Module policy

`renderer/vulkan.rs`:

- Vulkan backend の orchestration と protocol command handling だけを置く。
- command pool、command buffer、sync object、frame execution を足さない。
- 新しい resource family を追加したくなったら、先に module を切る。

`renderer/vulkan/debug.rs`:

- validation layer、debug utils messenger、debug callback logging だけを置く。
- renderer policy や frame execution を混ぜない。

`renderer/vulkan/swapchain.rs`:

- surface support query、swapchain、image view、swapchain dependent render pass、framebuffer、swapchain dependent pipeline だけを置く。
- command buffer や per-frame sync を置かない。

`renderer/vulkan/frame.rs`:

- frames-in-flight、command pools、command buffers、semaphores、fences、acquired image state を置く。
- swapchain recreation policy は持たない。resize は `vulkan.rs` から `swapchain.rs` を呼ぶ。
- compiled graph の pass/barrier を記録する。graph declaration や graph policy は持たない。

`renderer/vulkan/buffer.rs`:

- host-visible typed buffer upload helper を置く。
- mesh/triangle の所有方針は持たない。

`renderer/vulkan/mesh.rs`:

- backend-local mesh vertex/index buffers と mesh pipeline を置く。
- import parser、ECS、app policy を知らない。

`renderer/vulkan/material.rs`:

- backend-local texture image、sampler、material parameter buffer、descriptor set を置く。
- importer や renderer asset store の ownership policy を持たない。
- 暗黙 fallback texture を作らない。descriptor にない texture slot は shader/pipeline variant 側で扱う。

`renderer/vulkan/triangle.rs`:

- real draw packet ができるまでの temporary debug triangle resource だけを置く。
- vertex buffer、frame uniform、descriptor set layout、pipeline layout、shader module/pipeline 作成 helper を置く。
- asset handle、material system、scene extraction を混ぜない。
- debug triangle は `DrawPacket::DebugTriangle` から呼ばれる temporary path として維持する。

`renderer/assets/store.rs`:

- protocol handle allocation、scene ownership、active/stale check だけを置く。
- material slot 解決や texture payload details を直接増やさない。

`renderer/assets/material.rs`:

- imported material から renderer material descriptor を作る。
- material slot と shader binding の log/追跡は `shader_interface` から行う。

`renderer/assets/texture.rs`:

- validated texture payload record を置く。
- Vulkan `VkImage` / sampler / descriptor set は Vulkan backend 側に置き、payload と object owner を混ぜない。

`renderer/assets/mesh.rs`:

- imported mesh から renderer-owned vertex/index geometry を作る。
- Vulkan `VkBuffer` / memory は Vulkan backend 側に置き、geometry payload と object owner を混ぜない。

`protocol/`:

- Vulkan handle、raw pointer、ECS entity、component reference を入れない。
- command/event は network packet のように、owned data と stable id だけで表す。
- 新しい command を足すときは、backpressure と coalescing policy も決める。

## Boundary rules

- app / user code / renderer は direct call でつながない。
- winit callback では `await` しない。
- resize は latest state へ coalesce する。
- `Shutdown` のような lifecycle command は drop しない。
- Vulkan object は renderer thread の外へ出さない。
- ECS world は user task の外へ出さない。
- `FrameSnapshot` は `Send + 'static` な owned data にする。
- renderer core は暗黙 fallback をしない。

## Code style rules

- 1 function は 1 action。
- すべての関数に、具体的に何をする関数かを書く。
- guard は boundary に置く。validated type を受ける内部関数で同じ check を繰り返さない。
- `unwrap` は test、prototype、明確な invariant 以外で使わない。
- `unsafe` は薄い wrapper に閉じ込め、`Safety`、`Requires`、`Owner` を書く。
- `trace` は command/event、resource lifecycle、coalescing、Vulkan state transition に置く。
- `info` は起動、停止、renderer ready、swapchain created など手動で読む価値がある lifecycle event に絞る。
- `println!` を通常の状態報告に使わない。

## Test policy

unit test は、GPU なしで正誤を判定できるものだけを書く。

書いてよい test:

- protocol id / envelope / transport
- resize coalescing
- shutdown sequence
- constrained type の constructor validation
- graph dependency や state transition plan
- handle generation

書かない test:

- 実際に描画しないのに見た目の正しさを主張する test
- importer の field を大量に固定するだけの test
- renderer 内部状態に密着しすぎる test
- fixture を守ること自体が目的になった test

描画品質は validation layer、manual visual check、RenderDoc、将来の screenshot comparison で確認する。

## Stop signs

次を見つけたら、実装を止めて設計を確認する。

- `vulkan.rs` に新しい Vulkan resource family を足そうとしている。
- protocol payload に Vulkan handle、reference、ECS entity を入れようとしている。
- fallback cube や暗黙 placeholder を復活させようとしている。
- 同じ validation guard が 2 か所以上に出ている。
- function が 60 行を超え、責務名で分割できる。
- `info` log が高頻度 path に増えている。
- test の保証方法を説明できない。

## 完了チェック

実装 slice の最後に確認する。

- `cargo fmt --check`
- `cargo check`
- 関連する `cargo test`
- headless path に影響した場合は `cargo run -- --headless`
- window path に影響した場合は、必要に応じて `cargo run -- --window` を手動確認する

window event loop を自動実行しない場合は、最終報告でそのことを書く。
