# Render graph

## なぜ必要か

前作は shadow、scene、post の順序と image layout 遷移を手で管理していました。pass が増えるほど、resource がいつ writable で、いつ shader read なのかが見えにくくなります。

`rebuild1` では最初から小さな frame graph を置きます。大げさな汎用 engine ではなく、少なくとも pass の inputs/outputs/layout を一箇所に集めるための仕組みにします。

graph が入力として受け取るのは renderer 内部の `SceneCache` と `FrameSnapshot` から作った draw packet です。user code の object や ECS world を graph が直接読むことは禁止します。

## graph が管理するもの

- pass の実行順
- pass が読む resource
- pass が書く resource
- image layout 遷移
- render target の clear/load/store policy
- resize 時の graph 再構築
- pass plugin の宣言情報

## graph が最初は管理しないもの

- async compute
- transient memory aliasing
- render pass merge optimization
- GPU-driven rendering
- bindless resource residency
- visibility / culling の全体最適化

最初から賢くしすぎない方針です。手管理の散らばりを消すことを優先します。

## Resource model

Resource は graph 上の名前付き handle として扱います。

```text
SwapchainImage
SceneColor
SceneDepth
ShadowCascade0
ShadowCascade1
ShadowCascade2
TranslucentShadow0
TranslucentShadow1
TranslucentShadow2
FrameUniformBuffer
MaterialBuffer
TextureArray
```

`AssetStore` が持つ texture や mesh は graph resource ではありません。graph から見ると read-only external resource です。

protocol 上の `MeshHandle` や `TextureHandle` は graph の中では解決済み GPU resource として扱います。handle 解決は pass の record 中に散らさず、draw packet 作成時に寄せます。

## Pass model

Pass は次の情報を持ちます。

- name
- read resources
- write resources
- desired layout
- clear/load/store policy
- record callback

record callback は command buffer に実際の draw を記録します。ただし barrier や layout 遷移は callback の外に出します。

AAA 指向へ伸ばすため、pass は trait 境界にできます。

```text
RenderPass
  name()
  declare(builder)
  record(context)
```

最初は固定配列でよいです。runtime plugin 化は後でよいですが、`declare` と `record` の分離だけは最初から守ります。

## 現在の実行 pass 構成

今の Vulkan executor は fixed swapchain pass ではなく、compiled graph の barrier と pass 順で次を実行します。

| Order | Pass | Reads | Writes | Notes |
| --- | --- | --- | --- | --- |
| 1 | `shadow_cascade_0..2` | mesh packets | shadow cascade depth | cascade ごとに opaque/cutout caster を depth-only pipeline で描く。 |
| 2 | `translucent_shadow_0..2` | shadow cascade depth, mesh packets | transmittance color | opaque depth を shader sample し、transparent caster を multiplicative blend で描く。 |
| 3 | `scene` | render items, shadow cascades, transmittance maps | scene color, scene normal/roughness, scene depth | main camera。mesh indexed draw をここで描く。 |
| 4 | `post` | scene color, scene depth, scene normal/roughness | swapchain image | SSR、tone mapping、camera effects、gamma。 |
| 5 | `framebuffer_readback` | swapchain image | CPU readback buffer | request がある frame だけ final framebuffer summary を返す。 |
| 6 | `present` | swapchain image | external presentation | side effect pass として culling から守る。 |

transparent shadow は shader variant だけではなく、graph pass と resource として明示します。opaque depth と translucent transmittance を別 resource にすることで、layout transition と descriptor indexing を追える状態にします。

## Resize

resize では swapchain dependent resource だけを破棄して作り直します。

作り直すもの:

- swapchain images/views
- scene color/depth
- graph resource description
- swapchain format に依存する pipeline

作り直さないもの:

- device
- queue
- asset store
- imported scene
- CPU scene state
- shader source watcher
- fixed cascade shadow depth/transmittance targets

## Pass schedule

前作では pass update cadence が renderer に混ざりました。`rebuild1` では schedule を graph の外に置きます。

- graph: pass の依存関係と command 記録順
- schedule: その frame で pass を実行するか

shadow を light 変化時だけ更新する、readback を request frame だけ実行する、という判断は schedule の責務です。

## Stage 7.5 compiler gate

Stage 7.5 では fixed swapchain graph をやめ、pass/resource 宣言から plan を生成します。現在の executor はこの plan を実際に使い、cascade shadow、translucent shadow、scene、post、readback、present の image layout transition を graph barrier から記録します。

compiler が行うこと:

- resource name と pass name の validation
- read/write usage から dependency edge を自動生成
- topological sort と cycle detection
- resource の first/last use と lifetime を計算
- resource state transition から barrier plan を生成
- 最終出力に届かない pass を cull
- barrier merge / transient alias / render pass merge の候補を metadata として出す

現在の標準実行 graph:

```text
shadow_cascade_0
shadow_cascade_1
shadow_cascade_2
  writes: shadow_cascade_N

translucent_shadow_0
translucent_shadow_1
translucent_shadow_2
  reads: shadow_cascade_N as shader_read
  writes: translucent_shadow_N

scene
  reads: shadow_cascade_0..2, translucent_shadow_0..2
  writes: scene_color, scene_depth

post
  reads: scene_color
  writes: swapchain_image

present
  reads: swapchain_image
```

pass cadence は `pass_schedule.rs` に残します。graph compiler は「この frame で有効な pass 群がどう依存しているか」を解き、schedule は「その pass を今回更新するか」を決めます。

## Future TODO

現在の compiler はまだ小さい optimizer です。今後、async compute scheduling、transient resource aliasing の実 allocation、render pass merge の実 backend 反映、resource lifetime compression を行う graph executor に育てます。

ただし最初から汎用化しすぎません。まずは graph が pass、resource、barrier を一箇所に集め、Vulkan backend がそれを忠実に実行する状態を維持します。
