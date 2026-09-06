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
SceneNormalRoughness
SceneDirectionalShadowResponse
DirectionalShadowDepthArray
TranslucentShadow0
TranslucentShadow1
TranslucentShadow2
TranslucentShadow3
PcssShadowHistory0
PcssShadowHistory1
TaaHistory0
TaaHistory1
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
| 1 | `shadow_direction_0..3/cascade_0..3` | mesh packets | D16 array layer | 4 cascadeをstable light-space gridへ描き、各層は1回だけ更新する。 |
| 2 | `translucent_shadow_0..3` | CSM depth, mesh packets | deep log-transmittance/depth | 基準方向だけを使い、RGBを加算ログ透過率、AをMIN最前面深度として保存する。 |
| 3 | `scene` | D16 array, transmittance maps, previous `PcssShadowHistory` | scene HDR, normal/roughness, depth, current `PcssShadowHistory` | blocker探索はraw depth、PCSSの最終filterは比較サンプラのhardware 2x2 PCFで評価し、cascade境界をblendする。visibility と receiver view-depth だけを前フレームへ再投影して時間積分する。 |
| 4 | `taa` (optional) | scene HDR、opaque depth/normal、transparent metadata、previous TAA history | HDR TAA history | stable-grid reprojection、同一surfaceのhistory count、YCoCg variance clampで蓄積する。現在の production default は dormant で、post composition 後の専用 SMAA pass を使う。 |
| 5 | `bloom_*`, `god_ray_*` (optional) | resolved scene, CSM/translucent shadows, bloom/god-ray histories | post inputs | quality と光源に応じた bloom、専用 half-resolution camera-ray GodRay volume（48 strata）と履歴を作る。旧3Dボリューム経路は存在しない。 |
| 6 | `post` | resolved scene, metadata, bloom/god-ray histories | `PostColor` | stable output UVをjittered metadata UVへ明示変換し、SSAO、SSR、camera effects、tone mappingを完全な composition として書く。 |
| 7 | `smaa_edges` | `PostColor` | `edgesTex` (RG) | 公式 luma/local-contrast edge detection。 |
| 8 | `smaa_weights` | `edgesTex`, SearchTex, AreaTex | `blendTex` (RGBA) | quarter-pixel linear search と AreaTex pattern coverage。 |
| 9 | `smaa` | `PostColor`, `blendTex` | swapchain image | 公式 directional neighbourhood blending。 |
| 10 | `framebuffer_readback` | swapchain image | CPU readback buffer | requestがあるframeだけfinal framebuffer summaryを返す。 |
| 11 | `present` | swapchain image | external presentation | side effect passとしてcullingから守る。 |

Directional depthは一つのD16 arrayとして管理し、同じ方向を全cascadeで共有します。soft shadowは空間的な大半径filterではなく、太陽円盤上の方向jitterとhardware PCFで評価します。PCSS visibility と専用 GodRay はそれぞれ独立した2枚の ping-pong history を持ち、カメラ・light・surface・品質の discontinuity では共有 temporal state とともに破棄します。TAA は optional で、GodRay の temporal history は別の ping-pong resource として扱います。

## Resize

resize では swapchain dependent resource だけを破棄して作り直します。

作り直すもの:

- swapchain images/views
- scene color/depth
- dedicated GodRay half-resolution targets and histories, plus PCSS visibility/history targets, are recreated with the swapchain for descriptor compatibility
- graph resource description
- swapchain format に依存する pipeline

作り直さないもの:

- device
- queue
- asset store
- imported scene
- CPU scene state
- shader source watcher
- fixed directional D16 array and transmittance targets

## Pass schedule

前作では pass update cadence が renderer に混ざりました。`rebuild1` では schedule を graph の外に置きます。

- graph: pass の依存関係と command 記録順
- schedule: その frame で pass を実行するか

shadow をカメラ/光源/キャスター集合の変化時または resource invalidate 時だけ更新する、readback を request frame だけ実行する、という判断は schedule の責務です。CSM の深度配列は device-owned の永続資源として scene が継続利用します。

## Stage 7.5 compiler gate

Stage 7.5 では fixed swapchain graph をやめ、pass/resource 宣言から plan を生成します。現在のexecutorはこのplanを実際に使い、Stable CSM depth array、deep translucent shadow、scene、専用 GodRay volume、optional TAA、bloom、post composition、SMAA edge/weight/neighbourhood の3段、readback、presentのimage layout transitionをgraph barrierから記録します。

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
shadow_direction_0..3 / cascade_0..3
  writes: directional_shadow_depth_array[layer]

translucent_shadow_0
translucent_shadow_1
translucent_shadow_2
translucent_shadow_3
  reads: direction_0 depth layer as sampled D16
  writes: translucent_shadow_N

scene
  reads: stable_csm_depth_array, translucent_shadow_0..3, pcss_shadow_history[previous]
  writes: scene_color, scene_depth, scene_normal, pcss_shadow_history[current]

taa (optional; dormant in production default)
  reads: scene HDR, scene depth/normal/transparent metadata, previous TAA history
  writes: taa_history[current]

post
  reads: resolved scene color, scene metadata, bloom/god-ray histories
  writes: swapchain_image

present
  reads: swapchain_image
```

pass cadence は `pass_schedule.rs` に残します。graph compiler は「この frame で有効な pass 群がどう依存しているか」を解き、schedule は「その pass を今回更新するか」を決めます。

## Future TODO

現在の compiler はまだ小さい optimizer です。今後、async compute scheduling、transient resource aliasing の実 allocation、render pass merge の実 backend 反映、resource lifetime compression を行う graph executor に育てます。

ただし最初から汎用化しすぎません。まずは graph が pass、resource、barrier を一箇所に集め、Vulkan backend がそれを忠実に実行する状態を維持します。
