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

自動 smoke は画面の美しさを判定しません。既存 asset のロード、指定数の loaded mesh frame の present、Vulkan validation layer の有効化、validation/skipped-frame の診断、終了 timeout だけを契約として判定します。asset がない smoke は成功扱いにしません。

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

影の目視確認では、通常シーンで次を順に確認します。

- キー `1`〜`4` を切り替え、ログの `updated window renderer quality preset` と `updated Vulkan renderer quality feature switches` に連続値と機能ごとのON/OFFが出ること。キー1=全OFF＋performance、2=AA+SSAO＋interactive、3=SSR/Bloom/GodRayを追加＋balanced、4=Volumetric Fogも追加＋high。キー操作でPCSSの解像度、tap数、光源角半径、biasが各プロファイルへ変わり、必要な単独のON/OFF変更は `SetQualityFeatures` 側で別に確認する。
- 各splitの前後をカメラで横切り、境界でshadowが段差・点滅せず隣接cascadeへ滑らかにblendすること。
- 平面や斜面を太陽方向へ向け、PCSS tap数を高品質へ上げても規則的な横縞が増えず、receiver-plane補正後にacneが細い帯へ増幅されないこと。
- 影の自己接触部でD16量子化余裕を含むslope bias/normal offsetが効き、acneが消えつつ輪郭が大きく浮かないこと。キー`3`/`4`ではreceiver-planeの1 texel勾配も加わるため、横縞の低減と過剰なpeter-panningの両方を確認する。
- feature preset 4（または `high_quality()` の連続値設定）ではgod rayがCSMで遮蔽され、遮蔽された建物や柱の後ろで光束が切れること。feature preset 3では従来の画面空間近似、4では専用 half-resolution camera-ray volume routeとなり、放射ブラーの二重掛けや画面中心固定の光線が出ないこと。旧3Dボリューム経路は存在しない。
- SSAOの目視確認ではキー`4`で、平面が均一に汚れず、接触部だけが近距離で暗くなることを確認する。細いcutout/枝を横切っても一画素の深度段差が半径全体の黒いhaloにならず、画面端や背景境界でclamp由来の帯が出ないこと。`SsaoQualitySettings`のsample budgetは2/3/4 slices × 2/3/4 steps（8/18/32 depth taps）へ変わる。
- highのFogは太陽が画面外でも薄く残り、異方性の筋だけが光源方向へ変化すること。遮蔽物の手前ではFogが連続し、表面を貫通して明るくならないこと。
- 高品質god rayのカメラ移動時に、画面端・背景からジオメトリへ入る箇所で点滅や急な露出ジャンプがないこと（ユーザー確認済みのため CSM カスケード境界は今回の層状現象の原因候補から除外）。専用 GodRay volume は前フレームの共有 temporal state で履歴を再投影し、フレーム番号だけでなく光源状態・移動量を含む位相で 48 個別 strata を変えるため、光源とカメラが直交に近い角度でも shaft が層状に分離せず、画面端から入るときに過去フレームの矩形が残らないこと。
- 高品質god rayの平坦な霧で、固定深度サンプル由来の層状バンドが視認できず、カメラを静止したまま数フレーム待っても粒状感や履歴の残像が増えないこと。production default の TAA は dormant なので、TAA による平滑化を合否条件にしない。
- PCSS は受け手を横切る影の可視率だけが時間方向に滑らかになり、ハイライト・直接光・transmittance は現在フレームのまま追従すること。カメラ/光源/品質変更直後は PCSS と GodRay の ping-pong history が無効化され、旧位置の影や Sun Shaft が一瞬混ざらないこと。
- F12 の `pcss_visibility_raw` は時間履歴前の opaque PCSS 可視率、`pcss_history` は前フレームの PCSS 履歴Rを表示する。遠距離/近距離を同じカスケード内で比較し、生値も履歴もノイズ化するなら空間PCSS/D16/bias、生値は安定して履歴だけがノイズ化するなら履歴書き込み/再投影、履歴は安定して通常表示だけがノイズ化するなら現フレームへの反映を調べる。デバッグ表示は履歴attachmentをLOAD保持してalpha-zeroで書き換えない。静止時の履歴clampは可視率全域を許可し、rawよりhistoryが滑らかになることを確認する。
- 葉や細い枝のような複雑なcutoutシルエットを横切っても、葉の隙間だけFogが消えたり、輪郭の外側へ薄いFogの膜が漏れたりしないこと。特に前景・背景を交互に含む細かい葉群で、数フレーム後もエッジが安定していること。
- local light を含む snapshot（point/sphere/spot/rectangle）では、光源の範囲内だけ volume の散乱が増え、spot/rectangle の向きが反転せず、`casts_shadow` の cubemap 遮蔽が局所的に効くこと。translucent shadow の葉やガラス越しでは太陽散乱が deep RGB transmittance に従って減衰し、opaque CSM の影と重ねても矩形の cascade 層が出ないこと。

手動の画面確認は次を使います（これは unit test の代替ではなく visual acceptance です）。

```powershell
cargo run -- --headless
cargo run -- --window-smoke
$env:REBUILD1_WINDOW_ASSET='assets/stage8_textured_cutout.r1scene'; cargo run -- --window-smoke
$env:REBUILD1_WINDOW_ASSET='assets/stage9_translucent_shadow.r1scene'; cargo run -- --window-smoke
```

まとめて確認する場合（format check、workspace check、gr-render unit test、Vulkan smoke）:

```powershell
.\scripts\verify-renderer.ps1
```

個別実行やsmokeフレーム数の変更もできます。

```powershell
.\scripts\verify-renderer.ps1 -Mode check
.\scripts\verify-renderer.ps1 -Mode smoke -SmokeFrames 12
.\scripts\verify-renderer.ps1 -Mode smoke -SmokeFrames 6 -Asset assets/stage9_translucent_shadow.r1scene
.\scripts\verify-renderer.ps1 -Mode smoke -Quality high -SmokeFrames 1
.\scripts\verify-renderer.ps1 -Mode smoke -Quality balanced -QualitySequence '1,4,1,4' -SmokeFrames 5
```

確認対象:

- default `assets/model.glb`: GLB geometry, base-color texture import, normals, material draw path
- `stage8_textured_cutout.r1scene`: explicit checker texture, alpha cutout, shadow cutout, post camera effects
- `stage9_translucent_shadow.r1scene`: transparent materials, translucent shadow pass, transmittance sampling
- `-Quality high`: 専用 GodRay volume を初回 frame で実行
- `-QualitySequence 1,4,1,4`: asset load 後に機能ON/OFFとGodRay routeを反復して検証
- validation callback: Vulkan error / warning が出ていないこと

SMAA の入力順は graph で `post (全 post composition) -> smaa_edges (PostColor -> edgesTex) -> smaa_weights (edgesTex -> blendTex) -> smaa (PostColor + blendTex -> swapchain) -> framebuffer_readback -> present` に固定する。SMAA の color neighborhood は composition 済みの `PostColor` だけを読む。3つのpassで公式 1x の local-contrast edge、SearchTex/AreaTex の pattern coverage、directional neighborhood resolve を順に評価する。AA の輪郭、斜線の方向、細線の幅、bloom/GodRay の境界が正しいかは screenshot または RenderDoc で目視確認し、unit test の合否にはしない。

## Acceptance rule

「test があるから正しい」ではなく、次のどれで保証するかを明示します。

- unit test
- graph validation
- validation layer
- screenshot comparison
- manual RenderDoc capture
- manual visual check

保証方法が言えない test は書きません。
