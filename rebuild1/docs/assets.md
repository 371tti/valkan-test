# Assets

## 前作の問題

前作では `assets::cpu` が glTF/OBJ/MTL/image import を持ち、`assets::gpu` が mesh/material/texture upload に加えて render target や descriptor まで抱えていました。結果として「GPU に関係するもの」が全部 asset module に流れ込みました。

`rebuild1` では importer、CPU 中間表現、GPU asset store、render target を分けます。app/user code から renderer への asset 操作は `LoadAsset` / `UnloadAsset` message に限定します。

## Importer

`import/` はファイル形式を読むだけです。renderer の descriptor や pipeline を知りません。

現在は、最小の `.r1scene` manifest と GLB import を扱います。`.r1scene` は `LoadAsset` の成功/失敗、intermediate scene、handle 発行、fallback 禁止、material/texture slot の流れを固定するための小さい形式です。GLB は実モデルのロード確認用に triangle primitive、vertex normal、base-color texture を CPU intermediate data へ変換します。ファイル読み取りは worker task で行い、renderer thread では Vulkan object と asset store 登録だけを扱います。

```text
rebuild1-scene
texture solid 255 255 255 255
texture checker 255 255 255 255 0 0 0 0
material cutout base_color=0 alpha_cutoff=0.5
mesh plane
```

material が texture index を参照するため、現時点の `.r1scene` では参照される `texture` を先に書きます。

入力:

- path
- import options

出力:

- `ImportedScene`
- `ImportedMesh`
- `ImportedMaterial`
- `ImportedTexture`
- warning list

import 失敗時に cube は作りません。失敗は失敗として返します。

importer は trait 境界にします。

```text
AssetImporter
  import(request) -> ImportResult
```

glTF、OBJ、procedural debug asset は importer registry に登録できます。ただし renderer core が暗黙で procedural cube を選ぶことは禁止です。

## Intermediate scene

importer の出力は renderer に都合のよい中間表現に寄せます。ただし GPU object ではありません。

```text
ImportedScene
  meshes
  materials
  textures
  nodes
```

ここではファイル形式の都合を吸収します。

- glTF の node hierarchy
- OBJ の material group
- texture path 解決
- tangent/normal の欠落
- alpha mode

## GPU AssetStore

`AssetStore` は GPU 上にある asset だけを扱います。

現在の Stage 6 実装では、実 GPU image upload の前段として `GpuAssetStore` が protocol handle を発行し、material descriptor と texture payload を store に登録します。`UnloadAsset` は handle を active set から外し、破棄対象を deferred destroy queue に積みます。

Stage 7 では mesh handle も active set ではなく geometry record になりました。`.r1scene` の `mesh plane` は 4 vertices / 6 indices の renderer-owned geometry に変換されます。GLB triangle primitive は indexed geometry として登録されます。`ImportedScene` は bounds を持ち、app/user 側 camera framing に使えます。Vulkan `VkBuffer` / memory、mesh frame camera uniform、mesh pipeline は `vulkan/mesh.rs` が backend-local に所有します。

Stage 8 では imported texture payload が Vulkan sampled image へ upload され、material descriptor set が mesh pipeline に bind されます。base-color texture を持つ material だけ texture sampling shader variant を使い、texture がない material は untextured shader variant を使います。暗黙の white texture は作りません。

持つもの:

- mesh vertex/index buffer
- material parameter buffer
- texture image/view/sampler
- descriptor set または descriptor table

持たないもの:

- shadow map
- reflection target
- scene color/depth
- swapchain image
- window size dependent target
- file path 探索 policy
- user code の asset object

## Handles

`MeshId` や `TextureId` は単なる `usize` ではなく、generation 付き handle を基本にします。

削除しない append-only store で始める場合も、設計として「unload しない」と明記します。将来 unload を入れるなら stale handle を検出できる形にします。

protocol をまたぐ handle は `MeshHandle` / `TextureHandle` のような名前にします。renderer 内部 store の index と同じ型をそのまま外へ出さないようにします。

## Material slots

material の texture slot は名前付きで扱います。

```text
base_color
normal
metallic_roughness
occlusion
emissive
```

slot 数や binding 番号を scattered constants にしません。shader interface document と同じ場所で管理します。

Stage 6 の実装では、slot 名を `MaterialTextureSlot`、binding 番号を `shader_interface` に集約しています。`ImportedMaterial` は texture index を持ち、`GpuAssetStore` 登録時に `TextureHandle` へ解決します。

## Texture payloads

texture は `TextureDescriptor` として protocol-safe な owned data にします。

現在扱う形式:

```text
rgba8_srgb
```

`TextureDescriptor::rgba8_srgb` は width/height と byte count を検証します。読み込み失敗や texture 不足で renderer が暗黙 white texture を作ることはありません。必要な debug texture は `.r1scene` か user code 側で明示します。

Vulkan の sampled `VkImage` / `VkImageView` / `VkSampler` / material descriptor set への upload は `vulkan/material.rs` が backend-local に所有します。

## Fallback policy

renderer core は fallback asset を持ちません。

許可する fallback:

- shader debug 用の明示的 material
- user code が明示的に選んだ placeholder scene
- `.r1scene` で明示的に書かれた debug texture

禁止する fallback:

- model import 失敗時に renderer が勝手に cube を出す
- material import 失敗時に原因を隠して適当な material にする
- shader reload 失敗時に状態を黙って巻き戻す

shader reload 失敗時に最後の正常 pipeline を使い続けるのは許可します。ただし error は見える場所に出します。

## Loading flow

```text
UserApp
  -> LoadAsset command
  -> worker importer loads .r1scene or GLB to ImportedScene
  -> asset store allocates protocol handles
  -> AssetLoaded event with protocol handles
  -> UserApp/ECS asset system stores handles
  -> render extraction includes handles in FrameSnapshot
  -> renderer resolves handles to draw packets
  -> graph renders scene
```

path 探索は user code または application policy です。renderer core は `assets/model.glb` のような固定探索を知りません。
