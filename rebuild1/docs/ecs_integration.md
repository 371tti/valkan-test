# ECS integration

## 結論

将来 ECS を載せる前提でも、今の message / async / renderer thread 方針は使えます。ただし、次の境界を設計として固定します。

```text
ECS World
  -> simulation systems
  -> render extraction
  -> FrameSnapshot
  -> RendererCommand::SubmitFrame
  -> RendererTask
```

renderer は ECS crate、entity id、component storage を知りません。renderer が受け取るのは protocol handle と plain data だけです。

## そのままだと危ない点

### ECS Entity を protocol に出す

ECS の `Entity` は world 内の一時的な handle です。generation を持っていても、その world の外へ出すと lifetime と意味が曖昧になります。

protocol payload に ECS `Entity` を入れません。debug や picking 用に外へ出したい場合は、ECS 側で安定した `ExternalObjectId` を作り、それを optional metadata として `RenderItemPacket` に入れます。

### ECS World を async 境界越しに借用する

renderer thread が ECS world を読む設計にすると、lock、borrow、schedule が複雑になります。

ECS world は user task が所有します。`FrameSnapshot` は ECS schedule の最後に作る owned data です。snapshot は `Send + 'static` にして、renderer thread へ移動できる形にします。

### Component 更新を chatty message にする

`SetTransform(entity)`, `SetMaterial(entity)`, `SetVisibility(entity)` のような細かい message を毎 frame 大量に送ると、protocol が ECS の差分同期になってしまいます。

最初は per-frame batch として `FrameSnapshot` を送ります。将来最適化が必要になったら `SceneDelta` を追加しますが、その場合も ECS Entity ではなく renderer protocol handle と stable object id を使います。

## Render extraction

ECS から renderer へ行く唯一の入口は render extraction です。

```text
RenderExtract
  input:
    ECS World
    asset handle components
    visibility/camera/light components

  output:
    FrameSnapshot
```

`FrameSnapshot` は renderer が即 draw packet に変換できるようにします。

```text
FrameSnapshot
  frame_id
  views
  lights
  render_items
  camera_effects
  debug_draw

RenderItemPacket
  object_id optional ExternalObjectId
  transform
  mesh MeshHandle
  material MaterialHandle
  flags
  layer
```

`RenderItemPacket` は component の参照を持ちません。ECS の archetype、query、component id も持ちません。

ECS が入った後も auto exposure / white balance は extraction 側の camera system が `camera_effects` に畳み込みます。renderer は ECS world を読まず、暗い scene を明るい scene と誤認しないように「ほぼ黒の metering では exposure を上げない」制約を app/user 側で守ります。

## Asset flow with ECS

asset load は ECS system と renderer event の間で行います。

```text
AssetRequest component
  -> LoadAsset command
  -> AssetLoaded event
  -> ECS asset system writes MeshHandle/MaterialHandle component
  -> render extraction includes handles in FrameSnapshot
```

renderer は ECS component を更新しません。renderer は event を返すだけです。

## Visibility and culling

最初の culling は ECS/user 側の render extraction で行ってよいです。

将来の選択肢:

- ECS 側 visibility system
- renderer 側 CPU culling
- renderer 側 GPU culling
- hybrid culling

どれを選んでも、renderer が ECS world を直接読む必要はありません。境界は `FrameSnapshot` または将来の `VisibilityInput` / `DrawPacket` に保ちます。

## Trait

ECS を直接 trait に縛る必要はありません。ただし extract 境界は trait 候補です。

```text
trait RenderExtract {
    fn extract(&mut self, world: &mut EcsWorld, out: &mut FrameSnapshotBuilder);
}
```

async にしない方がよいです。ECS world を borrow している間に `await` すると schedule と borrow が壊れやすくなります。必要な async work は asset IO や renderer command 側に逃がし、extract 自体は同期的な短い処理にします。

## Rules

- renderer crate は ECS crate に依存しない。
- protocol crate は ECS crate に依存しない。
- ECS Entity は protocol payload に入れない。
- `FrameSnapshot` は owned data にする。
- `FrameSnapshot` は `Send + 'static` を満たす。
- renderer thread は ECS world を lock しない。
- component 変更を細かい renderer command にしない。
- asset result は `RendererEvent` として ECS 側に戻す。
