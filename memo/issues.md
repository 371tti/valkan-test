# 修正インベントリ

| 優先 | 問題 | 対処 | 状態 |
|---|---|---|---|
| B0 | smoke が asset 未ロードや validation error を見逃す | loaded frame、validation/skipped-frame 診断、timeout を gate 化 | fixed |
| B1 | 品質 preset が機能 toggle と見た目の強度を混在させる | feature bit と連続品質値を分離 | fixed |
| B1 | SMAA が最終画像の前に入り、方向を誤ってぼかす | PostColor 後に公式3 passを固定 | fixed; visual pending |
| B1 | SSAO が固定近傍の深度比較だけ | GTAO horizon search と解析的 arc integration に置換 | fixed; visual pending |
| B1 | PCSS 履歴がカメラ/光源移動で不安定 | visibility と receiver depth だけを専用履歴へ保存し、深度一致候補だけ再投影 | fixed; visual pending |
| B1 | PCSS 近距離ちらつき | raw/history/final の F12 比較を追加。CSM 境界は原因候補から除外済み | visual pending |
| B1 | Sun Shaft が層状・箱状になる | 専用 camera-ray 48 strata、光源位相、temporal history、bilateral upscale を接続 | visual pending |
| B1 | 高品質 GodRay の透過率が散乱光のない影で画面全体を暗くする | 散乱信号がある画素だけ scene attenuation を適用し、暗部では透過率を 1 に戻す | fixed; smoke verified; visual pending |
| B1 | 旧3Dボリューム経路が専用 GodRay に混入して見える | 旧 graph、target、pipeline、shader、診断、fallback を削除。専用 route を mask → temporal に限定 | fixed; smoke verified |
| B2 | 低品質 radial GodRay と専用 route の役割が曖昧 | radial は volumetric OFF 時だけ、専用 routeとは同一frameで合成しない | fixed |
| B2 | TAA は既定で未使用なのに資源を確保する | dormant sentinel とし、明示的に有効化した時だけ確保 | fixed |
| B2 | docs と実装の pass 順がずれる | README/docs/memo を現行 graph に同期 | fixed |

## 除外した仮説

- CSM cascade 境界: ユーザーの実画面確認で該当しないため除外
- shadow-map 最近傍、PCSS compare、footprint filter: 実画面差分がなく主因候補から除外
