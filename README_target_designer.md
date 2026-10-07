# Ideal Target Designer v1

Observed post-process masksを基に，直線と幾何学的bottom形状を使って
**desired target maskをgeometric idealizationとして作るGUI**です．

## 目的

元の実データmaskは変更しません．

- observed mask = 実験で得た形状
- desired target mask = observed maskを基に幾何学的に理想化した制御target

として別保存します．

## 8 target classes

1. Typical
2. Oblique-right
3. Oblique-left
4. Elongated
5. Wide & short (flat-bottom)
6. Rounded-bottom
7. Pointed-bottom
8. Near-parallel sidewall

Classを選択するとbottomの初期primitiveは次のようになります．

- Typical: circular arc
- Oblique-right: circular arc
- Oblique-left: circular arc
- Elongated: circular arc
- Wide & short: flat
- Rounded-bottom: elliptical arc
- Pointed-bottom: two straight segments / V
- Near-parallel sidewall: circular arc

bottom type自体はGUIから手動変更可能です．

## Geometry

5個のcontrol pointをドラッグします．

- TL: top-left / opening left
- BL: bottom-left shoulder
- BA: bottom apex
- BR: bottom-right shoulder
- TR: top-right / opening right

左右側壁は

- TL → BL
- TR → BR

の直線です．

### Circular arc

BL, BA, BRを通る円弧です．

### Elliptical arc

BL-BRを長軸方向とし，BAがbottom深さを決める半楕円です．
Rounded-bottom用です．

### Flat

BL-BRを直線で接続します．
Wide & short / flat-bottom用です．

### Pointed / V

BL → BA → BRを2本の直線で接続します．
Pointed-bottom用です．

## Three-mask overlay

最大3枚のobserved maskを読み込めます．

- 低opacityで半透明表示
- maskごとに別色
- 輪郭は少し濃く表示
- `Align observed masks to image X-center`をONにすると，
  各maskのopening centerだけX方向に画像中央へ揃えて表示

source file自体は変更されません．

## X centering

`Center ideal target X`でdesired targetを画像X中央へ移動できます．

Reference:

- `Opening midpoint`
  - TLとTRの中点を画像中央へ合わせる
- `Ideal centroid`
  - ideal maskの面積重心Xを画像中央へ合わせる

`Lock ideal target to X-center while editing`をONにすると，
control pointを動かすたびにX方向だけ再センタリングします．
Y形状は変更しません．

今回の用途ではデフォルトの
`Opening midpoint + Lock`
を推奨します．

## Convenience buttons

- `Auto initialize from masks`
  - source masksからcontrol pointの初期値を推定
- `Level opening`
  - TL/TRを同じYにする
- `Parallelize sidewalls`
  - 左右側壁の傾きを平均化して平行に近づける

Near-parallel sidewall targetでは
`Parallelize sidewalls`を利用できます．

## Live metrics

- ideal mask area fraction
- opening width
- depth
- centroid X
- ideal vs each displayed observed maskのIoU

を表示します．

これはidealizationで元形状から逸脱しすぎていないかを確認するための補助です．

## Saved files

`Save ideal target`で

`<output-dir>/<target-id>/`

以下へ，

- `desired_target_mask.npy`
- `desired_target_mask.png`
- `idealization_overlay.png`
- `ideal_target.json`
- `ideal_target_parameters.csv`

を保存します．

さらに`<output-dir>/ideal_target_registry.csv`へtargetを登録します．

`ideal_target.json`には

- source mask path
- source file SHA256
- control point座標
- target class
- bottom primitive
- centering設定
- source vs ideal IoU
- ideal mask metrics

が保存されるため，後から同じtargetを再現できます．

## Requirements

- Python 3.10+
- numpy
- Pillow
- tkinter

Windowsの通常のPythonではtkinterは含まれています．

## Example

```powershell
python ideal_target_designer.py `
  --output-dir "F:\261006_adaptive_control_data_test\target_selection\ideal_targets" `
  --channel 1 `
  --threshold 0.5 `
  --target-id T_TYPICAL `
  --target-class Typical
```

GUIを開いてから`Add masks...`で1～3個の`target.npy`を読み込めます．

CLIから直接3個まで渡すこともできます．

```powershell
python ideal_target_designer.py `
  --output-dir "F:\261006_adaptive_control_data_test\target_selection\ideal_targets" `
  --mask "F:\260910_ml_data\h0286\p0050\target.npy" `
  --mask "F:\260910_ml_data\h0215\p0050\target.npy" `
  --mask "F:\260910_ml_data\h0390\p0050\target.npy" `
  --target-id T_TYPICAL `
  --target-class Typical
```

## Recommended workflow

1. 同一target classのobserved masksを最大3枚読み込む
2. X-alignをONにして重ねる
3. Auto initialize
4. TL/BL/BA/BR/TRをドラッグ
5. 必要ならLevel opening / Parallelize sidewalls
6. desired targetの巨視的特徴が3 masksを代表しているか確認
7. IoU / area / depth / opening widthを確認
8. Center ideal target X
9. Save ideal target

targetを保存したあとに生成結果を見てtarget形状を変更することは避け，
target definitionを先に固定する運用を推奨します．
