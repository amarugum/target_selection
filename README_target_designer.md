# Ideal Target Designer v8

v7に加えて，control pointを動かしている間に座標を常時表示します．

## Coordinate convention

- X：画像中央pixelを `x = 0`
- 右方向：正
- 左方向：負
- Y：画像上端を `y = 0`
- 下方向：正

384 px幅では，absolute pixel `x=192` を `x=0` とします．

## Live coordinate display

TL / TR / BL / BR / BAをクリック・ドラッグすると，

`Active point: TL   x=-36 px   y=32 px`

のように画像下へ常時表示します．

ドラッグ中はcontrol pointのすぐ横にも

`TL  x=-36  y=32 px`

と表示します．

release後も最後に操作したpointの座標を上部表示に残します．

## Live metrics

`centroid x`も画像中央=0の相対X座標へ変更しました．
最後に操作したcontrol pointもLive metricsへ表示します．

## Saved parameters

`ideal_target.json`では各control pointについて，

- `x_abs`
- `x_centered`
- `y_from_top`

を保存します．

`ideal_target_parameters.csv`では従来の`*_x_px`に加えて，

- `TL_x_centered_px`
- `BL_x_centered_px`
- `BA_x_centered_px`
- `BR_x_centered_px`
- `TR_x_centered_px`

も保存します．
