# Target Selection GUI v2

v1 で Round 1 選択後に Round 2 が空に見えるケースを修正・診断しやすくしました。

## v2 の変更

- `Start Round 2` / `Start Round 3` ボタンを追加
- Round 切替時に Category filter を自動的に `All` へ戻す
- Round 2 header に `input from R1=N` を表示
- Round 3 header に `input from R2=N` を表示
- 入力候補が 0 件なら白画面ではなく原因を画面に表示
- CSV の 0/1, True/False を頑健に読み込む
- Round 切替時にも CSV を保存

## Round 2 の使い方

1. Round 1 で残したい画像をクリックし、画像カードに `✓` が付くことを確認する。
2. 画面上部の `R1=N` が 1 以上になっていることを確認する。
3. `Start Round 2` を押す。
4. Round 1 で選んだ N 件だけが Round 2 に表示される。
5. Round 2 でもさらに残したい画像をクリックする。これは `R2` 選択であり、Round 1 の選択とは別。
6. Round 2 の選別後に `Start Round 3` を押す。

既存の `target_selection_review.csv` はそのまま引き継げます。同じ `--output-dir` を指定してください。
