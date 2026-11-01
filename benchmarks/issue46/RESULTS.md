# #46: broccoli の grouped Auto 診断

**結論: 大ジョブ少数で Auto が遅い相対傾向は再現した (yes)。**
ただし、最大ジョブに合わせてグループ全体を Inner にする最小試作は、
all-small を維持しても mixed を壊した。この試作を本体に採用しない。
本 PR はベンチ・診断パッチ・証拠のみであり、#46 を解決済みにはしない。
**公式性能値ではない。Ryo さんの測定との絶対値比較は禁止。**

## 表と生データ

- **[全結果表: ms と最速=1.00 の相対値、ペア方向、レーン、CPU%](results/broccoli-20261101/tables.md)**
- [集計 JSON](results/broccoli-20261101/summary.json)
- [全 875 実行の生データ](results/broccoli-20261101/raw.jsonl)
- [CPU / L3 / affinity / コンパイラ / binary hash](results/broccoli-20261101/environment.json)
- [終了時の source 不変チェック・binary hash](results/broccoli-20261101/finished.json)
- [baseline / candidate の build identity](results/broccoli-20261101/build-evidence/builds.json)
- [事前プロトコル](PROTOCOL.md)、[診断パッチ](team.patch)、[独立レビュー](REVIEW.md)
- 進捗はリポジトリルートの [`report`](../../report)。計測中は timer で 10 分時点に追記した。
  準備中の一部の追記間隔は 10 分を超えており、この運用条件は完全には守れていない。

表の時間は 7 個の warm single-call 観測の中央値。ns を正確に ms に変換した。
相対値と CPU% は丸めを避けるため **既約分数**で記載する。
最速の `1.00` は同一 workload / width の baseline 四戦略と candidate Auto
の五列の最小値。異なる workload / width 間の比ではない。
`UNSTABLE` は事前定義どおり max/min > 1.25。サンプルは削除していない。

## ワークロード

f64、コピー不要な compact column-major GEMM:
`A[m,k] B[k,n] -> D[m,n]`、alpha=1、overwrite、入力・出力はジョブ間で別領域。
`PlanConfig.faer_limit=FaerLimit::NONE` に加え、各 geometry を独立に準備して
`Algorithm::Faer` を assert した。ベンチに独自の contraction kernel はない。

| 名称 | ジョブ構成 (m,k,n) | MAC 合計 | 通常の FLOP 換算 | A+B+D の論理 bytes |
|---|---|---:|---:|---:|
| A | 2 × (1024,256,1024) | 536870912 | 1073741824 | 25165824 |
| B | 16384 × (32,32,32) | 536870912 | 1073741824 | 402653184 |
| C (追加) | 1 × (1024,256,1024) + 8192 × (32,32,32) | 536870912 | 1073741824 | 213909504 |

A は contracted dimension=256、free dimension=4×256 の合成ブロックで、
Ryo さんの実際の symmetry block shape のコピーではない。
16T 以上では大ジョブ単体の cost-model width が 8 を超えることも assert した。
B と A は演算量だけが同じで、working set と dispatch 数は同じではない。
C は issue 本文の「大ジョブと小ジョブの混在」を別に検証する補足である。
さらに `large_job` / `small_job` として各 geometry 1 個を同じ全幅・戦略で測った。
これは **単独ジョブの時間**であり、group 内を instrument した per-job span ではない。

入力は行・列・ジョブで変化する符号付き rank-one 行列。全出力要素を解析解と
比較し、warmup 後と計測後に検証した。全 875 実行で検証・route assertion 成功、
実行失敗 0、条件欠落 0。functional-smoke は別ディレクトリに保存し、中央値には混ぜない。

## 環境・固定した CPU 集合

broccoli: AMD EPYC 7702P (Zen 2)、64 physical cores、SMT なし、1 NUMA node、
4-core L3 domain ×16、governor=performance、boost disabled。
測定セットは sysfs の `cache/index3/shared_cpu_list` から生成した。

| 要求幅 | 測定 CPU 集合 | 使用 L3 ドメイン数 | 観測 CPU 集合 | 注記 |
|---:|---|---:|---|---|
| 1 | 0 | 1 | 60-63 | 明示的 1T control |
| 8 | 0-7 | 2 | 60-63 | 観測側と分離 |
| 16 | 0-15 | 4 | 60-63 | 観測側と分離 |
| 32 | 0-31 | 8 | 60-63 | 観測側と分離 |
| 64 | 0-63 | 16 | 60-63 | **観測側と共有、参考値のみ** |

複数 L3 と 64T の観測側共有は、今回の診断に限ってユーザーが明示承認した例外。
親 pi も `taskset -apc 60-63` で固定。worker は全測定 mask を継承し、
`pin_participants(false)`。各実行の affinity と nice=5 を読み戻して assert した。
ビルドは `nice -n 5 ... -j8`、逐次実行。計測とビルドは重ねていない。

Rust 1.99.0、bench/release profile: opt-level=3、thin LTO、codegen-units=1。
cpueinsum baseline は `adcf0c4e090af1c7753f84facfb33342f59c3890` (#56 後)。
依存は lockfile の tcontract 0.9.0 **git 625ed7a7124beb9537660b607c3c453d822d4dc8**、
tcontract-exec 0.9.0、t4a-faer 0.27.0、threadteam 0.2.0。
`AGENTS.md` の「tcontract は registry / capi は release commit」という説明は、
現在の Cargo.toml の fresh-output 用 git patch と一致しない。
実際の Cargo.toml / Cargo.lock を優先し、依存を変更していない。
従って Ryo さんの cpueinsum 0.9.0 とも、完全に同一の依存状態とは主張しない。

## 測定境界・揺れ

plan / pool / Team / input / output 構築を除外。保持した owned Team の token から
Exec を作り、要求幅と carries_team を確認した。
**plain pool token では faer 内部並列が動かない**ため、この確認は必須である。
各 subprocess を 250ms 以上 warmup し、全出力チェック後にもう一度 execute して
warm 状態に戻し、complete `execute_into` 1 回を ns 単位で測定。
続けて 250ms 以上の repeated-call CPU 診断窓を取った。
process CPU ticks は全 worker を含み、CPU% は待機・spin も含む。
CPU 診断窓の平均時間を warm single-call 中央値と混ぜていない。

7 round の各 round で条件順を rotate、戦略順を forward / reverse と交互にした。
875 実行を逐次実行し、時間の大きなサンプルもすべて保持した。
A16T の Inner のうち 2 round は Auto/Inner >1.25 を満たさず、
Split2 と Split4 は同じ route でも時間が異なるサンプルがある。
64T の candidate A などにも大きな tail がある。これを selective retry していない。

round 前後の全 CPU aggregate idle 窓はすべて busy<5% だったが、
**round 4 開始時に測定 CPU 6 が 10% busy**、観測 CPU にも 5% 超の窓があった。
これは per-CPU idle warning として残す。短い外乱も残存 runtime tail も排除できず、
この診断を厳密な性能保証・採用ゲートに格上げしない。

## 判定・原因の切り分け

### (a) 大ジョブが 1 thread に固定されるか: 主因として支持

A の現行 Auto は 8–64T で **2 lanes × 1 item thread**（1T は 1 lane）、単独大ジョブ Auto は
**1 lane × 1**。要求幅を増やしても、この有効なジョブ同時数は増えない。
同じ shape / input / output のまま Inner / Split にすると、CPU 消費時間の増加と
wall time の短縮が対応する。単独大ジョブ control の serial 時間も A の
2 ジョブを同時実行した critical-path 時間と整合する（表参照）。
したがって単なる総 MAC 数や固定のメモリ帯域だけでは説明できない。

事前の相対傾向ゲートは **8T と 32T** で成立した:
Auto/Inner と Auto/best-Split の中央値がともに >1.25、かつ両比較の
7/7 round で同じ >1.25 方向。16T は中央値では大差だが Inner 側の tail により
Inner 比の方向ゲートは 5/7 なので、その幅だけで安定性を主張しない。
64T は参考値であり yes/no の根拠に使っていない。

### (b) レーン粒度: A の根因とは別、mixed では重要

A はジョブが 2 個なので Split2 と Split4 はどちらも 2 lanes に clamp され、
item width は T/2。同じ route のタイミング差を Split4 固有の優位とは解釈しない。
B はレーン数と仕事量が均等。C の現行 Auto はジョブ**数**で contiguous 分割するため、
最初の lane が大ジョブを抱える。lane MAC min/max は実際の job order と
返された lane 数から算出して表に記録した。この不均衡は width を増やしても残る。

### (c) メモリ帯域: 唯一の原因ではない。帯域値そのものは未測定

同じ A のデータを使い、item width の変更だけで改善したことは、
「ホストの帯域が固定の律速だから Auto が遅い」という単独説明に反する。
ただし B は working set が大きい。最低限の logical bytes は表の値で、
cache line / packing / repeated read / write-allocate を含む実測 traffic ではない。
`perf` は未インストール、perf_event_paranoid=3。DRAM 帯域カウンタは取っていない。
CPU% だけから bandwidth-bound と認定せず、キャッシュ／帯域の残余寄与は未確定とする。

## 試作と判断材料

診断パッチは `lane_decision` の Auto のみを変更し、最大ジョブを single-item の
`faer_auto` query に渡す。item width>1 なら 1 lane、それ以外は元の outer 計算を
そのまま使う。production source は build 後に復元し、計測の前後も不変だった。

- A: candidate は Inner route を取り、大ジョブ側の短縮を確認した。
- B: candidate は baseline Auto と同じ outer route。8/16/32T の中央値はいずれも
  candidate/Auto <=1.10。10% を超えて悪化した paired sample も残っているので、
  「誤差ゼロ」や Ryo さんの all-small gate 合格とは言わない。
- **C: candidate は 8/16/32T の全 round で baseline Auto より >10% 遅い。**
  大ジョブ 1 個の後に小ジョブを同じ single lane で流すと、small 側の job 並列を失う。
  all-small のみを見れば見逃す退行である。

よって「常時 Inner」も「最大ジョブだけでグループ全体を Inner」も採用しない。
今後の per-item / 大小を区別した実装は A/B に加えて mixed C でレビューする必要がある。
ここでは次の scheduler 設計や threshold tuning は実装していない。

## Ryo さんの測定と同じ点・違う点（傾向のみ）

同じ点: 大ジョブ少数では Auto の item-serial が並列資源を使い切らず、
Inner / Split で改善する。小ジョブ多数に Inner を強制すると逆に不利。

違う点: こちらは Zen 2 の 64 cores、合成 GEMM 群であり、symtn の
U(1)/SU(2) ブロックスパース縮約・96 コア専有機の測定ではない。
small size も実際の degeneracy 16 の symmetry workload を再現したものではない。
tenferro 0.7.1 を測っていないので、**「旧版比 2.1–3 倍」の倍率は未検証**。
絶対値、倍率の大きさ、最適幅を Ryo さんの環境へ外挿しない。

## ローカル検証

すべて逐次実行、Cargo build は nice=5 / -j8:

- `cargo fmt --all -- --check`
- `cargo clippy --workspace --all-targets -j8 -- -D warnings`
- `cargo test --workspace -j8`（doctest 含む）
- `cargo test --release --workspace -j8`（doctest 含む）
- `RUSTDOCFLAGS='-D warnings' cargo doc --workspace --no-deps -j8`
- `cargo +1.89.0 build --workspace --all-targets -j8`
- 既存 native / BLAS `overhead_smoke`: `OVERHEAD_GATE=1`、CPU 0、
  `RAYON_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1`、provider thread 読み戻し=1。
- recorder self-check（正確な単位・中央値・失敗条件の扱い）と、生データからの
  別計算による 875 sample / 125 median / binary・source hash 照合。
- `git diff --check`、production `grouped.rs` と Cargo.lock の差分なし。

[検証ログ](results/broccoli-20261101/validation/)を保存。
`cargo-deny` はこのホストにないためローカルでは未実行。macOS の実行検証も未実施。
これらは hosted CI の該当 gate に委ねるが、required CI や承認は迂回しない。
独立事後レビューの minor 指摘（1T のレーン数の表現）を修正した。
測定ソース・プロトコル・生データの変更や計測の再試行はしていない。

## crate 配布物からの分離

PR レビューに従い、診断資料一式を repository-root の `benchmarks/issue46/` に
移動した。crate 内には bench ターゲットのソースだけを残し、性能の結論は変更していない。
[package 一覧の前後比較とビルドログ](packaging/)を保存する。
相対リンクは同じ bundle 内を参照し、再実行コマンドとスクリプトの root 解決を修正した。
過去の `results/` 内の記録パス・hash・検証ログは当時の証拠として byte-identical に保持した。
bench ソースの変更も資料の場所を示すコメントのみで、計算や測定ロジックは不変。

## 再実行

```bash
# sysfs から観測用の最後の L3 domain を選ぶ（この記録では 60-63）。
# 親 pi もその集合に taskset -apc で固定してから、逐次実行する。
taskset -c 60-63 python3 benchmarks/issue46/build.py /tmp/issue46-build
python3 benchmarks/issue46/record.py self-test
taskset -c 60-63 python3 benchmarks/issue46/record.py run \
  --baseline /tmp/issue46-build/baseline --candidate /tmp/issue46-build/candidate \
  --output /tmp/issue46-run
# 集計のみなら計測を再実行しない。
python3 benchmarks/issue46/record.py summarize \
  benchmarks/issue46/results/broccoli-20261101
```

ビルド・測定は nice=5、build -j8 をスクリプト内で指定する。別のホストでは
上の観測用 mask を sysfs から選び直す。64T 観測共有の例外はこの依頼専用である。
