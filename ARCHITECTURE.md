# BruteForm アーキテクチャ

## 概要
BruteFormはClean Architectureの原則に基づいて設計されたぷよぷよ連鎖形総当たりツールです。

## レイヤー構造

### 1. Domain層 (`src/domain/`)
**ビジネスロジックの中核 - フレームワーク非依存**

- `domain/board/` - 盤面関連
  - `board.rs` - 高レベル盤面表現
  - `board_bits.rs` - ビットボード表現
  - `cell.rs` - セル型定義
  - `bitboard.rs` - ビットボード演算（旧search/board.rs）
  - `hash.rs` - ハッシュ計算（旧search/hash.rs）

- `domain/chain/` - 連鎖ロジック
  - `result.rs` - 連鎖結果の表現

- `domain/chain_legacy/` - 連鎖検出・生成（旧chain/）
  - `detector.rs` - 連鎖検出
  - `generator.rs` - 連鎖生成
  - `beam.rs` - ビーム探索
  - `grid.rs` - グリッド操作

- `domain/constraints/` - 制約・彩色ロジック（旧search/coloring.rs）
  - `coloring.rs` - 彩色列挙

- `domain/search/` - 探索設定
  - `config.rs` - 探索パラメータ

### 2. Application層 (`src/application/`)
**ユースケースの実装**

- `application/bruteforce/` - 総当たり探索
  - `service.rs` - 総当たりサービス
  - `search/` - 探索実装（旧search/engine.rs, dfs.rs, pruning.rs）
    - `engine.rs` - 探索エンジン
    - `dfs.rs` - DFS探索
    - `pruning.rs` - 枝刈り

- `application/chainplay/` - 連鎖プレイ
  - `service.rs` - 連鎖プレイサービス

- `application/progress/` - 進捗管理
  - `manager.rs` - 進捗トラッキング

### 3. Infrastructure層 (`src/infrastructure/`)
**技術的実装・外部システム接続**

- `infrastructure/executor/` - 並列実行
  - `parallel.rs` - 並列処理フレームワーク

- `infrastructure/storage/` - ストレージ
  - `writer.rs` - ファイル書き込み

- `infrastructure/cache/` - キャッシュ（旧search/lru.rs）
  - `lru.rs` - LRUキャッシュ

### 4. Presentation層 (`src/presentation/`)
**UI状態管理**

- `presentation/state/` - 状態管理
  - `bruteforce.rs` - 総当たり画面状態
  - `chainplay.rs` - 連鎖プレイ画面状態

### 5. UI層 (`src/app/`)
**eframe/eguiによるGUI実装**

- `app/` - メインアプリケーション
  - `state.rs` - アプリケーション状態
  - `ui/` - UI描画
    - `mod.rs` - UI更新ループ
    - `bruteforce.rs` - 総当たり画面
    - `chainplay.rs` - 連鎖プレイ画面
    - `helpers.rs` - UI描画ヘルパー
  - `chain_play/` - 連鎖プレイ機能
  - その他のUI関連ロジック

### 6. ユーティリティ
- `constants.rs` - 盤面定数とグローバル型定義
- `profiling.rs` - パフォーマンス計測
- `logging.rs` - ロギング機能

## 依存関係の流れ

```
UI層 (app/)
   ↓
Presentation層 (presentation/)
   ↓
Application層 (application/)
   ↓
Domain層 (domain/)
   ↑
Infrastructure層 (infrastructure/)
```

- **下位層は上位層に依存しない**（依存性逆転の原則）
- **Domain層は完全に独立**（他のレイヤーに依存しない）
- **Infrastructure層はDomain層のインターフェースを実装**

## リファクタリング履歴

### Phase 1: search/層の統合 ✅
- `search/board.rs` → `domain/board/bitboard.rs`
- `search/hash.rs` → `domain/board/hash.rs`
- `search/coloring.rs` → `domain/constraints/coloring.rs`
- `search/lru.rs` → `infrastructure/cache/lru.rs`
- `search/engine.rs, dfs.rs, pruning.rs` → `application/bruteforce/search/`

### Phase 2: chain/層の統合 ✅
- `chain/` → `domain/chain_legacy/`
  - `detector.rs`, `generator.rs`, `beam.rs`, `grid.rs`を移行
  - 後方互換性のため`chain/`は再エクスポートとして残存

### Phase 3: model/層の削除 ✅
- `model/cell.rs`の`cell_style`関数を`app/ui/helpers.rs`に移動
- `Cell`, `TCell`は`domain/board`から直接使用

## テスト構成
- **ユニットテスト**: 74個
- **統合テスト**: 13個
- **合計**: 87個

全てのテストが成功し、リファクタリング後も機能が保証されています。
