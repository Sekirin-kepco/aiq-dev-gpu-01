モデル差し替え手順（軽量→本番モデルへの移行ガイド）

目的
- このドキュメントは、現在の軽量テストサービス（`deploy/lightweight`）から実際の `aira` 実装や外部推論サービス（NIM）／より大きなローカルモデルへ安全に差し替える手順をまとめます。

適用範囲
- 軽量コンテナ（Flask placeholder）→ aiq_aira 実装（ローカル）
- 軽量コンテナ → litellm 等の小型ローカルモデル
- 軽量コンテナ → NVIDIA NIM を利用したリモート/セルフホスト推論

前提（必ず確認）
- Docker が動作すること（今回の EC2 は確認済み）。
- GPU を使う場合、ホストに NVIDIA ドライバと nvidia-container-toolkit が導入済みであること。`nvidia-smi` と `docker run --gpus all nvidia/cuda:... nvidia-smi` で確認。
- NIM を利用する場合、NGC/Build.NVIDIA の API キー（NGC API Key）または NIM のアクセス方法を用意。
- ライセンス確認：Llama 系などのモデルはライセンス条件があるため、利用前に確認。

概念設計（短く）
- 実行時に切り替えるキーは主に環境変数（例：`MODEL_BACKEND`）と `docker-compose` のイメージ／ビルド設定です。
- `MODEL_BACKEND` の想定値：`placeholder` | `aira_local` | `litellm_local` | `nim_remote`

変更パターン A — `aira` 実装（最小組込）
1. 依存をコンテナに追加
   - `deploy/lightweight/Dockerfile` の pip インストール部に `aiq_aira`（パッケージ名は `pyproject.toml` の名前に一致）と必要なランタイム依存を追加します。
   例（Dockerfile内の一行を置換または追加）:

```dockerfile
# 既存の requirements をインストールした後に
RUN pip install --no-cache-dir aiq_aira
```

2. 環境変数を追加
   - `docker-compose.yml` に以下を追加します（例）:

```yaml
environment:
  - MODEL_BACKEND=aira_local
  - AIRA_CONFIG_PATH=/etc/aira/config.yaml  # 必要なら
```

3. コンテナ内部から `aira` を起動する箇所を `server.py` に実装
   - 既に `server.py` は `aiq_aira` のクライアントを `try` して呼べるようになっています。`aiq_aira` 側のクライアント初期化（`.from_env()` 等）に合わせて環境変数や設定ファイルを渡してください。

4. ビルドと起動

```bash
cd deploy/lightweight
docker build -t aiq/aira-lite:aira_local .
docker rm -f aira-lite || true
docker run -d --name aira-lite -p 8080:8080 -e MODEL_BACKEND=aira_local aiq/aira-lite:aira_local
```

5. 検証
   - `/health` が `aira_available: true` を返すこと
   - `/query` に対して実際の回答（出典付き）が返ること

変更パターン B — `litellm` 等の軽量ローカルモデルへ差し替え
1. 依存追加
   - Dockerfile に `litellm`（または選定した軽量ランタイム）を追加。さらに、埋め込み用に `sentence-transformers` や `faiss-cpu` を追加するか、外部ベクトルDBの設定を行う。

```dockerfile
RUN pip install --no-cache-dir litellm sentence-transformers faiss-cpu aiq_aira
```

2. モデルファイルの配置方法
   - 小型モデルはコンテナ内にダウンロードして `/models` に置くか、ホスト上に置いてボリュームマウントします。
   - 例: `-v /home/ec2-user/models:/models`

3. 環境変数と起動

```yaml
environment:
  - MODEL_BACKEND=litellm_local
  - LITELLM_MODEL_PATH=/models/my-small-model.bin
volumes:
  - /home/ec2-user/models:/models
```

4. 検証
   - `/health` が `aira_available: true`（もし aira を経由するなら）または MODEL_BACKEND を反映する値を返す
   - レイテンシとメモリ使用量を監視（`docker stats`）

変更パターン C — NVIDIA NIM を使う（推奨：本番的運用）
1. 事前準備
   - NGC/Build.NVIDIA の API キーを用意し、NIM のエンドポイント/イメージを入手します。
   - NIM はコンテナとして起動するか、NIM がホストするサービスに接続します。

2. `server.py`／`aira` 側の設定
   - `aira` のクライアント設定を NIM のホスト/ポートと API キーに合わせます。
   - 環境変数例:

```yaml
environment:
  - MODEL_BACKEND=nim_remote
  - NIM_URL=https://nim-host:8000
  - NIM_API_KEY=<<YOUR_NIM_KEY>>
```

3. NIM コンテナ起動（例）

```bash
# これは NIM の指示に従ってください。例:
docker run -d --gpus all --name nim-llm -p 8000:8000 -e NIM_API_KEY=${NGC_API_KEY} nvcr.io/nim/llm:latest
```

4. 検証
   - NIM のヘルスを確認（NIM のドキュメントに従う）。
   - `aira` を通じて NIM に問い合わせが行けるか `/query` でテスト。

リソースとコストの注意
- 大型モデルは GPU / VRAM を大量に消費します。L4（24GB）は 8B〜13B の一部モデルが狙い目。49B 等は H100/A100 等高性能GPUが必要。
- 試験的に起動する場合はログ/モニタリング（`docker logs`, `docker stats`）と、ホストのメモリ・ディスクの監視を必ず行ってください。

検証チェックリスト（デプロイ後）
1. `/health` が有効（`aira_available` フラグなど）。
2. `/query` の簡易サンプルが期待した形式で応答する。出典（source/file/page）を含むか確認。  
3. ログにエラーが出ていないこと（`docker logs aira-lite`）。
4. リソース（GPU/CPU/RAM）の利用状況を確認。必要ならスケールダウン/停止。

ロールバック手順（問題発生時）
1. 元のイメージ（`aiq/aira-lite:latest` の旧タグ）を再度タグ付けして起動する、あるいはプレースホルダイメージを再度ビルドして置き換える。  

```bash
docker rm -f aira-lite || true
docker run -d --name aira-lite -p 8080:8080 aiq/aira-lite:latest
```

2. 環境変数を元に戻す（`docker-compose.yml` の差分を戻す等）。

セキュリティとライセンス留意点
- NIM や一部モデルは API キー／使用許諾が必要。必ず利用規約を満たすこと。  
- モデルの推論にはコストがかかる（GPU時間）。環境変数や監視で滅多なリクエストを防止してください。

短いテンプレート（よく使う env 変数）

```text
MODEL_BACKEND=litellm_local|aira_local|nim_remote
LITELLM_MODEL_PATH=/models/<model-file>
NIM_URL=https://nim-host:8000
NIM_API_KEY=...
VECTOR_URL=faiss://local or redis://host:port or http://vector-service
EMBEDDING_BACKEND=sentence-transformers|nim-embed

# ログ/設定
LOG_LEVEL=info
```

最後に
- まずは `aira` の最小組込（パターンA）で動作確認し、次に実際に使いたいモデルバックエンド（litellm か NIM）へ移行する流れが安全で簡潔です。
