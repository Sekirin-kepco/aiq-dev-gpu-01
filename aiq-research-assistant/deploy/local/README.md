このディレクトリは「軽量構成での素早い動作確認」を目的としたローカル/EC2向けの雛形です。

目的
- 最小限の aira バックエンドイメージをビルドして起動することで、AI-Q 全体の起動フローや接続先（NIM/軽量LLM）を検証します。

前提
- ホストに Docker と NVIDIA Container Toolkit（GPU利用時）がインストールされていること。
- 必要に応じて `pyproject.toml` / `requirements.txt` の依存を満たす必要があります。

手順（EC2 上での例）
1. リポジトリのルートに移動
   ```bash
   cd ~/aiq/aiq-research-assistant
   ```
2. 環境変数を設定（NIM を使う場合／ローカル lite-LLM を使う場合に応じて）
   ```bash
   # NIM を使う（既に NIM が稼働していれば）
   export NIM_ENDPOINT=http://localhost:8000

   # もしくはローカル軽量 LLM サービスがある場合
   export LOCAL_LLM_ENDPOINT=http://localhost:9000
   ```
3. ビルドして起動
   ```bash
   docker compose -f deploy/local/docker-compose-lite.yml up --build -d
   ```
4. 動作確認
   - バックエンドが 8080 ポートで起動しているか確認
     ```bash
     curl -sS http://localhost:8080/health || true
     ```
   - 実際の RAG/LLM 結合は、`AIRA` のコンフィグ（`configs/`）で `NIM_ENDPOINT` または `LOCAL_LLM_ENDPOINT` を指すように設定してください。

注意点
- この Dockerfile は開発・検証用の簡易イメージ作成を目的としており、実際のプロダクション向けビルドや最適化（キャッシュ、マルチステージビルド、不要パッケージ除去等）は行っていません。
- `pyproject.toml` の要求 Python バージョンは 3.12 以上ですが、ホストが 3.9 の場合でも Docker イメージ側は Python 3.12 を使うため問題ありません（コンテナ内部で実行します）。

次の作業（私が行います）
- 動作確認後に、`LOCAL_LLM` を起動するための小さなサンプルコンテナ（litellm ベース）を追加し、エンドツーエンドの簡易 RAG パイプラインを流せるようにします。
