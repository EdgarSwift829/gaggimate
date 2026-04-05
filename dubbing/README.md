# YouTube 日本語吹き替えパイプライン

YouTubeのURLまたはローカル動画ファイルを入力すると、英語音声を日本語に吹き替えたMP4ファイルと字幕ファイル（SRT）を出力します。

## 処理フロー

```
入力（URL or ローカルファイル）
 │
 ├─ yt-dlp: 動画ダウンロード + 音声抽出（URLの場合）
 │  ffmpeg: 音声抽出（ローカルの場合）
 │
 ├─ Whisper: 英語の文字起こし（タイムスタンプ付き）
 │
 ├─ Claude API: セグメントごと日本語翻訳
 │
 ├─ Qwen3-TTS: 日本語音声合成（セグメントごと）
 │
 ├─ ffmpeg: 動画合成（音声差し替え + 字幕）
 │
 └─ output_[タイトル].mp4 + output_[タイトル].srt
```

## 必要環境

- Python 3.11以上
- ffmpeg（システムインストール）
- CUDA対応GPU（RTX 4080推奨）

## セットアップ

```bash
pip install yt-dlp openai-whisper anthropic torch transformers soundfile flask
```

環境変数にClaude APIキーを設定：

```bash
export ANTHROPIC_API_KEY=sk-ant-xxxx
```

> Qwen3-TTSモデル（約3GB）は初回起動時に自動ダウンロードされます。

## 使い方

### ブラウザ（Web UI）

```bash
cd dubbing
python web.py
```

`http://127.0.0.1:8080` でブラウザが自動で開きます。

| オプション | 説明 |
|---|---|
| `--port` | ポート番号（デフォルト: 8080） |
| `--host` | バインドアドレス（デフォルト: 127.0.0.1） |
| `--no-browser` | ブラウザ自動起動を無効化 |

### コマンドライン（CLI）

```bash
cd dubbing

# YouTubeのURLを吹き替え（元音声消し・ソフトサブ）
python dubbing.py "https://youtu.be/VM4eaf3sksE"

# 元音声を小さく残して字幕焼き込み
python dubbing.py "https://youtu.be/VM4eaf3sksE" --audio lower --subtitle burn

# ローカルファイルを吹き替え
python dubbing.py "./video.mp4" --audio mute

# 中間ファイルを残してデバッグ
python dubbing.py "https://youtu.be/xxxx" --keep-tmp
```

### CLIオプション一覧

| オプション | 値 | 説明 |
|---|---|---|
| `<入力>`（必須） | URL or ファイルパス | YouTubeのURL、またはローカル動画ファイル |
| `--audio` | `mute` / `lower` | 元音声の処理。mute: 消音、lower: -30dB（デフォルト: mute） |
| `--subtitle` | `burn` / `soft` | 字幕モード。burn: 焼き込み、soft: ソフトサブ（デフォルト: soft） |
| `--whisper-model` | モデル名 | Whisperモデル（デフォルト: large-v3） |
| `--tts-model` | モデル名 | Qwen3-TTSモデル（デフォルト: Qwen/Qwen3-TTS-12Hz-1.7B-Base） |
| `--output-dir` | ディレクトリパス | 出力先（デフォルト: ./output） |
| `--keep-tmp` | フラグ | 中間ファイルを保持する |

## ディレクトリ構成

```
dubbing/
 ├── dubbing.py          # CLIエントリーポイント
 ├── web.py              # Web UIサーバー
 ├── templates/
 │   └── index.html      # Web UIフロントエンド
 ├── pipeline/
 │   ├── downloader.py   # yt-dlp / ffmpeg ラッパー
 │   ├── transcriber.py  # Whisper ラッパー
 │   ├── translator.py   # Claude API 翻訳
 │   ├── tts.py          # Qwen3-TTS ラッパー
 │   └── composer.py     # ffmpeg 合成
 ├── output/             # 出力先（自動生成）
 └── tmp/                # 中間ファイル（処理後に自動削除）
```

## 技術スタック

| コンポーネント | 技術 | 備考 |
|---|---|---|
| 動画ダウンロード | yt-dlp | URLの場合のみ |
| 音声抽出 | ffmpeg | MP4 → MP3 |
| 文字起こし | openai-whisper | large-v3推奨 |
| 翻訳 | Claude API | claude-sonnet-4-5 |
| 音声合成 | Qwen3-TTS 1.7B | CUDA対応 |
| 動画合成 | ffmpeg | 音声差し替え・字幕 |
| Web UI | Flask + SSE | リアルタイム進捗表示 |

## 処理時間の目安（RTX 4080）

10分の動画の場合：

| ステップ | 時間 |
|---|---|
| Whisper 文字起こし | 約1分 |
| Claude API 翻訳 | 約1分 |
| Qwen3-TTS 音声合成 | 約5分 |
| ffmpeg 合成 | 約30秒 |
| **合計** | **約10分** |

## 制約・注意事項

- **リップシンク**: 翻訳で文字数・話速が変わるため口パクは合いません
- **著作権**: 個人利用・学習目的に限定。YouTubeの利用規約に注意してください
- **対応言語**: 入力は多言語対応（Whisper）、出力は日本語のみ
- **初回起動**: Qwen3-TTSモデル（約3GB）のダウンロードに数分かかります
