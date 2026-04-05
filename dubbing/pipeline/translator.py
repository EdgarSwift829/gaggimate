"""Claude API wrapper for translating segments to Japanese."""

import json
import logging

import anthropic

logger = logging.getLogger(__name__)

BATCH_SIZE = 50  # segments per API call

SYSTEM_PROMPT = """\
あなたは英語から日本語への翻訳者です。動画の吹き替え用に翻訳してください。

ルール:
- 自然な日本語にすること（直訳ではなく意訳を優先）
- 技術用語（API, GPU, Python等）はそのままカタカナや英語で残す
- 話速を考慮し、元のセグメントの長さに収まる程度の文字数にする
- 各セグメントのstart/endタイムスタンプはそのまま保持する
- JSON形式で返すこと"""

USER_PROMPT_TEMPLATE = """\
以下の英語セグメントを日本語に翻訳してください。
JSONの配列として返してください。各要素は {{"start": float, "end": float, "text": "日本語テキスト"}} の形式です。

入力セグメント:
{segments_json}"""


def translate_segments(
    segments: list[dict],
    model: str = "claude-sonnet-4-5-20250514",
) -> list[dict]:
    """Translate segments from English to Japanese using Claude API.

    Args:
        segments: List of {"start", "end", "text"} dicts.
        model: Claude model to use.

    Returns:
        List of translated segments with same structure.
    """
    client = anthropic.Anthropic()
    translated = []

    # Process in batches to reduce API calls
    for i in range(0, len(segments), BATCH_SIZE):
        batch = segments[i:i + BATCH_SIZE]
        batch_num = i // BATCH_SIZE + 1
        total_batches = (len(segments) + BATCH_SIZE - 1) // BATCH_SIZE
        logger.info("Translating batch %d/%d (%d segments)", batch_num, total_batches, len(batch))

        segments_json = json.dumps(batch, ensure_ascii=False, indent=2)
        user_prompt = USER_PROMPT_TEMPLATE.format(segments_json=segments_json)

        response = client.messages.create(
            model=model,
            max_tokens=4096,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_prompt}],
        )

        response_text = response.content[0].text

        # Extract JSON from response (handle markdown code blocks)
        json_text = response_text
        if "```" in json_text:
            # Extract content between code block markers
            lines = json_text.split("\n")
            in_block = False
            block_lines = []
            for line in lines:
                if line.strip().startswith("```"):
                    if in_block:
                        break
                    in_block = True
                    continue
                if in_block:
                    block_lines.append(line)
            json_text = "\n".join(block_lines)

        batch_translated = json.loads(json_text)
        translated.extend(batch_translated)

    logger.info("Translated %d segments", len(translated))
    return translated
