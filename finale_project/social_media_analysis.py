"""
Sentiment-score the merged social-media CSV (Twitter + Reddit)
using the same LLM pipeline as news_analysis.py.

Input  : social_media_2015_2020.csv   (columns: post_id, author, datetime,
         text, num_comments, retweet_num, like_num, source, platform, trading_date)
Output : social_media_2015_2020_with_sentiment.csv
"""

import os
import pandas as pd
import re
import json
import time
import csv
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
from tqdm import tqdm
import os 
from dotenv import load_dotenv
load_dotenv("C:\\Users\\KANDARP\\OneDrive\\Desktop\\Desertation  Project\\.env")

# ── LLM client ───────────────────────────────────────────────────
client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=os.getenv("NV_API_KEY"),
)

# ── Tickers to track (same list as news_analysis.py) ─────────────
TICKERS = [
    "AAPL", "TSLA", "GME",  "AMC",  "AMZN",
    "MSFT", "NVDA", "AMD",  "PLTR", "BB",
    "GOOGL","META", "NFLX", "SPY",  "QQQ",
]
TICKERS_STR = ", ".join(TICKERS)


# ═══════════════════════════════════════════════════════════════════
#  LLM scoring  (identical logic to news_analysis.py)
# ═══════════════════════════════════════════════════════════════════

def _score_one_batch(batch, batch_index, delay=0.2, max_retries=4, base_backoff=1.0):
    """
    Send a batch of social-media posts to the LLM.
    Returns (batch_index, [{"ticker": …, "score": …}, …])
    """
    posts_str = "\n".join(
        f"{idx+1}. {str(txt)[:300]}" for idx, txt in enumerate(batch)
    )

    prompt = f"""You are a financial social-media analyst.

You track these stock tickers: {TICKERS_STR}

For each post below:
- If it is relevant to one of the tickers above, return the ticker symbol and a sentiment score.
- If it is NOT relevant to any ticker, return ticker "NONE" and score -100.

Sentiment score rules:
  -1.0 = very negative / bearish
   0.0 = neutral
  +1.0 = very positive / bullish

Posts:
{posts_str}

Return ONLY a JSON array of exactly {len(batch)} objects.
Format:
[
  {{"ticker": "AAPL", "score": 0.8}},
  {{"ticker": "NONE", "score": -100}},
  {{"ticker": "TSLA", "score": -0.6}}
]
Just the array. No explanation. No markdown."""

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="mistralai/devstral-2-123b-instruct-2512",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=50000,
                temperature=0.0,
            )

            raw = response.choices[0].message.content.strip()
            raw = raw.replace("```json", "").replace("```", "").strip()
            results = json.loads(raw)

            if not isinstance(results, list):
                raise ValueError(f"Expected list, got {type(results)}")

            validated = []
            for item in results:
                if isinstance(item, dict):
                    ticker = str(item.get("ticker", "NONE")).upper().strip()
                    score  = float(item.get("score", -100))
                    if ticker not in TICKERS:
                        ticker, score = "NONE", -100
                    if score != -100:
                        score = max(-1.0, min(1.0, score))
                    validated.append({"ticker": ticker, "score": score})
                else:
                    validated.append({"ticker": "NONE", "score": -100})

            # Fix length mismatch
            if len(validated) != len(batch):
                padding   = [{"ticker": "NONE", "score": -100}] * len(batch)
                validated = (validated + padding)[: len(batch)]

            return batch_index, validated

        except json.JSONDecodeError as e:
            print(f"  Batch {batch_index} JSON error (attempt {attempt+1}): {e}")
            if attempt < max_retries - 1:
                time.sleep(base_backoff * (2 ** attempt) + random.uniform(0, 0.5))
                continue

        except Exception as e:
            os._exit(0)
            print(f"  Batch {batch_index} error (attempt {attempt+1}): {e}")
            if attempt < max_retries - 1:
                backoff = base_backoff * (2 ** attempt)
                if "504" in str(e) or "Gateway Timeout" in str(e):
                    backoff *= 2
                time.sleep(backoff + random.uniform(0, 0.5))
                continue
            

    print(f"  Batch {batch_index} failed all {max_retries} attempts")
    return batch_index, [{"ticker": "NONE", "score": -100}] * len(batch)


def score_batch_bulk(texts, batch_size=100, delay=0.2, max_workers=10,
                     max_retries=4, base_backoff=1.0):
    total = len(texts)
    if total == 0:
        return []

    batch_size = max(1, int(batch_size))
    batches = [
        (i // batch_size, texts[i : i + batch_size])
        for i in range(0, total, batch_size)
    ]

    results = [None] * len(batches)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                _score_one_batch, batch, idx, delay, max_retries, base_backoff
            ): idx
            for idx, batch in batches
        }
        for future in as_completed(futures):
            batch_index, batch_results = future.result()
            results[batch_index] = batch_results

    flat = []
    for r in results:
        if r is not None:
            flat.extend(r)
    return flat[:total]


# ═══════════════════════════════════════════════════════════════════
#  Text cleaning
# ═══════════════════════════════════════════════════════════════════

def clean_text(text):
    text = str(text)
    text = re.sub(r"http\S+", "", text)                # URLs
    text = re.sub(r"@\w+", "", text)                    # @mentions
    text = re.sub(r"#", "", text)                       # hashtag symbols (keep word)
    text = re.sub(r"[^\w\s\.\,\!\?]", " ", text)       # special chars
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# ═══════════════════════════════════════════════════════════════════
#  Resume helpers  (keyed on post_id instead of URL)
# ═══════════════════════════════════════════════════════════════════

def _tail_last_nonempty_line(path, chunk_size=8192):
    """Read the last non-empty line of a file without loading it all."""
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        f.seek(0, os.SEEK_END)
        pos = f.tell()
        if pos == 0:
            return None
        data = b""
        while pos > 0:
            read_size = min(chunk_size, pos)
            pos -= read_size
            f.seek(pos)
            data = f.read(read_size) + data
            if b"\n" in data:
                break
        lines = [ln for ln in data.splitlines() if ln.strip()]
        if not lines:
            return None
        return lines[-1].decode("utf-8", errors="ignore")


def _get_last_post_id(output_path):
    """Return the post_id of the last row in the output CSV."""
    last_line = _tail_last_nonempty_line(output_path)
    if not last_line:
        return None
    try:
        row = next(csv.reader([last_line]))
        # post_id is the first column
        return str(row[1]).strip() if row else None
    except Exception:
        return None


def _find_resume_row(input_path, last_post_id, chunksize=50000):
    """
    Scan the input CSV to find the 1-based row number of `last_post_id`
    so we can skip everything up to (and including) that row.
    """
    if not last_post_id:
        return 0

    rows_seen = 0
    for chunk in pd.read_csv(input_path, chunksize=chunksize, dtype={"post_id": str}):
        matches = chunk["post_id"].astype(str).str.strip() == last_post_id
        if matches.any():
            first_match = matches.to_numpy().nonzero()[0][0]
            resume_at = rows_seen + first_match + 1   # +1 → skip this row
            print(f"Found last post_id '{last_post_id}' at input row {resume_at:,}")
            return resume_at
        rows_seen += len(chunk)
    

    print(f"Warning: post_id '{last_post_id}' not found in input — starting from 0")
    return 0


def get_total_rows(path, chunksize=100000):
    total = 0
    for chunk in pd.read_csv(path, chunksize=chunksize):
        total += len(chunk)
    return total


# ═══════════════════════════════════════════════════════════════════
#  Main processing pipeline
# ═══════════════════════════════════════════════════════════════════

def process_social_media_csv(input_path, output_path,
                             chunk_size=5000, resume=True):
    """
    Read the merged social-media CSV in chunks, score each post via LLM,
    keep only posts relevant to the tracked tickers, and save incrementally.
    """
    first_write = True
    total_read  = 0
    total_kept  = 0
    resume_row  = 0

    print(f"Tracking tickers : {TICKERS_STR}")
    print(f"Input file       : {input_path}")
    print(f"Output file      : {output_path}\n")

    # ── Resume logic ──────────────────────────────────────────────
    if resume and os.path.exists(output_path):
        last_post_id = _get_last_post_id(output_path)
        if last_post_id:
            
            resume_row = _find_resume_row(input_path, last_post_id)
            if resume_row > 0:
                first_write = False
                print(f"Resuming after row {resume_row:,}\n")

    print(f"Resume row : {resume_row}")

    # ── Count total rows for progress bar ─────────────────────────
    total_input_rows = get_total_rows(input_path)
    total_chunks = (
        (total_input_rows + chunk_size - 1) // chunk_size
        if total_input_rows > 0 else None
    )

    # ── Chunked processing ────────────────────────────────────────
    for chunk in tqdm(
        pd.read_csv(input_path, chunksize=chunk_size),
        desc="Processing chunks",
        total=total_chunks,
    ):
        # Skip rows we already processed
        if resume_row and resume_row > total_read:
            to_skip = min(resume_row - total_read, len(chunk))
            if to_skip >= len(chunk):
                total_read += len(chunk)
                continue
            chunk = chunk.iloc[to_skip:].copy()
            total_read += to_skip

        chunk["text"] = chunk["text"].apply(clean_text)
        total_read += len(chunk)

        print(f"Scoring {len(chunk):,} rows  "
              f"(total read so far: {total_read:,}) …")

        # ── LLM scoring ──────────────────────────────────────────
        llm_results = score_batch_bulk(
            chunk["text"].tolist(),
            batch_size=20,
            delay=0.1,
            max_workers=10,
        )

        chunk["ticker"]          = [r["ticker"] for r in llm_results]
        chunk["sentiment_score"] = [r["score"]  for r in llm_results]

        # ── Keep only relevant posts ─────────────────────────────
        relevant = chunk[
            (chunk["ticker"] != "NONE") &
            (chunk["sentiment_score"] != -100)
        ].copy()

        total_kept += len(relevant)

        if len(relevant) > 0:
            relevant.to_csv(
                output_path,
                mode="a" if not first_write else "w",
                header=first_write,
                index=False,
            )
            first_write = False

        print(f"  Kept {len(relevant):,} relevant  |  "
              f"Discarded {len(chunk) - len(relevant):,}  |  "
              f"Total saved: {total_kept:,}\n")

    # ── Final summary ─────────────────────────────────────────────
    print("=" * 55)
    print("DONE")
    print(f"Total posts read       : {total_read:,}")
    print(f"Relevant posts kept    : {total_kept:,}")
    print(f"Discarded              : {total_read - total_kept:,}")

    if not first_write:
        df = pd.read_csv(output_path)
        print(f"\nPosts per ticker:")
        print(df["ticker"].value_counts().to_string())
        print(f"\nPosts per platform:")
        print(df["platform"].value_counts().to_string())
        print(f"\nSentiment stats per ticker:")
        print(
            df.groupby("ticker")["sentiment_score"]
            .describe()
            .round(3)
            .to_string()
        )
    print("=" * 55)


# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    process_social_media_csv(
        input_path=r"C:\Users\KANDARP\OneDrive\Desktop\Desertation  Project\finale_project\meta_data\social_media_2015_2020.csv",
        output_path=r"C:\Users\KANDARP\OneDrive\Desktop\Desertation  Project\finale_project\meta_data\social_media_2015_2020_with_sentiment.csv",
        resume=True,
    )
