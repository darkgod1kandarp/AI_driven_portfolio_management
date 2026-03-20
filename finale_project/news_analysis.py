import os
import requests
import pandas as pd
import re
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import random
from openai import OpenAI
from tqdm import tqdm
import csv

from dotenv import load_dotenv
load_dotenv("C:\\Users\\KANDARP\\OneDrive\\Desktop\\Desertation  Project\\.env")

client = OpenAI(
    base_url = "https://integrate.api.nvidia.com/v1",
    api_key  = os.getenv("NV_API_KEY")
)

# ── Just the tickers — LLM figures out company names itself ───────
TICKERS = [
    "AAPL", "TSLA", "GME",  "AMC",  "AMZN",
    "MSFT", "NVDA", "AMD",  "PLTR", "BB",
    "GOOGL","META", "NFLX", "SPY",  "QQQ"
]

TICKERS_STR = ", ".join(TICKERS)


def _score_one_batch(batch, batch_index, delay=0.2, max_retries=4, base_backoff=1.0):
    """
    Score a batch of headlines.
    LLM identifies which company each headline is about
    purely from its own knowledge of the ticker symbols.

    Returns list of dicts:
      {"ticker": "AAPL", "score": 0.8}   — relevant
      {"ticker": "NONE", "score": -100}  — not relevant
    """
    headlines_str = "\n".join([
        f"{idx+1}. {str(txt)[:300]}"
        for idx, txt in enumerate(batch)
    ])

    prompt = f"""You are a financial news analyser.

You track these stock tickers: {TICKERS_STR}

For each headline below:
- If it is relevant to one of the tickers above, return its ticker symbol and a sentiment score
- If it is NOT relevant to any of the tickers above, return ticker "NONE" and score -100

Sentiment score rules:
  -1.0 = very negative / bearish
   0.0 = neutral
  +1.0 = very positive / bullish

Headlines:
{headlines_str}

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
                temperature=0.0
            )

            raw = response.choices[0].message.content.strip()
            raw = raw.replace("```json", "").replace("```", "").strip()

            results = json.loads(raw)

            if not isinstance(results, list):
                raise ValueError(f"Expected list got {type(results)}")

            # Validate each item
            validated = []
            for item in results:
                if isinstance(item, dict):
                    ticker = str(item.get("ticker", "NONE")).upper().strip()
                    score  = float(item.get("score", -100))

                    # Only accept tickers from our list or NONE
                    if ticker not in TICKERS:
                        ticker = "NONE"
                        score  = -100

                    # Clamp valid scores to -1 to +1
                    if score != -100:
                        score = max(-1.0, min(1.0, score))

                    validated.append({"ticker": ticker, "score": score})
                else:
                    validated.append({"ticker": "NONE", "score": -100})

            # Fix length mismatch
            if len(validated) != len(batch):
                padding   = [{"ticker": "NONE", "score": -100}] * len(batch)
                validated = (validated + padding)[:len(batch)]

            return batch_index, validated

        except json.JSONDecodeError as e:
            print(f"  Batch {batch_index} JSON error (attempt {attempt+1}): {e}")
            print(f"  Raw was: {raw[:200]}")
            if attempt < max_retries - 1:
                backoff = base_backoff * (2 ** attempt) + random.uniform(0, 0.5)
                time.sleep(backoff)
                continue

        except Exception as e:
            print(f"  Batch {batch_index} error (attempt {attempt+1}): {e}")
            if attempt < max_retries - 1:
                msg = str(e)
                is_504 = "504" in msg or "Gateway Timeout" in msg
                backoff = base_backoff * (2 ** attempt)
                if is_504:
                    backoff *= 2
                backoff += random.uniform(0, 0.5)
                time.sleep(backoff)
                continue

    # All attempts failed
    print(f"  Batch {batch_index} failed all {max_retries} attempts")
    return batch_index, [{"ticker": "NONE", "score": -100}] * len(batch)


def score_batch_bulk(texts, batch_size=100, delay=0.2, max_workers=10, max_retries=4, base_backoff=1.0):
    """
    Score texts in parallel batches.
    Returns list of dicts: [{"ticker": "AAPL", "score": 0.8}, ...]
    """
    total = len(texts)
    if total == 0:
        return []

    batch_size = max(1, int(batch_size))
    batches    = [
        (i // batch_size, texts[i:i + batch_size])
        for i in range(0, total, batch_size)
    ]

    results = [None] * len(batches)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {
            executor.submit(
                _score_one_batch, batch, batch_index, delay, max_retries, base_backoff
            ): batch_index
            for batch_index, batch in batches
        }
        for future in as_completed(future_map):
            batch_index, batch_results = future.result()
            results[batch_index] = batch_results

    all_results = []
    for r in results:
        if r is not None:
            all_results.extend(r)

    return all_results[:total]


def clean_text(text):
    text = str(text)
    text = re.sub(r"http\S+",          "",  text)
    text = re.sub(r"[^\w\s\.\,\!\?]", " ", text)
    text = re.sub(r"\s+",             " ", text)
    return text.strip()


def _tail_last_nonempty_line(path, chunk_size=8192):
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


def _find_resume_row_from_output(input_path, output_path):
    last_line = _tail_last_nonempty_line(output_path)
    print(f"Last line in output: {last_line}")
    if not last_line:
        return 0

    try:
        row = next(csv.reader([last_line]))
    except Exception:
        return 0

    if len(row) < 3:
        return 0

    extra_fields = row[2]
    url = None
    try:
        extra = json.loads(extra_fields)
        url = extra.get("url")
    except Exception:
        url = None

    if not url:
        return 0

    rows_seen = 0
    for chunk in pd.read_csv(input_path, chunksize=100000):
        direct_url_columns = [
            column for column in chunk.columns
            if str(column).strip().lower() in {"url", "link", "article_url", "source_url"}
        ]

        if direct_url_columns:
            for column in direct_url_columns:
                matches = chunk[column].astype(str).str.contains(re.escape(url), na=False)
                if matches.any():
                    first_match_position = matches.to_numpy().nonzero()[0][0]
                    return rows_seen + first_match_position + 1

        string_columns = chunk.select_dtypes(include=["object"]).columns
        for position, (_, chunk_row) in enumerate(chunk.iterrows()):
            for column in string_columns:
                value = chunk_row[column]
                if pd.isna(value):
                    continue

                value_str = str(value)
                if url in value_str:
                    return rows_seen + position + 1

                if value_str.startswith("{") and "url" in value_str:
                    try:
                        parsed = json.loads(value_str)
                    except Exception:
                        continue

                    if isinstance(parsed, dict) and parsed.get("url") == url:
                        return rows_seen + position + 1

        rows_seen += len(chunk)

    return 0


def get_total_rows(path):
    total_rows = 0   
    for chunk in pd.read_csv(path, chunksize=100000):
        total_rows += len(chunk)
    return total_rows


def analyze_csv(path, chunksize=100000):
    total_rows = 0
    missing_counts = {}
    average_text_length = 0.0
    non_null_text_rows = 0
    columns = None

    for chunk in pd.read_csv(path, chunksize=chunksize):
        if columns is None:
            columns = list(chunk.columns)
            missing_counts = {column: 0 for column in columns}

        total_rows += len(chunk)

        missing_in_chunk = chunk.isna().sum()
        for column, count in missing_in_chunk.items():
            missing_counts[column] = missing_counts.get(column, 0) + int(count)

        if "text" in chunk.columns:
            text_series = chunk["text"].dropna().astype(str)
            non_null_text_rows += len(text_series)
            average_text_length += float(text_series.str.len().sum())

    if non_null_text_rows > 0:
        average_text_length /= non_null_text_rows

    analysis = {
        "total_rows": total_rows,
        "columns": columns or [],
        "missing_counts": missing_counts,
        "average_text_length": round(average_text_length, 2),
    }

    print("=" * 55)
    print("INPUT DATA ANALYSIS")
    print(f"Total rows        : {analysis['total_rows']:,}")
    print(f"Columns           : {', '.join(analysis['columns'])}")
    print(f"Average text size : {analysis['average_text_length']}")
    print("Missing values per column:")
    for column, count in analysis["missing_counts"].items():
        print(f"  {column}: {count:,}")
    print("=" * 55)

    return analysis


def fetch_the_resumed_chunk(input_path, last_row_data, chunksize=5000):
    """
    Scan the input CSV to find the row matching last_row_data (parsed from
    the last line of the output file).  Returns the 1-based row number so
    processing can resume from the next row.

    last_row_data should be a list of field values (from csv.reader) whose
    third element is an extra_fields JSON string that may contain a "url".
    """
    if not last_row_data or len(last_row_data) < 3:
        return 0

    # Try to extract the URL from the extra_fields JSON column
    url = None
    try:
        extra = json.loads(last_row_data[2])
        url = extra.get("url")
        print(f"Extracted URL from last output line: {url}")
    except Exception:
        pass

    if not url:
        return 0

    total_read = 0
    for chunk in pd.read_csv(input_path, chunksize=chunksize):
        for position, row in enumerate(chunk.itertuples(index=False)):
            extra = row[2]
            if isinstance(extra, str) and extra.strip().startswith("{") and "url" in extra:
                try:
                    extra_json = json.loads(extra)
                except Exception:
                    continue
                if extra_json.get("url") == url:
                    match_row = total_read + position + 1
                    print("I found the matching URL in the input CSV!")
                    print("Total rows read to find match:", match_row)
                    # Round of next chunk boundary for cleaner resume point 
                    match_row = ((match_row + chunksize - 1) // chunksize) * chunksize
                    return match_row 

        total_read += len(chunk)

    return 0
        
def _get_last_processed_row(output_path):  
    last_line = _tail_last_nonempty_line(output_path)
    print(f"Last line in output: {last_line}")
    if not last_line:
        return 0

    try:
        row = next(csv.reader([last_line]))
        return row if row else 0
    except Exception:
        return 0


def process_and_score_csv(input_path, output_path, chunk_size=5000, resume=False):
    """
    Reads news CSV in chunks, scores each article,
    keeps only articles relevant to your tracked tickers.
    """
    first_write = True
    total_read  = 0
    total_kept  = 0
    

    print(f"Tracking tickers : {TICKERS_STR}")
    print(f"Input file       : {input_path}")
    print(f"Output file      : {output_path}\n")

    resume_row = 0
    mode = "w"
    if resume and os.path.exists(output_path):
        last_row_data = _get_last_processed_row(output_path)
        if last_row_data and last_row_data != 0:
            resume_row = fetch_the_resumed_chunk(input_path, last_row_data)
            if resume_row > 0:
                mode = "a"
                first_write = False
                print(f"Resuming from row {resume_row:,}")

    print(f"Resume row       : {resume_row}")

    total_input_rows = get_total_rows(input_path)
    total_chunks = None
    if total_input_rows > 0:
        total_chunks = (total_input_rows + chunk_size - 1) // chunk_size

    for chunk in tqdm(
        pd.read_csv(input_path, chunksize=chunk_size),
        desc="Processing chunks",
        total=total_chunks
    ):
        if resume and resume_row and resume_row > total_read:
            to_skip = min(resume_row - total_read, len(chunk))
            if to_skip >= len(chunk):
                total_read += len(chunk)
                continue
            chunk = chunk.iloc[to_skip:].copy()
            total_read += to_skip

        chunk["text"] = chunk["text"].apply(clean_text)
        total_read   += len(chunk)

        print(f"Scoring {len(chunk)} rows "
              f"(total read so far: {total_read:,})...")

        # Score with LLM — returns ticker + sentiment per article
        llm_results = score_batch_bulk(
            chunk["text"].tolist(),
            batch_size  = 20,
            delay       = 0.1,
            max_workers = 10
        )

        # Add ticker and score columns
        chunk["ticker"]          = [r["ticker"] for r in llm_results]
        chunk["sentiment_score"] = [r["score"]  for r in llm_results]

        # Keep only relevant articles
        relevant = chunk[
            (chunk["ticker"]          != "NONE") &
            (chunk["sentiment_score"] != -100)
        ].copy()

        total_kept += len(relevant)

        # Save to output
        if len(relevant) > 0:
            relevant.to_csv(
                output_path,
                mode   = "a" if not first_write else "w",
                header = first_write,
                index  = False
            )
            first_write = False

        print(f"  Kept {len(relevant):,} relevant  |  "
              f"Discarded {len(chunk) - len(relevant):,}  |  "
              f"Total saved: {total_kept:,}\n")

    # Final summary
    print("=" * 55)
    print(f"DONE")
    print(f"Total articles read    : {total_read:,}")
    print(f"Relevant articles kept : {total_kept:,}")
    print(f"Discarded              : {total_read - total_kept:,}")

    if not first_write:
        df = pd.read_csv(output_path)
        print(f"\nArticles per ticker:")
        print(df["ticker"].value_counts().to_string())
        print(f"\nSentiment stats per ticker:")
        print(df.groupby("ticker")["sentiment_score"]
                .describe()
                .round(3)
                .to_string())


if __name__ == "__main__":
    process_and_score_csv(
        input_path  = r"C:\Users\KANDARP\Downloads\financial_news_2015_2020.csv",
        output_path = r"C:\Users\KANDARP\OneDrive\Desktop\Desertation  Project\finale_project\meta_data\financial_news_2015_2020_with_sentiment.csv",
        resume=True
    )
