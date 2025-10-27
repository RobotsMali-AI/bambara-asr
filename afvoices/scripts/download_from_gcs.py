import argparse
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from google.cloud import storage
import requests
import urllib.parse
import threading

# Thread-local variable to store one client per thread
thread_local = threading.local()

def get_client():
    if not hasattr(thread_local, "client"):
        thread_local.client = storage.Client()
    return thread_local.client

def normalize_path(path: str) -> str:
    """Normalize different Firebase/GCS audio paths into a consistent form."""
    # Handle Firebase public URLs
    if "firebasestorage.googleapis.com" in path:
        match = re.search(r"/o/(.+)\?", path)
        if match:
            decoded = urllib.parse.unquote(match.group(1))
            return decoded.strip()

    # Handle GCS paths like gs://africa-voice-mali.firebasestorage.app/assist1/...
    match = re.search(r'africa-voice-mali\.firebasestorage\.app/(.+)', path)
    if match:
        return match.group(1).strip()

    # Otherwise, just strip and return
    return path.strip()

def sanitize_filename(name):
    """Remove invalid filename characters."""
    return re.sub(r'[^a-zA-Z0-9_\-\.]', '_', name)

def unique_path(base_path):
    """Generate a unique path if file already exists."""
    if not os.path.exists(base_path):
        return base_path
    print(f"Found existing file: {base_path}, generating unique name...")
    base, ext = os.path.splitext(base_path)
    suffix = 'a'
    while os.path.exists(f"{base}_{suffix}{ext}"):
        suffix = chr(ord(suffix) + 1)
    return f"{base}_{suffix}{ext}"

def download_from_gcs(client, uri, out_path):
    """Download a file from Google Cloud Storage."""
    match = re.match(r'gs://([^/]+)/(.+)', uri)
    if not match:
        raise ValueError(f"Invalid GCS URI: {uri}")
    bucket_name, blob_name = match.groups()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.download_to_filename(out_path)

def download_from_http(url, out_path):
    """Download a file from an HTTP(S) URL."""
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(out_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)

def download_audio(entry, out_dir):
    """Download a single entry and return updated entry."""
    uri = entry["audio_filepath"]
    duration = entry.get("duration", None)
    text_key = "text" if "text" in entry else "semi-label"
    text = entry.get("text", "")

    filename = sanitize_filename(normalize_path(uri))
    local_path = os.path.join(out_dir, filename)
    local_path = unique_path(local_path)

    try:
        if uri.startswith("gs://"):
            client = get_client()
            download_from_gcs(client, uri, local_path)
        elif uri.startswith("http"):
            download_from_http(uri, local_path)
        else:
            raise ValueError(f"Unsupported URI: {uri}")
    except Exception as e:
        print(f"❌ Failed to download {uri}: {e}")
        return None

    return {text_key: text, "duration": duration, "audio_filepath": local_path}

def main(manifest_path, out_dir, out_manifest, max_workers):
    os.makedirs(out_dir, exist_ok=True)
    entries = [json.loads(line) for line in open(manifest_path, "r", encoding="utf-8")]

    updated_entries = []
    with ThreadPoolExecutor(max_workers=int(max_workers)) as executor:
        futures = {executor.submit(download_audio, e, out_dir): e for e in entries}
        for future in tqdm(as_completed(futures), total=len(entries), desc="Downloading"):
            result = future.result()
            if result:
                updated_entries.append(result)

    with open(out_manifest, "w", encoding="utf-8") as f:
        for e in updated_entries:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")

    print(f"\n✅ Download complete: {len(updated_entries)} files saved to '{out_dir}'")
    print(f"📝 Updated manifest written to: {out_manifest}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download audios from a manifest.")
    parser.add_argument("manifest", help="Path to input JSONL manifest")
    parser.add_argument("--max-workers", help="Max number of parallel worker processes", default=os.cpu_count())
    parser.add_argument("--out-dir", default="audios", help="Folder to save audios")
    parser.add_argument("--out-manifest", default="original-full-manifest.jsonl", help="Output manifest path")
    args = parser.parse_args()

    main(args.manifest, args.out_dir, args.out_manifest, args.max_workers)
