#!/usr/bin/env python3

"""
Synchronize Hugging Face model/data cards with local documentation.

Examples
--------
Pull one model card:
    python scripts/hf_cards.py pull model RobotsMali/soloni-114m-tdt-ctc-v1

Pull one dataset card:
    python scripts/hf_cards.py pull dataset RobotsMali/jeli-asr

Pull every model card belonging to RobotsMali:
    python scripts/hf_cards.py pull model RobotsMali --all

Pull every dataset card belonging to diarray:
    python scripts/hf_cards.py pull dataset diarray --all

Push a model card:
    python scripts/hf_cards.py push model RobotsMali/soloni-114m-tdt-ctc-v1

Push a dataset card:
    python scripts/hf_cards.py push dataset RobotsMali/jeli-asr
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from typing import Any

import yaml
from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.errors import EntryNotFoundError, HfHubHTTPError


README_FILENAME = "README.md"
MANIFEST_RELATIVE_PATH = Path("docs/huggingface.yaml")

LOCAL_CARD_DIRS = {
    "model": Path("docs/models"),
    "dataset": Path("docs/data"),
}


def find_repo_root() -> Path:
    """
    Find the Git repository containing this script.

    This lets the command work regardless of the current working directory,
    provided the script itself lives somewhere inside the repository.
    """
    script_dir = Path(__file__).resolve().parent

    for candidate in (script_dir, *script_dir.parents):
        if (candidate / ".git").exists():
            return candidate

    # Useful fallback if the script is being tested before the repo is
    # initialized with Git.
    return Path.cwd().resolve()


REPO_ROOT = find_repo_root()
MANIFEST_PATH = REPO_ROOT / MANIFEST_RELATIVE_PATH


def load_manifest() -> dict[str, Any]:
    """Load docs/huggingface.yaml or initialize an empty manifest."""
    if not MANIFEST_PATH.exists():
        return {
            "version": 1,
            "cards": [],
        }

    with MANIFEST_PATH.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    data.setdefault("version", 1)
    data.setdefault("cards", [])

    if not isinstance(data["cards"], list):
        raise ValueError(
            f"{MANIFEST_PATH} is invalid: 'cards' must be a list."
        )

    return data


def save_manifest(manifest: dict[str, Any]) -> None:
    """Write the manifest in a stable, human-readable order."""
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)

    manifest["cards"] = sorted(
        manifest["cards"],
        key=lambda item: (
            item.get("repo_type", ""),
            item.get("repo_id", "").lower(),
        ),
    )

    with MANIFEST_PATH.open("w", encoding="utf-8") as f:
        yaml.safe_dump(
            manifest,
            f,
            sort_keys=False,
            allow_unicode=True,
            default_flow_style=False,
        )


def validate_repo_id(repo_id: str) -> tuple[str, str]:
    """
    Validate and split a full Hugging Face repository ID.

    Returns
    -------
    owner, repo_name
    """
    parts = repo_id.split("/", maxsplit=1)

    if len(parts) != 2 or not all(parts):
        raise ValueError(
            f"Expected a full Hugging Face repository ID like "
            f"'RobotsMali/my-model', got: {repo_id!r}"
        )

    return parts[0], parts[1]


def find_manifest_entry(
    manifest: dict[str, Any],
    repo_type: str,
    repo_id: str,
) -> dict[str, Any] | None:
    """Find the manifest entry for a Hugging Face repository."""
    for card in manifest["cards"]:
        if (
            card.get("repo_type") == repo_type
            and card.get("repo_id") == repo_id
        ):
            return card

    return None


def default_local_path(repo_type: str, repo_id: str) -> Path:
    """
    Generate a collision-safe local path for a new card.

    Example
    -------
    RobotsMali/jeli-asr
        -> docs/data/RobotsMali/jeli-asr.md
    """
    owner, repo_name = validate_repo_id(repo_id)

    return LOCAL_CARD_DIRS[repo_type] / owner / f"{repo_name}.md"


def get_local_path(
    manifest: dict[str, Any],
    repo_type: str,
    repo_id: str,
) -> Path:
    """
    Return the configured local path if the card already exists in the
    manifest; otherwise use the default layout.
    """
    entry = find_manifest_entry(manifest, repo_type, repo_id)

    if entry is not None:
        return Path(entry["local"])

    return default_local_path(repo_type, repo_id)


def upsert_manifest_entry(
    manifest: dict[str, Any],
    repo_type: str,
    repo_id: str,
    local_path: Path,
) -> bool:
    """
    Add/update a manifest entry.

    Returns True if a new entry was added, False if an existing entry
    was updated.
    """
    entry = find_manifest_entry(manifest, repo_type, repo_id)

    relative_path = local_path.as_posix()

    if entry is not None:
        entry["local"] = relative_path
        return False

    manifest["cards"].append(
        {
            "repo_id": repo_id,
            "repo_type": repo_type,
            "local": relative_path,
        }
    )

    return True


def pull_card(
    api: HfApi,
    repo_type: str,
    repo_id: str,
    manifest: dict[str, Any],
) -> bool:
    """
    Download only README.md from a Hugging Face repository.

    Returns True on success.
    """
    local_relative = get_local_path(manifest, repo_type, repo_id)
    local_absolute = REPO_ROOT / local_relative

    try:
        cached_readme = hf_hub_download(
            repo_id=repo_id,
            repo_type=repo_type,
            filename=README_FILENAME,
        )
    except EntryNotFoundError:
        print(
            f"SKIP  {repo_id}: no {README_FILENAME} found",
            file=sys.stderr,
        )
        return False
    except HfHubHTTPError as exc:
        print(
            f"ERROR {repo_id}: {exc}",
            file=sys.stderr,
        )
        return False

    local_absolute.parent.mkdir(parents=True, exist_ok=True)

    # hf_hub_download returns a file in Hugging Face's cache.
    # Never edit the cached copy directly.
    shutil.copyfile(cached_readme, local_absolute)

    is_new = upsert_manifest_entry(
        manifest=manifest,
        repo_type=repo_type,
        repo_id=repo_id,
        local_path=local_relative,
    )

    # Persist immediately so a later failure during --all does not lose
    # successful mappings.
    save_manifest(manifest)

    status = "NEW " if is_new else "PULL"

    print(
        f"{status}  {repo_id}\n"
        f"      -> {local_relative}"
    )

    return True


def list_owner_repositories(
    api: HfApi,
    repo_type: str,
    owner: str,
) -> list[str]:
    """Return all model or dataset repository IDs belonging to an owner."""
    if "/" in owner:
        raise ValueError(
            "--all expects an owner such as 'RobotsMali' or 'diarray', "
            "not a full repository ID."
        )

    if repo_type == "model":
        repos = api.list_models(author=owner)
    else:
        repos = api.list_datasets(author=owner)

    return sorted(repo.id for repo in repos)


def pull_all(
    api: HfApi,
    repo_type: str,
    owner: str,
    manifest: dict[str, Any],
) -> int:
    """Pull README.md from every repo of the selected type for an owner."""
    repo_ids = list_owner_repositories(api, repo_type, owner)

    if not repo_ids:
        print(f"No {repo_type} repositories found for {owner}.")
        return 0

    print(
        f"Found {len(repo_ids)} {repo_type} "
        f"{'repository' if len(repo_ids) == 1 else 'repositories'} "
        f"for {owner}.\n"
    )

    successful = 0
    skipped = 0

    for repo_id in repo_ids:
        if pull_card(
            api=api,
            repo_type=repo_type,
            repo_id=repo_id,
            manifest=manifest,
        ):
            successful += 1
        else:
            skipped += 1

    print(
        "\nDone.\n"
        f"  Pulled: {successful}\n"
        f"  Skipped/failed: {skipped}\n"
        f"  Manifest: {MANIFEST_RELATIVE_PATH}"
    )

    return 0 if skipped == 0 else 1


def push_card(
    api: HfApi,
    repo_type: str,
    repo_id: str,
    manifest: dict[str, Any],
    commit_message: str | None = None,
) -> int:
    """Upload a local card as README.md without cloning the HF repository."""
    validate_repo_id(repo_id)

    entry = find_manifest_entry(
        manifest=manifest,
        repo_type=repo_type,
        repo_id=repo_id,
    )

    if entry is None:
        print(
            f"ERROR: {repo_id} is not registered in "
            f"{MANIFEST_RELATIVE_PATH}.\n"
            f"Pull it first:\n\n"
            f"  python {Path(__file__).name} pull "
            f"{repo_type} {repo_id}",
            file=sys.stderr,
        )
        return 1

    local_relative = Path(entry["local"])
    local_absolute = REPO_ROOT / local_relative

    if not local_absolute.exists():
        print(
            f"ERROR: Local card does not exist:\n"
            f"  {local_relative}",
            file=sys.stderr,
        )
        return 1

    if commit_message is None:
        commit_message = (
            "Update model card"
            if repo_type == "model"
            else "Update dataset card"
        )

    try:
        commit = api.upload_file(
            path_or_fileobj=str(local_absolute),
            path_in_repo=README_FILENAME,
            repo_id=repo_id,
            repo_type=repo_type,
            commit_message=commit_message,
        )
    except HfHubHTTPError as exc:
        print(
            f"ERROR: Could not push {repo_id}: {exc}",
            file=sys.stderr,
        )
        return 1

    print(
        f"PUSH  {local_relative}\n"
        f"      -> {repo_id}/{README_FILENAME}"
    )

    # CommitInfo currently exposes commit_url, but don't make successful
    # uploads depend on that implementation detail.
    commit_url = getattr(commit, "commit_url", None)
    if commit_url:
        print(f"      {commit_url}")

    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Synchronize Hugging Face model and dataset cards with "
            "docs/ in the current Git repository."
        )
    )

    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
    )

    # pull
    pull_parser = subparsers.add_parser(
        "pull",
        help="Download model/data cards from Hugging Face.",
    )
    pull_parser.add_argument(
        "repo_type",
        choices=("model", "dataset"),
        help="Type of Hugging Face repository.",
    )
    pull_parser.add_argument(
        "target",
        help=(
            "Full repo ID for one card, e.g. RobotsMali/jeli-asr. "
            "With --all, this is the owner, e.g. RobotsMali."
        ),
    )
    pull_parser.add_argument(
        "--all",
        action="store_true",
        help="Pull every repository of this type belonging to TARGET.",
    )

    # push
    push_parser = subparsers.add_parser(
        "push",
        help="Upload a local model/data card to Hugging Face.",
    )
    push_parser.add_argument(
        "repo_type",
        choices=("model", "dataset"),
        help="Type of Hugging Face repository.",
    )
    push_parser.add_argument(
        "repo_id",
        help="Full repo ID, e.g. RobotsMali/jeli-asr.",
    )
    push_parser.add_argument(
        "-m",
        "--message",
        dest="commit_message",
        help="Hugging Face commit message.",
    )

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    manifest = load_manifest()
    api = HfApi()

    if args.command == "pull":
        if args.all:
            return pull_all(
                api=api,
                repo_type=args.repo_type,
                owner=args.target,
                manifest=manifest,
            )

        validate_repo_id(args.target)

        success = pull_card(
            api=api,
            repo_type=args.repo_type,
            repo_id=args.target,
            manifest=manifest,
        )

        return 0 if success else 1

    if args.command == "push":
        return push_card(
            api=api,
            repo_type=args.repo_type,
            repo_id=args.repo_id,
            manifest=manifest,
            commit_message=args.commit_message,
        )

    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
