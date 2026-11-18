"""Upload a first frame and submit the saved API graph to a running ComfyUI."""
import argparse
import json
import time
import uuid
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]


def request_json(base, endpoint, payload=None, content_type="application/json"):
    data = json.dumps(payload).encode() if isinstance(payload, dict) else payload
    headers = {"Content-Type": content_type} if data is not None else {}
    try:
        with urlopen(Request(base + endpoint, data=data, headers=headers), timeout=60) as response:
            return json.load(response)
    except HTTPError as exc:
        raise RuntimeError(f"HTTP {exc.code}: {exc.read().decode(errors='replace')}") from exc


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8188")
    parser.add_argument("--image", type=Path, default=ROOT / "assets/shot01-younger.png")
    parser.add_argument("--workflow", type=Path, default=ROOT / "workflows/shot01-api.json")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs")
    parser.add_argument("--seed", type=int, help="Override seed; default keeps workflow seeds")
    parser.add_argument("--wait", action="store_true", help="Wait and download saved output")
    parser.add_argument("--timeout", type=float, default=14400, help="Wait limit in seconds")
    parser.add_argument("--dry-run", action="store_true", help="Validate local inputs; no API calls")
    args = parser.parse_args()
    graph = json.loads(args.workflow.read_text())
    if not args.image.is_file():
        parser.error(f"image not found: {args.image}")
    for item in graph.values():
        if item["class_type"] == "KSamplerAdvanced" and args.seed is not None:
            item["inputs"]["noise_seed"] = args.seed
    if args.dry_run:
        seeds = sorted({item["inputs"]["noise_seed"] for item in graph.values()
                        if item["class_type"] == "KSamplerAdvanced"})
        print(f"OK: {len(graph)} nodes, image {args.image.name}, seeds {seeds}")
        return
    base = args.url.rstrip("/")
    boundary = uuid.uuid4().hex
    body = (
        f'--{boundary}\r\nContent-Disposition: form-data; name="image"; filename="{args.image.name}"\r\n'
        'Content-Type: image/png\r\n\r\n'
    ).encode() + args.image.read_bytes() + f"\r\n--{boundary}--\r\n".encode()
    uploaded = request_json(base, "/upload/image", body, f"multipart/form-data; boundary={boundary}")
    image_name = "/".join(filter(None, [uploaded.get("subfolder"), uploaded["name"]]))
    for item in graph.values():
        if item["class_type"] == "LoadImage":
            item["inputs"]["image"] = image_name
    result = request_json(base, "/prompt", {"prompt": graph, "client_id": str(uuid.uuid4())})
    prompt_id = result["prompt_id"]
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / f"{prompt_id}-job.json").write_text(json.dumps({"result": result, "prompt": graph}, indent=2))
    print(f"Submitted {prompt_id}", flush=True)
    if not args.wait:
        return
    deadline = time.monotonic() + args.timeout
    while time.monotonic() < deadline:
        history = request_json(base, f"/history/{prompt_id}").get(prompt_id)
        if history:
            (args.output / f"{prompt_id}-history.json").write_text(json.dumps(history, indent=2))
            status = history.get("status", {})
            if status.get("status_str") == "error":
                raise RuntimeError(f"ComfyUI generation failed: {status.get('messages')}")
            if status.get("completed"):
                for output in history.get("outputs", {}).values():
                    for value in output.values():
                        if not isinstance(value, list):
                            continue
                        for artifact in value:
                            if not isinstance(artifact, dict) or "filename" not in artifact:
                                continue
                            if artifact.get("type", "output") != "output":
                                continue
                            query = urlencode({key: artifact[key] for key in ("filename", "subfolder", "type") if key in artifact})
                            destination = args.output / (prompt_id + "-" + Path(artifact["filename"]).name)
                            with urlopen(base + "/view?" + query, timeout=120) as response:
                                destination.write_bytes(response.read())
                            print(f"Saved {destination}", flush=True)
                return
        time.sleep(10)
    raise TimeoutError(f"Wait timeout. Job {prompt_id} may still be running; do not resubmit blindly.")


if __name__ == "__main__":
    main()
