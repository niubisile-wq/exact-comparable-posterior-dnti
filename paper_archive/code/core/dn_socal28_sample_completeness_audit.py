import json
from pathlib import Path
import tempfile
from urllib.request import urlopen


URL = "https://caltech.box.com/s/5baxy2ogbalqohpidh1lyxgnnxmv5tuc"
LOCAL_ROOT = Path(r"<LOCAL_WORKSPACE>\digital-twin-dataset\sample_dataset")
OUT_FILE = Path(r"<LOCAL_WORKSPACE>\socal28_sample_completeness_audit.txt")
CACHED_HTML = Path(tempfile.gettempdir()) / "socal_sample_dataset_box.html"


def main():
    if CACHED_HTML.exists():
        html = CACHED_HTML.read_text(encoding="utf-8", errors="ignore")
    else:
        html = urlopen(URL, timeout=15).read().decode("utf-8", errors="ignore")
    marker = "Box.postStreamData = "
    start = html.find(marker)
    end = html.find(";</script>", start)
    payload = html[start + len(marker):end].strip()
    data = json.loads(payload)
    shared = data["/app-api/enduserapp/shared-folder"]
    remote_items = {
        item["name"]: {"filesCount": item["filesCount"], "itemSize": item["itemSize"]}
        for item in shared["items"]
        if item["type"] == "folder"
    }
    local_dirs = sorted([p.name for p in LOCAL_ROOT.iterdir() if p.is_dir()]) if LOCAL_ROOT.exists() else []
    missing = [name for name in sorted(remote_items) if name not in local_dirs]

    lines = [
        "SoCal 28-bus sample completeness audit",
        f"shared_sample_folder={shared['currentFolderName']}",
        f"remote_folders={', '.join(sorted(remote_items))}",
        f"local_folders={', '.join(local_dirs)}",
    ]
    for name in sorted(remote_items):
        info = remote_items[name]
        lines.append(f"remote_{name}=files:{info['filesCount']} size_bytes:{info['itemSize']}")
    lines.append(f"missing_local_folders={', '.join(missing)}")
    lines.append("boundary=The public shared sample includes measurement folders, but the current workspace sample only contains topology.")
    lines.append("boundary=The SoCal measurement risk is therefore narrowed from data unavailability to incomplete local sample acquisition.")
    OUT_FILE.write_text("\n".join(lines), encoding="utf-8")
    print(OUT_FILE)


if __name__ == "__main__":
    main()
