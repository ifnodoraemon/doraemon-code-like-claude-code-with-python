from pathlib import Path


def test_static_bundle_references_existing_assets():
    static_dir = Path("src/webui/static")
    index_html = (static_dir / "index.html").read_text(encoding="utf-8")

    assert "/vite.svg" not in index_html

    for marker in ('src="/assets/', 'href="/assets/'):
        start = 0
        while True:
            offset = index_html.find(marker, start)
            if offset == -1:
                break
            asset_start = offset + len(marker)
            asset_end = index_html.find('"', asset_start)
            asset_path = index_html[asset_start:asset_end]
            assert (static_dir / "assets" / asset_path).exists()
            start = asset_end + 1
