"""Install a user-scoped desktop launcher without writing outside the project."""
from pathlib import Path
import os
import stat

root = Path(__file__).resolve().parents[1]
desktop = Path.home() / "Desktop"
desktop.mkdir(parents=True, exist_ok=True)
target = desktop / "Lumina Guide Mono.desktop"
content = f"""[Desktop Entry]
Type=Application
Name=LUMINA Guide Mono
Comment=Launch the monocular perception and distance dashboard
Exec=bash -lc 'cd {root} && ./run.sh --host 0.0.0.0 --port 8766; exec bash'
Terminal=true
Icon=utilities-terminal
Categories=Development;
"""
target.write_text(content, encoding="utf-8")
target.chmod(target.stat().st_mode | stat.S_IXUSR)
print(target)
