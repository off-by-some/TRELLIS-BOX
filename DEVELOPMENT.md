# 🔥 Development Mode Guide

This document explains how to use hot reloading for faster development.

## Quick Start

### Option 1: Quick Dev Mode (Recommended for Remote)
```bash
./trellis.sh dev
```
Directly starts development mode with hot reloading. Requires `docker-compose` or `docker compose`.

### Option 2: Full Run with Dev Flag
```bash
./trellis.sh run --dev
```
Runs all GPU checks and validations, then starts in dev mode.

### Option 3: Direct Compose (Manual)
```bash
docker-compose -f docker-compose.yml -f docker-compose.dev.yml up --build
```

## What Gets Hot-Reloaded

✅ **Files that reload automatically:**
- `app.py` - Main Streamlit application
- `webui/` - UI components and logic
- `docs/` - Documentation files
- `assets/` - Static assets and images

❌ **Files that require rebuild:**
- `trellis/` - Core algorithms (installed as Python package)
- `extensions/` - Compiled C++ extensions (nvdiffrast, etc.)
- `pyproject.toml` - Dependencies

## When to Use Dev Mode

### ✅ Use Dev Mode For:
- 🎨 UI/UX changes
- 📊 Layout adjustments
- 🔧 App logic modifications
- 🐛 Debugging Streamlit issues
- 📝 Documentation updates

### ❌ Rebuild Required For:
- 🧠 Algorithm changes in `trellis/`
- 🔧 C++ extension modifications
- 📦 Dependency updates
- ⚙️ Core pipeline changes

## Development Workflow

```bash
# 1. Start development mode
./trellis.sh dev

# 2. Edit files (app.py, webui/*.py)
#    Changes reload automatically in browser!

# 3. For core algorithm changes:
#    - Stop dev mode (Ctrl+C)
#    - Rebuild: ./trellis.sh build
#    - Restart: ./trellis.sh dev
```

## Troubleshooting

### "Docker Compose not found"
Install Docker Compose on your system:

**Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install docker-compose-plugin
```

**Or standalone:**
```bash
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose
```

### Container dies immediately
This usually means you're mounting incompatible files. The dev mode only mounts:
- Python source files (app.py, webui/)
- Static assets (docs/, assets/)

It does NOT mount compiled code or installed packages.

### Changes not reflecting
1. Check Streamlit shows "Source file changed, rerunning" in browser
2. Make sure you're editing the file on the host (not inside container)
3. Try hard refresh (Ctrl+Shift+R)
4. Check file isn't in `trellis/` or `extensions/` (these need rebuild)

## Performance Notes

- **Dev mode** has slightly slower startup (mounts volumes)
- **Hot reload** is instant (no rebuild needed)
- **Production mode** is faster at runtime (no volume overhead)

## Remote Development

For remote GPU servers:

```bash
# On remote server
./trellis.sh dev

# Access from local machine
ssh -L 8501:localhost:8501 user@remote-server
# Then open http://localhost:8501 in local browser
```

## Docker Compose Detection

The scripts automatically detect:
1. `docker-compose` (standalone binary)
2. `docker compose` (Docker plugin)

If neither is found, `./trellis.sh run --dev` falls back to production mode.

