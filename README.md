# FFNN-Scratch

Project ini menggunakan `uv` untuk manajemen environment dan dependency Python.

## Setup dengan uv

1. Install `uv` (sekali saja):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

2. Sinkronkan environment dari `pyproject.toml`:

```bash
uv sync
```

## Menjalankan Main Runner

Main runner ada di `src/main.py`.

```bash
uv run src/main.py
```

Atau ekuivalen:

```bash
uv run python src/main.py
```

## Dependency Utama

- `numpy`
- `pandas`
- `scikit-learn`

Dependency tambahan untuk development (opsional):

- `jupyter`