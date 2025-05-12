DATA_DIR=$HOME/librispeech_local
mkdir -p $DATA_DIR

# ── адреса исходных архивов (OpenSLR зеркало) ───────────────────
BASE_URL=http://www.openslr.org/resources/12
FILES=(
  train-clean-100.tar.gz   # 6.5 GB  – хватит для домашки
  dev-clean.tar.gz         # 337 MB  – валидация
)

# ── multi‑thread download via aria2 (если нет, sudo apt install aria2) ─
for f in "${FILES[@]}"; do
  if [ ! -f "$DATA_DIR/$f" ]; then
    echo ">>> downloading $f"
    aria2c -x16 -s16 -c "$BASE_URL/$f" -d "$DATA_DIR"
  fi
done

# ── распаковка (один раз) ───────────────────────────────────────
for f in "${FILES[@]}"; do
  dir="${f%.tar.gz}"
  if [ ! -d "$DATA_DIR/LibriSpeech/$dir" ]; then
    echo ">>> extracting $f"
    tar -xf "$DATA_DIR/$f" -C "$DATA_DIR"
  fi
done