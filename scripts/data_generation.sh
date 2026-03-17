source .venv/bin/activate
python -m models.train \
    --parquet-dir /home/qw3460/wp/huggingface/Wrigggy/crypto-ohlcv-1m/data\
    --resample-minutes 5 --seq-len 240 --forward-window 12 \
    --save-dataset artifacts/dataset_65sym_5m_240seq.npz
