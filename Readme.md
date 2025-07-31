### Lệnh dùng với vectordb:
- Chạy lần đầu init vectordb: python build_index.py data/ja_vi.jsonl indexes/
- Chạy để thêm data vào vectordb: python build_index.py --extend data/ja_vi.jsonl indexes/

### Lệnh chạy api translate:
- uvicorn translate_api:app --host 0.0.0.0 --port 8000 --reload

### Lệnh chạy streamlist giao diện
- streamlit run translate_streamlist.py