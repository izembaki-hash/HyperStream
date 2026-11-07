import json
import struct

tokenizer_path = r"H:\ai model\qwen_tokenizer\tokenizer.json"
print("Reading merges from Qwen tokenizer.json...")
with open(tokenizer_path, "r", encoding="utf-8") as f:
    tok_data = json.load(f)

merges = tok_data["model"].get("merges", [])
print(f"Total merges found: {len(merges)}")

valid_merges = []
for rank, item in enumerate(merges):
    parts = item.split(" ")
    if len(parts) == 2:
        p1 = parts[0].encode("utf-8")
        p2 = parts[1].encode("utf-8")
        if len(p1) <= 255 and len(p2) <= 255:
            valid_merges.append((p1, p2, rank))

out_path = r"H:\asurrini\qwen_merges_table.bin"
with open(out_path, "wb") as f:
    f.write(struct.pack("<I", len(valid_merges)))
    for p1, p2, rank in valid_merges:
        f.write(struct.pack("<B", len(p1)))
        f.write(p1)
        f.write(struct.pack("<B", len(p2)))
        f.write(p2)
        f.write(struct.pack("<I", rank))

print(f"Exported {len(valid_merges)} merges to {out_path}")
