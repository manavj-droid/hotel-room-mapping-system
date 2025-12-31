import json

input_file = "hotelbeds_rooms.json"
output_file = "room_mapping_code_name_only.json"

with open(input_file, "r", encoding="utf-8") as f:
    data = json.load(f)

# Extract only code and name from source and target rooms
simplified = []
for m in data["mappings"]:
    source = m.get("source_room", {})
    target = m.get("target_room", {})
    
    simplified.append({
        "source": {
            "code": source.get("code", ""),
            "name": source.get("name", "")
        },
        "target": {
            "code": target.get("code", "") if target else "",
            "name": target.get("name", "") if target else ""
        },
        "match_score": m.get("match_score"),
        "match_type": m.get("match_type")
    })

# Save to new file
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(simplified, f, ensure_ascii=False, indent=2)

print(f"Simplified results saved to: {output_file}")
print(f"\nSample output:")
print(json.dumps(simplified[:2], ensure_ascii=False, indent=2))
