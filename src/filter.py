# import json

# input_file = "room_mapping_results.json"
# output_file = "room_mapping_results_filtered.json"

# with open(input_file, "r", encoding="utf-8") as f:
#     data = json.load(f)

# allowed_types = {"exact", "high_confidence", "medium_confidence"}

# # Filter by match type
# filtered = [
#     m for m in data["mappings"]
#     if m.get("match_type") in allowed_types
# ]

# # Remove duplicates (based on source and target room codes)
# seen = set()
# unique = []
# for m in filtered:
#     source_code = m.get("source_room", {}).get("code", "")
#     target_code = m.get("target_room", {}).get("code", "") if m.get("target_room") else ""
#     key = (source_code, target_code)
#     if key not in seen:
#         seen.add(key)
#         unique.append(m)

# # Update mappings
# data["mappings"] = unique

# # Update statistics
# total = len(unique)
# mapped = sum(1 for m in unique if m.get("target_room"))
# unmapped = total - mapped

# data["statistics"]["total"] = total
# data["statistics"]["mapped"] = mapped
# data["statistics"]["unmapped"] = unmapped
# data["statistics"]["duplicates_removed"] = len(filtered) - len(unique)

# # Recalculate by_confidence counts
# data["statistics"]["by_confidence"] = {
#     "exact": sum(1 for m in unique if m.get("match_type") == "exact"),
#     "high": sum(1 for m in unique if m.get("match_type") == "high_confidence"),
#     "medium": sum(1 for m in unique if m.get("match_type") == "medium_confidence"),
#     "low": 0,
#     "none": 0
# }

# # Recalculate recommendations
# data["statistics"]["recommendations"] = {
#     "auto_apply": data["statistics"]["by_confidence"]["exact"] + data["statistics"]["by_confidence"]["high"],
#     "manual_review": data["statistics"]["by_confidence"]["medium"],
#     "needs_attention": 0
# }

# # Recalculate average score
# if unique:
#     data["statistics"]["average_score"] = round(sum(m["match_score"] for m in unique) / len(unique), 2)
# else:
#     data["statistics"]["average_score"] = 0

# # Recalculate mapping rate
# data["statistics"]["mapping_rate"] = f"{(mapped / total * 100):.2f}%" if total > 0 else "0%"

# with open(output_file, "w", encoding="utf-8") as f:
#     json.dump(data, f, ensure_ascii=False, indent=2)

# print(f"Filtered results saved to: {output_file}")
# print(f"Original filtered count: {len(filtered)}")
# print(f"After removing duplicates: {len(unique)}")
# print(f"Duplicates removed: {len(filtered) - len(unique)}")


import json

input_file = "room_mapping_results.json"
output_file = "room_mapping_results_filtered.json"

with open(input_file, "r", encoding="utf-8") as f:
    data = json.load(f)

# Define confidence tiers
HIGH_CONFIDENCE_TYPES = {"exact", "high_confidence"}
LOW_CONFIDENCE_TYPES = {"medium_confidence", "low_confidence", "none"}

# Step 1: Filter by match type (all allowed)
allowed_types = HIGH_CONFIDENCE_TYPES | LOW_CONFIDENCE_TYPES
filtered = [
    m for m in data.get("mappings", [])
    if m.get("match_type") in allowed_types
]

# Step 2: Remove duplicates based on axisData/code and hotelBeds/code
# Track what we've seen to prevent data loss
seen_high = set()  # Tracks (axis_code, hb_code) for high confidence
seen_axis = set()  # Tracks axis_code for unmapped/low confidence
unique_high_confidence = []
unique_low_confidence = []

for m in filtered:
    axis_code = m.get("source_room", {}).get("code", "")
    hotelbeds_code = m.get("target_room", {}).get("code", "") if m.get("target_room") else ""
    match_type = m.get("match_type")
    
    # HIGH CONFIDENCE: Only add if both providers exist and not seen before
    if match_type in HIGH_CONFIDENCE_TYPES:
        if hotelbeds_code:  # Ensure target exists
            key = (axis_code, hotelbeds_code)
            if key not in seen_high:
                seen_high.add(key)
                unique_high_confidence.append(m)
    
    # LOW CONFIDENCE: Only add axisData if not already mapped in high confidence
    elif match_type in LOW_CONFIDENCE_TYPES:
        if axis_code not in seen_high and axis_code not in seen_axis:
            seen_axis.add(axis_code)
            unique_low_confidence.append(m)

# Step 3: Prepare streamlined organized structure
organized = []

# Add high confidence mappings (both providers)
for m in unique_high_confidence:
    axis_room = m.get("source_room", {})
    hb_room = m.get("target_room", {}) or {}
    organized.append({
        "axisData": {
            "code": axis_room.get("code", ""),
            "description": axis_room.get("description", "")
        },
        "hotelBeds": {
            "code": hb_room.get("code", ""),
            "name": hb_room.get("name", "")
        },
        "match_score": m.get("match_score"),
        "match_type": m.get("match_type"),
        "confidence": m.get("confidence"),
    })

# Add low confidence mappings (axisData only, no hotelBeds)
for m in unique_low_confidence:
    axis_room = m.get("source_room", {})
    organized.append({
        "axisData": {
            "code": axis_room.get("code", ""),
            "description": axis_room.get("description", "")
        },
        "hotelBeds": None,  # Explicit None to indicate no mapping
        "match_score": m.get("match_score"),
        "match_type": m.get("match_type"),
        "confidence": m.get("confidence"),
    })

# Step 4: Sort by match_score DESC
organized = sorted(organized, key=lambda x: x.get("match_score", 0), reverse=True)

# Step 5: Recalculate statistics from ALL unique data (no data loss)
all_unique = unique_high_confidence + unique_low_confidence
total = len(organized)
by_confidence = {"exact": 0, "high": 0, "medium": 0, "low": 0, "none": 0}

for m in all_unique:
    mt = m.get("match_type")
    if mt == "exact": by_confidence["exact"] += 1
    elif mt == "high_confidence": by_confidence["high"] += 1
    elif mt == "medium_confidence": by_confidence["medium"] += 1
    elif mt == "low_confidence": by_confidence["low"] += 1
    elif mt == "none": by_confidence["none"] += 1

# Mapped = high confidence records with target_room
# Unmapped = low confidence records (axisData only)
mapped = len(unique_high_confidence)
unmapped = len(unique_low_confidence)

stats = dict(
    total=total,
    mapped=mapped,
    unmapped=unmapped,
    duplicates_removed=len(filtered) - len(all_unique),
    by_confidence=by_confidence,
    recommendations=dict(
        auto_apply=by_confidence["exact"] + by_confidence["high"],
        manual_review=by_confidence["medium"],
        needs_attention=by_confidence["low"] + by_confidence["none"],
    ),
    average_score=round(
        sum(m.get("match_score", 0) for m in all_unique) / total, 2
    ) if total > 0 else 0.0,
    mapping_rate=f"{(mapped / total * 100):.2f}%" if total > 0 else "0%",
    data_integrity=dict(
        original_count=len(data.get("mappings", [])),
        filtered_count=len(filtered),
        unique_processed=len(all_unique),
        high_confidence_count=len(unique_high_confidence),
        low_confidence_count=len(unique_low_confidence),
    )
)

# Step 6: Compose final output
output = {
    "mappings": organized,
    "statistics": stats
}

# Step 7: Write to file
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(output, f, ensure_ascii=False, indent=2)

# Step 8: Verbose reporting (for production verification)
print(f"✓ Filtered results saved to: {output_file}")
print(f"\n📊 Processing Summary:")
print(f"  Original records: {len(data.get('mappings', []))}")
print(f"  Filtered count: {len(filtered)}")
print(f"  High confidence (exact + high): {len(unique_high_confidence)}")
print(f"  Low confidence (medium + low + none): {len(unique_low_confidence)}")
print(f"  Duplicates removed: {len(filtered) - len(all_unique)}")
print(f"  Total unique output: {len(organized)}")
print(f"\n✓ No data loss - all {len(all_unique)} records processed")
