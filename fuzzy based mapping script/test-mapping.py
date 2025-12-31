# #!/usr/bin/env python3
# """
# All-in-One Room Mapping Script - IMPROVED with Number & Code Normalization (removes last 2 chars from all AxisData codes)
# Zero Hardcoded Rules - 100% Data-Driven Pattern Learning

# IMPROVEMENTS:
# - Numeric word conversion (1 <-> one, 2 <-> two, etc.)
# - Better token filtering (don't discard numbers!)
# - Suite 1 Bedroom now matches SUITE ONE BEDROOM ✓
# - AxisData: Removes last 2 chars from ALL room codes (e.g. STSD0000A0 -> STSD0000)

# Usage:
#     python room_mapping_complete_v2.py

# Requirements:
#     pip install fuzzywuzzy python-Levenshtein nltk
# """

# import re
# import json
# import logging
# from collections import Counter
# from typing import Dict, List, Tuple
# from pathlib import Path

# # Try importing required packages
# try:
#     from fuzzywuzzy import fuzz
#     import nltk
#     from nltk.tokenize import word_tokenize
#     from nltk.corpus import stopwords
#     from nltk.stem import PorterStemmer
# except ImportError as e:
#     print(f"❌ Missing required package: {e}")
#     print("\n📦 Install dependencies with:")
#     print("pip install fuzzywuzzy python-Levenshtein nltk")
#     exit(1)

# # Download NLTK data quietly
# try:
#     nltk.download('punkt', quiet=True)
#     nltk.download('stopwords', quiet=True)
# except:
#     pass

# # Setup logging
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(levelname)s - %(message)s'
# )
# logger = logging.getLogger(__name__)


# # ============================================================================
# # NUMBER CONVERSION
# # ============================================================================

# NUMBER_MAPPING = {
#     '0': 'zero', 'zero': '0',
#     '1': 'one', 'one': '1',
#     '2': 'two', 'two': '2',
#     '3': 'three', 'three': '3',
#     '4': 'four', 'four': '4',
#     '5': 'five', 'five': '5',
#     '6': 'six', 'six': '6',
#     '7': 'seven', 'seven': '7',
#     '8': 'eight', 'eight': '8',
#     '9': 'nine', 'nine': '9',
#     '10': 'ten', 'ten': '10',
#     '11': 'eleven', 'eleven': '11',
#     '12': 'twelve', 'twelve': '12',
# }


# # ============================================================================
# # PATTERN LEARNING CLASS
# # ============================================================================

# class PatternLearning:
#     """Auto-learn patterns from room data without hardcoded rules"""
    
#     def __init__(self):
#         self.stemmer = PorterStemmer()
#         try:
#             self.stop_words = set(stopwords.words('english'))
#             # Keep room-related words and numbers
#             self.stop_words -= {'with', 'view', 'one', 'two', 'three', 'four'}
#         except:
#             self.stop_words = set()
    
#     def normalize_number_words(self, text: str) -> str:
#         """Convert number words to digits and vice versa for matching"""
#         normalized = text.lower()
#         # Convert number words to digits
#         for word, digit in NUMBER_MAPPING.items():
#             if word.isalpha():  # word -> digit conversion
#                 pattern = r'\b' + word + r'\b'
#                 normalized = re.sub(pattern, digit, normalized)
#         return normalized
    
#     def learn_patterns(self, rooms: List[Dict]) -> Dict:
#         """Learn all patterns from room data automatically"""
#         logger.info(f"Learning patterns from {len(rooms)} rooms...")
        
#         patterns = {
#             'categories': {},
#             'tokens': Counter(),
#             'bigrams': Counter(),
#             'trigrams': Counter(),
#             'numeric_patterns': Counter(),
#             'word_clusters': {},
#             'code_prefixes': {}
#         }
        
#         # Extract all room texts
#         room_texts = [self._get_room_text(room).lower() for room in rooms]
        
#         # Learn all patterns
#         self._learn_token_frequencies(room_texts, patterns)
#         self._learn_ngrams(room_texts, patterns)
#         self._learn_numeric_patterns(room_texts, patterns)
#         self._learn_code_patterns(rooms, patterns)
#         self._cluster_similar_words(patterns)
#         self._extract_categories(rooms, room_texts, patterns)
        
#         logger.info(f"✓ Learned {len(patterns['categories'])} categories, "
#                    f"{len(patterns['tokens'])} unique tokens")
        
#         return patterns
    
#     def _get_room_text(self, room: Dict) -> str:
#         """Extract text from any room field"""
#         return (room.get('name') or 
#                 room.get('description') or 
#                 room.get('code') or '').strip()
    
#     def _learn_token_frequencies(self, texts: List[str], patterns: Dict):
#         """Learn token frequencies - IMPROVED: Keep numbers!"""
#         for text in texts:
#             # Normalize number words
#             text_normalized = self.normalize_number_words(text)
#             try:
#                 tokens = word_tokenize(text_normalized.lower())
#             except:
#                 tokens = text_normalized.lower().split()
#             for token in tokens:
#                 if (len(token) > 2 and token not in self.stop_words) or (token.isdigit() and len(token) == 1):
#                     patterns['tokens'][token] += 1
    
#     def _learn_ngrams(self, texts: List[str], patterns: Dict):
#         """Learn bigram and trigram patterns"""
#         for text in texts:
#             text_normalized = self.normalize_number_words(text)
#             try:
#                 tokens = word_tokenize(text_normalized.lower())
#             except:
#                 tokens = text_normalized.lower().split()
#             tokens = [t for t in tokens if len(t) > 2 or t.isdigit()]
#             for i in range(len(tokens) - 1):
#                 bigram = f"{tokens[i]} {tokens[i+1]}"
#                 patterns['bigrams'][bigram] += 1
#             for i in range(len(tokens) - 2):
#                 trigram = f"{tokens[i]} {tokens[i+1]} {tokens[i+2]}"
#                 patterns['trigrams'][trigram] += 1
    
#     def _learn_numeric_patterns(self, texts: List[str], patterns: Dict):
#         """Learn numeric patterns like '1 bed', '2 queen'"""
#         numeric_regex = re.compile(r'(\d+)\s+([a-z]+)')
#         for text in texts:
#             text_normalized = self.normalize_number_words(text)
#             matches = numeric_regex.findall(text_normalized)
#             for num, word in matches:
#                 pattern = f"{num} {word}"
#                 patterns['numeric_patterns'][pattern] += 1
    
#     def _learn_code_patterns(self, rooms: List[Dict], patterns: Dict):
#         """Learn patterns from room codes (with normalization for AxisData)"""
#         for room in rooms:
#             code = room.get('code', '')
#             if code:
#                 normalized_code = code[:-2] if len(code) > 2 else code  # Remove last 2 chars for ALL AxisData codes
#                 for length in [2, 3, 4]:
#                     if len(normalized_code) >= length:
#                         prefix = normalized_code[:length]
#                         room_text = self._get_room_text(room).lower()
#                         if prefix not in patterns['code_prefixes']:
#                             patterns['code_prefixes'][prefix] = room_text
#             room_type = room.get('type', '')
#             if room_type:
#                 if room_type not in patterns['categories']:
#                     patterns['categories'][room_type] = []
#                 patterns['categories'][room_type].append(self._get_room_text(room))
    
#     def _cluster_similar_words(self, patterns: Dict):
#         """Cluster similar words using edit distance"""
#         try:
#             import Levenshtein
#             top_words = [word for word, _ in patterns['tokens'].most_common(200)]
#             clustered = set()
#             for word1 in top_words:
#                 if word1 in clustered or len(word1) <= 3:
#                     continue
#                 cluster = [word1]
#                 for word2 in top_words:
#                     if word1 != word2 and word2 not in clustered:
#                         distance = Levenshtein.distance(word1, word2)
#                         max_len = max(len(word1), len(word2))
#                         similarity = 1 - (distance / max_len)
#                         if similarity > 0.8:
#                             cluster.append(word2)
#                             clustered.add(word2)
#                 if len(cluster) > 1:
#                     patterns['word_clusters'][word1] = cluster
#                 clustered.add(word1)
#         except ImportError:
#             logger.warning("Levenshtein not available, skipping word clustering")
    
#     def _extract_categories(self, rooms: List[Dict], texts: List[str], patterns: Dict):
#         """Extract categories from clustering"""
#         min_frequency = len(rooms) * 0.05
#         category_indicators = [
#             word for word, freq in patterns['tokens'].items()
#             if len(word) > 3 and freq > min_frequency
#         ]
#         category_indicators = sorted(
#             category_indicators,
#             key=lambda w: patterns['tokens'][w],
#             reverse=True
#         )[:30]
#         for indicator in category_indicators:
#             matching_rooms = []
#             for i, text in enumerate(texts):
#                 if indicator in text:
#                     matching_rooms.append(self._get_room_text(rooms[i]))
#             if matching_rooms:
#                 patterns['categories'][indicator] = matching_rooms


# # ============================================================================
# # ROOM MAPPER CLASS
# # ============================================================================

# class DynamicRoomMapper:
#     """Provider-agnostic room mapping using learned patterns"""
    
#     def __init__(self):
#         self.pattern_learner = PatternLearning()
#         self.stemmer = PorterStemmer()
    
#     def map_rooms(self, source_rooms: List[Dict], target_rooms: List[Dict],
#                   provider_name: str = "unknown") -> Dict:
#         logger.info(f"🔄 Mapping {len(source_rooms)} source → {len(target_rooms)} target rooms")
#         source_patterns = self.pattern_learner.learn_patterns(source_rooms)
#         target_patterns = self.pattern_learner.learn_patterns(target_rooms)
#         normalized_source = [self._normalize_room(r) for r in source_rooms]
#         normalized_target = [self._normalize_room(r) for r in target_rooms]
#         mappings = []
#         for i, source_room in enumerate(normalized_source):
#             if i % 100 == 0:
#                 logger.info(f"  Processing {i}/{len(normalized_source)}...")
#             match = self._find_best_match(
#                 source_room, normalized_target,
#                 source_patterns, target_patterns
#             )
#             mappings.append(match)
#         statistics = self._calculate_statistics(mappings)
#         return {
#             'mappings': mappings,
#             'statistics': statistics,
#             'provider_name': provider_name
#         }
    
#     def _normalize_room(self, room: Dict) -> Dict:
#         text = (room.get('name') or 
#                 room.get('description') or 
#                 room.get('code') or '').lower().strip()
#         pl = PatternLearning()
#         text_normalized = pl.normalize_number_words(text)
#         try:
#             tokens = word_tokenize(text_normalized)
#         except:
#             tokens = text_normalized.split()
#         tokens = [t for t in tokens if len(t) > 2 or t.isdigit()]
#         stems = [self.stemmer.stem(t) for t in tokens]
#         numeric_features = self._extract_numeric_features(text_normalized)
#         code = room.get('code', '')
#         normalized_code = code[:-2] if len(code) > 2 else code  # Remove last 2 for ALL AxisData
#         code_prefix = normalized_code[:4] if len(normalized_code) >= 4 else normalized_code[:2] if normalized_code else ''
#         return {
#             'original': room,
#             'text': text_normalized,
#             'tokens': tokens,
#             'stems': stems,
#             'numeric_features': numeric_features,
#             'code_prefix': code_prefix,
#             'normalized_code': normalized_code,
#             'original_code': code
#         }
    
#     def _extract_numeric_features(self, text: str) -> List[str]:
#         numeric_regex = re.compile(r'(\d+)\s*([a-z]+)')
#         matches = numeric_regex.findall(text)
#         return [f"{num}_{word}" for num, word in matches]
    
#     def _find_best_match(self, source_room: Dict, target_rooms: List[Dict],
#                         source_patterns: Dict, target_patterns: Dict) -> Dict:
#         best_match = None
#         best_score = 0
#         best_features = []
#         for target_room in target_rooms:
#             score, features = self._calculate_similarity(
#                 source_room, target_room,
#                 source_patterns, target_patterns
#             )
#             if score > best_score:
#                 best_score = score
#                 best_match = target_room
#                 best_features = features
#         if best_score >= 95:
#             match_type = 'exact'
#         elif best_score >= 90:
#             match_type = 'high_confidence'
#         elif best_score >= 85:
#             match_type = 'medium_confidence'
#         elif best_score >= 75:
#             match_type = 'low_confidence'
#         else:
#             match_type = 'none'
#         return {
#             'source_room': source_room['original'],
#             'target_room': best_match['original'] if best_match else None,
#             'match_score': round(best_score, 2),
#             'match_type': match_type,
#             'confidence': round(best_score / 100, 2),
#             'matching_features': best_features,
#             'source_normalized_code': source_room.get('normalized_code', ''),
#             'target_normalized_code': best_match.get('normalized_code', '') if best_match else ''
#         }
    
#     def _calculate_similarity(self, room1: Dict, room2: Dict,
#                              patterns1: Dict, patterns2: Dict) -> Tuple[float, List[str]]:
#         matching_features = []
#         total_score = 0
#         total_weight = 0
#         token_weight = 35
#         token_overlap = self._calculate_jaccard(room1['tokens'], room2['tokens'])
#         if token_overlap > 0:
#             matching_features.append(f"token:{token_overlap:.2f}")
#         total_score += token_overlap * token_weight
#         total_weight += token_weight
#         stem_weight = 30
#         stem_overlap = self._calculate_jaccard(room1['stems'], room2['stems'])
#         if stem_overlap > 0:
#             matching_features.append(f"stem:{stem_overlap:.2f}")
#         total_score += stem_overlap * stem_weight
#         total_weight += stem_weight
#         numeric_weight = 15
#         numeric_match = self._calculate_numeric_match(
#             room1['numeric_features'],
#             room2['numeric_features']
#         )
#         if numeric_match > 0:
#             matching_features.append(f"numeric:{numeric_match:.2f}")
#         total_score += numeric_match * numeric_weight
#         total_weight += numeric_weight
#         fuzzy_weight = 10
#         fuzzy_score = fuzz.ratio(room1['text'], room2['text']) / 100
#         if fuzzy_score > 0.5:
#             matching_features.append(f"fuzzy:{fuzzy_score:.2f}")
#         total_score += fuzzy_score * fuzzy_weight
#         total_weight += fuzzy_weight
#         code_weight = 10
#         code_score = 0
#         if room1['code_prefix'] and room2['code_prefix']:
#             if room1['code_prefix'] == room2['code_prefix']:
#                 code_score = 1.0
#             elif room1['code_prefix'][:2] == room2['code_prefix'][:2]:
#                 code_score = 0.5
#             if code_score > 0:
#                 matching_features.append(f"code:{room1['code_prefix']}")
#         total_score += code_score * code_weight
#         total_weight += code_weight
#         final_score = (total_score / total_weight) * 100
#         return final_score, matching_features
    
#     def _calculate_jaccard(self, list1: List[str], list2: List[str]) -> float:
#         if not list1 or not list2:
#             return 0
#         set1 = set(list1)
#         set2 = set(list2)
#         intersection = len(set1 & set2)
#         union = len(set1 | set2)
#         return intersection / union if union > 0 else 0
    
#     def _calculate_numeric_match(self, features1: List[str], features2: List[str]) -> float:
#         if not features1 and not features2:
#             return 1.0
#         if not features1 or not features2:
#             return 0
#         set1 = set(features1)
#         set2 = set(features2)
#         intersection = len(set1 & set2)
#         max_size = max(len(set1), len(set2))
#         return intersection / max_size
    
#     def _calculate_statistics(self, mappings: List[Dict]) -> Dict:
#         total = len(mappings)
#         mapped = sum(1 for m in mappings if m['target_room'] is not None)
#         by_type = {
#             'exact': sum(1 for m in mappings if m['match_type'] == 'exact'),
#             'high': sum(1 for m in mappings if m['match_type'] == 'high_confidence'),
#             'medium': sum(1 for m in mappings if m['match_type'] == 'medium_confidence'),
#             'low': sum(1 for m in mappings if m['match_type'] == 'low_confidence'),
#             'none': sum(1 for m in mappings if m['match_type'] == 'none')
#         }
#         avg_score = sum(m['match_score'] for m in mappings) / total if total > 0 else 0
#         return {
#             'total': total,
#             'mapped': mapped,
#             'unmapped': total - mapped,
#             'mapping_rate': f"{(mapped / total * 100):.2f}%" if total > 0 else "0%",
#             'by_confidence': by_type,
#             'average_score': round(avg_score, 2),
#             'recommendations': {
#                 'auto_apply': by_type['exact'] + by_type['high'],
#                 'manual_review': by_type['medium'],
#                 'needs_attention': by_type['low'] + by_type['none']
#             }
#         }

# # ============================================================================
# # MAIN TEST FUNCTION
# # ============================================================================

# def main():
#     print("\n" + "="*70)
#     print("🚀 DYNAMIC ROOM MAPPING SYSTEM - V2 (IMPROVED)")
#     print("   ✨ Number Normalization Enabled (1 <-> one)")
#     print("   ✨ AxisData Code Normalization Enabled (last 2 chars REMOVED for ALL AxisData codes)")
#     print("="*70 + "\n")
#     axisdata_file = "mapped_room_codes_structured.json"
#     hotelbeds_file = "hotelbeds_rooms.json"
#     if not Path(axisdata_file).exists():
#         logger.error(f"❌ File not found: {axisdata_file}")
#         logger.info("\n💡 Please place your JSON files in the same directory:")
#         logger.info(f"   - {axisdata_file}")
#         logger.info(f"   - {hotelbeds_file}")
#         return
#     if not Path(hotelbeds_file).exists():
#         logger.error(f"❌ File not found: {hotelbeds_file}")
#         logger.info("\n💡 Please place your JSON files in the same directory:")
#         logger.info(f"   - {axisdata_file}")
#         logger.info(f"   - {hotelbeds_file}")
#         return
#     logger.info(f"📂 Loading {axisdata_file}...")
#     with open(axisdata_file, 'r', encoding='utf-8') as f:
#         axisdata_rooms = json.load(f)
#     logger.info(f"📂 Loading {hotelbeds_file}...")
#     with open(hotelbeds_file, 'r', encoding='utf-8') as f:
#         hotelbeds_rooms = json.load(f)
#     logger.info(f"✓ Loaded {len(axisdata_rooms)} AxisData rooms")
#     logger.info(f"✓ Loaded {len(hotelbeds_rooms)} Hotelbeds rooms")
#     if axisdata_rooms:
#         sample_code = axisdata_rooms[0].get('code', 'N/A')
#         normalized = sample_code[:-2] if sample_code and len(sample_code) > 2 else sample_code
#         logger.info(f"📝 Code normalization example: {sample_code} → {normalized}")
#     sample_size = min(17000, len(axisdata_rooms))
#     logger.info(f"\n🧪 Testing with first {sample_size} AxisData rooms")
#     logger.info("\n" + "="*70)
#     logger.info("Starting Room Mapping Process")
#     logger.info("="*70 + "\n")
#     mapper = DynamicRoomMapper()
#     result = mapper.map_rooms(
#         axisdata_rooms[:sample_size],
#         hotelbeds_rooms,
#         "axisdata_to_hotelbeds"
#     )
#     stats = result['statistics']
#     print("\n" + "="*70)
#     print("✅ MAPPING COMPLETE!")
#     print("="*70 + "\n")
#     print("📊 RESULTS:")
#     print(f"  Total mappings: {stats['total']}")
#     print(f"  Successfully mapped: {stats['mapped']}")
#     print(f"  Unmapped: {stats['unmapped']}")
#     print(f"  Mapping rate: {stats['mapping_rate']}")
#     print(f"  Average score: {stats['average_score']}")
#     print("\n🎯 BY CONFIDENCE LEVEL:")
#     for conf_type, count in stats['by_confidence'].items():
#         print(f"  {conf_type.upper()}: {count}")
#     print("\n💡 RECOMMENDATIONS:")
#     print(f"  ✓ Auto-apply: {stats['recommendations']['auto_apply']} mappings")
#     print(f"  ⚠ Manual review: {stats['recommendations']['manual_review']} mappings")
#     print(f"  ⚡ Needs attention: {stats['recommendations']['needs_attention']} mappings")
#     print("\n" + "="*70)
#     print("📋 SAMPLE HIGH-CONFIDENCE MAPPINGS (Top 5)")
#     print("="*70 + "\n")
#     high_conf = [m for m in result['mappings'] 
#                  if m['match_type'] in ['exact', 'high_confidence']][:5]
#     if high_conf:
#         for i, mapping in enumerate(high_conf, 1):
#             source = mapping['source_room']
#             target = mapping['target_room']
#             print(f"{i}. SCORE: {mapping['match_score']} ({mapping['match_type'].upper()})")
#             source_code = source.get('code', 'N/A')
#             source_desc = source.get('description') or source.get('name', 'N/A')
#             source_normalized = mapping.get('source_normalized_code', 'N/A')
#             print(f"   SOURCE: [{source_code}] → [{source_normalized}] {source_desc}")
#             if target:
#                 target_code = target.get('code', 'N/A')
#                 target_name = target.get('name') or target.get('description', 'N/A')
#                 print(f"   TARGET: [{target_code}] {target_name}")
#             features = ', '.join(mapping['matching_features'][:4])
#             print(f"   FEATURES: {features}")
#             print()
#     else:
#         print("  No high-confidence mappings found.")
#     output_file = "room_mapping_results.json"
#     logger.info(f"💾 Saving full results to: {output_file}")
#     with open(output_file, 'w', encoding='utf-8') as f:
#         json.dump(result, f, indent=2, ensure_ascii=False)
#     print("\n" + "="*70)
#     print(f"✅ TEST COMPLETE! Results saved to: {output_file}")
#     print("="*70 + "\n")
#     print("📝 NEW FEATURES:")
#     print("  • Number normalization (1 <-> one, 2 <-> two, etc.)")
#     print("  • Suite 1 Bedroom now matches SUITE ONE BEDROOM ✓")
#     print("  • Numeric digits are kept in token analysis")
#     print("  • Last 2 chars are removed from ALL AxisData room codes")
#     print()
#     print("ℹ️  CODE NORMALIZATION:")
#     print("  • AxisData codes now always have last 2 chars removed (e.g., SU00T10000 -> SU00T100)")
#     print("  • Example: STSD0000A0 -> STSD0000")
#     print()

# if __name__ == "__main__":
#     main()
#!/usr/bin/env python3
"""
All-in-One Room Mapping Script - IMPROVED with Number & Code Normalization
Maps ONLY based on name/description fields (not entire JSON objects)

IMPROVEMENTS:
- Numeric word conversion (1 <-> one, 2 <-> two, etc.)
- Better token filtering (don't discard numbers!)
- Suite 1 Bedroom now matches SUITE ONE BEDROOM ✓
- AxisData: Removes last 2 chars from ALL room codes (e.g. STSD0000A0 -> STSD0000)
- Maps ONLY on description/name fields

Usage:
    python room_mapping_complete_v3.py

Requirements:
    pip install fuzzywuzzy python-Levenshtein nltk
"""

import re
import json
import logging
from collections import Counter
from typing import Dict, List, Tuple
from pathlib import Path

# Try importing required packages
try:
    from fuzzywuzzy import fuzz
    import nltk
    from nltk.tokenize import word_tokenize
    from nltk.corpus import stopwords
    from nltk.stem import PorterStemmer
except ImportError as e:
    print(f"❌ Missing required package: {e}")
    print("\n📦 Install dependencies with:")
    print("pip install fuzzywuzzy python-Levenshtein nltk")
    exit(1)

# Download NLTK data quietly
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt_tab', quiet=True)
except:
    pass

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# NUMBER CONVERSION
# ============================================================================

NUMBER_MAPPING = {
    '0': 'zero', 'zero': '0',
    '1': 'one', 'one': '1',
    '2': 'two', 'two': '2',
    '3': 'three', 'three': '3',
    '4': 'four', 'four': '4',
    '5': 'five', 'five': '5',
    '6': 'six', 'six': '6',
    '7': 'seven', 'seven': '7',
    '8': 'eight', 'eight': '8',
    '9': 'nine', 'nine': '9',
    '10': 'ten', 'ten': '10',
    '11': 'eleven', 'eleven': '11',
    '12': 'twelve', 'twelve': '12',
}

# ============================================================================
# PATTERN LEARNING CLASS
# ============================================================================

class PatternLearning:
    """Auto-learn patterns from room data without hardcoded rules"""
    
    def __init__(self):
        self.stemmer = PorterStemmer()
        try:
            self.stop_words = set(stopwords.words('english'))
            # Keep room-related words and numbers
            self.stop_words -= {'with', 'view', 'one', 'two', 'three', 'four'}
        except:
            self.stop_words = set()
    
    def normalize_number_words(self, text: str) -> str:
        """Convert number words to digits and vice versa for matching"""
        normalized = text.lower()
        # Convert number words to digits
        for word, digit in NUMBER_MAPPING.items():
            if word.isalpha():  # word -> digit conversion
                pattern = r'\b' + word + r'\b'
                normalized = re.sub(pattern, digit, normalized)
        return normalized
    
    def learn_patterns(self, rooms: List[Dict]) -> Dict:
        """Learn all patterns from room data automatically"""
        logger.info(f"Learning patterns from {len(rooms)} rooms...")
        
        patterns = {
            'categories': {},
            'tokens': Counter(),
            'bigrams': Counter(),
            'trigrams': Counter(),
            'numeric_patterns': Counter(),
            'word_clusters': {},
            'code_prefixes': {}
        }
        
        # Extract all room texts - ONLY from name/description
        room_texts = [self._get_room_text(room).lower() for room in rooms]
        
        # Learn all patterns
        self._learn_token_frequencies(room_texts, patterns)
        self._learn_ngrams(room_texts, patterns)
        self._learn_numeric_patterns(room_texts, patterns)
        self._learn_code_patterns(rooms, patterns)
        self._cluster_similar_words(patterns)
        self._extract_categories(rooms, room_texts, patterns)
        
        logger.info(f"✓ Learned {len(patterns['categories'])} categories, "
                   f"{len(patterns['tokens'])} unique tokens")
        
        return patterns
    
    def _get_room_text(self, room: Dict) -> str:
        """Extract text ONLY from name or description field"""
        return (room.get('name') or room.get('description') or '').strip()
    
    def _learn_token_frequencies(self, texts: List[str], patterns: Dict):
        """Learn token frequencies - IMPROVED: Keep numbers!"""
        for text in texts:
            # Normalize number words
            text_normalized = self.normalize_number_words(text)
            try:
                tokens = word_tokenize(text_normalized.lower())
            except:
                tokens = text_normalized.lower().split()
            for token in tokens:
                if (len(token) > 2 and token not in self.stop_words) or (token.isdigit() and len(token) == 1):
                    patterns['tokens'][token] += 1
    
    def _learn_ngrams(self, texts: List[str], patterns: Dict):
        """Learn bigram and trigram patterns"""
        for text in texts:
            text_normalized = self.normalize_number_words(text)
            try:
                tokens = word_tokenize(text_normalized.lower())
            except:
                tokens = text_normalized.lower().split()
            tokens = [t for t in tokens if len(t) > 2 or t.isdigit()]
            for i in range(len(tokens) - 1):
                bigram = f"{tokens[i]} {tokens[i+1]}"
                patterns['bigrams'][bigram] += 1
            for i in range(len(tokens) - 2):
                trigram = f"{tokens[i]} {tokens[i+1]} {tokens[i+2]}"
                patterns['trigrams'][trigram] += 1
    
    def _learn_numeric_patterns(self, texts: List[str], patterns: Dict):
        """Learn numeric patterns like '1 bed', '2 queen'"""
        numeric_regex = re.compile(r'(\d+)\s+([a-z]+)')
        for text in texts:
            text_normalized = self.normalize_number_words(text)
            matches = numeric_regex.findall(text_normalized)
            for num, word in matches:
                pattern = f"{num} {word}"
                patterns['numeric_patterns'][pattern] += 1
    
    def _learn_code_patterns(self, rooms: List[Dict], patterns: Dict):
        """Learn patterns from room codes (with normalization for AxisData)"""
        for room in rooms:
            code = room.get('code', '')
            if code:
                normalized_code = code[:-2] if len(code) > 2 else code
                for length in [2, 3, 4]:
                    if len(normalized_code) >= length:
                        prefix = normalized_code[:length]
                        room_text = self._get_room_text(room).lower()
                        if prefix not in patterns['code_prefixes']:
                            patterns['code_prefixes'][prefix] = room_text
            room_type = room.get('type', '')
            if room_type:
                if room_type not in patterns['categories']:
                    patterns['categories'][room_type] = []
                patterns['categories'][room_type].append(self._get_room_text(room))
    
    def _cluster_similar_words(self, patterns: Dict):
        """Cluster similar words using edit distance"""
        try:
            import Levenshtein
            top_words = [word for word, _ in patterns['tokens'].most_common(200)]
            clustered = set()
            for word1 in top_words:
                if word1 in clustered or len(word1) <= 3:
                    continue
                cluster = [word1]
                for word2 in top_words:
                    if word1 != word2 and word2 not in clustered:
                        distance = Levenshtein.distance(word1, word2)
                        max_len = max(len(word1), len(word2))
                        similarity = 1 - (distance / max_len)
                        if similarity > 0.8:
                            cluster.append(word2)
                            clustered.add(word2)
                if len(cluster) > 1:
                    patterns['word_clusters'][word1] = cluster
                clustered.add(word1)
        except ImportError:
            logger.warning("Levenshtein not available, skipping word clustering")
    
    def _extract_categories(self, rooms: List[Dict], texts: List[str], patterns: Dict):
        """Extract categories from clustering"""
        min_frequency = len(rooms) * 0.05
        category_indicators = [
            word for word, freq in patterns['tokens'].items()
            if len(word) > 3 and freq > min_frequency
        ]
        category_indicators = sorted(
            category_indicators,
            key=lambda w: patterns['tokens'][w],
            reverse=True
        )[:30]
        for indicator in category_indicators:
            matching_rooms = []
            for i, text in enumerate(texts):
                if indicator in text:
                    matching_rooms.append(self._get_room_text(rooms[i]))
            if matching_rooms:
                patterns['categories'][indicator] = matching_rooms

# ============================================================================
# ROOM MAPPER CLASS
# ============================================================================

class DynamicRoomMapper:
    """Provider-agnostic room mapping using learned patterns"""
    
    def __init__(self):
        self.pattern_learner = PatternLearning()
        self.stemmer = PorterStemmer()
    
    def map_rooms(self, source_rooms: List[Dict], target_rooms: List[Dict],
                  provider_name: str = "unknown") -> Dict:
        logger.info(f"🔄 Mapping {len(source_rooms)} source → {len(target_rooms)} target rooms")
        source_patterns = self.pattern_learner.learn_patterns(source_rooms)
        target_patterns = self.pattern_learner.learn_patterns(target_rooms)
        normalized_source = [self._normalize_room(r) for r in source_rooms]
        normalized_target = [self._normalize_room(r) for r in target_rooms]
        mappings = []
        for i, source_room in enumerate(normalized_source):
            if i % 100 == 0:
                logger.info(f"  Processing {i}/{len(normalized_source)}...")
            match = self._find_best_match(
                source_room, normalized_target,
                source_patterns, target_patterns
            )
            mappings.append(match)
        statistics = self._calculate_statistics(mappings)
        return {
            'mappings': mappings,
            'statistics': statistics,
            'provider_name': provider_name
        }
    
    def _normalize_room(self, room: Dict) -> Dict:
        """Normalize room - ONLY using name/description"""
        # Get text ONLY from name or description
        text = (room.get('name') or room.get('description') or '').lower().strip()
        
        pl = PatternLearning()
        text_normalized = pl.normalize_number_words(text)
        try:
            tokens = word_tokenize(text_normalized)
        except:
            tokens = text_normalized.split()
        tokens = [t for t in tokens if len(t) > 2 or t.isdigit()]
        stems = [self.stemmer.stem(t) for t in tokens]
        numeric_features = self._extract_numeric_features(text_normalized)
        code = room.get('code', '')
        normalized_code = code[:-2] if len(code) > 2 else code
        code_prefix = normalized_code[:4] if len(normalized_code) >= 4 else normalized_code[:2] if normalized_code else ''
        return {
            'original': room,
            'text': text_normalized,
            'tokens': tokens,
            'stems': stems,
            'numeric_features': numeric_features,
            'code_prefix': code_prefix,
            'normalized_code': normalized_code,
            'original_code': code
        }
    
    def _extract_numeric_features(self, text: str) -> List[str]:
        numeric_regex = re.compile(r'(\d+)\s*([a-z]+)')
        matches = numeric_regex.findall(text)
        return [f"{num}_{word}" for num, word in matches]
    
    def _find_best_match(self, source_room: Dict, target_rooms: List[Dict],
                        source_patterns: Dict, target_patterns: Dict) -> Dict:
        best_match = None
        best_score = 0
        best_features = []
        for target_room in target_rooms:
            score, features = self._calculate_similarity(
                source_room, target_room,
                source_patterns, target_patterns
            )
            if score > best_score:
                best_score = score
                best_match = target_room
                best_features = features
        if best_score >= 95:
            match_type = 'exact'
        elif best_score >= 90:
            match_type = 'high_confidence'
        elif best_score >= 85:
            match_type = 'medium_confidence'
        elif best_score >= 75:
            match_type = 'low_confidence'
        else:
            match_type = 'none'
        return {
            'source_room': source_room['original'],
            'target_room': best_match['original'] if best_match else None,
            'match_score': round(best_score, 2),
            'match_type': match_type,
            'confidence': round(best_score / 100, 2),
            'matching_features': best_features,
            'source_normalized_code': source_room.get('normalized_code', ''),
            'target_normalized_code': best_match.get('normalized_code', '') if best_match else ''
        }
    
    def _calculate_similarity(self, room1: Dict, room2: Dict,
                             patterns1: Dict, patterns2: Dict) -> Tuple[float, List[str]]:
        """Calculate similarity - ONLY based on name/description text"""
        matching_features = []
        total_score = 0
        total_weight = 0
        
        # Token overlap (35% weight)
        token_weight = 35
        token_overlap = self._calculate_jaccard(room1['tokens'], room2['tokens'])
        if token_overlap > 0:
            matching_features.append(f"token:{token_overlap:.2f}")
        total_score += token_overlap * token_weight
        total_weight += token_weight
        
        # Stem overlap (30% weight)
        stem_weight = 30
        stem_overlap = self._calculate_jaccard(room1['stems'], room2['stems'])
        if stem_overlap > 0:
            matching_features.append(f"stem:{stem_overlap:.2f}")
        total_score += stem_overlap * stem_weight
        total_weight += stem_weight
        
        # Numeric features (20% weight - increased for better number matching)
        numeric_weight = 20
        numeric_match = self._calculate_numeric_match(
            room1['numeric_features'],
            room2['numeric_features']
        )
        if numeric_match > 0:
            matching_features.append(f"numeric:{numeric_match:.2f}")
        total_score += numeric_match * numeric_weight
        total_weight += numeric_weight
        
        # Fuzzy string matching (15% weight)
        fuzzy_weight = 15
        fuzzy_score = fuzz.ratio(room1['text'], room2['text']) / 100
        if fuzzy_score > 0.5:
            matching_features.append(f"fuzzy:{fuzzy_score:.2f}")
        total_score += fuzzy_score * fuzzy_weight
        total_weight += fuzzy_weight
        
        # Code prefix removed from similarity calculation (optional bonus only)
        # We focus purely on name/description matching
        
        final_score = (total_score / total_weight) * 100
        return final_score, matching_features
    
    def _calculate_jaccard(self, list1: List[str], list2: List[str]) -> float:
        if not list1 or not list2:
            return 0
        set1 = set(list1)
        set2 = set(list2)
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        return intersection / union if union > 0 else 0
    
    def _calculate_numeric_match(self, features1: List[str], features2: List[str]) -> float:
        if not features1 and not features2:
            return 1.0
        if not features1 or not features2:
            return 0
        set1 = set(features1)
        set2 = set(features2)
        intersection = len(set1 & set2)
        max_size = max(len(set1), len(set2))
        return intersection / max_size
    
    def _calculate_statistics(self, mappings: List[Dict]) -> Dict:
        total = len(mappings)
        mapped = sum(1 for m in mappings if m['target_room'] is not None)
        by_type = {
            'exact': sum(1 for m in mappings if m['match_type'] == 'exact'),
            'high': sum(1 for m in mappings if m['match_type'] == 'high_confidence'),
            'medium': sum(1 for m in mappings if m['match_type'] == 'medium_confidence'),
            'low': sum(1 for m in mappings if m['match_type'] == 'low_confidence'),
            'none': sum(1 for m in mappings if m['match_type'] == 'none')
        }
        avg_score = sum(m['match_score'] for m in mappings) / total if total > 0 else 0
        return {
            'total': total,
            'mapped': mapped,
            'unmapped': total - mapped,
            'mapping_rate': f"{(mapped / total * 100):.2f}%" if total > 0 else "0%",
            'by_confidence': by_type,
            'average_score': round(avg_score, 2),
            'recommendations': {
                'auto_apply': by_type['exact'] + by_type['high'],
                'manual_review': by_type['medium'],
                'needs_attention': by_type['low'] + by_type['none']
            }
        }

# ============================================================================
# MAIN TEST FUNCTION
# ============================================================================

def main():
    print("\n" + "="*70)
    print("🚀 DYNAMIC ROOM MAPPING SYSTEM - V3 (NAME/DESCRIPTION ONLY)")
    print("   ✨ Maps ONLY based on name/description fields")
    print("   ✨ Number Normalization Enabled (1 <-> one)")
    print("   ✨ AxisData Code Normalization (last 2 chars removed)")
    print("="*70 + "\n")
    
    axisdata_file = "mapped_room_codes_structured.json"
    hotelbeds_file = "hotelbeds_rooms.json"
    
    if not Path(axisdata_file).exists():
        logger.error(f"❌ File not found: {axisdata_file}")
        logger.info("\n💡 Please place your JSON files in the same directory:")
        logger.info(f"   - {axisdata_file}")
        logger.info(f"   - {hotelbeds_file}")
        return
    
    if not Path(hotelbeds_file).exists():
        logger.error(f"❌ File not found: {hotelbeds_file}")
        logger.info("\n💡 Please place your JSON files in the same directory:")
        logger.info(f"   - {axisdata_file}")
        logger.info(f"   - {hotelbeds_file}")
        return
    
    logger.info(f"📂 Loading {axisdata_file}...")
    with open(axisdata_file, 'r', encoding='utf-8') as f:
        axisdata_rooms = json.load(f)
    
    logger.info(f"📂 Loading {hotelbeds_file}...")
    with open(hotelbeds_file, 'r', encoding='utf-8') as f:
        hotelbeds_rooms = json.load(f)
    
    logger.info(f"✓ Loaded {len(axisdata_rooms)} AxisData rooms")
    logger.info(f"✓ Loaded {len(hotelbeds_rooms)} Hotelbeds rooms")
    
    if axisdata_rooms:
        sample = axisdata_rooms[0]
        sample_desc = sample.get('description', 'N/A')
        logger.info(f"📝 AxisData sample description: {sample_desc}")
    
    if hotelbeds_rooms:
        sample = hotelbeds_rooms[0]
        sample_name = sample.get('name', 'N/A')
        logger.info(f"📝 Hotelbeds sample name: {sample_name}")
    
    sample_size = min(17000, len(axisdata_rooms))
    logger.info(f"\n🧪 Testing with first {sample_size} AxisData rooms")
    
    logger.info("\n" + "="*70)
    logger.info("Starting Room Mapping Process (Name/Description Only)")
    logger.info("="*70 + "\n")
    
    mapper = DynamicRoomMapper()
    result = mapper.map_rooms(
        axisdata_rooms[:sample_size],
        hotelbeds_rooms,
        "axisdata_to_hotelbeds"
    )
    
    stats = result['statistics']
    
    print("\n" + "="*70)
    print("✅ MAPPING COMPLETE!")
    print("="*70 + "\n")
    
    print("📊 RESULTS:")
    print(f"  Total mappings: {stats['total']}")
    print(f"  Successfully mapped: {stats['mapped']}")
    print(f"  Unmapped: {stats['unmapped']}")
    print(f"  Mapping rate: {stats['mapping_rate']}")
    print(f"  Average score: {stats['average_score']}")
    
    print("\n🎯 BY CONFIDENCE LEVEL:")
    for conf_type, count in stats['by_confidence'].items():
        print(f"  {conf_type.upper()}: {count}")
    
    print("\n💡 RECOMMENDATIONS:")
    print(f"  ✓ Auto-apply: {stats['recommendations']['auto_apply']} mappings")
    print(f"  ⚠ Manual review: {stats['recommendations']['manual_review']} mappings")
    print(f"  ⚡ Needs attention: {stats['recommendations']['needs_attention']} mappings")
    
    print("\n" + "="*70)
    print("📋 SAMPLE HIGH-CONFIDENCE MAPPINGS (Top 5)")
    print("="*70 + "\n")
    
    high_conf = [m for m in result['mappings'] 
                 if m['match_type'] in ['exact', 'high_confidence']][:5]
    
    if high_conf:
        for i, mapping in enumerate(high_conf, 1):
            source = mapping['source_room']
            target = mapping['target_room']
            print(f"{i}. SCORE: {mapping['match_score']} ({mapping['match_type'].upper()})")
            
            source_code = source.get('code', 'N/A')
            source_desc = source.get('description', 'N/A')
            print(f"   SOURCE: [{source_code}] {source_desc}")
            
            if target:
                target_code = target.get('code', 'N/A')
                target_name = target.get('name', 'N/A')
                print(f"   TARGET: [{target_code}] {target_name}")
            
            features = ', '.join(mapping['matching_features'][:4])
            print(f"   FEATURES: {features}")
            print()
    else:
        print("  No high-confidence mappings found.")
    
    output_file = "room_mapping_results.json"
    logger.info(f"💾 Saving full results to: {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    
    print("\n" + "="*70)
    print(f"✅ TEST COMPLETE! Results saved to: {output_file}")
    print("="*70 + "\n")
    
    print("📝 KEY CHANGES:")
    print("  • Mapping based ONLY on name/description fields")
    print("  • AxisData uses 'description' field")
    print("  • Hotelbeds uses 'name' field")
    print("  • Code fields preserved but not used for matching")
    print("  • Number normalization (1 <-> one, 2 <-> two, etc.)")
    print()

if __name__ == "__main__":
    main()
