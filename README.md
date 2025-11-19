# Hotel Room Mapping System

[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-Proprietary-red.svg)](LICENSE)

A production-grade, modular system for intelligent matching and mapping of hotel rooms across diverse data sources using semantic similarity, fuzzy matching, and structured decision logging.

## 📋 Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [API Reference](#api-reference)
- [Output Format](#output-format)
- [Logging & Audit Trail](#logging--audit-trail)
- [Development](#development)
- [Performance](#performance)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)

## Overview

This system solves the complex problem of matching hotel rooms across multiple data sources with varying schemas and naming conventions. It combines:

- **Geometric name-based matching** for hotels using fuzzy string similarity
- **Semantic embeddings** for room descriptions and amenities
- **Multi-level matching strategies** (direct ID matching → name similarity → semantic similarity)
- **Complete audit trails** for all matching decisions and confidence scores
- **Production-ready error handling** and structured logging

### Problem Statement

Hotel aggregation systems often need to correlate rooms from different suppliers:
- **Hotelbeds**: Global accommodation network with rich room descriptions
- **Axisdata**: Regional catalog with geographic identifiers (placeids)

Each source has different naming conventions, metadata granularity, and data quality levels. This system intelligently bridges these differences.

## Key Features

### 🎯 Intelligent Matching

- **Multi-Strategy Hotel Matching**
  - Direct placeid matching (highest confidence)
  - Name similarity matching with configurable thresholds
  - Normalized text comparison

- **Semantic Room Matching**
  - Uses pre-trained transformer models for embeddings
  - Computes cosine similarity between room descriptions
  - Caches embeddings for performance

- **Room Classification**
  - Automatic room type detection (single, double, suite, etc.)
  - Amenity extraction from descriptions
  - Metadata enrichment

### 🔍 Quality Assurance

- **Confidence Scoring**: Every match includes a confidence score
- **Audit Logging**: Complete decision trail in JSONL format
- **Error Tracking**: Structured error logging with context
- **Validation**: Output validation before file write

### ⚡ Performance

- **Embedding Caching**: Avoid recomputing embeddings across runs
- **Batch Processing**: Efficient similarity calculation for multiple rooms
- **Memory Efficient**: Streaming JSON parsing for large datasets
- **Configurable Thresholds**: Fine-tune matching sensitivity

### 🏗️ Architecture

- **Modular Design**: Each component is independent and testable
- **Clean Separation**: Clear responsibility boundaries between modules
- **Extensible**: Easy to add new matchers or enrichment logic
- **Type Hints**: Full Python type annotations for IDE support

## Architecture

The system implements a **pipeline orchestration pattern**:

```
Input Files
    ↓
[Data Loader] → Load JSON files
    ↓
[Data Normalizer] → Text cleaning, standardization
    ↓
[Hotel Matcher] → Match hotels by placeid/name
    ↓
[Room Enricher] → Extract amenities, classify types
    ↓
[Embedding Service] → Generate/cache embeddings
    ↓
[Room Matcher] → Semantic similarity matching
    ↓
[Output Builder] → Structure results with confidence
    ↓
[Audit Logger] → Log all decisions
    ↓
Output Files
```

**Key Design Patterns:**
- Singleton: Embedding service for shared cache
- Strategy: Multiple matching strategies in HotelMatcher
- Pipeline: Sequential processing in PipelineOrchestrator
- Factory: DataLoader for different file types

## Project Structure

```
hotel_room_mapping_system/
│
├── config/                      # Configuration and logging setup
│   ├── __init__.py
│   ├── settings.py              # 🔧 Configuration, thresholds, file paths
│   └── logging_config.py        # 📝 Structured logging configuration
│
├── data/                        # Data directory (inputs, outputs, logs)
│   ├── inputs/                  # 📥 Input JSON files
│   │   ├── hotelbeds_hotels.json
│   │   ├── axisdata_hotels.json
│   │   └── hotelbeds_room_types.json
│   ├── outputs/                 # 📤 Generated mapping results
│   │   └── hotel_room_mapping.json
│   └── logs/                    # 📋 Audit and error logs
│       ├── mapping_audit.jsonl  # Structured event logs
│       └── mapping_errors.log   # Error log file
│
├── src/                         # Core application code
│   ├── __init__.py
│   ├── data_loader.py           # 📖 Load and parse JSON input files
│   ├── data_normalizer.py       # 🧹 Data cleaning and normalization
│   ├── hotel_matcher.py         # 🏨 Hotel matching logic
│   ├── room_enricher.py         # ✨ Room metadata enrichment
│   ├── embedding_service.py     # 🧠 Embedding generation and caching
│   ├── room_matcher.py          # 🎯 Room semantic similarity matching
│   ├── output_builder.py        # 📦 Final output structuring
│   ├── audit_logger.py          # 📊 Production audit logging
│   └── pipeline_orchestrator.py # 🔄 Pipeline orchestration
│
├── tests/                       # Comprehensive test suite
│   ├── __init__.py
│   ├── test_data_loader.py      # Tests for data loading
│   ├── test_hotel_matcher.py    # Tests for hotel matching
│   ├── test_room_matcher.py     # Tests for room matching
│   └── test_integration.py      # End-to-end pipeline tests
│
├── main.py                      # ▶️ Application entry point
├── requirements.txt             # 📦 Python dependencies
└── README.md                    # 📖 This file
```

## Prerequisites

- **Python**: 3.8 or higher
- **pip**: Latest version
- **Virtual Environment**: Recommended (venv, conda, etc.)
- **Disk Space**: ~2GB for embeddings cache
- **RAM**: Minimum 4GB for production runs

## Installation

### 1. Clone or Setup

```bash
cd hotel_room_mapping_system
```

### 2. Create Virtual Environment (Recommended)

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Verify Installation

```bash
python -c "from src.data_loader import DataLoader; print('✓ Installation successful')"
```

## Configuration

### Environment Settings

Edit `config/settings.py` to customize:

```python
# File paths
HOTELBEDS_HOTELS_FILE = INPUT_DIR / "hotelbeds_hotels.json"
AXISDATA_HOTELS_FILE = INPUT_DIR / "axisdata_hotels.json"

# Matching thresholds (0.0 - 1.0)
HOTEL_NAME_SIMILARITY_THRESHOLD = 0.85  # Stricter matching
ROOM_SEMANTIC_SIMILARITY_THRESHOLD = 0.80

# Embedding configuration
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_BATCH_SIZE = 32

# Logging
LOG_LEVEL = "INFO"  # DEBUG, INFO, WARNING, ERROR, CRITICAL
```

### Environment Variables

```bash
# Set logging level via environment variable
export LOG_LEVEL=DEBUG

# Set embedding model cache directory
export SENTENCE_TRANSFORMERS_HOME=/custom/cache/path
```

## Usage

### Running the Pipeline

```bash
# Standard execution
python main.py

# With debug logging
LOG_LEVEL=DEBUG python main.py

# With timing information
time python main.py
```

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_hotel_matcher.py -v

# Run with coverage report
pytest tests/ --cov=src --cov-report=html

# Run in parallel (faster)
pytest tests/ -n auto
```

### Running Specific Test

```bash
pytest tests/test_hotel_matcher.py::test_match_by_name -v
```

## API Reference

### DataLoader

```python
from src.data_loader import DataLoader

# Load hotels from file
hotels = DataLoader.load_hotelbeds_hotels(Path("data/inputs/hotelbeds_hotels.json"))

# Load room types
rooms = DataLoader.load_room_types(Path("data/inputs/hotelbeds_room_types.json"))
```

### DataNormalizer

```python
from src.data_normalizer import DataNormalizer

# Normalize text
text = DataNormalizer.normalize_text("Grand Hotel Paris™")
# Result: "grand hotel paris"

# Clean hotel data
clean_hotels = DataNormalizer.clean_hotels(hotels)
```

### HotelMatcher

```python
from src.hotel_matcher import HotelMatcher

# Match hotels
matches = HotelMatcher.match_hotels(hotelbeds_hotels, axisdata_hotels)

# Access results
for hotelbeds_hotel, axisdata_match in matches:
    if axisdata_match:
        print(f"Matched: {hotelbeds_hotel['name']}")
```

### RoomMatcher

```python
from src.room_matcher import RoomMatcher

matcher = RoomMatcher()

# Match rooms
room_matches = matcher.match_rooms(hotelbeds_rooms, enriched_rooms)

# Calculate similarity
similarity = matcher.cosine_similarity(vec1, vec2)  # Returns 0.0 - 1.0
```

### PipelineOrchestrator

```python
from src.pipeline_orchestrator import PipelineOrchestrator

orchestrator = PipelineOrchestrator()
output = orchestrator.run()  # Returns complete mapping result
```

## Output Format

### Structure

```json
{
  "metadata": {
    "generated_at": "2024-11-18T10:30:00.123456",
    "total_hotels": 150,
    "matched_hotels": 142,
    "total_rooms": 1250,
    "pipeline_version": "1.0.0"
  },
  "mappings": [
    {
      "hotelbeds_hotel_id": "HB001",
      "hotelbeds_hotel_name": "Grand Hotel Paris",
      "axisdata_hotel_id": "AD001",
      "axisdata_hotel_name": "Grand Hotel Paris",
      "matched": true,
      "confidence": 0.95,
      "rooms": [
        {
          "hotelbeds_room_code": "DBL01",
          "hotelbeds_room_name": "Double Room",
          "room_class": "double",
          "matched_room": "ER01",
          "confidence": 0.87
        }
      ]
    }
  ]
}
```

### Interpretation

- **confidence** (0.0-1.0): Higher values indicate more reliable matches
  - 0.9+: Excellent match
  - 0.8-0.9: Good match
  - 0.7-0.8: Fair match, review recommended
  - <0.7: Poor match, manual review needed

## Logging & Audit Trail

### Console Logs

Real-time output during pipeline execution:

```
2024-11-18 10:30:00 - root - INFO - Hotel Room Mapping System starting...
2024-11-18 10:30:01 - src.data_loader - INFO - Successfully loaded data from data/inputs/hotelbeds_hotels.json
2024-11-18 10:30:02 - src.data_normalizer - INFO - Cleaned 150 hotels from 150 total
```

### Audit Logs (`data/logs/mapping_audit.jsonl`)

Each line is a structured event for compliance and analysis:

```json
{"timestamp": "2024-11-18T10:30:02.123456", "event_type": "hotel_match", "data": {"hotelbeds_id": "HB001", "axisdata_id": "AD001", "confidence": 0.95}}
{"timestamp": "2024-11-18T10:30:02.234567", "event_type": "room_match", "data": {"hotel_id": "HB001", "matched": true, "confidence": 0.87}}
```

### Error Logs (`data/logs/mapping_errors.log`)

Errors and warnings from pipeline execution:

```
2024-11-18 10:30:03 - src.hotel_matcher - WARNING - No match found for hotel "Unknown Place" (best score: 0.65)
```

## Development

### Code Style

The project follows PEP 8 with:
- Type hints on all functions
- Docstrings for modules and functions
- Comprehensive error handling
- Logging at appropriate levels

### Adding New Features

#### 1. Add a New Matcher

```python
# src/custom_matcher.py
from typing import Dict, Any

class CustomMatcher:
    """Custom matching strategy."""
    
    @staticmethod
    def match(item1: Dict[str, Any], item2: Dict[str, Any]) -> float:
        """Return similarity score 0.0-1.0"""
        return 0.5  # Implement logic
```

#### 2. Integrate into Pipeline

```python
# src/pipeline_orchestrator.py
from custom_matcher import CustomMatcher

# Add to run() method
custom_score = CustomMatcher.match(item1, item2)
```

#### 3. Add Tests

```python
# tests/test_custom_matcher.py
def test_custom_matcher():
    score = CustomMatcher.match(item1, item2)
    assert 0 <= score <= 1
```

### Extending Room Enrichment

Enhance `room_enricher.py`:

```python
# Add to ROOM_PATTERNS
"executive": ["executive", "premium", "luxury"],
"accessible": ["accessible", "wheelchair", "mobility"]

# Add to extract_amenities()
"parking": ["parking", "garage", "lot"],
"spa": ["spa", "jacuzzi", "sauna"],
```

## Performance

### Benchmarks

On standard hardware (4-core CPU, 8GB RAM):

- **Hotel Matching**: ~1,000 hotels/second
- **Room Matching**: ~500 rooms/second (with embeddings)
- **Full Pipeline**: 150 hotels (1,250 rooms) in ~3-5 seconds

### Optimization Tips

1. **Increase Thresholds**: Skip low-confidence matches
   ```python
   HOTEL_NAME_SIMILARITY_THRESHOLD = 0.90  # More selective
   ```

2. **Batch Processing**: Process large datasets in chunks
   ```python
   for batch in chunks(hotels, 100):
       process(batch)
   ```

3. **Cache Embeddings**: Reuse across runs
   ```python
   embedding_service.get_embeddings_batch(texts)  # Auto-caches
   ```

4. **Parallel Testing**: Run tests concurrently
   ```bash
   pytest tests/ -n auto
   ```

## Troubleshooting

### Issue: "No matching hotels found"

**Cause**: Similarity threshold too high

**Solution**:
```python
# Lower threshold in config/settings.py
HOTEL_NAME_SIMILARITY_THRESHOLD = 0.75  # Default: 0.85
```

### Issue: "Memory error during embedding generation"

**Cause**: Large dataset processing

**Solution**:
```python
# Reduce batch size in config/settings.py
EMBEDDING_BATCH_SIZE = 8  # Default: 32
```

### Issue: "Import errors for pytest"

**Cause**: Dev dependencies not installed

**Solution**:
```bash
pip install pytest pytest-cov
```

### Issue: "Slow performance on first run"

**Cause**: First run generates all embeddings

**Solution**: This is normal. Subsequent runs use cached embeddings and are 10x faster.

## Contributing

### Guidelines

1. Fork the repository
2. Create feature branch: `git checkout -b feature/my-feature`
3. Write tests for new functionality
4. Ensure all tests pass: `pytest tests/`
5. Submit pull request with description

### Testing Requirements

- Minimum 80% code coverage
- All tests must pass
- Type hints required
- Docstrings for public methods

## License

Proprietary - Hotel Room Mapping System

## Support

For issues, questions, or contributions, please contact the development team.
