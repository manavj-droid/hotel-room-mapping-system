"""
Data Loader Module - REVISED WITH OCCUPANCY SUPPORT
===================================================
Loads and validates input JSON files for hotel and room data from multiple providers.
Includes comprehensive occupancy data extraction and normalization.

Author: Senior Engineer
Date: 2025-11-19
"""
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class ValidationResult:
    """Result of data validation."""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    total_records: int
    valid_records: int

class DataLoaderException(Exception):
    """Custom exception for data loading errors."""
    pass

class DataLoader:
    """
    Load and validate input data from JSON files.

    HOTELBEDS FORMAT:
    [
        {
            "placeId": "ChIJ...",
            "hotelCode": "HB005662",
            "name": "Apartamentos Soldoiro",
            "rooms": [
                {
                    "id": "...",
                    "name": "APT",
                    "roomCode": "APT-VM-1",
                    "providers": [
                        {
                            "name": "hotelbeds",
                            "code": "9524"
                        }
                    ],
                    "occupancies": {
                        "minOccupancy": 1,
                        "maxOccupancy": 4,
                        "minAdult": 1,
                        "maxAdults": 4,
                        "maxChildren": 2
                    }
                }
            ]
        }
    ]

    AXISDATA FORMAT:
    [
        {
            "placeId": "ChIJ...",
            "hotelCode": "HB000004",
            "name": "Soldoiro",
            "rooms": [
                {
                    "id": "...",
                    "name": "1 Bedroom Apartment (SeaView, Balcony)",
                    "code": "AP00T1SVB0",
                    "providers": [
                        {
                            "name": "axisData",
                            "code": "AMTSPT0042"
                        }
                    ],
                    "occupancies": {
                        "MaxOccupancy": "4",
                        "MinOccupancy": "1",
                        "MinAdultOccupancy": "1",
                        "MaxAdultOccupancy": "4",
                        "MinChildOccupancy": "0",
                        "MaxChildOccupancy": "2",
                        "MaxChildAge": "2",
                        "InfantOccupancy": "1",
                        "MinInfantOccupancy": "0",
                        "MaxInfantOccupancy": "1",
                        "StandardOccupancy": "1"
                    },
                    "description": "1 Bedroom Apartment (SeaView, Balcony)"
                }
            ]
        }
    ]
    """

    def __init__(self, strict_mode: bool = False):
        self.strict_mode = strict_mode
        self.stats = {
            'hotelbeds_loaded': 0,
            'axisdata_loaded': 0,
            'hotelbeds_skipped': 0,
            'axisdata_skipped': 0,
            'hotelbeds_rooms_total': 0,
            'axisdata_rooms_total': 0,
            'hotelbeds_rooms_with_occupancy': 0,
            'axisdata_rooms_with_occupancy': 0,
        }

    # ========================================================================
    # HOTELBEDS HOTELS LOADER
    # ========================================================================

    def load_hotelbeds_hotels(self, file_path: Path) -> Tuple[List[Dict[str, Any]], ValidationResult]:
        logger.info(f"Loading Hotelbeds hotels from: {file_path}")
        raw_data = self._load_json_file(file_path, "Hotelbeds hotels")
        
        if not isinstance(raw_data, list):
            raise DataLoaderException(
                f"Hotelbeds hotels data must be a list, got {type(raw_data).__name__}"
            )
        
        validated_data = []
        errors = []
        warnings = []
        
        for idx, hotel in enumerate(raw_data):
            validation_errors = self._validate_hotelbeds_hotel(hotel, idx)
            
            if validation_errors:
                error_msg = f"Hotel #{idx}: {', '.join(validation_errors)}"
                errors.append(error_msg)
                self.stats['hotelbeds_skipped'] += 1
                
                if self.strict_mode:
                    logger.error(error_msg)
                else:
                    logger.warning(error_msg)
                    warnings.append(error_msg)
            else:
                room_warnings = self._validate_hotelbeds_rooms(hotel, idx)
                if room_warnings:
                    warnings.extend(room_warnings)
                
                validated_data.append(hotel)
                self.stats['hotelbeds_loaded'] += 1
                self.stats['hotelbeds_rooms_total'] += len(hotel.get('rooms', []))
        
        validation_result = ValidationResult(
            is_valid=len(errors) == 0 if self.strict_mode else True,
            errors=errors,
            warnings=warnings,
            total_records=len(raw_data),
            valid_records=len(validated_data),
        )
        
        logger.info(
            f"Loaded {len(validated_data)}/{len(raw_data)} Hotelbeds hotels. "
            f"Total rooms: {self.stats['hotelbeds_rooms_total']}. "
            f"Rooms with occupancy: {self.stats['hotelbeds_rooms_with_occupancy']}. "
            f"Errors: {len(errors)}, Warnings: {len(warnings)}"
        )
        
        if self.strict_mode and errors:
            raise DataLoaderException(
                f"Hotelbeds hotel validation failed with {len(errors)} errors"
            )
        
        return validated_data, validation_result

    def _validate_hotelbeds_hotel(self, hotel: Any, idx: int) -> List[str]:
        errors = []
        
        if not isinstance(hotel, dict):
            errors.append(f"Must be a dict, got {type(hotel).__name__}")
            return errors
        
        required_fields = ['placeId', 'hotelCode', 'name', 'rooms']
        for field in required_fields:
            if field not in hotel:
                errors.append(f"Missing required field: {field}")
            elif not hotel[field] and field != 'rooms':
                errors.append(f"Empty value for required field: {field}")
        
        if 'placeId' in hotel and not isinstance(hotel['placeId'], str):
            errors.append(f"placeId must be string, got {type(hotel['placeId']).__name__}")
        
        if 'hotelCode' in hotel and not isinstance(hotel['hotelCode'], str):
            errors.append(f"hotelCode must be string, got {type(hotel['hotelCode']).__name__}")
        
        if 'name' in hotel and not isinstance(hotel['name'], str):
            errors.append(f"name must be string, got {type(hotel['name']).__name__}")
        
        if 'rooms' in hotel and not isinstance(hotel['rooms'], list):
            errors.append(f"rooms must be list, got {type(hotel['rooms']).__name__}")
        
        return errors

    def _validate_hotelbeds_rooms(self, hotel: Dict[str, Any], hotel_idx: int) -> List[str]:
        warnings = []
        rooms = hotel.get('rooms', [])
        
        if not rooms:
            warnings.append(
                f"Hotel #{hotel_idx} ({hotel.get('hotelCode', 'UNKNOWN')}): No rooms defined"
            )
            return warnings
        
        for room_idx, room in enumerate(rooms):
            if not isinstance(room, dict):
                warnings.append(
                    f"Hotel #{hotel_idx}, Room #{room_idx}: Must be a dict"
                )
                continue
            
            # Validate basic room fields
            if 'id' not in room:
                warnings.append(
                    f"Hotel #{hotel_idx}, Room #{room_idx}: Missing id"
                )
            
            if 'name' not in room or not room['name']:
                warnings.append(
                    f"Hotel #{hotel_idx}, Room #{room_idx}: Missing or empty name"
                )
            
            if 'roomCode' not in room or not room['roomCode']:
                warnings.append(
                    f"Hotel #{hotel_idx}, Room #{room_idx}: Missing or empty roomCode"
                )
            
            # Validate providers
            providers = room.get('providers', [])
            if not isinstance(providers, list):
                warnings.append(
                    f"Hotel #{hotel_idx}, Room #{room_idx}: providers must be a list"
                )
                continue
            
            if not providers:
                warnings.append(
                    f"Hotel #{hotel_idx}, Room #{room_idx}: No providers defined"
                )
                continue
            
            provider = providers[0]
            if not isinstance(provider, dict):
                warnings.append(
                    f"Hotel #{hotel_idx}, Room #{room_idx}: provider must be a dict"
                )
                continue
            
            if provider.get('name') != 'hotelbeds':
                warnings.append(
                    f"Hotel #{hotel_idx}, Room #{room_idx}: First provider should be 'hotelbeds'"
                )
            
            if 'code' not in provider or not provider['code']:
                warnings.append(
                    f"Hotel #{hotel_idx}, Room #{room_idx}: Missing provider code"
                )
            
            # Validate occupancies
            occupancy_warnings = self._validate_hotelbeds_occupancies(
                room, hotel_idx, room_idx
            )
            if occupancy_warnings:
                warnings.extend(occupancy_warnings)
            else:
                self.stats['hotelbeds_rooms_with_occupancy'] += 1
        
        return warnings

    def _validate_hotelbeds_occupancies(
        self, room: Dict[str, Any], hotel_idx: int, room_idx: int
    ) -> List[str]:
        """Validate Hotelbeds occupancy data."""
        warnings = []
        occupancies = room.get('occupancies')
        
        if not occupancies:
            warnings.append(
                f"Hotel #{hotel_idx}, Room #{room_idx}: Missing occupancies data"
            )
            return warnings
        
        if not isinstance(occupancies, dict):
            warnings.append(
                f"Hotel #{hotel_idx}, Room #{room_idx}: occupancies must be a dict"
            )
            return warnings
        
        # Expected fields in Hotelbeds occupancies
        expected_fields = {
            'minOccupancy': int,
            'maxOccupancy': int,
            'minAdult': int,
            'maxAdults': int,
            'maxChildren': int,
        }
        
        for field, expected_type in expected_fields.items():
            if field not in occupancies:
                warnings.append(
                    f"Hotel #{hotel_idx}, Room #{room_idx}: "
                    f"Missing occupancy field '{field}'"
                )
            else:
                value = occupancies[field]
                if not isinstance(value, expected_type):
                    warnings.append(
                        f"Hotel #{hotel_idx}, Room #{room_idx}: "
                        f"occupancy field '{field}' should be {expected_type.__name__}, "
                        f"got {type(value).__name__}"
                    )
                elif value < 0:
                    warnings.append(
                        f"Hotel #{hotel_idx}, Room #{room_idx}: "
                        f"occupancy field '{field}' cannot be negative"
                    )
        
        # Logical validations
        if 'minOccupancy' in occupancies and 'maxOccupancy' in occupancies:
            if occupancies['minOccupancy'] > occupancies['maxOccupancy']:
                warnings.append(
                    f"Hotel #{hotel_idx}, Room #{room_idx}: "
                    f"minOccupancy ({occupancies['minOccupancy']}) > "
                    f"maxOccupancy ({occupancies['maxOccupancy']})"
                )
        
        if 'minAdult' in occupancies and 'maxAdults' in occupancies:
            if occupancies['minAdult'] > occupancies['maxAdults']:
                warnings.append(
                    f"Hotel #{hotel_idx}, Room #{room_idx}: "
                    f"minAdult ({occupancies['minAdult']}) > "
                    f"maxAdults ({occupancies['maxAdults']})"
                )
        
        return warnings

    # ========================================================================
    # AXISDATA HOTELS LOADER
    # ========================================================================

    def load_axisdata_hotels(self, file_path: Path) -> Tuple[List[Dict[str, Any]], ValidationResult]:
        logger.info(f"Loading AxisData hotels from: {file_path}")
        raw_data = self._load_json_file(file_path, "AxisData hotels")
        
        if not isinstance(raw_data, list):
            raise DataLoaderException(
                f"AxisData hotels data must be a list, got {type(raw_data).__name__}"
            )
        
        validated_data = []
        errors = []
        warnings = []
        
        for idx, hotel in enumerate(raw_data):
            validation_errors = self._validate_axisdata_hotel(hotel, idx)
            
            if validation_errors:
                error_msg = f"Hotel #{idx}: {', '.join(validation_errors)}"
                errors.append(error_msg)
                self.stats['axisdata_skipped'] += 1
                
                if self.strict_mode:
                    logger.error(error_msg)
                else:
                    logger.warning(error_msg)
                    warnings.append(error_msg)
            else:
                room_warnings = self._validate_axisdata_rooms(hotel, idx)
                if room_warnings:
                    warnings.extend(room_warnings)
                
                validated_data.append(hotel)
                self.stats['axisdata_loaded'] += 1
                self.stats['axisdata_rooms_total'] += len(hotel.get('rooms', []))
        
        validation_result = ValidationResult(
            is_valid=len(errors) == 0 if self.strict_mode else True,
            errors=errors,
            warnings=warnings,
            total_records=len(raw_data),
            valid_records=len(validated_data),
        )
        
        logger.info(
            f"Loaded {len(validated_data)}/{len(raw_data)} AxisData hotels. "
            f"Total rooms: {self.stats['axisdata_rooms_total']}. "
            f"Rooms with occupancy: {self.stats['axisdata_rooms_with_occupancy']}. "
            f"Errors: {len(errors)}, Warnings: {len(warnings)}"
        )
        
        if self.strict_mode and errors:
            raise DataLoaderException(
                f"AxisData hotel validation failed with {len(errors)} errors"
            )
        
        return validated_data, validation_result

    def _validate_axisdata_hotel(self, hotel: Any, idx: int) -> List[str]:
        errors = []
        
        if not isinstance(hotel, dict):
            errors.append(f"Must be a dict, got {type(hotel).__name__}")
            return errors
        
        required_fields = ['placeId', 'hotelCode', 'name', 'rooms']
        for field in required_fields:
            if field not in hotel:
                errors.append(f"Missing required field: {field}")
            elif not hotel[field] and field != 'rooms':
                errors.append(f"Empty value for required field: {field}")
        
        if 'placeId' in hotel and not isinstance(hotel['placeId'], str):
            errors.append(f"placeId must be string, got {type(hotel['placeId']).__name__}")
        
        if 'hotelCode' in hotel and not isinstance(hotel['hotelCode'], str):
            errors.append(f"hotelCode must be string, got {type(hotel['hotelCode']).__name__}")
        
        if 'name' in hotel and not isinstance(hotel['name'], str):
            errors.append(f"name must be string, got {type(hotel['name']).__name__}")
        
        if 'rooms' in hotel and not isinstance(hotel['rooms'], list):
            errors.append(f"rooms must be list, got {type(hotel['rooms']).__name__}")
        
        return errors

    def _validate_axisdata_rooms(self, hotel: Dict[str, Any], hotel_idx: int) -> List[str]:
        warnings = []
        rooms = hotel.get('rooms', [])
        
        if not rooms:
            warnings.append(
                f"Hotel #{hotel_idx} ({hotel.get('hotelCode', 'UNKNOWN')}): No rooms defined"
            )
            return warnings
        
        for room_idx, room in enumerate(rooms):
            if not isinstance(room, dict):
                warnings.append(
                    f"Hotel #{hotel_idx}, Room #{room_idx}: Must be a dict"
                )
                continue
            
            # Validate basic room fields
            if 'id' not in room:
                warnings.append(
                    f"Hotel #{hotel_idx}, Room #{room_idx}: Missing id"
                )
            
            if 'name' not in room or not room['name']:
                warnings.append(
                    f"Hotel #{hotel_idx}, Room #{room_idx}: Missing or empty name"
                )
            
            if 'code' not in room or not room['code']:
                warnings.append(
                    f"Hotel #{hotel_idx}, Room #{room_idx}: Missing or empty code"
                )
            
            # Validate providers
            providers = room.get('providers', [])
            if not isinstance(providers, list):
                warnings.append(
                    f"Hotel #{hotel_idx}, Room #{room_idx}: providers must be a list"
                )
                continue
            
            if not providers:
                warnings.append(
                    f"Hotel #{hotel_idx}, Room #{room_idx}: No providers defined"
                )
                continue
            
            provider = providers[0]
            if not isinstance(provider, dict):
                warnings.append(
                    f"Hotel #{hotel_idx}, Room #{room_idx}: provider must be a dict"
                )
                continue
            
            if provider.get('name') != 'axisData':
                warnings.append(
                    f"Hotel #{hotel_idx}, Room #{room_idx}: First provider should be 'axisData'"
                )
            
            if 'code' not in provider or not provider['code']:
                warnings.append(
                    f"Hotel #{hotel_idx}, Room #{room_idx}: Missing provider code"
                )
            
            # Validate occupancies
            occupancy_warnings = self._validate_axisdata_occupancies(
                room, hotel_idx, room_idx
            )
            if occupancy_warnings:
                warnings.extend(occupancy_warnings)
            else:
                self.stats['axisdata_rooms_with_occupancy'] += 1
        
        return warnings

    def _validate_axisdata_occupancies(
        self, room: Dict[str, Any], hotel_idx: int, room_idx: int
    ) -> List[str]:
        """Validate AxisData occupancy data."""
        warnings = []
        occupancies = room.get('occupancies')
        
        if not occupancies:
            warnings.append(
                f"Hotel #{hotel_idx}, Room #{room_idx}: Missing occupancies data"
            )
            return warnings
        
        if not isinstance(occupancies, dict):
            warnings.append(
                f"Hotel #{hotel_idx}, Room #{room_idx}: occupancies must be a dict"
            )
            return warnings
        
        # Expected fields in AxisData occupancies (note: values are strings!)
        expected_fields = [
            'MaxOccupancy',
            'MinOccupancy',
            'MinAdultOccupancy',
            'MaxAdultOccupancy',
            'MinChildOccupancy',
            'MaxChildOccupancy',
            'MaxChildAge',
            'StandardOccupancy',
        ]
        
        for field in expected_fields:
            if field not in occupancies:
                warnings.append(
                    f"Hotel #{hotel_idx}, Room #{room_idx}: "
                    f"Missing occupancy field '{field}'"
                )
            else:
                value = occupancies[field]
                # AxisData stores numbers as strings
                if not isinstance(value, str):
                    warnings.append(
                        f"Hotel #{hotel_idx}, Room #{room_idx}: "
                        f"occupancy field '{field}' should be string, "
                        f"got {type(value).__name__}"
                    )
                else:
                    # Try to parse as integer
                    try:
                        int_value = int(value)
                        if int_value < 0:
                            warnings.append(
                                f"Hotel #{hotel_idx}, Room #{room_idx}: "
                                f"occupancy field '{field}' cannot be negative"
                            )
                    except ValueError:
                        warnings.append(
                            f"Hotel #{hotel_idx}, Room #{room_idx}: "
                            f"occupancy field '{field}' is not a valid integer: '{value}'"
                        )
        
        # Logical validations (convert strings to ints for comparison)
        try:
            if 'MinOccupancy' in occupancies and 'MaxOccupancy' in occupancies:
                min_occ = int(occupancies['MinOccupancy'])
                max_occ = int(occupancies['MaxOccupancy'])
                if min_occ > max_occ:
                    warnings.append(
                        f"Hotel #{hotel_idx}, Room #{room_idx}: "
                        f"MinOccupancy ({min_occ}) > MaxOccupancy ({max_occ})"
                    )
            
            if 'MinAdultOccupancy' in occupancies and 'MaxAdultOccupancy' in occupancies:
                min_adult = int(occupancies['MinAdultOccupancy'])
                max_adult = int(occupancies['MaxAdultOccupancy'])
                if min_adult > max_adult:
                    warnings.append(
                        f"Hotel #{hotel_idx}, Room #{room_idx}: "
                        f"MinAdultOccupancy ({min_adult}) > MaxAdultOccupancy ({max_adult})"
                    )
        except ValueError:
            pass  # Already warned about invalid integers above
        
        return warnings

    # ========================================================================
    # OCCUPANCY NORMALIZATION
    # ========================================================================

    def normalize_hotelbeds_occupancy(self, occupancies: Dict[str, Any]) -> Dict[str, int]:
        """
        Normalize Hotelbeds occupancy data to standard format.
        
        Returns:
            {
                'min_occupancy': int,
                'max_occupancy': int,
                'min_adults': int,
                'max_adults': int,
                'max_children': int,
            }
        """
        return {
            'min_occupancy': occupancies.get('minOccupancy', 0),
            'max_occupancy': occupancies.get('maxOccupancy', 0),
            'min_adults': occupancies.get('minAdult', 0),
            'max_adults': occupancies.get('maxAdults', 0),
            'max_children': occupancies.get('maxChildren', 0),
        }

    def normalize_axisdata_occupancy(self, occupancies: Dict[str, str]) -> Dict[str, int]:
        """
        Normalize AxisData occupancy data to standard format.
        Note: AxisData stores numbers as strings.
        
        Returns:
            {
                'min_occupancy': int,
                'max_occupancy': int,
                'min_adults': int,
                'max_adults': int,
                'max_children': int,
                'max_child_age': int,
                'min_infants': int,
                'max_infants': int,
                'max_infant_age': int,
                'standard_occupancy': int,
            }
        """
        def safe_int(value: str, default: int = 0) -> int:
            try:
                return int(value)
            except (ValueError, TypeError):
                return default
        
        return {
            'min_occupancy': safe_int(occupancies.get('MinOccupancy')),
            'max_occupancy': safe_int(occupancies.get('MaxOccupancy')),
            'min_adults': safe_int(occupancies.get('MinAdultOccupancy')),
            'max_adults': safe_int(occupancies.get('MaxAdultOccupancy')),
            'min_children': safe_int(occupancies.get('MinChildOccupancy')),
            'max_children': safe_int(occupancies.get('MaxChildOccupancy')),
            'max_child_age': safe_int(occupancies.get('MaxChildAge')),
            'min_infants': safe_int(occupancies.get('MinInfantOccupancy')),
            'max_infants': safe_int(occupancies.get('MaxInfantOccupancy')),
            'max_infant_age': safe_int(occupancies.get('MaxInfantAge')),
            'standard_occupancy': safe_int(occupancies.get('StandardOccupancy')),
        }

    # ========================================================================
    # UTILITY METHODS
    # ========================================================================

    def _load_json_file(self, file_path: Path, data_type: str) -> Any:
        """
        Load and parse JSON file with error handling.
        
        Args:
            file_path: Path to JSON file
            data_type: Description of the data type (for error messages)
            
        Returns:
            Parsed JSON data
            
        Raises:
            DataLoaderException: If file cannot be loaded or parsed
        """
        if not isinstance(file_path, Path):
            file_path = Path(file_path)
        
        if not file_path.exists():
            raise DataLoaderException(
                f"{data_type} file not found: {file_path}"
            )
        
        if not file_path.is_file():
            raise DataLoaderException(
                f"{data_type} path is not a file: {file_path}"
            )
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return data
        
        except json.JSONDecodeError as e:
            raise DataLoaderException(
                f"Invalid JSON in {data_type} file: {file_path}. Error: {e}"
            )
        
        except UnicodeDecodeError as e:
            raise DataLoaderException(
                f"Encoding error in {data_type} file: {file_path}. Error: {e}"
            )
        
        except Exception as e:
            raise DataLoaderException(
                f"Error loading {data_type} file: {file_path}. Error: {e}"
            )

    def get_stats(self) -> Dict[str, int]:
        """Get loading statistics."""
        return self.stats.copy()

    def reset_stats(self):
        """Reset loading statistics."""
        for key in self.stats:
            self.stats[key] = 0


# ========================================================================
# CONVENIENCE FUNCTIONS
# ========================================================================

def load_all_data(
    hotelbeds_path: Path,
    axisdata_path: Path,
    strict_mode: bool = False,
) -> Tuple[List[Dict], List[Dict], Dict[str, ValidationResult]]:
    """
    Convenience function to load both input files.
    
    Args:
        hotelbeds_path: Path to Hotelbeds hotels JSON
        axisdata_path: Path to AxisData hotels JSON
        strict_mode: Whether to use strict validation
        
    Returns:
        Tuple of:
        - hotelbeds_hotels (list)
        - axisdata_hotels (list)
        - validation_results (dict with keys: 'hotelbeds', 'axisdata')
    """
    loader = DataLoader(strict_mode=strict_mode)
    
    hotelbeds_hotels, hb_validation = loader.load_hotelbeds_hotels(hotelbeds_path)
    axisdata_hotels, ad_validation = loader.load_axisdata_hotels(axisdata_path)
    
    validation_results = {
        'hotelbeds': hb_validation,
        'axisdata': ad_validation,
    }
    
    logger.info("\n" + "="*80)
    logger.info("LOADING STATISTICS")
    logger.info("="*80)
    stats = loader.get_stats()
    logger.info(f"Hotelbeds: {stats['hotelbeds_loaded']} hotels, "
                f"{stats['hotelbeds_rooms_total']} rooms, "
                f"{stats['hotelbeds_rooms_with_occupancy']} with occupancy data")
    logger.info(f"AxisData: {stats['axisdata_loaded']} hotels, "
                f"{stats['axisdata_rooms_total']} rooms, "
                f"{stats['axisdata_rooms_with_occupancy']} with occupancy data")
    
    return hotelbeds_hotels, axisdata_hotels, validation_results


def print_validation_summary(validation_results: Dict[str, ValidationResult]):
    print("\n" + "=" * 80)
    print("VALIDATION SUMMARY")
    print("=" * 80)
    
    for data_type, result in validation_results.items():
        print(f"\n{data_type.upper()}:")
        print(f"  Total Records: {result.total_records}")
        print(f"  Valid Records: {result.valid_records}")
        print(f"  Errors: {len(result.errors)}")
        print(f"  Warnings: {len(result.warnings)}")
        
        if result.errors and len(result.errors) <= 5:
            print("  Error Details:")
            for error in result.errors:
                print(f"    - {error}")
        
        if result.warnings and len(result.warnings) <= 10:
            print("  Warning Details:")
            for warning in result.warnings[:10]:
                print(f"    - {warning}")
            if len(result.warnings) > 10:
                print(f"    ... and {len(result.warnings) - 10} more warnings")


def extract_occupancy_data(
    hotels: List[Dict[str, Any]], 
    provider: str
) -> List[Dict[str, Any]]:
    """
    Extract occupancy data from all rooms in hotel list.
    
    Args:
        hotels: List of hotel dicts
        provider: 'hotelbeds' or 'axisdata'
    
    Returns:
        List of dicts with structure:
        {
            'place_id': str,
            'hotel_code': str,
            'hotel_name': str,
            'room_id': str,
            'room_name': str,
            'room_code': str,
            'occupancy': dict (normalized occupancy data)
        }
    """
    loader = DataLoader()
    occupancy_records = []
    
    for hotel in hotels:
        place_id = hotel.get('placeId', '')
        hotel_code = hotel.get('hotelCode', '')
        hotel_name = hotel.get('name', '')
        
        for room in hotel.get('rooms', []):
            room_id = room.get('id', '')
            room_name = room.get('name', '')
            room_code = room.get('roomCode' if provider == 'hotelbeds' else 'code', '')
            
            occupancies = room.get('occupancies')
            if not occupancies:
                continue
            
            if provider == 'hotelbeds':
                normalized = loader.normalize_hotelbeds_occupancy(occupancies)
            else:
                normalized = loader.normalize_axisdata_occupancy(occupancies)
            
            occupancy_records.append({
                'place_id': place_id,
                'hotel_code': hotel_code,
                'hotel_name': hotel_name,
                'room_id': room_id,
                'room_name': room_name,
                'room_code': room_code,
                'occupancy': normalized,
            })
    
    return occupancy_records


# ========================================================================
# MAIN EXECUTION
# ========================================================================

if __name__ == "__main__":
    import sys
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    hotelbeds_path = Path("data/inputs/hotelbeds_hotels.json")
    axisdata_path = Path("data/inputs/axisdata_hotels.json")
    
    try:
        hotelbeds, axisdata, validation = load_all_data(
            hotelbeds_path, axisdata_path, strict_mode=False
        )
        
        print_validation_summary(validation)
        
        print(f"\n{'='*80}")
        print("SUCCESSFULLY LOADED")
        print(f"{'='*80}")
        print(f"  - Hotelbeds: {len(hotelbeds)} hotels")
        print(f"  - AxisData: {len(axisdata)} hotels")
        
        # Extract and display occupancy data samples
        print(f"\n{'='*80}")
        print("OCCUPANCY DATA SAMPLES")
        print(f"{'='*80}")
        
        hb_occupancies = extract_occupancy_data(hotelbeds, 'hotelbeds')
        ad_occupancies = extract_occupancy_data(axisdata, 'axisdata')
        
        print(f"\nHotelbeds rooms with occupancy data: {len(hb_occupancies)}")
        if hb_occupancies:
            sample = hb_occupancies[0]
            print(f"\nSample Hotelbeds Room:")
            print(f"  Hotel: {sample['hotel_name']} ({sample['hotel_code']})")
            print(f"  Room: {sample['room_name']} ({sample['room_code']})")
            print(f"  Occupancy: {sample['occupancy']}")
        
        print(f"\nAxisData rooms with occupancy data: {len(ad_occupancies)}")
        if ad_occupancies:
            sample = ad_occupancies[0]
            print(f"\nSample AxisData Room:")
            print(f"  Hotel: {sample['hotel_name']} ({sample['hotel_code']})")
            print(f"  Room: {sample['room_name']} ({sample['room_code']})")
            print(f"  Occupancy: {sample['occupancy']}")
        
    except Exception as e:
        logger.error(f"ERROR: {e}")
        sys.exit(1)
