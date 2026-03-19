"""
Shared Feature Preparation Module

This module contains shared functions for feature preparation that are used
in both training (build_dataset.py) and inference (test_models.py) to ensure
exact consistency in text formatting, embedding generation, and feature scaling.
"""

import json
import re
import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
import joblib
import os


def combine_text_fields(row, text_fields):
    """
    Combine text fields into a single string using exact same format as training.
    
    This function MUST match the exact format used in build_dataset.py to ensure
    embeddings are identical between training and inference.
    
    Args:
        row: Row from DataFrame (dict-like)
        text_fields: List of field names to combine
        
    Returns:
        str: Combined text string
    """
    text_parts = []
    for field in text_fields:
        value = row.get(field, '')
        
        # Handle different value types safely
        # Check if value is array/list first (these can't use pd.notna() directly)
        if isinstance(value, (list, np.ndarray)):
            # Convert array/list to string representation
            if len(value) > 0:
                value_str = str(value).strip()
                if value_str and value_str.lower() not in ['nan', 'none', '[]', '']:
                    text_parts.append(f"{field}: {value_str}")
        else:
            # For scalar values, check if not NA
            try:
                if pd.notna(value):
                    value_str = str(value).strip()
                    if value_str and value_str.lower() not in ['nan', 'none', '']:
                        text_parts.append(f"{field}: {value_str}")
            except (ValueError, TypeError):
                # If pd.notna() fails (e.g., for arrays), just try to convert to string
                try:
                    value_str = str(value).strip()
                    if value_str and value_str.lower() not in ['nan', 'none', '']:
                        text_parts.append(f"{field}: {value_str}")
                except:
                    pass  # Skip this field if we can't convert it
    
    combined_text = " | ".join(text_parts)
    return combined_text


def _format_experience_date(date_val):
    """Format ISO date (e.g. 2017-01-01T00:00:00.000Z) to YYYY-MM-DD. Returns empty string if invalid."""
    if date_val is None or (isinstance(date_val, float) and pd.isna(date_val)):
        return ""
    s = str(date_val).strip()
    if not s or s.lower() in ("nan", "none"):
        return ""
    # Take first 10 chars for YYYY-MM-DD if present
    if len(s) >= 10 and s[4] == "-" and s[7] == "-":
        return s[:10]
    # Try parsing common ISO-like formats
    m = re.match(r"(\d{4})-(\d{2})-(\d{2})", s)
    if m:
        return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
    return s[:10] if len(s) >= 10 else s


def _normalize_experiences_to_list(experiences):
    """Return a list of experience items from various input types (list, ndarray, dict, JSON string)."""
    if experiences is None:
        return []
    try:
        if pd.isna(experiences):
            return []
    except (ValueError, TypeError):
        pass
    if isinstance(experiences, (list, np.ndarray)):
        return list(experiences) if isinstance(experiences, list) else experiences.tolist()
    if isinstance(experiences, str):
        s = experiences.strip()
        if not s or s.lower() in ("nan", "none", "null", ""):
            return []
        try:
            parsed = json.loads(experiences)
            return _normalize_experiences_to_list(parsed)
        except (json.JSONDecodeError, TypeError):
            return []
    if isinstance(experiences, dict):
        for key in ("experiences", "data", "items", "positions", "experience"):
            if key in experiences and experiences[key] is not None:
                return _normalize_experiences_to_list(experiences[key])
        return list(experiences.values()) if experiences else []
    return []


def _exp_get(exp, key, *alt_keys):
    """Get value from experience item (dict or dict-like). Tries key then alternate keys."""
    if exp is None:
        return ""
    if not hasattr(exp, "get") or not callable(getattr(exp, "get")):
        return ""
    val = exp.get(key)
    if val is not None and str(val).strip() and str(val).lower() not in ("nan", "none"):
        return str(val).strip()
    for k in alt_keys:
        val = exp.get(k)
        if val is not None and str(val).strip() and str(val).lower() not in ("nan", "none"):
            return str(val).strip()
    return ""


def _format_experience_location(loc_obj):
    """Format experience location dict (locality, region, country) to a single string. Returns '' if empty."""
    if not loc_obj or not isinstance(loc_obj, dict):
        return ""
    parts = [
        loc_obj.get("locality"),
        loc_obj.get("region"),
        loc_obj.get("country"),
    ]
    parts = [str(p).strip() for p in parts if p is not None and str(p).strip()]
    return ", ".join(parts) if parts else ""


def format_experiences_json(experiences):
    """
    Format all_experiences JSON into a text representation.
    Dates are shown as YYYY-MM-DD only.
    Handles list, np.ndarray, dict-wrapped list, JSON string, and alternate key names.
    
    This function MUST match the exact format used in build_dataset.py to ensure
    embeddings are identical between training and inference.
    
    Args:
        experiences: JSON object or string containing experiences
        
    Returns:
        str: Formatted text representation of experiences
    """
    exp_list = _normalize_experiences_to_list(experiences)
    if not exp_list:
        return ""

    formatted_parts = []
    for exp in exp_list:  # All experiences, no limit
        if not hasattr(exp, "get") or not callable(getattr(exp, "get")):
            continue
        company = _exp_get(exp, "company_name", "company")
        position = _exp_get(exp, "position", "title", "role")
        description = _exp_get(exp, "description")
        start_date = _format_experience_date(exp.get("start_date") if hasattr(exp, "get") else None)
        end_date = _format_experience_date(exp.get("end_date") if hasattr(exp, "get") else None)
        loc_str = _format_experience_location(exp.get("location") if hasattr(exp, "get") else None)
        industry_tags = exp.get("industry_tags") if hasattr(exp, "get") else None
        if isinstance(industry_tags, (list, np.ndarray)) and len(industry_tags) > 0:
            industry_str = ", ".join(str(t).strip() for t in industry_tags if t is not None and str(t).strip())
        else:
            industry_str = ""

        parts = []
        if position:
            parts.append(position)
        if company:
            parts.append(f"at {company}")
        if start_date or end_date:
            date_str = f"{start_date} - {end_date}" if end_date else f"{start_date} - Present"
            parts.append(f"({date_str})")
        if loc_str:
            parts.append(f"in {loc_str}")
        if industry_str:
            parts.append(f"[{industry_str}]")
        if description:
            parts.append(f": {description}")

        if parts:
            formatted_parts.append(" ".join(parts))

    return " | ".join(formatted_parts)


def convert_boolean_field(value):
    """
    Convert a value to 0/1 integer, handling various boolean representations.
    
    Args:
        value: Can be bool, int, str ('true'/'false'), or None
        
    Returns:
        int: 0 or 1
    """
    if pd.isna(value) or value is None:
        return 0
    
    # If already an integer (0 or 1)
    if isinstance(value, (int, np.integer)):
        return int(value) if value in [0, 1] else 0
    
    # If already a boolean
    if isinstance(value, bool):
        return 1 if value else 0
    
    # If string, check for various representations
    if isinstance(value, str):
        value_lower = value.lower().strip()
        if value_lower in ['true', '1', 'yes', 't']:
            return 1
        elif value_lower in ['false', '0', 'no', 'f', '']:
            return 0
    
    # Default to 0 for unknown values
    return 0


def convert_building_since_to_years(building_since):
    """
    Convert building_since date to years since building started.
    
    Handles various date formats:
    - "2024-05-31" (ISO format)
    - "Dec 2019" (month year)
    - "2025-02-21" (ISO format)
    - etc.
    
    Args:
        building_since: Date string or None
        
    Returns:
        float: Years since building started (0 if invalid/None)
    """
    if pd.isna(building_since) or building_since is None:
        return 0.0
    
    from datetime import datetime
    import re
    
    try:
        building_since_str = str(building_since).strip()
        
        # Try to parse as ISO date (YYYY-MM-DD)
        if re.match(r'^\d{4}-\d{2}-\d{2}', building_since_str):
            date_obj = datetime.strptime(building_since_str[:10], '%Y-%m-%d')
        # Try to parse as "Month YYYY" format (e.g., "Dec 2019", "December 2019")
        elif re.match(r'^[A-Za-z]+\s+\d{4}', building_since_str):
            # Try abbreviated month first, then full month name
            for fmt in ['%b %Y', '%B %Y']:
                try:
                    date_obj = datetime.strptime(building_since_str, fmt)
                    break
                except ValueError:
                    continue
            else:
                return 0.0
        # Try to parse as just year (YYYY)
        elif re.match(r'^\d{4}$', building_since_str):
            date_obj = datetime.strptime(building_since_str, '%Y')
        # Try other common formats
        else:
            for fmt in ['%Y-%m-%d', '%Y-%m-%d %H:%M:%S', '%m/%d/%Y', '%d/%m/%Y']:
                try:
                    date_obj = datetime.strptime(building_since_str, fmt)
                    break
                except ValueError:
                    continue
            else:
                return 0.0
        
        # Calculate years since that date
        now = datetime.now()
        delta = now - date_obj
        years = delta.days / 365.25
        
        # Return 0 if negative (future dates) or cap at reasonable max (e.g., 50 years)
        return max(0.0, min(years, 50.0))
        
    except (ValueError, TypeError, AttributeError):
        return 0.0


def get_numeric_fields():
    """Get list of numeric field names."""
    return [
        'years_of_experience',
        'years_building',  # years since building started
        'lastroundvaluation',
        'latestdealamount',
        'headcount'
    ]


def get_text_fields_group1():
    """Get list of text field names for first embedding group (company/founder info)."""
    return ['about', 'post_data', 'location', 'company_tags', 'school_tags']


def get_text_fields_group2():
    """Get list of text field names for second embedding group (product/market info)."""
    return ['product', 'market', 'embeddednews', 'funding', 'tree_path', 'tree_thesis']


def get_text_fields():
    """Get all text field names (for backward compatibility)."""
    return get_text_fields_group1() + get_text_fields_group2()


def get_boolean_fields():
    """Get list of boolean field names."""
    return ['technical', 'repeat_founder']


def save_scaler(scaler, scaler_path):
    """
    Save a fitted scaler to disk.
    
    Args:
        scaler: Fitted RobustScaler
        scaler_path: Path to save the scaler
    """
    os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
    joblib.dump(scaler, scaler_path)
    print(f"  Saved scaler to {scaler_path}")


def load_scaler(scaler_path):
    """
    Load a fitted scaler from disk.
    
    Args:
        scaler_path: Path to the saved scaler
        
    Returns:
        RobustScaler: Loaded scaler
    """
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scaler not found: {scaler_path}. Please train models first.")
    scaler = joblib.load(scaler_path)
    print(f"  Loaded scaler from {scaler_path}")
    return scaler


def prepare_numeric_features(df, scaler=None, fit_scaler=False, scaler_path=None):
    """
    Prepare numeric features with Winsorization and scaling.
    
    Args:
        df: DataFrame with numeric fields
        scaler: Pre-fitted scaler (if None, will create new one)
        fit_scaler: Whether to fit the scaler (True for training, False for inference)
        scaler_path: Path to save/load scaler
        
    Returns:
        tuple: (df_with_scaled_features, scaler)
    """
    df = df.copy()
    numeric_fields = get_numeric_fields()
    
    # Ensure numeric types
    for f in numeric_fields:
        if f in df.columns:
            df[f] = pd.to_numeric(df[f], errors='coerce').fillna(0)
        else:
            df[f] = 0
    
    # 1. Clip extreme outliers (Winsorize at 1st and 99th percentiles)
    print("    Clipping outliers (Winsorization at 1st and 99th percentiles)...")
    for f in numeric_fields:
        if df[f].notna().sum() > 0:
            lower = df[f].quantile(0.01)
            upper = df[f].quantile(0.99)
            clipped_count = ((df[f] < lower) | (df[f] > upper)).sum()
            if clipped_count > 0 and fit_scaler:
                print(f"      {f}: clipped {clipped_count} outliers (bounds: [{lower:.2f}, {upper:.2f}])")
            df[f] = df[f].clip(lower=lower, upper=upper)
    
    # 2. Robust scaling (median=0, IQR=1)
    if scaler is None:
        scaler = RobustScaler()
    
    if fit_scaler:
        print("    Fitting RobustScaler (median=0, IQR=1)...")
        scaled = scaler.fit_transform(df[numeric_fields])
        if scaler_path:
            save_scaler(scaler, scaler_path)
    else:
        print("    Applying RobustScaler transform (using saved scaler)...")
        scaled = scaler.transform(df[numeric_fields])
    
    for i, f in enumerate(numeric_fields):
        df[f + "_scaled"] = scaled[:, i]
    
    print(f"    Standardized {len(numeric_fields)} numeric features")
    
    return df, scaler


def _safe_str(value):
    """Return a non-empty string from a row value, or empty string."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    if isinstance(value, (list, np.ndarray)):
        if len(value) == 0:
            return ""
        return " ".join(str(x).strip() for x in value if x is not None and str(x).strip())
    s = str(value).strip()
    if not s or s.lower() in ("nan", "none", ""):
        return ""
    return s


def _get_founder_title(row):
    """Derive founder title from first experience's position in all_experiences, or row['title'] if present."""
    if hasattr(row, "get") and row.get("title"):
        return _safe_str(row.get("title"))
    exp_list = _normalize_experiences_to_list(row.get("all_experiences") if hasattr(row, "get") else None)
    if not exp_list:
        return ""
    first = exp_list[0]
    if hasattr(first, "get") and callable(getattr(first, "get")):
        return _safe_str(_exp_get(first, "position", "title", "role"))
    return ""


def _format_education_from_school_fields(row):
    """
    Build Educational background from school_name_1/degree_1/details_1/school_dates_1
    and same for 2 and 3. No processing, just pull and format cleanly.
    """
    if not hasattr(row, "get"):
        return ""
    parts = []
    for i in range(1, 4):
        school = _safe_str(row.get(f"school_name_{i}", ""))
        if not school:
            continue
        degree = _safe_str(row.get(f"degree_{i}", ""))
        details = _safe_str(row.get(f"details_{i}", ""))
        dates = _safe_str(row.get(f"school_dates_{i}", ""))
        line_parts = [school]
        if degree:
            line_parts.append(f"Degree: {degree}")
        if dates:
            line_parts.append(f"Dates: {dates}")
        if details:
            line_parts.append(details)
        parts.append(" | ".join(line_parts))
    return " | ".join(parts) if parts else ""


def build_structured_profile_text(row):
    """
    Build a single structured text block for a founder/company profile (no priority label).
    Used as input for Tinker finetuning; the model predicts the priority label as completion.
    """
    lines = []
    name = _safe_str(row.get("name", "")) if hasattr(row, "get") else ""
    if name:
        lines.append(f"Founder name: {name}")
    title = _get_founder_title(row)
    if title:
        lines.append(f"Founder title: {title}")
    location = _safe_str(row.get("location", "")) if hasattr(row, "get") else ""
    if location:
        lines.append(f"Location: {location}")
    about = _safe_str(row.get("about", "")) if hasattr(row, "get") else ""
    if about:
        lines.append(f"Founder bio: {about}")
    company_name = _safe_str(row.get("company_name", "")) if hasattr(row, "get") else ""
    if company_name:
        lines.append(f"Company name: {company_name}")
    years = convert_building_since_to_years(
        row.get("building_since") if hasattr(row, "get") else None
    )
    if years is not None and years > 0:
        lines.append(f"Years in current company: {years:.1f}")
    product = _safe_str(row.get("product", "")) if hasattr(row, "get") else ""
    market = _safe_str(row.get("market", "")) if hasattr(row, "get") else ""
    if product:
        lines.append(f"Product: {product}")
    if market:
        lines.append(f"Market: {market}")
    background = format_experiences_json(row.get("all_experiences") if hasattr(row, "get") else None)
    if background:
        lines.append(f"Professional experience: {background}")
    education = _format_education_from_school_fields(row)
    if education:
        lines.append(f"Educational background: {education}")
    funding = _safe_str(row.get("funding", "")) if hasattr(row, "get") else ""
    if funding:
        lines.append(f"Company funding: {funding}")
    return "\n".join(lines) if lines else ""

