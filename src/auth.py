"""
Authentication Module for DemandWise
====================================
Local academic-prototype authentication system using PBKDF2-HMAC-SHA256
password hashing and SQLite-backed user credentials.
"""

import os
import hashlib
import secrets
from typing import Tuple, Optional, Dict, Any

from src.database import (
    create_user,
    get_user_by_username,
    get_user_by_id,
    upsert_business_profile,
    get_business_profile,
    seed_demo_stores_for_user
)


def hash_password(password: str, salt: Optional[str] = None) -> Tuple[str, str]:
    """
    Hashes a password with PBKDF2-HMAC-SHA256 (100,000 iterations).
    Returns (hex_hash, hex_salt).
    """
    if salt is None:
        salt_bytes = secrets.token_bytes(16)
        salt_hex = salt_bytes.hex()
    else:
        salt_hex = salt
        salt_bytes = bytes.fromhex(salt)

    pwd_bytes = password.encode('utf-8')
    key = hashlib.pbkdf2_hmac('sha256', pwd_bytes, salt_bytes, 100_000)
    return key.hex(), salt_hex


def verify_password(password: str, stored_hash: str, salt: str) -> bool:
    """Verifies candidate password against stored PBKDF2 hash."""
    candidate_hash, _ = hash_password(password, salt)
    return secrets.compare_digest(candidate_hash, stored_hash)


def validate_registration_inputs(username: str, email: str, password: str, confirm_password: str) -> Tuple[bool, str]:
    """Validates registration input fields."""
    u = username.strip()
    if not u or len(u) < 3:
        return False, "Username must be at least 3 characters long."
    if not u.isalnum() and "_" not in u:
        return False, "Username can only contain alphanumeric characters and underscores."
    
    e = email.strip()
    if not e or "@" not in e or "." not in e:
        return False, "Please enter a valid email address."
        
    if len(password) < 6:
        return False, "Password must be at least 6 characters long."
    if password != confirm_password:
        return False, "Passwords do not match."
        
    return True, ""


def register_user(
    username: str,
    email: str,
    password: str,
    confirm_password: str,
    business_name: str,
    owner_name: str,
    currency: str = "USD",
    seed_demo: bool = True,
    db_path: Optional[str] = None
) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
    """Registers a new user, creates their business profile, and seeds demo stores."""
    valid, err = validate_registration_inputs(username, email, password, confirm_password)
    if not valid:
        return False, err, None
        
    b_name = business_name.strip() or f"{username.strip()}'s Retail"
    o_name = owner_name.strip() or username.strip()
    
    # Check if username already exists
    existing = get_user_by_username(username, db_path=db_path)
    if existing:
        return False, "Username is already taken. Please choose another.", None
        
    pwd_hash, salt = hash_password(password)
    
    try:
        user_id = create_user(username, email, pwd_hash, salt, db_path=db_path)
        upsert_business_profile(user_id, b_name, o_name, currency, db_path=db_path)
        
        if seed_demo:
            seed_demo_stores_for_user(user_id, db_path=db_path)
            
        user_data = get_user_by_id(user_id, db_path=db_path)
        return True, "Account created successfully!", user_data
    except Exception as e:
        return False, f"Registration failed: {str(e)}", None


def authenticate_user(
    username: str,
    password: str,
    db_path: Optional[str] = None
) -> Tuple[bool, str, Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """
    Authenticates a user by username and password.
    Returns (success, message, user_dict, business_profile_dict).
    """
    u = username.strip()
    if not u or not password:
        return False, "Username and password are required.", None, None
        
    user = get_user_by_username(u, db_path=db_path)
    if not user:
        return False, "Invalid username or password.", None, None
        
    if not verify_password(password, user['password_hash'], user['salt']):
        return False, "Invalid username or password.", None, None
        
    business = get_business_profile(user['id'], db_path=db_path)
    return True, "Authentication successful.", user, business

