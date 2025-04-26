# conftest.py
import pytest
import asyncio
from database import init_db

@pytest.fixture(scope="session", autouse=True)
def initialize_database():
    """Create tables before all tests"""
    asyncio.run(init_db())