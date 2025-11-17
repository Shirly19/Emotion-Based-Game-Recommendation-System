# TODO: Separate Unit and Integration Tests

## Steps to Complete
- [x] Create subdirectories `tests/unit/` and `tests/integration/`
- [x] Move `test_db.py` and `test_model.py` to `tests/unit/`
- [x] Move `test_routes.py` to `tests/integration/`
- [x] Split `test_unit.py` into unit and integration parts
- [x] Move unit part of `test_unit.py` to `tests/unit/test_unit.py`
- [x] Move integration part of `test_unit.py` to `tests/integration/test_unit.py`
- [x] Remove original `test_unit.py` from `tests/`
- [x] Update `pytest.ini` if needed (currently testpaths = tests, should work with subdirs)
- [x] Test running unit tests with `pytest tests/unit/`
- [x] Test running integration tests with `pytest tests/integration/`
- [x] Fix import issues in unit tests (mock app imports or dependencies)
- [x] Run unit tests successfully
- [x] Run integration tests successfully
- [x] Clean up original test files from tests/ directory
- [x] Remove __pycache__ directory
