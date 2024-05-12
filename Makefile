# Define the Python interpreter to use
PYTHON = python3

# Create a virtual environment if it doesn't exist and install dependencies
setup:
	@test -d venv || virtualenv -p python3 venv
	. venv/bin/activate; \
	$(PYTHON) -m pip install -r requirements.txt

# Run Scenario1.py with optional flag
run1:
	. venv/bin/activate; \
	$(PYTHON) Scenario1.py $(FLAG)

# Run Scenario2.py with optional flag
run2:
	. venv/bin/activate; \
	$(PYTHON) Scenario2.py $(FLAG)

# Run Scenario3.py with optional flag
run3:
	. venv/bin/activate; \
	$(PYTHON) Scenario3.py $(FLAG)

# Clean up virtual environment and .pyc files
clean:
	rm -rf venv
	find -iname "*.pyc" -delete