.PHONY: setup generate validate test all clean help

PYTHON ?= python
VENV_DIR ?= venv
OUTPUT_DIR ?= data_set
SEED ?= 42

help: ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

setup: ## Create virtual environment and install dependencies
	$(PYTHON) -m venv $(VENV_DIR)
	. $(VENV_DIR)/bin/activate && pip install --upgrade pip && pip install -r requirements.txt
	@echo ""
	@echo "Setup complete. Activate with: source $(VENV_DIR)/bin/activate"

generate: ## Generate dataset (SEED=42 by default)
	$(PYTHON) tools/generate_dataset.py --output_dir $(OUTPUT_DIR) --seed $(SEED)

generate-small: ## Generate small dataset (2 scenes each) for testing
	$(PYTHON) tools/generate_dataset.py --output_dir $(OUTPUT_DIR) --r1_scenes 2 --r2_scenes 2 --conflict_scenes 2 --seed $(SEED)

validate: ## Validate dataset integrity and information equivalence
	$(PYTHON) tools/validate_dataset.py --base_dir $(OUTPUT_DIR) --validate --check_equivalence --verbose

stats: ## Show dataset statistics
	$(PYTHON) tools/validate_dataset.py --base_dir $(OUTPUT_DIR) --verbose

test: ## Run unit tests
	$(PYTHON) tests/test_scene_serializer.py
	$(PYTHON) tests/test_generate_dataset.py

test-verbose: ## Run unit tests with verbose output
	$(PYTHON) -m pytest tests/ -v 2>/dev/null || \
		($(PYTHON) tests/test_scene_serializer.py && $(PYTHON) tests/test_generate_dataset.py)

verify: ## Verify query ground truths with symbolic solvers
	$(PYTHON) -c "\
	import json, csv, glob; \
	from tools.verify_queries import verify_scene_queries; \
	total_correct = 0; total = 0; \
	for sf in sorted(glob.glob('$(OUTPUT_DIR)/*/scene/*.json')): \
	    qf = sf.replace('/scene/', '/queries/').replace('.json', '_questions.csv'); \
	    scene = json.load(open(sf, encoding='utf-8')); \
	    queries = list(csv.DictReader(open(qf, encoding='utf-8'))); \
	    results = verify_scene_queries(scene, queries); \
	    c = sum(1 for v in results.values() if v); \
	    total_correct += c; total += len(results); \
	print(f'Verified: {total_correct}/{total} ({100*total_correct/total:.1f}%)'); \
	"

all: setup generate validate test ## Run complete pipeline (setup + generate + validate + test)

clean: ## Remove generated files and virtual environment
	rm -rf $(OUTPUT_DIR)
	rm -rf $(VENV_DIR)
	rm -rf __pycache__ tools/__pycache__ tests/__pycache__
	rm -rf .pytest_cache htmlcov .coverage
	rm -rf final_stats
	@echo "Cleaned."

clean-data: ## Remove generated dataset only
	rm -rf $(OUTPUT_DIR)
	@echo "Dataset removed."
