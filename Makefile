.PHONY: all smoke setup clean-smoke
all:        ; ./run_all.sh
smoke:      ; ./run_all.sh --smoke
setup:      ; python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
clean-smoke:; rm -rf outputs/smoke figures/smoke
