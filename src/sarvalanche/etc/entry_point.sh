#!/bin/bash --login
set -e
conda activate sarvalanche
exec python -um sarvalanche "$@"