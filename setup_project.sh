#!/bin/bash

PROJECT_NAME="AI-Project-EA-MT"

mkdir -p $PROJECT_NAME
cd $PROJECT_NAME

# Root files
touch README.md
touch requirements.txt
touch .gitignore

# configs
mkdir -p configs/{data,model,inference,experiment}

# data
mkdir -p data/raw/{train,validation,test}
mkdir -p data/interim/{parsed,qid_lists,kb_cache,prompt_ready}
mkdir -p data/processed/{sft/ja,sft/multilingual,inference_inputs}
mkdir -p data/processed/cerc/{corruption_inputs,correction_targets}
mkdir -p data/external/wikidata

# notebooks
mkdir -p notebooks
touch notebooks/01_data_inspection.ipynb
touch notebooks/02_kb_cache_analysis.ipynb
touch notebooks/03_baseline_error_analysis.ipynb
touch notebooks/04_ablation_analysis.ipynb

# scripts
mkdir -p scripts/{setup,data,train,inference,eval,utils}

touch scripts/setup/install.sh
touch scripts/setup/set_env.sh

touch scripts/data/parse_eamt.py
touch scripts/data/split_by_language.py
touch scripts/data/extract_qids.py
touch scripts/data/build_kb_cache.py
touch scripts/data/build_sft_data.py
touch scripts/data/build_cerc_data.py

touch scripts/train/train_lora.py
touch scripts/train/train_cerc.py

touch scripts/inference/run_plain.py
touch scripts/inference/run_entity_plan.py
touch scripts/inference/run_bias.py
touch scripts/inference/run_cerc.py

touch scripts/eval/format_predictions.py
touch scripts/eval/score_local.py
touch scripts/eval/collect_results.py

touch scripts/utils/check_data.py
touch scripts/utils/merge_jsonl.py

# src
mkdir -p src/eamt/{data,kb,prompt,prompt/templates,model,inference,postprocess,evaluation,common}

touch src/eamt/__init__.py

touch src/eamt/data/dataset.py
touch src/eamt/data/loader.py
touch src/eamt/data/preprocessing.py
touch src/eamt/data/collator.py

touch src/eamt/kb/wikidata_client.py
touch src/eamt/kb/kb_cache.py
touch src/eamt/kb/entity_normalizer.py
touch src/eamt/kb/alias_selector.py

touch src/eamt/prompt/plain.py
touch src/eamt/prompt/entity_plan.py
touch src/eamt/prompt/cerc_prompt.py

touch src/eamt/model/lora.py
touch src/eamt/model/trainer.py
touch src/eamt/model/generation.py
touch src/eamt/model/decoding.py

touch src/eamt/inference/pipeline_plain.py
touch src/eamt/inference/pipeline_entity_plan.py
touch src/eamt/inference/pipeline_bias.py
touch src/eamt/inference/pipeline_cerc.py

touch src/eamt/postprocess/regex_cleaner.py
touch src/eamt/postprocess/output_parser.py
touch src/eamt/postprocess/entity_consistency.py

touch src/eamt/evaluation/metrics.py
touch src/eamt/evaluation/analyzer.py
touch src/eamt/evaluation/reporter.py

touch src/eamt/common/io.py
touch src/eamt/common/logging.py
touch src/eamt/common/seed.py
touch src/eamt/common/config.py

# outputs
mkdir -p outputs/{logs,checkpoints/lora,checkpoints/cerc,predictions/{plain,entity_plan,bias,cerc},evaluations,figures}

# reports
mkdir -p reports/{weekly,ablation,slides,final}

# tests
mkdir -p tests
touch tests/test_loader.py
touch tests/test_kb_cache.py
touch tests/test_prompt.py
touch tests/test_inference.py
touch tests/test_postprocess.py

echo "EA-MT project directory created successfully!"