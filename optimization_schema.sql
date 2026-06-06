-- File: J:\Praca wyniki\results\optimization_schema.sql

CREATE TABLE IF NOT EXISTS optimization_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    
    -- IDs
    problem_idx INTEGER NOT NULL,
    task_name TEXT NOT NULL,
    model_id TEXT NOT NULL,
    strategy TEXT NOT NULL,                    -- "zero_shot" | "cot" | "self_refine"
    iteration INTEGER NOT NULL DEFAULT 0,      -- 0 = initial, 1 = first refine, 2 = second
    sample_idx INTEGER NOT NULL,               -- 0..n-1 (n=2)
    
    -- Stratyfikacja
    canonical_time_bucket TEXT,                -- "FAST" | "MEDIUM" | "SLOW"
    keyword_category TEXT,                     -- "array_sort" | "string" | etc.
    
    -- Prompt + Generacja
    prompt_template_version TEXT,
    raw_response TEXT,
    extracted_code TEXT,
    extraction_status TEXT,                    -- "success" | "no_code_block" | etc.
    tokens_input INTEGER,
    tokens_output INTEGER,
    api_cost_usd REAL,
    generation_duration_s REAL,
    finish_reason TEXT,                        -- "stop" | "MAX_TOKENS" | etc.
    api_error_type TEXT,                       -- "API_ERROR" | "REFUSAL" | etc.
    api_error_message TEXT,
    
    -- Walidacja semantyczna
    pytest_canonical_pass INTEGER,             -- 0/1
    pytest_small_pass INTEGER,                 -- 0/1
    functional_status TEXT,                    -- "SUCCESS" | "LOGICAL_REGRESSION" | etc.
    validation_error TEXT,
    
    -- Pomiary
    generated_time_median_ms REAL,
    generated_time_std_ms REAL,
    generated_time_min_ms REAL,
    canonical_time_median_ms REAL,
    eta_efficiency REAL,
    n_warmup INTEGER DEFAULT 5,
    n_measurements INTEGER DEFAULT 15,
    
    -- Metadane
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    hardware_id TEXT,
    python_version TEXT,
    
    -- Unikalność
    UNIQUE(problem_idx, model_id, strategy, iteration, sample_idx)
);

CREATE INDEX IF NOT EXISTS idx_opt_model ON optimization_results(model_id);
CREATE INDEX IF NOT EXISTS idx_opt_strategy ON optimization_results(strategy);
CREATE INDEX IF NOT EXISTS idx_opt_problem ON optimization_results(problem_idx);
CREATE INDEX IF NOT EXISTS idx_opt_status ON optimization_results(functional_status);