# Function to determine whether to include a rule based on config
def should_run_rule(rule_name):
    return config["run_rules"] == 'all' or rule_name in config["run_rules"]

# Top-level rule that chains everything if needed
rule all:
    input:
        # Only include these rules if 'run_rules' is 'all' or specifically requested
        expand("output/{runname}_umap.png", runname=config["runname"]) if should_run_rule("umap_plot") else [],
        expand("output/{runname}_embeddings.npz", runname=config["runname"]) if should_run_rule("umap_plot") else [],
        expand("output/{runname}_metrics.csv", runname=config["runname"]) if should_run_rule("scib_metrics") else [],
        expand("output/{runname}_crossgen_mouse_cis.png", runname=config["runname"]) if should_run_rule("crossgen_info") else [],
        expand("output/{runname}_crossgen_mouse_cross.png", runname=config["runname"]) if should_run_rule("crossgen_info") else [],
        expand("output/{runname}_mouse_cis_human_cross.npy", runname=config["runname"]) if should_run_rule("crossgen_info") else [],
        expand("output/{runname}_human_cis_mouse_cross.npy", runname=config["runname"]) if should_run_rule("crossgen_info") else [],
        expand("output/{runname}_crossgen_stats.txt", runname=config["runname"]) if should_run_rule("crossgen_info") else []

# UMAPs
rule umap_plot:
    input:
        config["model_input_npz"],
        config["model_input_pkl"]
    output:
        expand("output/{runname}_umap.png", runname=config["runname"]),
        expand("output/{runname}_embeddings.npz", runname=config["runname"])
    script:
        "scripts/umap_latent.py"

# SCIB metrics
rule scib_metrics:
    input: 
        config["model_input_npz"],
        config["model_input_pkl"],
        config["model_input_h5ad"]
    output:
        expand("output/{runname}_metrics.csv", runname=config["runname"])
    script:
        "scripts/scib_metrics.py"

# Crossgen UMAP & summary statistics
rule crossgen_info:
    input: 
        config["human_cis_npz"],
        config["human_cross_npz"],
        config["mouse_cis_npz"],
        config["mouse_cross_npz"]
    output:
        expand("output/{runname}_crossgen_mouse_cis.png", runname=config["runname"]), 
        expand("output/{runname}_crossgen_mouse_cross.png", runname=config["runname"]),
        expand("output/{runname}_mouse_cis_human_cross.npy", runname=config["runname"]), 
        expand("output/{runname}_human_cis_mouse_cross.npy", runname=config["runname"]),
        expand("output/{runname}_crossgen_stats.txt", runname=config["runname"])
    script:
        "scripts/cross_umap_stats.py"
