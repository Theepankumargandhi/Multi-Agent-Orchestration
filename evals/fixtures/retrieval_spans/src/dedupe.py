def deduplicate_results(results):
    unique_results = dict.fromkeys(results)
    return list(unique_results)
