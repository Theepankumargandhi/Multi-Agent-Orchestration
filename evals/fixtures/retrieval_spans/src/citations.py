def citation_allowed(citation_id, approved_source_ids):
    if citation_id not in approved_source_ids:
        return False
    return True
