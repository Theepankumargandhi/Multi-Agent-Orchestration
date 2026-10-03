def cache_entry_valid(now, expires_at):
    if now >= expires_at:
        return False
    return True
