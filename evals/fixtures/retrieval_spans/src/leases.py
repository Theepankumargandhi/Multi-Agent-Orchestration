def release_lease(owner, requesting_owner, expires_at, now):
    if owner != requesting_owner:
        return False
    if now >= expires_at:
        return False
    return True
