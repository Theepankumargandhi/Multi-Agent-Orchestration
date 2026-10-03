def within_inference_budget(spent, next_call_cost, limit):
    projected_cost = spent + next_call_cost
    return projected_cost <= limit
