def total(cart):
    """Integer units; zero-quantity items contribute nothing."""
    value = sum(item["quantity"] * item["price"] for item in cart["items"])
    return value - (1 if cart.get("coupon") else 0)


def customer_label(name):
    return name.lower()
