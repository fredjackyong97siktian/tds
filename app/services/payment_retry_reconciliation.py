"""Separate unchanged pending checkout retries from unresolved transaction issues."""

from collections import defaultdict
from decimal import Decimal, InvalidOperation


def _number(value):
    if value is None or isinstance(value, bool):
        return None
    try:
        number = Decimal(str(value))
        return number if number.is_finite() else None
    except (InvalidOperation, ValueError):
        return None


def _cart(row):
    amount, count = _number(row.get("total_amount")), _number(row.get("total_items"))
    if amount is None or amount < 0 or count is None or count <= 0:
        return None
    details = row.get("details")
    if not isinstance(details, list) or not details:
        return None
    items = defaultdict(Decimal)
    for detail in details:
        if not isinstance(detail, dict):
            return None
        barcode = str(detail.get("barcode") or "").strip()
        name = " ".join(str(detail.get("item_name") or "").casefold().split())
        identity = ("barcode", barcode) if barcode else ("name", name)
        quantity = _number(detail.get("quantity"))
        if not identity[1] or quantity is None or quantity <= 0:
            return None
        items[identity] += quantity
    return amount, count, tuple(sorted(items.items()))


def classify(issues, assigned_paid, *, event_time, machine_name):
    paid_carts = []
    for paid in assigned_paid:
        cart, stamp = _cart(paid), event_time(paid)
        if cart is not None and stamp is not None:
            paid_carts.append((paid, cart, stamp))
    unresolved, normal = [], []
    for issue in issues:
        cart, stamp = _cart(issue), event_time(issue)
        resolved = None
        if str(issue.get("status") or "").strip().lower() == "pending" and cart is not None and stamp is not None:
            machine = str(machine_name(issue) or "").strip().casefold()
            for paid, paid_cart, paid_at in paid_carts:
                paid_machine = str(machine_name(paid) or "").strip().casefold()
                if machine and paid_machine and machine != paid_machine:
                    continue
                if cart == paid_cart and stamp <= paid_at:
                    resolved = paid
                    break
        if resolved is None:
            unresolved.append(issue)
        else:
            normal.append({**issue, "resolved_by_receipt": resolved.get("receipt_number") or resolved.get("transaction_id")})
    return unresolved, normal
