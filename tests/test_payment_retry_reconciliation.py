from copy import deepcopy
from datetime import datetime
import unittest

from app.services import payment_retry_reconciliation as retries
from app.services import workflow_service as workflow


def transaction(receipt, minute, status="pending", **values):
    return {"receipt_number": receipt, "status": status, "total_amount": 207, "total_items": 1,
            "transaction_time": datetime(2026, 10, 23, 11, minute), "machine_name": "Edumetro-right-1",
            "details": [{"barcode": "123", "item_name": "Bottle", "quantity": 1}], **values}


class PaymentRetryReconciliationTests(unittest.TestCase):
    def classify(self, issues, paid):
        return retries.classify(issues, paid, event_time=lambda row: row.get("transaction_time"),
                                machine_name=lambda row: row.get("machine_name"))

    def test_two_pending_attempts_then_unchanged_paid_are_normal(self):
        issues = [transaction("pending-1", 7), transaction("pending-2", 8)]
        original = deepcopy(issues)
        unresolved, normal = self.classify(issues, [transaction("paid", 8, "paid")])
        self.assertEqual(unresolved, [])
        self.assertEqual([row["resolved_by_receipt"] for row in normal], ["paid", "paid"])
        self.assertEqual(issues, original)
        self.assertFalse(workflow._has_short_period_transaction_issues(unresolved, 300)[0])
        self.assertFalse(any(workflow._transaction_event_time(row) for row in unresolved))

    def test_different_amount_or_product_or_quantity_stays_an_issue(self):
        paid = transaction("paid", 8, "paid")
        for changes in ({"total_amount": 208}, {"total_items": 2},
                        {"details": [{"barcode": "456", "item_name": "Other", "quantity": 1}]},
                        {"details": [{"barcode": "123", "quantity": 2}]}):
            with self.subTest(changes=changes):
                issue = transaction("pending", 7, **changes)
                unresolved, normal = self.classify([issue], [paid])
                self.assertEqual(unresolved, [issue])
                self.assertEqual(normal, [])

    def test_missing_or_invalid_item_evidence_is_not_cleared(self):
        for changes in ({"details": []}, {"details": None}, {"details": [{}]},
                        {"total_amount": None}, {"total_amount": "NaN"}, {"total_items": None},
                        {"details": [{"barcode": "123", "quantity": None}]}):
            with self.subTest(changes=changes):
                issue = transaction("pending", 7, **changes)
                self.assertEqual(self.classify([issue], [transaction("paid", 8, "paid")]), ([issue], []))

    def test_failed_attempts_and_pending_after_payment_stay_issues(self):
        for issue in [transaction("failed", 7, "failed"), transaction("later", 9)]:
            with self.subTest(receipt=issue["receipt_number"]):
                self.assertEqual(self.classify([issue], [transaction("paid", 8, "paid")]), ([issue], []))

    def test_no_assigned_payment_cannot_resolve_pending(self):
        issue = transaction("pending", 7)
        self.assertEqual(self.classify([issue], []), ([issue], []))

    def test_different_kiosk_cannot_resolve_pending(self):
        issue = transaction("pending", 7, machine_name="Edumetro-left-1")
        self.assertEqual(self.classify([issue], [transaction("paid", 8, "paid")]), ([issue], []))

    def test_same_count_and_amount_but_different_cart_is_not_normal(self):
        issue = transaction("pending", 7)
        paid = transaction("paid", 8, "paid", details=[{"barcode": "999", "quantity": 1}])
        self.assertEqual(self.classify([issue], [paid]), ([issue], []))

    def test_detail_order_and_numeric_format_do_not_matter(self):
        issue = transaction("pending", 7, total_items=2,
            details=[{"barcode": "123", "quantity": "1.0"}, {"barcode": "456", "quantity": 1}])
        paid = transaction("paid", 8, "paid", total_items="2.00", total_amount="207.0",
            details=[{"barcode": "456", "quantity": 1}, {"barcode": "123", "quantity": 1}])
        self.assertEqual(self.classify([issue], [paid])[0], [])

    def test_paid_receipt_need_not_be_the_last_receipt(self):
        issue = transaction("pending", 7)
        paid = [transaction("matching-paid", 8, "paid"), transaction("other-paid", 9, "paid", total_amount=900)]
        unresolved, normal = self.classify([issue], paid)
        self.assertEqual(unresolved, [])
        self.assertEqual(normal[0]["resolved_by_receipt"], "matching-paid")

    def test_workflow_normalizes_audit_utc_when_pos_time_is_missing(self):
        issue = transaction("pending", 7, transaction_time=None, created_at=datetime(2026, 10, 23, 3, 7))
        unresolved, normal = workflow._reconcile_pending_payment_retries([issue], [transaction("paid", 8, "paid")])
        self.assertEqual(unresolved, [])
        self.assertEqual(workflow._transaction_summary(normal)[0]["resolved_by_receipt"], "paid")

    def test_raw_payload_machine_names_are_checked(self):
        issue = transaction("pending", 7, machine_name=None, raw_payload={"machineName": "left"})
        paid = transaction("paid", 8, "paid", machine_name=None, raw_payload={"machineName": "right"})
        self.assertEqual(workflow._reconcile_pending_payment_retries([issue], [paid]), ([issue], []))
