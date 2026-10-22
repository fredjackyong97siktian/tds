import ast
from pathlib import Path
import unittest


class LongStayLowPurchaseTests(unittest.TestCase):
    def test_batch_and_single_group_require_positive_purchase_value(self):
        # Exercise the actual predicates in both scoring paths without DB/AI work.
        source = Path(__file__).resolve().parents[1] / "app/services/workflow_service.py"
        tree = ast.parse(source.read_text())
        predicates = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            keywords = {keyword.arg: keyword.value for keyword in node.keywords}
            code = keywords.get("factor_code")
            if isinstance(code, ast.Constant) and code.value == "long_stay_low_purchase":
                predicates.append(compile(ast.Expression(keywords["hit"]), str(source), "eval"))
        self.assertEqual(len(predicates), 2)
        for predicate in predicates:
            for long_stay, low_purchase, total_value, expected in (
                (True, True, 0, False),
                (True, True, -1, False),
                (True, True, 100, True),
                (False, True, 100, False),
                (True, False, 100, False),
            ):
                with self.subTest(long_stay=long_stay, low_purchase=low_purchase, total_value=total_value):
                    self.assertEqual(eval(predicate, {"__builtins__": {}}, {
                        "long_stay": long_stay, "low_purchase": low_purchase,
                        "total_value": total_value,
                    }), expected)
