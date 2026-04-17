import ast
import operator
import re
from statistics import mean
from typing import Any, List, Optional


class CalculationAgent:
    """Agent for performing numeric calculations."""

    _binary_ops = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Mod: operator.mod,
        ast.Pow: operator.pow,
        ast.FloorDiv: operator.floordiv,
    }

    _unary_ops = {
        ast.UAdd: operator.pos,
        ast.USub: operator.neg,
    }

    def _eval_node(self, node: ast.AST):
        """Recursively evaluate a restricted AST node."""
        if isinstance(node, ast.BinOp):
            op = self._binary_ops.get(type(node.op))
            if op is None:
                raise ValueError("Unsupported operator")
            return op(self._eval_node(node.left), self._eval_node(node.right))
        if isinstance(node, ast.UnaryOp):
            op = self._unary_ops.get(type(node.op))
            if op is None:
                raise ValueError("Unsupported unary operator")
            return op(self._eval_node(node.operand))
        if isinstance(node, ast.Num):
            return node.n
        if isinstance(node, ast.Constant):  # for Python 3.8+
            if isinstance(node.value, (int, float)):
                return node.value
            raise ValueError("Unsupported constant")
        raise ValueError("Unsupported expression")

    def compute(
        self,
        expression: Optional[str] = None,
        *,
        question: Optional[str] = None,
        table: Optional[List[List[Any]]] = None,
        reasoning_steps: Optional[List[dict]] = None,
        title: Optional[str] = None,
        operator_chain: Optional[List[dict]] = None,
        base_value: Optional[float] = None,
    ):
        """Return the evaluated result of an expression or derive a value from question/table/operator chain."""

        expression = (expression or "").strip()
        if expression:
            try:
                tree = ast.parse(expression, mode="eval")
                return self._eval_node(tree.body)
            except Exception:
                pass

        chain_result = self._derive_from_operator_chain(base_value, operator_chain)
        if chain_result is not None:
            return chain_result

        crt_result = self._derive_from_crt_steps(question, table, reasoning_steps, title=title)
        if crt_result is not None:
            return crt_result

        derived = self._derive_from_table(question, table)
        return derived

    def _derive_from_crt_steps(
        self,
        question: Optional[str],
        table: Optional[List[List[Any]]],
        reasoning_steps: Optional[List[dict]],
        *,
        title: Optional[str] = None,
    ):
        if not question or not table or not isinstance(table, list) or len(table) < 2:
            return None
        headers = [str(h).strip().lower() for h in (table[0] or [])]
        rows = [r for r in table[1:] if isinstance(r, list) and r]
        if not headers or not rows:
            return None

        q = question.lower()
        step_details = []
        for step in reasoning_steps or []:
            if isinstance(step, dict):
                detail = str(step.get("detail", "")).strip().lower()
                if detail:
                    step_details.append(detail)

        numeric_cols = self._numeric_columns(rows, len(headers))
        if not numeric_cols:
            return None

        target_col = self._pick_target_column(headers, q, step_details, numeric_cols)
        if target_col is None:
            target_col = numeric_cols[0]

        values = []
        for row in rows:
            if len(row) <= target_col:
                continue
            num = self._extract_first_number(row[target_col])
            if num is not None:
                values.append(num)
        if not values:
            return None

        if self._expects_yes_no(q):
            if "outlier" in q:
                return "Yes" if self._has_outlier(values) else "No"
            if "higher" in q or "lower" in q or "similar" in q or "same" in q:
                return "Yes" if self._simple_yes_no_compare(q, headers, rows, target_col) else "No"

        if self._expects_more_less_equal(q):
            comp = self._more_less_equal_compare(q, headers, rows, target_col)
            if comp:
                return comp

        top_n = self._extract_top_n(q)
        nums = sorted(values, reverse=True)[:top_n] if top_n else values
        if "average" in q or "mean" in q:
            return round(mean(nums), 3)
        if "sum" in q or "total" in q:
            return round(sum(nums), 3)
        if "highest" in q or "largest" in q or "max" in q:
            return round(max(values), 3)
        if "lowest" in q or "smallest" in q or "min" in q:
            return round(min(values), 3)

        # Helpful hint for the LLM summarizer to follow operation schema.
        if step_details:
            return f"CRT operation hints: title={title or ''}; target_column={headers[target_col]}; steps={'; '.join(step_details)}"
        return None

    def _numeric_columns(self, rows: List[List[Any]], num_cols: int) -> List[int]:
        numeric_cols = []
        for c in range(num_cols):
            hits = 0
            total = 0
            for row in rows:
                if len(row) <= c:
                    continue
                total += 1
                if self._extract_first_number(row[c]) is not None:
                    hits += 1
            if total and hits / total >= 0.4:
                numeric_cols.append(c)
        return numeric_cols

    def _pick_target_column(
        self,
        headers: List[str],
        question: str,
        step_details: List[str],
        numeric_cols: List[int],
    ) -> Optional[int]:
        tokens = set(re.findall(r"[a-zA-Z]+", question))
        for detail in step_details:
            detail_tokens = set(re.findall(r"[a-zA-Z]+", detail))
            tokens |= detail_tokens
        best = None
        best_score = -1
        for idx in numeric_cols:
            if idx >= len(headers):
                continue
            ht = set(re.findall(r"[a-zA-Z]+", headers[idx]))
            score = len(tokens & ht)
            if score > best_score:
                best_score = score
                best = idx
        return best

    def _extract_first_number(self, value: Any) -> Optional[float]:
        text = str(value).replace(",", "").replace("%", " ").strip()
        m = re.search(r"[-+]?\d+(?:\.\d+)?", text)
        if not m:
            return None
        try:
            return float(m.group(0))
        except ValueError:
            return None

    def _expects_yes_no(self, question: str) -> bool:
        return "answer with only 'yes' or 'no'" in question or "answer with only \"yes\" or \"no\"" in question

    def _expects_more_less_equal(self, question: str) -> bool:
        return "answer with only 'more', 'less' or 'equal'" in question or "answer with only \"more\", \"less\" or \"equal\"" in question

    def _has_outlier(self, values: List[float]) -> bool:
        if len(values) < 4:
            return False
        vals = sorted(values)
        q1 = vals[len(vals) // 4]
        q3 = vals[(3 * len(vals)) // 4]
        iqr = q3 - q1
        if iqr <= 0:
            return False
        lo = q1 - 1.5 * iqr
        hi = q3 + 1.5 * iqr
        return any(v < lo or v > hi for v in vals)

    def _extract_top_n(self, question: str) -> Optional[int]:
        m = re.search(r"top\s+(\d+)", question)
        if m:
            try:
                return int(m.group(1))
            except ValueError:
                return None
        return None

    def _simple_yes_no_compare(self, question: str, headers: List[str], rows: List[List[Any]], target_col: int) -> bool:
        comp = self._more_less_equal_compare(question, headers, rows, target_col)
        if comp is None:
            return False
        if "higher" in question:
            return comp == "more"
        if "lower" in question:
            return comp == "less"
        if "similar" in question or "same" in question:
            return comp == "equal"
        return False

    def _more_less_equal_compare(self, question: str, headers: List[str], rows: List[List[Any]], target_col: int) -> Optional[str]:
        # Try splitting by common comparison markers.
        marker = None
        for m in (" versus ", " vs ", " between "):
            if m in question:
                marker = m
                break
        if marker is None:
            return None

        # Identify two phrase anchors.
        if marker.strip() == "between":
            parts = question.split("between", 1)[-1].split(" and ", 1)
        else:
            parts = question.split(marker, 1)
        if len(parts) < 2:
            return None
        left = re.findall(r"[a-zA-Z]+", parts[0].lower())
        right = re.findall(r"[a-zA-Z]+", parts[1].lower())
        left_tokens = {t for t in left if len(t) > 2}
        right_tokens = {t for t in right if len(t) > 2}
        if not left_tokens or not right_tokens:
            return None

        left_vals = []
        right_vals = []
        for row in rows:
            row_text = " ".join(str(c).lower() for c in row)
            if len(row) <= target_col:
                continue
            val = self._extract_first_number(row[target_col])
            if val is None:
                continue
            if any(tok in row_text for tok in left_tokens):
                left_vals.append(val)
            if any(tok in row_text for tok in right_tokens):
                right_vals.append(val)
        if not left_vals or not right_vals:
            return None
        l = mean(left_vals)
        r = mean(right_vals)
        if abs(l - r) <= 1e-9:
            return "equal"
        return "more" if l > r else "less"

    def _derive_from_operator_chain(
        self,
        base_value: Optional[float],
        operator_chain: Optional[List[dict]],
    ) -> Optional[float]:
        """Execute deterministic synthetic operator chains (5–8 steps)."""

        if base_value is None or not operator_chain:
            return None

        result = float(base_value)
        for step in operator_chain:
            if not isinstance(step, dict):
                continue
            op = step.get("op")
            try:
                val = float(step.get("value"))
            except (TypeError, ValueError):
                continue

            if op == "add":
                result += val
            elif op == "subtract":
                result -= val
            elif op == "multiply":
                result *= val
            elif op == "divide":
                if val != 0:
                    result /= val
            # ignore unknown ops silently

        return result

    def _derive_from_table(self, question: Optional[str], table: Optional[List[List[Any]]]):
        """Heuristically answer lookup-style questions from a simple table."""

        if not question or not table or not isinstance(table, list):
            return None

        if not table:
            return None

        headers = None
        rows = table
        first_row = table[0]
        if isinstance(first_row, list) and all(isinstance(item, (str, int, float)) for item in first_row):
            headers = [str(x) for x in first_row]
            rows = table[1:]

        if not rows:
            return None

        q_lower = question.lower()
        numbers = re.findall(r"\d{3,4}", q_lower)
        tokens = {tok for tok in re.findall(r"[a-zA-Z]+", q_lower) if len(tok) > 2}

        question_lower = question.lower()
        derived = self._aggregate_table(question_lower, headers, rows)
        if derived:
            return derived

        target_row = None
        for row in rows:
            if not row:
                continue
            row_label = str(row[0]).lower()
            if numbers and any(num in row_label for num in numbers):
                target_row = row
                break
            if tokens and any(tok in row_label for tok in tokens):
                target_row = row
                break

        if target_row is None:
            target_row = rows[0]

        col_index = self._determine_column(headers, target_row, tokens)
        if col_index is None or col_index >= len(target_row):
            return None

        value = target_row[col_index]
        header = headers[col_index] if headers and col_index < len(headers) else f"column {col_index}"
        row_label = target_row[0]
        return f"{header} for {row_label} = {value}"

    def _determine_column(self, headers: Optional[List[str]], row: List[Any], tokens: set) -> Optional[int]:
        """Pick the column that overlaps with question tokens or contains numeric data."""

        if headers:
            for idx, header in enumerate(headers[1:], start=1):
                header_tokens = set(re.findall(r"[a-zA-Z]+", header.lower()))
                if tokens and header_tokens & tokens:
                    return idx

        for idx, cell in enumerate(row[1:], start=1):
            if isinstance(cell, (int, float)):
                return idx
            try:
                float(str(cell).replace(",", ""))
                return idx
            except ValueError:
                continue
        return None

    def _aggregate_table(self, question: str, headers: Optional[List[str]], rows: List[List[Any]]):
        """Handle sum/difference/average questions over a column."""

        if not headers or len(headers) < 2:
            return None

        target_col = None
        for idx, header in enumerate(headers[1:], start=1):
            if header and header.strip():
                if header.lower() in question or any(tok in header.lower() for tok in re.findall(r"[a-zA-Z]+", question)):
                    target_col = idx
                    break
        if target_col is None:
            target_col = 1

        numeric_values = []
        labels = []
        for row in rows:
            if len(row) <= target_col:
                continue
            try:
                numeric_values.append(float(str(row[target_col]).replace(",", "")))
                labels.append(str(row[0]))
            except ValueError:
                continue

        if not numeric_values:
            return None

        if "average" in question or "mean" in question:
            avg = sum(numeric_values) / len(numeric_values)
            return f"Average {headers[target_col]} = {avg:.2f}"
        if "sum" in question or "total" in question:
            total = sum(numeric_values)
            return f"Total {headers[target_col]} = {total}"
        if "difference" in question or "decrease" in question or "increase" in question:
            if len(numeric_values) >= 2:
                diff = numeric_values[-1] - numeric_values[0]
                return f"Difference in {headers[target_col]} from {labels[0]} to {labels[-1]} = {diff}"
        if "max" in question or "highest" in question:
            max_val = max(numeric_values)
            label = labels[numeric_values.index(max_val)]
            return f"Highest {headers[target_col]} is {max_val} for {label}"
        if "min" in question or "lowest" in question:
            min_val = min(numeric_values)
            label = labels[numeric_values.index(min_val)]
            return f"Lowest {headers[target_col]} is {min_val} for {label}"
        return None
