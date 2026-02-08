"""
Code Validator
==============
Validates generated C++ code for syntax and API usage.
"""

import re
from typing import Dict, List


class CodeValidator:
    """Validates C++ code for BakkesMod plugins."""

    def __init__(self):
        """Initialize validator."""
        # Common syntax patterns
        self.bracket_pairs = {'{': '}', '(': ')', '[': ']'}

        # BakkesMod API patterns
        self.api_patterns = {
            "gamewrapper": r"gameWrapper->",
            "hook_event": r"HookEvent\(",
            "server_wrapper": r"ServerWrapper",
            "car_wrapper": r"CarWrapper",
        }

    def validate_syntax(self, code: str) -> Dict:
        """
        Validate C++ syntax.

        Args:
            code: C++ code to validate

        Returns:
            Dict with 'valid' bool and 'errors' list
        """
        errors = []

        # Check bracket matching
        bracket_errors = self._check_brackets(code)
        errors.extend(bracket_errors)

        # Check for unclosed strings
        string_errors = self._check_strings(code)
        errors.extend(string_errors)

        return {
            "valid": len(errors) == 0,
            "errors": errors
        }

    def validate_bakkesmod_api(self, code: str) -> Dict:
        """
        Validate BakkesMod API usage.

        Args:
            code: C++ code to check

        Returns:
            Dict with API usage flags
        """
        return {
            "uses_gamewrapper": bool(re.search(self.api_patterns["gamewrapper"], code)),
            "hooks_events": bool(re.search(self.api_patterns["hook_event"], code)),
            "uses_server": bool(re.search(self.api_patterns["server_wrapper"], code)),
            "uses_car": bool(re.search(self.api_patterns["car_wrapper"], code)),
        }

    def _check_brackets(self, code: str) -> List[str]:
        """Check for unmatched brackets."""
        errors = []
        stack = []

        for i, char in enumerate(code):
            if char in self.bracket_pairs.keys():
                stack.append((char, i))
            elif char in self.bracket_pairs.values():
                if not stack:
                    errors.append(f"Unmatched closing bracket '{char}' at position {i}")
                else:
                    open_char, _ = stack.pop()
                    if self.bracket_pairs[open_char] != char:
                        errors.append(f"Mismatched bracket: expected '{self.bracket_pairs[open_char]}' but got '{char}'")

        # Check for unclosed brackets
        for open_char, pos in stack:
            errors.append(f"Unclosed bracket '{open_char}' at position {pos}")

        return errors

    def _check_strings(self, code: str) -> List[str]:
        """Check for unclosed strings."""
        errors = []
        in_string = False
        escape_next = False

        for i, char in enumerate(code):
            if escape_next:
                escape_next = False
                continue

            if char == '\\':
                escape_next = True
                continue

            if char == '"':
                in_string = not in_string

        if in_string:
            errors.append("Unclosed string literal")

        return errors
