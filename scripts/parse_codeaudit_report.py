#!/usr/bin/env python3
"""Parse codeaudit HTML report and output a text summary for CI."""

import re
import sys
from pathlib import Path
from typing import Any


def parse_codeaudit_html(html_path: str) -> dict[str, Any]:
    with open(html_path) as f:
        html = f.read()

    sections = re.split(r"<h3[^>]*>\s*Security scan:\s*([^<]+)</h3>", html)

    issues = []
    files_scanned = 0
    files_without_issues = 0

    for i in range(1, len(sections), 2):
        filename = sections[i].strip()
        body = sections[i + 1] if i + 1 < len(sections) else ""
        files_scanned += 1

        file_issues = []
        tables = re.findall(r"<table[^>]*>.*?</table>", body, re.DOTALL)
        for table in tables:
            if "<th>line</th>" not in table or "<th>found</th>" not in table:
                continue
            rows = re.findall(r"<tr[^>]*>(.*?)</tr>", table, re.DOTALL)
            for row in rows:
                if "<th>" in row:
                    continue
                cells = re.findall(r"<td>(.*?)</td>", row, re.DOTALL)
                if len(cells) >= 3:
                    line = re.sub(r"<[^>]+>", "", cells[0]).strip()
                    found = re.sub(r"<[^>]+>", "", cells[1]).strip()
                    code = re.sub(r"<[^>]+>", "", cells[2]).strip()
                    file_issues.append(
                        {
                            "line": line,
                            "finding": found,
                            "code": code[:100],
                        }
                    )

        if file_issues:
            issues.append(
                {
                    "file": filename,
                    "count": len(file_issues),
                    "details": file_issues,
                }
            )

    no_issues_match = re.search(
        r"Total Python files\s+without\s+detected security issues:\s*(\d+)",
        html,
        re.I,
    )
    if no_issues_match:
        files_without_issues = int(no_issues_match.group(1))

    return {
        "files_scanned": files_scanned,
        "files_with_issues": len(issues),
        "files_without_issues": files_without_issues,
        "issues": issues,
        "total_issues": sum(i["count"] for i in issues),
    }


def main() -> int:
    html_path = sys.argv[1] if len(sys.argv) > 1 else "codeaudit-report.html"
    if not Path(html_path).exists():
        print(f"ERROR: {html_path} not found")
        return 1

    result = parse_codeaudit_html(html_path)

    print("=" * 70)
    print("CODEAUDIT SAST SUMMARY")
    print("=" * 70)
    print(f"Files scanned: {result['files_scanned']}")
    print(f"Files with issues: {result['files_with_issues']}")
    print(f"Files without issues: {result['files_without_issues']}")
    print(f"Total findings: {result['total_issues']}")
    print()

    for file_issues in result["issues"]:
        print(f"--- {file_issues['file']} ({file_issues['count']} finding(s)) ---")
        for detail in file_issues["details"]:
            print(f"  line {detail['line']}: [{detail['finding']}] {detail['code']}")
        print()

    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
