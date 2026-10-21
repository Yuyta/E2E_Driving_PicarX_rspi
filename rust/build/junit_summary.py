"""Write a compact GitHub Actions summary from a JUnit XML report."""

from pathlib import Path
import os
import sys
import xml.etree.ElementTree as ET


def markdown_escape(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ").strip()


def append_summary(lines: list[str]) -> None:
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    rendered = "\n".join(lines) + "\n"
    if summary_path:
        with Path(summary_path).open("a", encoding="utf-8") as summary:
            summary.write(rendered)
    else:
        sys.stdout.buffer.write(rendered.encode("utf-8"))


def main() -> int:
    if len(sys.argv) != 2:
        print(f"usage: {Path(sys.argv[0]).name} JUNIT_XML", file=sys.stderr)
        return 0

    report_path = Path(sys.argv[1])
    if not report_path.exists():
        append_summary(
            [
                "## Unit test report",
                "⚠️ JUnit XML was not generated. Check the test runner step output.",
            ]
        )
        return 0

    try:
        root = ET.parse(report_path).getroot()
    except ET.ParseError as error:
        append_summary(
            [
                "## Unit test report",
                f"⚠️ JUnit XML could not be parsed: `{markdown_escape(str(error))}`",
            ]
        )
        return 0

    testcases = root.findall(".//testcase")
    failed_cases = []
    failure_count = 0
    skipped = 0
    errors = 0
    duration = 0.0

    for testcase in testcases:
        duration += float(testcase.get("time", "0") or 0)
        status = testcase.get("status", "").lower()
        if testcase.find("skipped") is not None or status in {"skip", "skipped"}:
            skipped += 1
            continue

        failure = testcase.find("failure")
        error = testcase.find("error")
        is_error = error is not None or status in {"error", "errored"}
        is_failure = failure is not None or status in {"fail", "failed"}
        if is_error:
            errors += 1
        elif is_failure:
            failure_count += 1

        if is_error or is_failure:
            detail_node = failure if failure is not None else error
            detail = ""
            if detail_node is not None and detail_node.text:
                detail = detail_node.text.strip().splitlines()[0]
            name = testcase.get("name", "unknown test")
            classname = testcase.get("classname", "")
            display_name = f"{classname}::{name}" if classname else name
            failed_cases.append((display_name, detail))

    total = len(testcases)
    passed = total - failure_count - errors - skipped
    status = "✅ Passed" if not failure_count and not errors else "❌ Failed"

    lines = [
        "## Unit test report",
        "",
        "| Status | Total | Passed | Failed | Errors | Skipped | Duration |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        f"| {status} | {total} | {passed} | {failure_count} | {errors} | {skipped} | {duration:.2f}s |",
    ]

    if failed_cases:
        lines.extend(
            [
                "",
                "### Failed tests",
                "",
                "| Test | Details |",
                "| --- | --- |",
            ]
        )
        lines.extend(
            f"| `{markdown_escape(name)}` | {markdown_escape(detail) or '-'} |"
            for name, detail in failed_cases
        )

    append_summary(lines)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
