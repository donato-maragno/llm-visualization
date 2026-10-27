"""Optional real-browser smoke test. Start run_web.py first; requires Playwright/Edge.

Runs ten Python problems and a magnitude-pruned Coder variant. No judge/API calls.
"""

from pathlib import Path


def main():
    from playwright.sync_api import sync_playwright, expect

    output = Path(__file__).resolve().parents[1] / "results"
    errors = []
    with sync_playwright() as p:
        browser = p.chromium.launch(channel="msedge", headless=True)
        page = browser.new_page(viewport={"width": 1440, "height": 1100})
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.goto("http://127.0.0.1:8000/experiments")
        page.wait_for_selector(".variant-row")
        assert page.locator(".variant-row").count() == 3
        page.screenshot(path=str(output / "web-desktop.png"), full_page=True)
        assert page.locator("#judge").count() == page.locator("#tokens").count() == 0
        assert "Coder" in page.locator("#model").input_value()
        page.locator(".variant-row").first.locator(".remove").click()
        page.locator(".variant-row").first.locator(".remove").click()
        page.locator(".variant-row input").fill("5")
        page.locator("#execution-consent").check()
        page.locator("#run").click()
        expect(page.locator("#run-status")).to_have_text("COMPLETED", timeout=600000)
        assert page.locator("#matrix tbody tr").count() == 10
        assert page.locator("#matrix .score-button").count() == 10
        assert all(text in ("0", "1") for text in page.locator("#matrix .score-button").all_text_contents())
        page.locator("#matrix .score-button").first.click()
        assert page.locator("#baseline-answer").inner_text()
        assert page.locator("#pruned-answer").inner_text()
        assert page.locator("#outcome-reason").inner_text()
        assert page.locator("#case-results tbody tr").count() == 4
        assert "Expected-test correctness" in page.locator("#correctness").inner_text()
        page.screenshot(path=str(output / "web-answers.png"), full_page=True)
        # Add another vetted coding problem without any model call.
        page.locator("#generate-question").click()
        expect(page.locator(".question-row")).to_have_count(11, timeout=120000)
        page.get_by_text("Add a custom Python problem with tests", exact=True).click()
        page.locator("#custom-question").fill('{"id":"double","title":"Double a number","category":"arrays","description":"Implement solve(n). Return twice n.","tests":[{"args":[3],"expected":6}],"normalize":"exact"}')
        page.locator("#add-question").click()
        assert page.locator(".question-row").count() == 12
        page.set_viewport_size({"width": 390, "height": 844})
        assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
        page.screenshot(path=str(output / "web-mobile.png"), full_page=True)
        assert not errors, errors
        print("Coding browser smoke passed: 10 Python problems, Coder generation without user token caps, actual 0/1 execution, per-case outputs, no judge, new/custom problems, mobile layout.")
        browser.close()


if __name__ == "__main__":
    main()
