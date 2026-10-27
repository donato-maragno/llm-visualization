"""Optional simplified Learn smoke check. Start run_web.py first; requires Edge/Playwright."""

from pathlib import Path


def main():
    from playwright.sync_api import sync_playwright, expect

    output = Path(__file__).resolve().parents[1] / "results"
    errors = []
    with sync_playwright() as p:
        browser = p.chromium.launch(channel="msedge", headless=True)
        page = browser.new_page(viewport={"width":1440,"height":1100})
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.on("console", lambda message: errors.append(message.text) if "Content Security Policy" in message.text else None)
        page.goto("http://127.0.0.1:8000/")
        sentence = page.request.get("http://127.0.0.1:8000/api/walkthrough/config").json()["sentence"]
        expect(page.locator("#fixed-sentence")).to_have_text(sentence)
        expect(page.locator(".preset-card")).to_have_count(4)
        expect(page.locator(".learn-network")).to_have_count(1, timeout=240000)
        assert page.locator("textarea,input,[contenteditable=true],#generate-walkthrough").count() == 0
        assert page.locator(".learn-inspector").count() == 0
        page.locator('[data-model-state="pruned"]').click()
        expect(page.locator(".bypass-route")).to_have_count(1)
        page.locator('[data-preset="1"]').click()
        expect(page.locator("#learn-layer")).to_have_value("0")
        expect(page.locator(".pruned-route")).not_to_have_count(0)
        page.locator('[data-learn-open="attention"]').first.click()
        expect(page.locator(".learn-inspector")).to_have_count(1)
        expect(page.locator(".learn-attention-matrix")).to_have_count(1)
        page.locator('[data-learn-matrix="masked"]').click()
        page.locator('[data-learn-close]').click()
        expect(page.locator(".learn-inspector")).to_have_count(0)
        page.locator('[data-preset="2"]').click()
        expect(page.locator(".learn-network")).to_have_count(1)
        page.locator('[data-learn-open="mlp"]').first.click()
        expect(page.locator(".learn-inspector")).to_contain_text("MLP parameters")
        page.locator('[data-learn-close]').click()
        assert page.locator(".prediction-bar").count() == 8
        widths = page.locator(".prediction-bar").evaluate_all("nodes => nodes.map(n => +n.getAttribute('width'))")
        assert len(set(widths)) > 4 and all(0 <= value <= 102 for value in widths)
        assert page.locator("#gpt2-view [style]").count() == 0
        page.screenshot(path=str(output / "simplified-transformer-explainer.png"), full_page=True)
        page.set_viewport_size({"width":390,"height":844})
        assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
        page.locator('.page-menu a[href="/experiments"]').click()
        expect(page.locator("#model option")).to_have_count(1)
        expect(page.locator("#coding-problems")).to_be_visible()
        assert not errors, errors
        print("Learn smoke passed: continuous Transformer Explainer layout, model toggle, preset masks, click-to-expand measurements, cache loading, Qwen isolation, mobile pan.")
        browser.close()


if __name__ == "__main__":
    main()
