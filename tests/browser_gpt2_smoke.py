"""Optional upstream Learn smoke check. Start run_web.py first; requires Edge/Playwright."""

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
        expect(page.locator("#learn-network-frame")).to_have_count(1, timeout=240000)
        diagram = page.frame_locator('#learn-network-frame')
        expect(diagram.locator('.head-card')).to_have_count(12)
        assert page.locator("textarea,input,[contenteditable=true],#generate-walkthrough").count() == 0
        assert page.locator(".learn-inspector").count() == 0
        page.locator('[data-model-state="pruned"]').click()
        expect(diagram.locator(".bypass-route").first).to_be_attached()
        page.locator('[data-preset="1"]').click()
        expect(page.locator("#learn-layer")).to_have_value("0")
        expect(diagram.locator(".head-mask-label")).to_contain_text('output zeroed')
        diagram.locator('[id="self-attention"]').click()
        expect(page.locator(".learn-inspector")).to_have_count(1)
        expect(page.locator(".learn-attention-matrix")).to_have_count(1)
        page.locator('[data-learn-matrix="masked"]').click()
        page.locator('[data-learn-close]').click()
        expect(page.locator(".learn-inspector")).to_have_count(0)
        page.locator('[data-preset="2"]').click()
        expect(page.locator("#learn-network-frame")).to_have_count(1)
        diagram.locator('[id="mlp"]').first.click()
        expect(page.locator(".learn-inspector")).to_contain_text("MLP parameters")
        page.locator('[data-learn-close]').click()
        assert diagram.locator('g.probability rect.bar').count() == 8
        widths = diagram.locator('g.probability rect.bar').evaluate_all("nodes => nodes.map(n => +n.getAttribute('width'))")
        assert len(set(widths)) > 4 and all(value >= 0 for value in widths)
        assert page.locator("#gpt2-view [style]").count() == 0
        page.screenshot(path=str(output / "upstream-transformer-explainer.png"), full_page=True)
        page.set_viewport_size({"width":390,"height":844})
        assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
        page.locator('.page-menu a[href="/experiments"]').click()
        expect(page.locator("#model option")).to_have_count(1)
        expect(page.locator("#coding-problems")).to_be_visible()
        assert not errors, errors
        print("Learn smoke passed: actual upstream Transformer Explainer components, model toggle, pruning overlays, measured inspector, cache loading, Qwen isolation, mobile pan.")
        browser.close()


if __name__ == "__main__":
    main()
