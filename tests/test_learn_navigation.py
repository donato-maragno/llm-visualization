"""Optional real-browser navigation checks; no model downloads or generation.

Install Playwright and Edge (or Playwright's Chromium) to run these tests.
The fixture data is synthetic and used only by these UI regression tests.
"""

from copy import deepcopy
from urllib.parse import urlsplit

import pytest
from fastapi.testclient import TestClient

from src.web_service import ExperimentService
from web_app import create_app


def navigation_run():
    vectors = [[0.1, -0.2], [0.3, -0.4]]
    matrix = [[1, 0], [0.4, 0.6]]
    layer = {
        "input": vectors, "output": vectors, "mlp_output": vectors,
        "q": [vectors] * 12, "k": [vectors] * 12, "v": [vectors] * 12,
        "head_output": [vectors] * 12,
        "scores": [matrix] * 12, "attention": [matrix] * 12,
        "bypassed": False, "masked_heads": [], "masked_weights": 0,
    }
    pipeline = {
        "prompt": "Test input", "tokens": [{"id": 1, "text": "Test"}, {"id": 2, "text": " input"}],
        "head_count": 12, "head_dim": 2, "hidden_size": 24,
        "embeddings": {field: vectors for field in ("token", "position", "combined")},
        "layers": [deepcopy(layer) for _ in range(12)],
        "predictions": [{"candidates": [{"id": 3, "text": " next", "probability": 0.4}],
                         "other_probability": 0.6}] * 2,
    }
    pruned = deepcopy(pipeline)
    pruned["layers"][-1]["bypassed"] = True
    return {
        "id": "navigation-test", "revision": 1, "status": "completed",
        "config": {"prompt": pipeline["prompt"]},
        "walkthrough_presets": [{"technique": "layers", "title": "Bypass last block",
                                 "inspection_layer": 11, "inspection_head": 0}],
        "baselines": {"0": {"pipeline": pipeline, "text": "Original test output", "tokens": 3}},
        "comparisons": {"0:0": {"answer": {"pipeline": pruned, "text": "Pruned test output", "tokens": 3}}},
    }


@pytest.fixture(scope="module")
def navigation_browser():
    playwright = pytest.importorskip("playwright.sync_api")
    with playwright.sync_playwright() as driver:
        try:
            browser = driver.chromium.launch(channel="msedge", headless=True)
        except playwright.Error:
            try:
                browser = driver.chromium.launch(headless=True)
            except playwright.Error:
                pytest.skip("Install Microsoft Edge or run: python -m playwright install chromium")
        yield browser
        browser.close()


@pytest.fixture
def learn_page(navigation_browser, tmp_path):
    client = TestClient(create_app(ExperimentService(tmp_path)))
    page = navigation_browser.new_page(viewport={"width": 1440, "height": 1000})
    errors = []
    page.on("pageerror", lambda error: errors.append(str(error)))

    def serve(route):
        path = urlsplit(route.request.url).path
        if path == "/static/walkthrough.js":
            # Keep the real HTML, renderer and CSP, without starting model preparation.
            route.fulfill(status=200, content_type="text/javascript", body="")
        else:
            response = client.get(path)
            route.fulfill(status=response.status_code, headers=dict(response.headers), body=response.content)

    page.route("http://llm.test/**", serve)
    page.goto("http://llm.test/")
    page.clock.install()
    page.evaluate("run => window.gpt2View.render(run, true)", navigation_run())
    yield page
    page.close()
    assert not errors, errors


def wheel(page, selector, delta, **options):
    page.clock.run_for(150)
    return page.locator(selector).evaluate(
        "(node, options) => {const event = new WheelEvent('wheel', {bubbles:true, cancelable:true, ...options});"
        "node.dispatchEvent(event); return event.defaultPrevented;}",
        {"deltaY": delta, **options},
    )


def test_scroll_layer_head_limits_and_trackpad(learn_page):
    page = learn_page
    layer, head = page.locator("#learn-layer"), page.locator("#learn-head")
    assert layer.input_value() == "11" and head.input_value() == "0"
    assert wheel(page, "#learn-layer", -100)
    assert layer.input_value() == "10" and head.input_value() == "0"
    assert wheel(page, "#learn-head", 100)
    assert head.input_value() == "1" and layer.input_value() == "10"
    assert wheel(page, "#learn-head", -100)
    assert wheel(page, "#learn-head", -100)
    assert head.input_value() == "0"
    layer.select_option("0")
    assert wheel(page, "#learn-layer", -100)
    assert layer.input_value() == "0"
    assert wheel(page, "#learn-layer", 12)
    assert layer.input_value() == "0"
    assert wheel(page, "#learn-layer", 12)
    assert layer.input_value() == "1"
    assert not wheel(page, "#learn-layer", 100, ctrlKey=True)
    assert not wheel(page, "#learn-layer", 100, shiftKey=True)
    assert not wheel(page, "#learn-layer", 10, deltaX=100)
    assert layer.input_value() == "1"
    head.select_option("11")
    assert wheel(page, "#learn-head", 100)
    assert head.input_value() == "11"
    # An ordinary vertical wheel gesture over the controls must not move the page.
    page.clock.run_for(150)
    layer.hover()
    before = page.evaluate("scrollY")
    page.mouse.wheel(0, 100)
    from playwright.sync_api import expect
    expect(layer).to_have_value("2")
    assert page.evaluate("scrollY") == before


def test_diagram_arrows_sync_details_and_keyboard(learn_page):
    page = learn_page
    layer = page.locator("#learn-layer")
    assert page.locator('.learn-layer-arrow[aria-disabled="true"]').count() == 4
    page.locator('[data-learn-open="attention"]').click()
    page.locator('[data-learn-nav="attention--1"]').click()
    assert layer.input_value() == "10"
    assert page.locator(".learn-inspector h3").inner_text().endswith("layer 11, head 1")
    assert page.locator('[data-learn-nav="attention--1"]').evaluate("node => node === document.activeElement")
    page.locator('[data-learn-nav="mlp-1"]').focus()
    page.keyboard.press("Enter")
    assert layer.input_value() == "11"
    assert layer.evaluate("node => node === document.activeElement")
    page.locator('[data-learn-nav="qkv--1"]').focus()
    page.keyboard.press("Space")
    assert layer.input_value() == "10"
    assert page.locator(".learn-layer-navigation > text").all_text_contents() == ["Layer 11 / 12"] * 4
    page.locator('[data-model-state="pruned"]').click()
    assert layer.input_value() == "10"
    page.locator('[data-learn-nav="blocks-1"]').click()
    assert page.locator(".bypass-route").count() == 1
    assert wheel(page, '[data-learn-nav="attention--1"]', -100)
    assert layer.input_value() == "10"
    page.locator('[data-learn-nav="toolbar-head-1"]').click()
    assert page.locator("#learn-head").input_value() == "1"
    assert page.locator(".learn-inspector h3").inner_text().endswith("layer 11, head 2")


def test_mobile_navigation_keeps_pan_and_shared_brand(learn_page):
    page = learn_page
    page.set_viewport_size({"width": 390, "height": 844})
    assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
    assert "LLM visualization" in page.title()
    assert page.locator(".brand-logo").evaluate("node => node.complete && node.naturalWidth > 0")
    canvas = page.locator(".learn-canvas-scroll")
    canvas.evaluate("node => node.scrollLeft = 300")
    before = canvas.evaluate("node => node.scrollLeft")
    page.locator('[data-learn-nav="attention--1"]').click()
    assert canvas.evaluate("node => node.scrollLeft") == before
    assert page.locator("#learn-layer").input_value() == "10"
    assert page.locator("#gpt2-view [style]").count() == 0
