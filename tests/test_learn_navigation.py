"""Optional real-browser navigation checks; no model downloads or generation.

Install Playwright and Edge (or Playwright's Chromium) to run these tests.
The fixture data is synthetic and used only by these UI regression tests.
"""

from copy import deepcopy
import re
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
    pruned["layers"][0]["bypassed"] = True
    heads = deepcopy(pipeline)
    magnitude = deepcopy(pipeline)
    random = deepcopy(pipeline)
    for block in heads['layers']:
        block['masked_heads'] = [0, 1, 2]
        block['head_output'][:3] = [[[0, 0], [0, 0]]] * 3
    for block in magnitude['layers'] + random['layers']:
        block.update(masked_weights=4, total_mlp_weights=8)
    topologies = [{"layers": [{"mlp_weights": [{"name": "mlp.c_fc.weight", "sample_values": [0.1] * 8,
                                               "sample": [i % 2 == 0 and variant >= 2 for i in range(8)]}]} for _ in range(12)]}
                  for variant in range(4)]
    return {
        "id": "navigation-test", "revision": 1, "status": "completed",
        "config": {"prompt": pipeline["prompt"]},
        "walkthrough_presets": [{"technique": technique, "title": technique,
                                 "inspection_layer": 0, "inspection_head": 0}
                                for technique in ['layers', 'heads', 'magnitude', 'random']],
        "baselines": {"0": {"pipeline": pipeline, "text": "Original test output", "tokens": 3}},
        "comparisons": {f"{i}:0": {"answer": {"pipeline": p, "text": "Pruned test output", "tokens": 3}}
                        for i, p in enumerate([pruned, heads, magnitude, random])},
        "baseline_view": topologies[0], "pruning_views": topologies,
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
    page.on('console', lambda message: errors.append(message.text) if message.type == 'error' else None)

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
    from playwright.sync_api import expect
    expect(page.frame_locator('#learn-network-frame').locator('.multi-head')).to_be_visible()
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
    assert layer.input_value() == "0" and head.input_value() == "0"
    layer.select_option("11")
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
    from playwright.sync_api import expect
    diagram = page.frame_locator('#learn-network-frame')
    assert diagram.locator('.layer-navigation button:disabled').count() == 3
    layer.select_option("11")
    diagram.locator('[id="self-attention"]').click()
    expect(page.locator('.learn-inspector h3')).to_contain_text('Masked self-attention')
    diagram.locator('[data-nav="attention"]').get_by_role('button', name='Previous layer in Self Attention').click()
    expect(layer).to_have_value("10")
    expect(page.locator(".learn-inspector h3")).to_contain_text("layer 11, head 1")
    diagram.get_by_role('button', name='Next layer in MLP').focus()
    page.keyboard.press("Enter")
    expect(layer).to_have_value("11")
    diagram.get_by_role('button', name='Previous layer in QKV').focus()
    page.keyboard.press("Space")
    expect(layer).to_have_value("10")
    expect(diagram.locator('.layer-navigation').first).to_contain_text('Layer 11 / 12')
    page.locator('[data-model-state="pruned"]').click()
    assert layer.input_value() == "10"
    diagram.get_by_role('button', name='Next layer in MLP').click()
    expect(diagram.locator('.bypass-route')).to_have_count(0)
    expect(diagram.locator('.bypassed-layer-context')).to_be_visible()
    assert wheel(page, '#learn-layer', -100)
    expect(layer).to_have_value("10")
    page.locator('[data-learn-nav="toolbar-head-1"]').click()
    assert page.locator("#learn-head").input_value() == "1"
    assert page.locator(".learn-inspector h3").inner_text().endswith("layer 11, head 2")


def test_mobile_navigation_keeps_pan_and_shared_brand(learn_page):
    page = learn_page
    page.set_viewport_size({"width": 390, "height": 844})
    assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
    assert "LLM visualization" in page.title()
    assert page.locator('.brand img,.brand svg').count() == 0
    canvas = page.locator(".learn-canvas-scroll")
    page.locator('#learn-layer').select_option('11')
    canvas.evaluate("node => node.scrollLeft = 390")
    before = canvas.evaluate("node => node.scrollLeft")
    page.frame_locator('#learn-network-frame').get_by_role('button', name='Previous layer in Self Attention').click()
    assert canvas.evaluate("node => node.scrollLeft") == before
    from playwright.sync_api import expect
    expect(page.locator("#learn-layer")).to_have_value("10")
    assert page.locator("#gpt2-view [style]").count() == 0


def test_upstream_components_use_measured_data_and_persist(learn_page):
    from playwright.sync_api import expect
    page = learn_page
    diagram = page.frame_locator('#learn-network-frame')
    expect(diagram.locator('.head-card')).to_have_count(12)
    expect(diagram.locator('path.sankey-path').first).to_have_attribute('d', re.compile(r'.*M.*'))
    sample = diagram.locator('.head-out canvas').first
    expect(sample).to_have_attribute('data-sample', '[0.1,-0.2]')
    diagram.locator('[data-click="embedding-step-title"]').click()
    expect(diagram.locator('.embedding.expanded')).to_have_count(1)
    page.locator('#learn-layer').select_option('5')
    expect(diagram.locator('.embedding.expanded')).to_have_count(1)
    expect(diagram.locator('[data-nav="attention"]')).to_contain_text('Layer 6 / 12')
    # Only the real parent frame may drive the embedded visualization.
    page.evaluate("window.postMessage({type:'llm-visualizer:navigate',axis:'layer',delta:1}, location.origin)")
    assert page.locator('#learn-layer').input_value() == '5'


def test_actual_pruning_overlays_and_raw_probabilities(learn_page):
    from playwright.sync_api import expect
    page = learn_page
    diagram = page.frame_locator('#learn-network-frame')
    expect(diagram.locator('text.label').first).to_have_text('40.00%')
    page.locator('[data-model-state="pruned"]').click()
    expect(diagram.locator('.bypass-route')).to_have_count(2)
    bounds = diagram.locator('.bypass-route').first.evaluate('node => {const b=node.getBBox();return b.y+b.height;}')
    assert bounds < int(page.locator('#learn-network-frame').get_attribute('height'))
    page.evaluate('run => window.gpt2View.render(run, true, 0, 1)', navigation_run())
    expect(diagram.locator('.head-out canvas').first).to_have_attribute('data-sample', '[0,0]')
    expect(diagram.locator('.head-mask-label')).to_contain_text('output zeroed')
    page.locator('[data-model-state="original"]').click()
    expect(diagram.locator('.head-out canvas').first).to_have_attribute('data-sample', '[0.1,-0.2]')
    for variant in [2, 3]:
        page.evaluate('(args) => window.gpt2View.render(args.run, true, 0, args.variant)', {'run': navigation_run(), 'variant': variant})
        expect(diagram.locator('.mlp-mask-label')).to_contain_text('4 / 8')
        expect(diagram.locator('.weight-sample rect[fill="#d98e78"]')).to_have_count(4)


def test_expanded_attention_is_reachable_without_document_overflow(learn_page):
    from playwright.sync_api import expect
    page = learn_page
    diagram = page.frame_locator('#learn-network-frame')
    diagram.locator('[data-click="attention-matrix"]').click()
    page.clock.run_for(8000)
    expect(diagram.locator('.attention-qk')).to_be_visible()
    expect(diagram.locator('.attention-softmax .main')).to_have_css('opacity', '1')
    required = diagram.locator('.head-content').evaluate('node => Math.ceil(node.getBoundingClientRect().right)')
    assert int(page.locator('#learn-network-frame').get_attribute('width')) >= required
    assert page.evaluate('document.documentElement.scrollWidth <= innerWidth')
    diagram.get_by_role('button', name='Close expanded view ×').click()
    page.clock.run_for(1000)
    expect(diagram.locator('.attention.expanded')).to_have_count(0)


def test_first_bypassed_layer_is_transparent_and_stays_in_background(learn_page):
    from playwright.sync_api import expect
    page = learn_page
    diagram = page.frame_locator('#learn-network-frame')
    expect(page.locator('#learn-layer')).to_have_value('0')
    expect(diagram.locator('.bypassed-layer-contour')).to_have_count(0)
    page.locator('[data-model-state="pruned"]').click()
    contour = diagram.locator('.bypassed-layer-contour')
    expect(contour).to_have_attribute('data-bypassed-layer', '1')
    expect(contour.locator('rect')).to_have_css('fill', 'none')
    expect(contour.locator('rect')).to_have_css('stroke', 'rgb(228, 148, 53)')
    expect(contour.locator('rect')).to_have_css('stroke-dasharray', '8px, 5px')
    expect(diagram.locator('.head-card').first).to_have_css('background-color', 'rgba(0, 0, 0, 0)')
    expect(diagram.locator('.blocks .attention')).to_have_css('opacity', '0.25')
    expect(diagram.locator('.bypassed-layer-context')).to_have_count(0)
    for index in [1, 5, 11]:
        page.locator('#learn-layer').select_option(str(index))
        expect(contour).to_have_count(0)
        context = diagram.locator('.bypassed-layer-context')
        expect(context).to_be_visible()
        expect(context).to_have_attribute('data-bypassed-layer', '1')
        expect(context.locator('.background-layer text')).to_have_text('Layer 1 · bypassed')
        expect(context.locator('.active-context-title')).to_have_text(f'Layer {index + 1} · inspecting')
        expect(context.locator('.background-depth')).to_have_css('fill', 'none')
        expect(context.locator('.background-depth')).to_have_css('stroke-dasharray', '7px, 4px')
        expect(diagram.locator('.blocks .attention')).to_have_css('opacity', '1')
    page.locator('[data-model-state="original"]').click()
    expect(diagram.locator('.bypassed-layer-context')).to_have_count(0)
    page.locator('[data-model-state="pruned"]').click()
    diagram.locator('.bypassed-layer-context').click()
    expect(page.locator('#learn-layer')).to_have_value('0')
    expect(contour).to_have_attribute('data-bypassed-layer', '1')
    expect(diagram.locator('.bypassed-layer-context')).to_have_count(0)
