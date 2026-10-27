"""The simplified Learn renderer is isolated from the Qwen experiment UI."""

from pathlib import Path
import re

from fastapi.testclient import TestClient

from src.web_service import ExperimentService
from web_app import create_app


ROOT = Path(__file__).resolve().parents[1]


def test_learn_uses_separate_continuous_renderer_and_preserves_qwen(tmp_path):
    client = TestClient(create_app(ExperimentService(tmp_path)))
    learn = client.get("/").text
    qwen = client.get("/experiments").text
    assert '/static/learn-explainer.js' in learn
    assert '/static/gpt2.js' not in learn
    assert '/static/gpt2.js' in qwen
    assert '/static/learn-explainer.js' not in qwen
    assert 'walkthrough-intro' not in learn and 'walkthrough-purpose' not in learn
    assert 'id="fixed-sentence"' in learn
    assert 'id="walkthrough-presets"' in learn


def test_learn_assets_are_local_and_csp_compatible(tmp_path):
    client = TestClient(create_app(ExperimentService(tmp_path)))
    response = client.get('/static/learn-explainer.js')
    assert response.status_code == 200
    source = response.text
    assert 'style=' not in source
    assert 'innerHTML' in source
    assert 'onnxruntime' not in source and 'googletagmanager' not in source
    assert 'learn-network' in source and 'learn-inspector' in source
    assert 'data-model-state' in source and 'data-learn-cell' in source
    assert 'Content-Security-Policy' in response.headers
    assert "style-src 'self'" in response.headers['Content-Security-Policy']


def test_upstream_attribution_and_mit_notice_are_packaged(tmp_path):
    client = TestClient(create_app(ExperimentService(tmp_path)))
    page = client.get('/').text
    assert 'github.com/poloclub/transformer-explainer' in page
    license_response = client.get('/static/third-party/transformer-explainer-LICENSE.txt')
    assert license_response.status_code == 200
    assert 'Copyright (c) 2022 Polo Club of Data Science' in license_response.text
    assert 'Permission is hereby granted' in license_response.text
    assert 'THE SOFTWARE IS PROVIDED "AS IS"' in license_response.text


def test_shared_branding_without_logo(tmp_path):
    client = TestClient(create_app(ExperimentService(tmp_path)))
    for path in ('/', '/experiments'):
        page = client.get(path).text
        assert 'LLM visualization' in page
        assert 'Pruning Lab' not in page
        assert 'brand-logo' not in page
        assert 'logo.svg' not in page
    assert client.get('/static/logo.svg').status_code == 404


def test_upstream_visualization_is_local_and_isolated(tmp_path):
    client = TestClient(create_app(ExperimentService(tmp_path)))
    frame = client.get('/static/transformer/index.html')
    assert frame.status_code == 200
    assert '/static/transformer/explainer.js' in frame.text
    assert '/static/transformer/explainer.css' in frame.text
    policy = frame.headers['content-security-policy']
    assert "script-src 'self'" in policy
    assert "style-src 'self' 'unsafe-inline'" in policy
    assert "frame-ancestors 'self'" in policy
    outer = client.get('/').headers['content-security-policy']
    assert "style-src 'self';" in outer
    assert "frame-ancestors 'none'" in outer
    assert 'unsafe-inline' not in outer


def test_learning_guide_follows_visualization_and_links_to_its_chapters(tmp_path):
    client = TestClient(create_app(ExperimentService(tmp_path)))
    page = client.get('/').text
    assert page.index('id="gpt2-view"') < page.index('id="learn-guide"') < page.index('<footer')
    chapters = ['transformer', 'embedding', 'blocks', 'attention', 'mlp', 'probabilities', 'supporting', 'pruning', 'explore']
    for chapter in chapters:
        assert f'id="guide-{chapter}"' in page
        assert f'href="#guide-{chapter}"' in page
        assert f'aria-labelledby="guide-{chapter}-title"' in page
    ids = set(re.findall(r'\bid="([^"]+)"', page))
    anchors = set(re.findall(r'href="#([^"]+)"', page))
    assert anchors <= ids
    assert '/static/learn-guide.css' in page
    assert client.get('/static/learn-guide.css').status_code == 200
    assert 'id="learn-guide"' not in client.get('/experiments').text
    assert '<input' not in page and '<textarea' not in page


def test_learning_guide_explains_actual_controls_and_pruning_limits(tmp_path):
    client = TestClient(create_app(ExperimentService(tmp_path)))
    page = client.get('/').text
    assert '50,257' in page and '3,072' in page and '768' in page
    assert 'head 1 is masked in every layer' in page
    assert 'Heads 2 through 12 are not masked' in page
    assert '—' not in page and '–' not in page
    assert 'activations or attention weights' in page
    assert 'does not permanently erase' in page
    assert 'not raw logits' in page
    assert 'not controls on this page' in page
    assert 'do not establish a smaller checkpoint' in page
    assert 'without rescaling them to sum to 100%' in page
    assert 'no editable prompt or temperature/top-k/top-p control' in page
    assert 'Explanations paraphrased and adapted from' in page
