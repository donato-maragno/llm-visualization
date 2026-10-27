"""The simplified Learn renderer is isolated from the Qwen experiment UI."""

from pathlib import Path

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


def test_shared_branding_and_local_logo(tmp_path):
    client = TestClient(create_app(ExperimentService(tmp_path)))
    for path in ('/', '/experiments'):
        page = client.get(path).text
        assert 'LLM visualization' in page
        assert 'Pruning Lab' not in page
        assert 'class="brand-logo"' in page
        assert '<link rel="icon" href="/static/logo.svg"' in page
    logo = client.get('/static/logo.svg')
    assert logo.status_code == 200
    assert logo.headers['content-type'].startswith('image/svg+xml')
    assert '<circle' in logo.text
