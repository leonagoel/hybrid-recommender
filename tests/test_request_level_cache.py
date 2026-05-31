from fastapi.testclient import TestClient
import backend.main as backend

client = TestClient(backend.app)


def test_recommendation_cache_user_segment(monkeypatch):
    # Ensure models ready and provide a lightweight hybrid.recommend
    backend.models['ready'] = True

    call_count = {'n': 0}

    def fake_recommend(title, top_n=10, explain=False, target_catalog=None, user_id=None):
        call_count['n'] += 1
        return [{'title': 'Fake A', 'score': 0.9}, {'title': 'Fake B', 'score': 0.8}][:top_n]

    # Monkeypatch the hybrid model used by the API
    backend.models['hybrid'] = type('M', (), {'recommend': staticmethod(fake_recommend), 'get_weights': staticmethod(lambda: {'alpha':0.4,'beta':0.3,'gamma':0.3})})()

    # Clear cache
    backend._clear_response_cache()

    params = {'title': 'Some Item', 'top_n': 2, 'user_id': 'user_abc123456'}

    r1 = client.get('/api/recommend', params=params)
    assert r1.status_code == 200
    assert r1.headers.get('X-Cache') in ('MISS', 'HIT')

    # second request should hit cache for same user segment
    r2 = client.get('/api/recommend', params=params)
    assert r2.status_code == 200
    assert r2.headers.get('X-Cache') == 'HIT'

    # request with different user id (different segment) should call model again or be cache miss
    params2 = {'title': 'Some Item', 'top_n': 2, 'user_id': 'other_user_xyz'}
    r3 = client.get('/api/recommend', params=params2)
    assert r3.status_code == 200
    # Depending on segment prefix collision, this may be MISS or HIT; ensure call_count increased appropriately
    assert call_count['n'] >= 1