"""Smoke test for ET-only read ablation in IterativeQueryHead."""
import torch
from tmew1_queries import IterativeQueryHead

def test_backward_compat():
    head = IterativeQueryHead(d_input=64, d_memory=32, max_entities=4,
                              num_query_types=10, d_entity=16)
    seq = torch.randn(2, 8, 64)
    qt = torch.randint(0, 8, (2, 3))
    qy = torch.randint(0, 10, (2, 3))
    et = torch.randn(2, 4, 16)
    tape = torch.randn(2, 8, 32)
    mask = torch.ones(2, 8, dtype=torch.bool)
    e, b = head(seq, qt, qy, entity_state=et, event_tape=tape, event_tape_mask=mask)
    assert e.shape == (2, 3, 4), f"Expected (2,3,4) got {e.shape}"
    assert b.shape == (2, 3, 2), f"Expected (2,3,2) got {b.shape}"
    print("Test 1 PASS: backward compat")

def test_et_only_mixed():
    head = IterativeQueryHead(d_input=64, d_memory=32, max_entities=4,
                              num_query_types=10, d_entity=16,
                              et_only_qtypes={0, 2})
    seq = torch.randn(2, 8, 64)
    qt = torch.randint(0, 8, (2, 3))
    qy = torch.tensor([[0, 1, 2], [5, 0, 1]])
    et = torch.randn(2, 4, 16)
    tape = torch.randn(2, 8, 32)
    mask = torch.ones(2, 8, dtype=torch.bool)
    e, b = head(seq, qt, qy, entity_state=et, event_tape=tape, event_tape_mask=mask)
    assert e.shape == (2, 3, 4)
    print("Test 2 PASS: ET-only mixed routing")

def test_all_et_only():
    head = IterativeQueryHead(d_input=64, d_memory=32, max_entities=4,
                              num_query_types=10, d_entity=16,
                              et_only_qtypes={0, 2})
    seq = torch.randn(2, 8, 64)
    qt = torch.randint(0, 8, (2, 3))
    qy = torch.tensor([[0, 2, 0], [2, 0, 2]])
    et = torch.randn(2, 4, 16)
    e, b = head(seq, qt, qy, entity_state=et)
    assert e.shape == (2, 3, 4)
    print("Test 3 PASS: all-ET-only")

def test_none_et_only():
    head = IterativeQueryHead(d_input=64, d_memory=32, max_entities=4,
                              num_query_types=10, d_entity=16,
                              et_only_qtypes={0, 2})
    seq = torch.randn(2, 8, 64)
    qt = torch.randint(0, 8, (2, 3))
    qy = torch.tensor([[1, 3, 5], [7, 8, 9]])
    et = torch.randn(2, 4, 16)
    tape = torch.randn(2, 8, 32)
    mask = torch.ones(2, 8, dtype=torch.bool)
    e, b = head(seq, qt, qy, entity_state=et, event_tape=tape, event_tape_mask=mask)
    assert e.shape == (2, 3, 4)
    print("Test 4 PASS: none-ET-only (all fused)")

def test_no_entity_state():
    head = IterativeQueryHead(d_input=64, d_memory=32, max_entities=4,
                              num_query_types=10, d_entity=16,
                              et_only_qtypes={0, 2})
    seq = torch.randn(2, 8, 64)
    qt = torch.randint(0, 8, (2, 3))
    qy = torch.tensor([[0, 1, 2], [5, 0, 1]])
    e, b = head(seq, qt, qy)
    assert e.shape == (2, 3, 4)
    print("Test 5 PASS: no entity state fallback")

def test_gradients_flow():
    """Verify gradients flow through both ET-only and fused paths."""
    head = IterativeQueryHead(d_input=64, d_memory=32, max_entities=4,
                              num_query_types=10, d_entity=16,
                              et_only_qtypes={0})
    seq = torch.randn(2, 8, 64, requires_grad=True)
    qt = torch.tensor([[0, 3], [1, 5]])
    qy = torch.tensor([[0, 1], [0, 1]])  # 0=ET-only, 1=fused
    et = torch.randn(2, 4, 16, requires_grad=True)
    tape = torch.randn(2, 8, 32, requires_grad=True)
    mask = torch.ones(2, 8, dtype=torch.bool)
    e, b = head(seq, qt, qy, entity_state=et, event_tape=tape, event_tape_mask=mask)
    loss = e.sum() + b.sum()
    loss.backward()
    assert seq.grad is not None and seq.grad.abs().sum() > 0
    assert et.grad is not None and et.grad.abs().sum() > 0
    print("Test 6 PASS: gradients flow")


if __name__ == "__main__":
    test_backward_compat()
    test_et_only_mixed()
    test_all_et_only()
    test_none_et_only()
    test_no_entity_state()
    test_gradients_flow()
    print("\nAll 6 tests PASSED")
