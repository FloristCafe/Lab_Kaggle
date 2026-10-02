import torch

from otto_recommender.sasrec import SASRecConfig, SASRecEncoder, masked_in_batch_nce


def test_sasrec_padding_and_type_aware_forward():
    model = SASRecEncoder(SASRecConfig(n_items=100, d_model=16, n_heads=4, n_layers=1, max_len=4))
    items = torch.tensor([[1, 2, 0, 0], [3, 4, 5, 0]])
    types = torch.zeros_like(items)
    result = model(items, types)
    assert result.shape == (2, 16)


def test_in_batch_nce_masks_duplicate_positive_ids():
    embedding = torch.nn.Embedding(20, 8)
    session = torch.randn(3, 8, requires_grad=True)
    ids = torch.tensor([2, 2, 3])
    loss = masked_in_batch_nce(session, ids, embedding)
    assert torch.isfinite(loss)
    loss.backward()
