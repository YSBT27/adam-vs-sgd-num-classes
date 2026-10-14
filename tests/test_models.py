import torch
from torch.nn.utils import parameters_to_vector

from config import RunConfig
from models import FCN, MaxPool2x2, SmallCNN, build_model, class_block_indices, param_count


def test_forward_is_head_of_features():
    for model, x in ((SmallCNN(10), torch.randn(4, 3, 32, 32)), (FCN(5), torch.randn(4, 1, 28, 28))):
        model.eval()
        assert torch.allclose(model(x), model.head(model.features(x)))
        assert model.last_layer is model.head
        assert model.head.in_features == model.features(x).shape[1]


def test_class_block_indices_cover_head_exactly():
    for model in (SmallCNN(7), FCN(10)):
        flat = parameters_to_vector(model.parameters())
        blocks = class_block_indices(model)
        C, d = model.head.weight.shape
        assert len(blocks) == C and all(len(b) == d + 1 for b in blocks)
        allidx = torch.cat(blocks)
        assert len(torch.unique(allidx)) == len(allidx)  # disjoint
        # values match head parameters
        for c, b in enumerate(blocks):
            expect = torch.cat([model.head.weight[c].detach(), model.head.bias[c].detach().view(1)])
            assert torch.equal(flat[b], expect)
        # nothing outside the head
        head_numel = model.head.weight.numel() + model.head.bias.numel()
        assert len(allidx) == head_numel


def test_param_counts():
    assert param_count(FCN(10)) == 196 * 32 + 32 + 32 * 10 + 10
    assert param_count(FCN(10)) < 20_000
    assert 250_000 < param_count(SmallCNN(100)) < 400_000


def test_same_seed_same_init_across_optimizers():
    a = RunConfig(protocol="cifar_merge", C=10, superclass=False, seed=1, optimizer="sgd", lr=0.1, steps=10)
    b = RunConfig(protocol="cifar_merge", C=10, superclass=False, seed=1, optimizer="adam", lr=1e-3, steps=10)
    c = RunConfig(protocol="cifar_merge", C=10, superclass=False, seed=2, optimizer="adam", lr=1e-3, steps=10)
    va, vb, vc = (parameters_to_vector(build_model(cfg).parameters()) for cfg in (a, b, c))
    assert torch.equal(va, vb) and not torch.equal(va, vc)


def test_maxpool2x2_matches_torch():
    x = torch.randn(3, 4, 8, 8)
    assert torch.equal(MaxPool2x2()(x), torch.nn.MaxPool2d(2)(x))
