# Author: Theo Gnassounou <theo.gnassounou@inria.fr>
#         Remi Flamary <remi.flamary@polytechnique.edu>
#         Oleksii Kachaiev <kachayev@gmail.com>
#         Antoine Collas <contact@antoinecollas.fr>
#
# License: BSD 3-Clause

import numpy as np
import pytest
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_random_state

from skada import (
    DensityReweight,
    DensityReweightAdapter,
    DiscriminatorReweight,
    DiscriminatorReweightAdapter,
    GaussianReweight,
    GaussianReweightAdapter,
    KLIEPReweight,
    KLIEPReweightAdapter,
    KMMReweight,
    KMMReweightAdapter,
    MMDTarSReweight,
    MMDTarSReweightAdapter,
    NearestNeighborReweight,
    NearestNeighborReweightAdapter,
    make_da_pipeline,
    uLSIFReweight,
    uLSIFReweightAdapter,
)
from skada.base import (
    BaseAdapter,
    BaseEstimator,
    SelectSource,
    SelectSourceTarget,
    SelectTarget,
    Shared,
)
from skada.datasets import make_shifted_datasets
from skada.utils import source_target_split


@pytest.mark.parametrize(
    "estimator",
    [
        make_da_pipeline(
            DensityReweightAdapter(),
            LogisticRegression().set_fit_request(sample_weight=True),
        ),
        DensityReweight(),
        make_da_pipeline(
            GaussianReweightAdapter(),
            LogisticRegression().set_fit_request(sample_weight=True),
        ),
        GaussianReweight(),
        make_da_pipeline(
            DiscriminatorReweightAdapter(),
            LogisticRegression().set_fit_request(sample_weight=True),
        ),
        DiscriminatorReweight(),
        make_da_pipeline(
            KLIEPReweightAdapter(gamma=[0.1, 1, "auto", "scale"], random_state=42),
            LogisticRegression().set_fit_request(sample_weight=True),
        ),
        KLIEPReweight(gamma=[0.1, 1], random_state=42),
        KLIEPReweight(gamma=0.2),
        NearestNeighborReweight(
            LogisticRegression().set_fit_request(sample_weight=True),
            laplace_smoothing=True,
            n_neighbors=3,
        ),
        NearestNeighborReweight(laplace_smoothing=True),
        make_da_pipeline(
            NearestNeighborReweightAdapter(
                laplace_smoothing=True,
                n_neighbors=1,
            ),
            LogisticRegression().set_fit_request(sample_weight=True),
        ),
        make_da_pipeline(
            KMMReweightAdapter(gamma=0.1),
            LogisticRegression().set_fit_request(sample_weight=True),
        ),
        KMMReweight(),
        KMMReweight(eps=0.1),
        KMMReweight(solver="frank-wolfe"),
        KMMReweight(solver="scipy"),
        make_da_pipeline(
            uLSIFReweightAdapter(random_state=42),
            LogisticRegression().set_fit_request(sample_weight=True),
        ),
        uLSIFReweight(random_state=42),
        uLSIFReweight(gamma=1.0, reg=0.1, random_state=42),
        make_da_pipeline(
            MMDTarSReweightAdapter(gamma=1.0),
            LogisticRegression().set_fit_request(sample_weight=True),
        ),
        MMDTarSReweight(gamma=1.0),
    ],
)
def test_reweight_estimator(estimator, da_dataset):
    X_train, y_train, sample_domain = da_dataset.pack(
        as_sources=["s"], as_targets=["t"], mask_target_labels=True
    )
    estimator.fit(X_train, y_train, sample_domain=sample_domain)
    X_test, y_test, sample_domain = da_dataset.pack(
        as_sources=[], as_targets=["t"], mask_target_labels=False
    )
    y_pred = estimator.predict(X_test, sample_domain=sample_domain)
    assert np.mean(y_pred == y_test) > 0.9
    score = estimator.score(X_test, y_test, sample_domain=sample_domain)
    assert score > 0.9


@pytest.mark.parametrize(
    "estimator",
    [
        make_da_pipeline(
            DensityReweightAdapter(), Ridge().set_fit_request(sample_weight=True)
        ),
        DensityReweight(Ridge().set_fit_request(sample_weight=True)),
        make_da_pipeline(
            GaussianReweightAdapter(),
            Ridge().set_fit_request(sample_weight=True),
        ),
        GaussianReweight(Ridge().set_fit_request(sample_weight=True)),
        make_da_pipeline(
            DiscriminatorReweightAdapter(),
            Ridge().set_fit_request(sample_weight=True),
        ),
        DiscriminatorReweight(Ridge().set_fit_request(sample_weight=True)),
        make_da_pipeline(
            KLIEPReweightAdapter(gamma=[0.1, 1, "auto", "scale"], random_state=42),
            Ridge().set_fit_request(sample_weight=True),
        ),
        KLIEPReweight(
            Ridge().set_fit_request(sample_weight=True), gamma=[0.1, 1], random_state=42
        ),
        KLIEPReweight(Ridge().set_fit_request(sample_weight=True), gamma=0.2),
        make_da_pipeline(
            KMMReweightAdapter(gamma=0.1), Ridge().set_fit_request(sample_weight=True)
        ),
        KMMReweight(Ridge().set_fit_request(sample_weight=True)),
        KMMReweight(Ridge().set_fit_request(sample_weight=True), eps=0.1),
        make_da_pipeline(
            uLSIFReweightAdapter(random_state=42),
            Ridge().set_fit_request(sample_weight=True),
        ),
        uLSIFReweight(Ridge().set_fit_request(sample_weight=True), random_state=42),
        make_da_pipeline(
            MMDTarSReweightAdapter(gamma=1.0),
            Ridge().set_fit_request(sample_weight=True),
        ),
        MMDTarSReweight(Ridge().set_fit_request(sample_weight=True), gamma=1.0),
    ],
)
def test_reg_reweight_estimator(estimator):
    dataset = make_shifted_datasets(
        n_samples_source=20,
        n_samples_target=21,
        shift="conditional_shift",
        mean=0.5,
        noise=0.3,
        label="regression",
        random_state=43,
        return_dataset=True,
    )
    X_train, y_train, sample_domain_train = dataset.pack(
        as_sources=["s"], as_targets=["t"], mask_target_labels=True
    )
    estimator.fit(X_train, y_train, sample_domain=sample_domain_train)
    X_test, y_test, _ = dataset.pack(
        as_sources=[], as_targets=["t"], mask_target_labels=False
    )
    score = estimator.score(X_test, y_test)
    assert score >= 0


def _base_test_new_X_adapt(estimator, da_dataset):
    X_train, y_train, sample_domain = da_dataset.pack(
        as_sources=["s"], as_targets=["t"], mask_target_labels=True
    )

    # fit works with no errors
    estimator.fit(X_train, y_train, sample_domain=sample_domain)

    # fit_transform returns additional parameters
    _, res1 = estimator.fit_transform(X_train, y_train, sample_domain=sample_domain)
    rng = check_random_state(43)
    idx = rng.choice(X_train.shape[0], 10)
    true_weights = res1["sample_weight"][idx]

    # Adapt with new X, i.e. same domain, different samples
    sample_weight_2 = estimator.compute_weights(
        X_train[idx, :] + 1e-8, y_train[idx], sample_domain=sample_domain[idx]
    )

    # Check that the normalized weights are the same
    true_weights = true_weights / np.sum(true_weights)
    sample_weight_2 /= np.sum(sample_weight_2)
    assert np.allclose(true_weights, sample_weight_2)

    # Check it adapts even if some target classes are not present in the new X
    classes = np.unique(y_train)[::2]
    mask = np.isin(y_train, classes)
    X_train = X_train[mask]
    y_train = y_train[mask]
    sample_domain = sample_domain[mask]
    sample_weight_3 = estimator.compute_weights(
        X_train, y_train, sample_domain=sample_domain
    )

    # Check that the normalized weights are the same
    true_weights = res1["sample_weight"][mask]
    true_weights = true_weights / np.sum(true_weights)
    sample_weight_3 /= np.sum(sample_weight_3)
    assert np.allclose(true_weights, sample_weight_3)


@pytest.mark.parametrize(
    "estimator",
    [
        (DensityReweightAdapter()),
        (DensityReweightAdapter()),
        (GaussianReweightAdapter()),
        (GaussianReweightAdapter()),
        (DiscriminatorReweightAdapter()),
        (DiscriminatorReweightAdapter()),
        (KLIEPReweightAdapter(gamma=[0.1, 1, "auto", "scale"], random_state=42)),
        (KLIEPReweightAdapter(gamma=[0.1, 1, "auto", "scale"], random_state=42)),
        (KMMReweightAdapter(gamma=0.1, smooth_weights=True)),
        (KMMReweightAdapter(gamma=0.1, smooth_weights=True)),
        (uLSIFReweightAdapter(random_state=42)),
        (uLSIFReweightAdapter(random_state=42)),
        (MMDTarSReweightAdapter(gamma=1.0)),
        (MMDTarSReweightAdapter(gamma=1.0)),
    ],
)
def test_new_X_adapt(estimator, da_reg_datasets):
    for dataset in da_reg_datasets:
        _base_test_new_X_adapt(estimator, dataset)


@pytest.mark.parametrize(
    "estimator",
    [
        DensityReweightAdapter(),
        GaussianReweightAdapter(),
        DiscriminatorReweightAdapter(),
        KLIEPReweightAdapter(gamma=[0.1, 1, "auto", "scale"], random_state=42),
        KMMReweightAdapter(gamma=0.1, smooth_weights=True),
        uLSIFReweightAdapter(random_state=42),
        MMDTarSReweightAdapter(gamma=1.0),
    ],
)
def test_reg_new_X_adapt(estimator, da_reg_dataset):
    _base_test_new_X_adapt(estimator, da_reg_dataset)


def test_reweight_warning(da_dataset):
    X_train, y_train, sample_domain = da_dataset.pack(
        as_sources=["s"], as_targets=["t"], mask_target_labels=True
    )
    estimator = KLIEPReweightAdapter(gamma=0.1, max_iter=0)
    estimator.fit(X_train, y_train, sample_domain=sample_domain)

    with pytest.warns(
        UserWarning, match="Maximum iteration reached before convergence."
    ):
        estimator.fit(X_train, y_train, sample_domain=sample_domain)


def test_KMMReweight_kernel_error():
    with pytest.raises(ValueError, match="got 'hello'"):
        KMMReweightAdapter(kernel="hello")


def test_uLSIFReweight_reg_error(da_dataset):
    X_train, y_train, sample_domain = da_dataset.pack(
        as_sources=["s"], as_targets=["t"], mask_target_labels=True
    )
    estimator = uLSIFReweightAdapter(gamma=1.0, reg=-1)
    with pytest.raises(ValueError, match="`reg` argument should be positive"):
        estimator.fit(X_train, y_train, sample_domain=sample_domain)


def test_uLSIFReweight_weights_match_true_ratio():
    rng = check_random_state(42)
    n_samples = 1000
    # source and target are Gaussians with the same covariance and shifted
    # means, so the true importance w(x) = p_target(x) / p_source(x) is known
    mean_source = np.zeros(2)
    mean_target = np.array([1.0, 0.0])
    cov = np.eye(2)
    X_source = rng.multivariate_normal(mean_source, cov, n_samples)
    X_target = rng.multivariate_normal(mean_target, cov, n_samples)

    true_ratio = np.exp(
        X_source @ (mean_target - mean_source)
        - 0.5 * (mean_target @ mean_target - mean_source @ mean_source)
    )

    X = np.concatenate([X_source, X_target])
    sample_domain = np.concatenate([np.ones(n_samples), -np.ones(n_samples)])

    estimator = uLSIFReweightAdapter(random_state=0)
    estimator.fit(X, sample_domain=sample_domain)
    weights = estimator.compute_weights(X, sample_domain=sample_domain)
    source_weights = weights[:n_samples]

    # the estimated weights are non-negative
    assert np.all(source_weights >= 0)
    # and they follow the true ratio: same ranking
    corr, _ = spearmanr(source_weights, true_ratio)
    assert corr > 0.9
    # and close values (both normalized to mean 1 over the source samples)
    true_ratio /= true_ratio.mean()
    assert np.mean((source_weights - true_ratio) ** 2) < 0.5


def test_uLSIFReweight_improves_target_accuracy():
    rng = check_random_state(0)
    n_samples = 400
    # covariate shift: the source is concentrated on x[0] < 0 and the target
    # on x[0] > 0, while the labeling function sign(x[0]) is shared
    X_source = rng.normal([-2.5, 0.0], 1.0, size=(n_samples, 2))
    X_target = rng.normal([2.5, 0.0], 1.0, size=(n_samples, 2))
    y_source = (X_source[:, 0] > 0).astype(int)
    y_target = (X_target[:, 0] > 0).astype(int)

    # without adaptation, the source classifier is biased towards the
    # majority class of the source domain
    estimator = LogisticRegression().fit(X_source, y_source)
    score_unweighted = estimator.score(X_target, y_target)
    assert score_unweighted < 0.9

    X = np.concatenate([X_source, X_target])
    # target labels are masked, as they are not available at fit time
    y = np.concatenate([y_source, -np.ones(n_samples, dtype=int)])
    sample_domain = np.concatenate([np.ones(n_samples), -np.ones(n_samples)])

    estimator = uLSIFReweight(random_state=0)
    estimator.fit(X, y, sample_domain=sample_domain)
    score_reweighted = estimator.score(
        X_target, y_target, sample_domain=-np.ones(n_samples)
    )
    assert score_reweighted > score_unweighted
    assert score_reweighted > 0.9


def test_uLSIFReweight_densratio_cross_check():
    # optional parity cross-check against the MIT-licensed densratio package
    densratio = pytest.importorskip("densratio")
    rng = check_random_state(42)
    n_samples = 300
    X_source = rng.normal(np.zeros(2), 1.0, size=(n_samples, 2))
    X_target = rng.normal(np.array([1.0, 0.0]), 1.0, size=(n_samples, 2))

    X = np.concatenate([X_source, X_target])
    sample_domain = np.concatenate([np.ones(n_samples), -np.ones(n_samples)])

    estimator = uLSIFReweightAdapter(random_state=0)
    estimator.fit(X, sample_domain=sample_domain)
    weights = estimator.compute_weights(X, sample_domain=sample_domain)[:n_samples]

    # densratio(x, y) estimates p_x / p_y, so the target comes first
    result = densratio(X_target, X_source)
    ref_weights = result.compute_density_ratio(X_source)

    corr, _ = spearmanr(weights, ref_weights)
    assert corr > 0.8


# KMMReweight.adapt behavior should be the same when smooth weights is True or
# when X_source differs between fit and adapt.
def test_KMMReweight_new_X_adapt(da_dataset):
    X_train, y_train, sample_domain = da_dataset.pack(
        as_sources=["s"], as_targets=["t"], mask_target_labels=True
    )
    estimator = KMMReweightAdapter(smooth_weights=True)
    _, res1 = estimator.fit_transform(X_train, sample_domain=sample_domain)

    estimator = KMMReweightAdapter(smooth_weights=False)
    _, res2 = estimator.fit_transform(X_train, sample_domain=sample_domain)
    weights3 = estimator.compute_weights(X_train + 1e-8, sample_domain=sample_domain)

    assert np.allclose(res1["sample_weight"], weights3)
    assert not np.allclose(res1["sample_weight"], res2["sample_weight"])


@pytest.mark.parametrize(
    "mediator",
    [
        StandardScaler(),
        SelectSource(StandardScaler()),
        SelectTarget(StandardScaler()),
        SelectSourceTarget(StandardScaler()),
    ],
)
def test_adaptation_output_propagation_multiple_steps(da_reg_dataset, mediator):
    X, y, sample_domain = da_reg_dataset.pack(
        as_sources=["s"], as_targets=["t"], mask_target_labels=False
    )
    _, X_target, _, target_domain = source_target_split(
        X, sample_domain, sample_domain=sample_domain
    )

    class FakeEstimator(BaseEstimator):
        __metadata_request__fit = {"sample_weight": True}
        __metadata_request__predict = {"sample_weight": True}

        def fit(self, _X, _y, sample_weight=None):
            assert sample_weight.shape[0] > 0
            return self

        def predict(self, X, sample_weight=None):
            # xxx(okachaiev): i need to come up with a more accurate test
            assert sample_weight is None
            return X

    clf = make_da_pipeline(
        Shared(DensityReweightAdapter(), mask_target_labels=False),
        mediator,
        FakeEstimator(),
    )

    # check no errors are raised
    clf.fit(X, y, sample_domain=sample_domain)
    clf.predict(X_target, sample_domain=target_domain)


def test_select_source_target_output_merge(da_reg_dataset):
    X, y, sample_domain = da_reg_dataset.pack(
        as_sources=["s"], as_targets=["t"], mask_target_labels=False
    )
    _, X_target, _, target_domain = source_target_split(
        X, sample_domain, sample_domain=sample_domain
    )

    class FakeAdapter(BaseAdapter):
        def __init__(self, multiplier):
            self.multiplier = multiplier

        def fit_transform(self, X, y=None, *, sample_domain=None):
            self.fitted_ = True
            return X, dict(sample_weight=np.ones(X.shape[0]) * self.multiplier)

    clf = make_da_pipeline(
        SelectSourceTarget(FakeAdapter(1.0), FakeAdapter(2.0)),
        Ridge().set_fit_request(sample_weight=True),
    )

    # check no errors are raised
    clf.fit(X, y, sample_domain=sample_domain)
    clf.predict(X_target, sample_domain=target_domain)
