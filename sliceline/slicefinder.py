"""
The slicefinder module implements the Slicefinder class.
"""
import logging
from typing import Tuple, Union

import numpy as np
from scipy import sparse as sp
from scipy.stats import rankdata
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils.validation import check_is_fitted

from sliceline.validation import check_array, check_X_e

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class Slicefinder(BaseEstimator, TransformerMixin):
    """Slicefinder class.

    SliceLine is a fast, linear-algebra-based slice finding for ML Model Debugging.

    Given an input dataset (`X`) and a model error vector (`errors`), SliceLine finds
    the `k` slices in `X` that identify where the model performs significantly worse.
    A slice is a subspace of `X` defined by one or more predicates.
    The maximal dimension of this subspace is controlled by `max_l`.

    The slice scoring function is the linear combination of two objectives:
        - Find sufficiently large slices, with more than `min_sup` elements
          (high impact on the overall model)
        - With substantial errors
          (high negative impact on sub-group/model)
    The importance of each objective is controlled through a single parameter `alpha`.

    Slice enumeration and pruning techniques are done via sparse linear algebra.

    Parameters
    ----------
    alpha: float, default=0.6
        Weight parameter for the importance of the average slice error.
        0 < `alpha` <= 1.

    k: int, default=1
        Maximum number of slices to return.
        Note: in case of equality between `k`-th slice score and the following ones,
        all those slices are returned, leading to `_n_features_out` slices returned.
        (`_n_features_out` >= `k`)

    max_l: int, default=4
        Maximum lattice level.
        In other words: the maximum number of predicate to define a slice.

    min_sup: int or float, default=10
        Minimum support threshold.
        Inspired by frequent itemset mining, it ensures statistical significance.
        If `min_sup` is a float (0 < `min_sup` < 1),
            it represents the faction of the input dataset (`X`).

    verbose: bool, default=True
        Controls the verbosity.

    Attributes
    ----------
    top_slices_: np.ndarray of shape (_n_features_out, number of columns of the input dataset)
        The `_n_features_out` slices with the highest score.
        `None` values in slices represent unused column in the slice.

    average_error_: float
        Mean value of the input error.

    References
    ----------
    `SliceLine: Fast, Linear-Algebra-based Slice Finding for ML Model Debugging
    <https://mboehm7.github.io/resources/sigmod2021b_sliceline.pdf>`__,
    from *Svetlana Sagadeeva* and *Matthias Boehm* of Graz University of Technology.
    """

    def __init__(
        self,
        alpha: float = 0.6,
        k: int = 1,
        max_l: int = 4,
        min_sup: Union[int, float] = 10,
        verbose: bool = True,
    ):
        self.alpha = alpha
        self.k = k
        self.max_l = max_l
        self.min_sup = min_sup
        self.verbose = verbose

        self._one_hot_encoder = self._top_slices_enc = None
        self.top_slices_ = self.average_error_ = None

        if self.verbose:
            logger.setLevel(logging.DEBUG)
        else:
            logger.setLevel(logging.INFO)

    def _check_params(self):
        """Check transformer parameters."""
        if not 0 < self.alpha <= 1:
            raise ValueError(f"Invalid 'alpha' parameter: {self.alpha}")

        if self.k <= 0:
            raise ValueError(f"Invalid 'k' parameter: {self.k}")

        if self.max_l <= 0:
            raise ValueError(f"Invalid 'max_l' parameter: {self.max_l}")

        if self.min_sup < 0 or (
            isinstance(self.min_sup, float) and self.min_sup >= 1
        ):
            raise ValueError(f"Invalid 'min_sup' parameter: {self.min_sup}")

    def _check_top_slices(self):
        """Check if slices have been found."""
        # Check if fit has been called
        check_is_fitted(self)

        # Check if a slice has been found
        if self.top_slices_.size == 0:
            raise ValueError("No transform: Sliceline did not find any slice.")

    def fit(self, X, errors):
        """Search for slice(s) on `X` based on `errors`.

        Parameters
        ----------
        X: array-like of shape (n_samples, n_features)
            Training data, where `n_samples` is the number of samples
            and `n_features` is the number of features.

        errors: array-like of shape (n_samples, )
            Errors of a machine learning model.

        Returns
        -------
        self: object
            Returns the instance itself.
        """
        self._check_params()

        # Update min_sup for a fraction of the input dataset size
        if 0 < self.min_sup < 1:
            self.min_sup = int(self.min_sup * len(X))

        # Check that X and e have correct shape
        X_array, errors = check_X_e(X, errors, y_numeric=True)

        self._check_feature_names(X, reset=True)

        self._search_slices(X_array, errors)

        return self

    def transform(self, X):
        """Generate slices masks for `X`.

        Parameters
        ----------
        X: array-like of shape (n_samples, n_features)
            Training data, where `n_samples` is the number of samples
            and `n_features` is the number of features.

        Returns
        -------
        slices_masks: np.ndarray of shape (n_samples, _n_features_out)
            `slices_masks[i, j] == 1`: the `i`-th sample of `X` is in the `j`-th `top_slices_`.
        """
        self._check_top_slices()

        # Input validation
        X = check_array(X)

        slices_masks = self._get_slices_masks(X)

        return slices_masks.T

    def get_slice(self, X, slice_index: int):
        """Filter `X` samples according to the `slice_index`-th slice.

        Parameters
        ----------
        X: array-like of shape (n_samples, n_features)
            Training data, where `n_samples` is the number of samples
            and `n_features` is the number of features.

        slice_index: int
            Index of the slice to get from `top_slices_`.

        Returns
        -------
        X_slice: np.ndarray of shape (n_samples in the `slice_index`-th slice, n_features)
            Filter `X` samples that are in the `slice_index`-th slice.
        """
        self._check_top_slices()

        # Input validation
        X = check_array(X)

        slices_masks = self._get_slices_masks(X)

        return X[np.where(slices_masks[slice_index])[0], :]

    def get_feature_names_out(self):
        """Get output feature names for transformation.

        Returns
        -------
        feature_names_out : ndarray of str objects
            The following output feature names are generated:
            `["slice_0", "slice_1", ..., "slice_(_n_features_out)"]`.
        """
        check_is_fitted(self)

        feature_names = [f"slice_{i}" for i in range(self._n_features_out)]

        return np.array(feature_names, dtype=object)

    def _get_slices_masks(self, X):
        """Private utilities function generating slices masks for `X`."""
        X_encoded = self._one_hot_encoder.transform(X)

        # Shape X_encoded: (X.shape[0], total number of modalities in _one_hot_encoder.categories_)
        # Shape _top_slices_enc: (top_slices_.shape[0], X_encoded[1])
        slice_candidates = self._top_slices_enc @ X_encoded.T

        # self._top_slices_enc.sum(axis=1) is the number of predicate(s) for each top_slices_
        slices_masks = (
            slice_candidates == self._top_slices_enc.sum(axis=1)
        ).A.astype(int)

        return slices_masks

    @property
    def _n_features_out(self):
        """Number of transformed output features."""
        return self.top_slices_.shape[0]

    @staticmethod
    def _dummify(array: np.ndarray, n_col_x_encoded: int) -> sp.csr_matrix:
        """Dummify `array` with respect to `n_col_x_encoded`.
        Assumption: v does not contain any 0."""
        assert (
            0 not in array
        ), "Modality 0 is not expected to be one-hot encoded."
        one_hot_encoding = sp.lil_matrix(
            (array.size, n_col_x_encoded), dtype=bool
        )
        one_hot_encoding[np.arange(array.size), array - 1] = True
        return one_hot_encoding.tocsr()

    def _maintain_top_k(
        self,
        slices: sp.csr_matrix,
        statistics: np.ndarray,
        top_k_slices: sp.csr_matrix,
        top_k_statistics: np.ndarray,
    ) -> Tuple[sp.csr_matrix, np.ndarray]:
        """Add new `slices` to `top_k_slices` and update the top-k slices."""
        # prune invalid min_sup and scores
        valid_slices_mask = (statistics[:, 3] >= self.min_sup) & (
            statistics[:, 0] > 0
        )
        if np.sum(valid_slices_mask) != 0:
            slices, statistics = (
                slices[valid_slices_mask],
                statistics[valid_slices_mask],
            )

            if (slices.shape[1] != top_k_slices.shape[1]) & (
                slices.shape[1] == 1
            ):
                slices, statistics = slices.T, statistics.T

            # evaluated candidates and previous top-k
            slices = sp.vstack([top_k_slices, slices])
            statistics = np.concatenate([top_k_statistics, statistics])

            # extract top-k
            top_slices_bool = (
                rankdata(-statistics[:, 0], method="min") <= self.k
            )
            top_k_slices, top_k_statistics = (
                slices[top_slices_bool],
                statistics[top_slices_bool],
            )
            top_slices_indices = np.argsort(-top_k_statistics[:, 0])
            top_k_slices, top_k_statistics = (
                top_k_slices[top_slices_indices],
                top_k_statistics[top_slices_indices],
            )
        return top_k_slices, top_k_statistics

    def _score_ub(
        self,
        slice_sizes_ub: np.ndarray,
        slice_errors_ub: np.ndarray,
        max_slice_errors_ub: np.ndarray,
        n_col_x_encoded: int,
    ) -> np.ndarray:
        """Compute the upper-bound score for all the slices."""
        # Since slice_scores is either monotonically increasing or decreasing, we
        # probe interesting points of slice_scores in the interval [min_sup, ss],
        # and compute the maximum to serve as the upper bound
        potential_solutions = np.column_stack(
            (
                self.min_sup * np.ones(slice_sizes_ub.shape[0]),
                np.maximum(
                    slice_errors_ub / max_slice_errors_ub, self.min_sup
                ),
                slice_sizes_ub,
            )
        )
        slice_scores_ub = np.amax(
            (
                self.alpha
                * (
                    np.minimum(
                        potential_solutions.T * max_slice_errors_ub,
                        slice_errors_ub,
                    ).T
                    / self.average_error_
                    - potential_solutions
                )
                - (1 - self.alpha) * (n_col_x_encoded - potential_solutions)
            )
            / potential_solutions,
            axis=1,
        )
        return slice_scores_ub

    @staticmethod
    def _analyse_top_k(top_k_statistics: np.ndarray) -> tuple:
        """Get the maximum and the minimum slices scores."""
        max_slice_scores = min_slice_scores = -np.inf
        if top_k_statistics.shape[0] > 0:
            max_slice_scores = top_k_statistics[0, 0]
            min_slice_scores = top_k_statistics[
                top_k_statistics.shape[0] - 1, 0
            ]
        return max_slice_scores, min_slice_scores

    def _score(
        self,
        slice_sizes: np.ndarray,
        slice_errors: np.ndarray,
        n_row_x_encoded: int,
    ) -> np.ndarray:
        """Compute the score for all the slices."""
        slice_scores = self.alpha * (
            (slice_errors / slice_sizes) / self.average_error_ - 1
        ) - (1 - self.alpha) * (n_row_x_encoded / slice_sizes - 1)
        return np.nan_to_num(slice_scores, nan=-np.inf)

    def _eval_slice(
        self,
        x_encoded: sp.csr_matrix,
        errors: np.ndarray,
        slices: sp.csr_matrix,
        level: int,
    ) -> np.ndarray:
        """Compute several statistics for all the slices."""
        slice_candidates = x_encoded @ slices.T == level
        slice_sizes = slice_candidates.sum(axis=0).A[0]
        slice_errors = errors @ slice_candidates
        max_slice_errors = slice_candidates.T.multiply(errors).max(axis=1).A

        # score of relative error and relative size
        slice_scores = self._score(
            slice_sizes, slice_errors, x_encoded.shape[0]
        )
        return np.column_stack(
            [slice_scores, slice_errors, max_slice_errors, slice_sizes]
        )

    def _create_and_score_basic_slices(
        self,
        x_encoded: sp.csr_matrix,
        n_col_x_encoded: int,
        errors: np.ndarray,
    ) -> Tuple[sp.csr_matrix, np.ndarray]:
        """Initialise 1-slices, i.e. slices with one predicate."""
        slice_sizes = x_encoded.sum(axis=0).A[0]
        slice_errors = errors @ x_encoded
        max_slice_errors = x_encoded.T.multiply(errors).max(axis=1).A[:, 0]

        # working set of active slices (#attr x #slices) and top-k
        valid_slices_mask = (slice_sizes >= self.min_sup) & (slice_errors > 0)
        attr = np.arange(1, n_col_x_encoded + 1)[valid_slices_mask]
        slice_sizes = slice_sizes[valid_slices_mask]
        slice_errors = slice_errors[valid_slices_mask]
        max_slice_errors = max_slice_errors[valid_slices_mask]
        slices = self._dummify(attr, n_col_x_encoded)

        # score 1-slices and create initial top-k
        slice_scores = self._score(
            slice_sizes, slice_errors, x_encoded.shape[0]
        )
        statistics = np.column_stack(
            (slice_scores, slice_errors, max_slice_errors, slice_sizes)
        )

        n_col_dropped = n_col_x_encoded - sum(valid_slices_mask)
        logger.debug(
            "Dropping %i/%i features below min_sup = %i."
            % (n_col_dropped, n_col_x_encoded, self.min_sup)
        )

        return slices, statistics

    def _get_pruned_s_r(
        self, slices: sp.csr_matrix, statistics: np.ndarray
    ) -> Tuple[sp.csr_matrix, np.ndarray]:
        """Prune invalid slices.
        Do not affect overall pruning effectiveness due to handling of missing parents."""
        valid_slices_mask = (statistics[:, 3] >= self.min_sup) & (
            statistics[:, 1] > 0
        )
        return slices[valid_slices_mask], statistics[valid_slices_mask]

    @staticmethod
    def _join_compatible_slices(
        slices: sp.csr_matrix, level: int
    ) -> np.ndarray:
        """Join compatible slices according to `level`."""
        slices_int = slices.astype(int)
        join = (slices_int @ slices_int.T).A == level - 2
        return np.triu(join, 1) * join

    @staticmethod
    def _combine_slices(
        slices: sp.csr_matrix,
        statistics: np.ndarray,
        compatible_slices: np.ndarray,
    ) -> Tuple[sp.csr_matrix, np.ndarray, np.ndarray, np.ndarray]:
        """Combine slices by exploiting parents node statistics."""
        parent_1_idx, parent_2_idx = np.where(compatible_slices == 1)
        pair_candidates = slices[parent_1_idx] + slices[parent_2_idx]

        slice_errors = np.minimum(
            statistics[parent_1_idx, 1], statistics[parent_2_idx, 1]
        )
        max_slice_errors = np.minimum(
            statistics[parent_1_idx, 2], statistics[parent_2_idx, 2]
        )
        slice_sizes = np.minimum(
            statistics[parent_1_idx, 3], statistics[parent_2_idx, 3]
        )
        return pair_candidates, slice_sizes, slice_errors, max_slice_errors

    @staticmethod
    def _prune_invalid_self_joins(
        feature_offset_start: np.ndarray,
        feature_offset_end: np.ndarray,
        pair_candidates: sp.csr_matrix,
        slice_sizes: np.ndarray,
        slice_errors: np.ndarray,
        max_slice_errors: np.ndarray,
    ) -> Tuple[sp.csr_matrix, np.ndarray, np.ndarray, np.ndarray]:
        """Prune invalid self joins (>1 bit per feature)."""
        valid_slices_mask = np.full(pair_candidates.shape[0], True)
        for start, end in zip(feature_offset_start, feature_offset_end):
            valid_slices_mask = (
                valid_slices_mask
                * (pair_candidates[:, start:end].sum(axis=1) <= 1).A[:, 0]
            )
        return (
            pair_candidates[valid_slices_mask],
            slice_sizes[valid_slices_mask],
            slice_errors[valid_slices_mask],
            max_slice_errors[valid_slices_mask],
        )

    @staticmethod
    def _prepare_deduplication_and_pruning(
        feature_offset_start: np.ndarray,
        feature_offset_end: np.ndarray,
        feature_domains: np.ndarray,
        pair_candidates: sp.csr_matrix,
    ) -> np.ndarray:
        """Prepare IDs for deduplication and pruning."""
        ids = np.zeros(pair_candidates.shape[0])
        dom = feature_domains + 1
        for j, (start, end) in enumerate(
            zip(feature_offset_start, feature_offset_end)
        ):
            sub_pair_candidates = pair_candidates[:, start:end]
            # sub_p should not contain multiple True on the same line
            i = sub_pair_candidates.argmax(axis=1).T + np.any(
                sub_pair_candidates.A, axis=1
            )
            ids = ids + i.A * np.prod(dom[(j + 1) : dom.shape[0]])
        return ids

    def _get_pair_candidates(
        self,
        slices: sp.csr_matrix,
        statistics: np.ndarray,
        top_k_statistics: np.ndarray,
        level: int,
        n_col_x_encoded: int,
        feature_domains: np.ndarray,
        feature_offset_start: np.ndarray,
        feature_offset_end: np.ndarray,
    ) -> sp.csr_matrix:
        """Compute and prune plausible slices candidates."""
        compatible_slices = self._join_compatible_slices(slices, level)

        if np.sum(compatible_slices) == 0:
            return sp.csr_matrix(np.empty((0, slices.shape[1])))

        (
            pair_candidates,
            slice_sizes,
            slice_errors,
            max_slice_errors,
        ) = self._combine_slices(slices, statistics, compatible_slices)

        (
            pair_candidates,
            slice_sizes,
            slice_errors,
            max_slice_errors,
        ) = self._prune_invalid_self_joins(
            feature_offset_start,
            feature_offset_end,
            pair_candidates,
            slice_sizes,
            slice_errors,
            max_slice_errors,
        )

        if pair_candidates.shape[0] == 0:
            return sp.csr_matrix(np.empty((0, slices.shape[1])))

        ids = self._prepare_deduplication_and_pruning(
            feature_offset_start,
            feature_offset_end,
            feature_domains,
            pair_candidates,
        )

        # remove duplicate candidates and select corresponding statistics
        _, unique_candidate_indices, duplicate_counts = np.unique(
            ids, return_index=True, return_counts=True
        )

        # Slices at level i normally have i parents (cf. section 3.1 in the paper)
        # We want to keep only slices whose parents have not been pruned.
        # If all the parents are present they are going to get combined 2 by 2 in i*(i-1)/2 ways
        # So, we select only candidates which appear with the correct cardinality.
        all_parents_mask = duplicate_counts == level * (level - 1) / 2
        unique_candidate_indices = unique_candidate_indices[all_parents_mask]

        pair_candidates = pair_candidates[unique_candidate_indices]
        slice_sizes = slice_sizes[unique_candidate_indices]
        slice_errors = slice_errors[unique_candidate_indices]
        max_slice_errors = max_slice_errors[unique_candidate_indices]

        slice_scores = self._score_ub(
            slice_sizes,
            slice_errors,
            max_slice_errors,
            n_col_x_encoded,
        )

        # Seems to be always fully True
        # Due to maintain_top_k that apply slice_sizes filter
        pruning_sizes = slice_sizes >= self.min_sup

        _, min_slice_scores = self._analyse_top_k(top_k_statistics)

        pruning_scores = (slice_scores > min_slice_scores) & (slice_scores > 0)

        return pair_candidates[pruning_scores & pruning_sizes]

    def _search_slices(
        self,
        input_x: np.ndarray,
        errors: np.ndarray,
    ) -> None:
        """Main function of the SliceLine algorithm."""
        # prepare offset vectors and one-hot encoded input_x
        self._one_hot_encoder = OneHotEncoder(handle_unknown="ignore")
        x_encoded = self._one_hot_encoder.fit_transform(input_x)
        feature_domains: np.ndarray = np.array(
            [len(sub_array) for sub_array in self._one_hot_encoder.categories_]
        )
        feature_offset_end = np.cumsum(feature_domains)
        feature_offset_start = feature_offset_end - feature_domains

        # initialize statistics and basic slices
        n_col_x_encoded = x_encoded.shape[1]
        self.average_error_ = float(np.mean(errors))
        slices, statistics = self._create_and_score_basic_slices(
            x_encoded,
            n_col_x_encoded,
            errors,
        )

        # initialize top-k
        top_k_slices, top_k_statistics = self._maintain_top_k(
            slices,
            statistics,
            sp.csr_matrix((0, n_col_x_encoded)),
            np.zeros((0, 4)),
        )

        max_slice_scores, min_slice_scores = self._analyse_top_k(
            top_k_statistics
        )
        logger.debug(
            "Initial top-K: count=%i, max=%f, min=%f"
            % (top_k_slices.shape[0], max_slice_scores, min_slice_scores)
        )

        # lattice enumeration w/ size/error pruning, one iteration per level
        # termination condition (max #feature levels)
        level = 1
        min_condition = min(input_x.shape[1], self.max_l)
        while (
            (slices.shape[0] > 0)
            & (slices.sum() > 0)
            & (level < min_condition)
        ):
            level += 1

            # enumerate candidate join pairs, including size/error pruning
            slices, statistics = self._get_pruned_s_r(slices, statistics)
            nr_s = slices.shape[0]
            slices = self._get_pair_candidates(
                slices,
                statistics,
                top_k_statistics,
                level,
                n_col_x_encoded,
                feature_domains,
                feature_offset_start,
                feature_offset_end,
            )

            logger.debug("Level %i:" % level)
            logger.debug(
                " -- generated paired slice candidates: %i -> %i"
                % (nr_s, slices.shape[0])
            )

            # extract and evaluate candidate slices
            statistics = self._eval_slice(x_encoded, errors, slices, level)

            # maintain top-k after evaluation
            top_k_slices, top_k_statistics = self._maintain_top_k(
                slices, statistics, top_k_slices, top_k_statistics
            )

            max_slice_scores, min_slice_scores = self._analyse_top_k(
                top_k_statistics
            )
            valid = np.sum(
                (statistics[:, 3] >= self.min_sup) & (statistics[:, 1] > 0)
            )
            logger.debug(
                " -- valid slices after eval: %s/%i" % (valid, slices.shape[0])
            )
            logger.debug(
                " -- top-K: count=%i, max=%f, min=%f"
                % (top_k_slices.shape[0], max_slice_scores, min_slice_scores)
            )

        self._top_slices_enc = top_k_slices.copy()
        if top_k_slices.shape[0] == 0:
            self.top_slices_ = np.empty((0, input_x.shape[1]))
        else:
            self.top_slices_ = self._one_hot_encoder.inverse_transform(
                top_k_slices
            )

        # compute slices' average errors
        top_k_statistics = np.column_stack(
            (
                top_k_statistics,
                np.divide(top_k_statistics[:, 1], top_k_statistics[:, 3]),
            )
        )

        # transform statistics to a list of dict
        statistics_names = [
            "slice_score",
            "sum_slice_error",
            "max_slice_error",
            "slice_size",
            "slice_average_error",
        ]
        self.top_slices_statistics_ = [
            {
                stat_name: stat_value
                for stat_value, stat_name in zip(statistic, statistics_names)
            }
            for statistic in top_k_statistics
        ]

        logger.debug("Terminated at level %i." % level)
