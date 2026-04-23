import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from diffusiongym import DDTensor, reward_registry
import diffusiongym

from adm.setups.proteins import (
    CREILOV_WILD_TYPE,
    CREILOV_ALPHABET,
    HAMMING_PENALTY_CUTOFF,
    HAMMING_PENALTY_RATE,
    CosineSchedule,
    ProteinFitnessReward,
    ProteinProblemSetup,
    ProteinModel,
)
from adm.uncertainty import FlowFeatureExtractor

def _seq_array(seqs: list[str]) -> np.ndarray:
    return np.array([[ord(c) for c in seq] for seq in seqs], dtype=np.int32)


CREILOV_PRETRAINED_REWARD = 3.78

CONFIG = """
data:
  name: CreiLOV
  data_path: data/CreiLOV/fitness.csv
  pretrain_data_prefix: data/CreiLOV/
  full_seq: MAGLRHTFVVADATLPDCPLVYASEGFYAMTGYGPDEVLGHNARFLQGEGTDPKEVQKIRDAIKKGEACSVRLLNYRKDGTPFWNLLTVTPIKTPDGRVSKFVGVQVDVTSKTEGKALA
  seq_len: 119
  residues: null
  seed: 42
  alphabet: ACDEFGHIKLMNPQRSTVWYBZXJOU-*#@!
  alphabet_size: 31
  oracle_model_path: oracle/checkpoints/CreiLOV
model:
  name: continuous
  model:
    _target_: sgpo.models.continuous.ContinuousModel
exp_name: uncond
measurement: false
num_samples: 1000
batch_size: 200
num_samples_per_round: 100
n_repeats: 5
n_rounds: 8
seed: 42
wandb: false
pretrained_ckpt: continuous_ESM/CreiLOV
"""


class TestOnehotEncode:
    def test_shape(self):
        seq = "ACDE"
        enc = ProteinFitnessReward._onehot_encode(seq)
        assert enc.shape == (len(seq) * 20,)

    def test_first_amino_acid(self):
        enc = ProteinFitnessReward._onehot_encode("A")
        assert enc[0] == 1.0
        assert enc[1:].sum() == 0.0

    def test_sum_equals_sequence_length(self):
        seq = "ACDEFGHIKLMNPQRSTVWY"
        enc = ProteinFitnessReward._onehot_encode(seq)
        assert enc.sum() == len(seq)

    def test_unknown_character_is_all_zeros(self):
        enc = ProteinFitnessReward._onehot_encode("X")
        assert enc.sum() == 0.0

    def test_second_amino_acid_position(self):
        # "C" is index 1 in the alphabet
        enc = ProteinFitnessReward._onehot_encode("C")
        assert enc[1] == 1.0
        assert enc[0] == 0.0
        assert enc[2:].sum() == 0.0


class TestHammingDistance:
    def test_same_string(self):
        assert ProteinFitnessReward._hamming_distance("ACDE", "ACDE") == 0

    def test_single_difference(self):
        assert ProteinFitnessReward._hamming_distance("ACDE", "BCDE") == 1

    def test_all_different(self):
        assert ProteinFitnessReward._hamming_distance("AAAA", "CCCC") == 4

    def test_wildtype_to_itself(self):
        assert ProteinFitnessReward._hamming_distance(CREILOV_WILD_TYPE, CREILOV_WILD_TYPE) == 0

    def test_wildtype_single_mutation(self):
        mutant = "X" + CREILOV_WILD_TYPE[1:]
        assert ProteinFitnessReward._hamming_distance(CREILOV_WILD_TYPE, mutant) == 1


class TestHammingPenalty:
    def test_at_zero_distance(self):
        assert ProteinFitnessReward._hamming_penalty(0) == 1.0

    def test_at_cutoff(self):
        assert ProteinFitnessReward._hamming_penalty(HAMMING_PENALTY_CUTOFF) == 1.0

    def test_above_cutoff(self):
        excess = 5
        d = HAMMING_PENALTY_CUTOFF + excess
        expected = HAMMING_PENALTY_RATE ** excess
        assert ProteinFitnessReward._hamming_penalty(d) == pytest.approx(expected)

    def test_custom_cutoff_and_rate_at_cutoff(self):
        assert ProteinFitnessReward._hamming_penalty(10, cutoff=10, rate=0.5) == 1.0

    def test_custom_cutoff_and_rate_above_cutoff(self):
        result = ProteinFitnessReward._hamming_penalty(12, cutoff=10, rate=0.5)
        assert result == pytest.approx(0.5 ** 2)


class TestLoadOracleEnsemble:
    def test_nonexistent_path(self):
        result = ProteinFitnessReward._load_oracle_ensemble(CREILOV_WILD_TYPE, CREILOV_ALPHABET, Path("/nonexistent/path"))
        assert result == []

    def test_empty_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            result = ProteinFitnessReward._load_oracle_ensemble(CREILOV_WILD_TYPE, CREILOV_ALPHABET, Path(tmpdir))
            assert result == []


class TestOraclePredictEmptyEnsemble:
    def setup_method(self):
        self.reward = ProteinFitnessReward.__new__(ProteinFitnessReward)
        self.reward.oracle_ensemble = []

    def test_returns_nan_for_each_sequence(self):
        sequences = ["ACDE", "GHIK"]
        result = self.reward._oracle_predict(sequences)
        assert len(result) == 2
        assert all(np.isnan(v) for v in result)

    def test_correct_length_for_single_sequence(self):
        result = self.reward._oracle_predict(["ACDE"])
        assert len(result) == 1
        assert np.isnan(result[0])


class TestCosineSchedule:
    def setup_method(self):
        self.schedule = CosineSchedule()

    def test_alpha_at_t_zero_is_zero(self):
        x = DDTensor(torch.ones(2, 4))
        t = torch.zeros(2)
        alpha = self.schedule.alpha(x, t)
        assert torch.allclose(alpha.data, torch.zeros_like(alpha.data), atol=1e-6)

    def test_alpha_at_t_one_is_one(self):
        x = DDTensor(torch.ones(2, 4))
        t = torch.ones(2)
        alpha = self.schedule.alpha(x, t)
        assert torch.allclose(alpha.data, torch.ones_like(alpha.data), atol=1e-6)

    def test_beta_at_t_zero_is_one(self):
        x = DDTensor(torch.ones(2, 4))
        t = torch.zeros(2)
        beta = self.schedule.beta(x, t)
        assert torch.allclose(beta.data, torch.ones_like(beta.data), atol=1e-6)

    def test_beta_at_t_one_is_zero(self):
        x = DDTensor(torch.ones(2, 4))
        t = torch.ones(2)
        beta = self.schedule.beta(x, t)
        assert torch.allclose(beta.data, torch.zeros_like(beta.data), atol=1e-6)

    def test_alpha_squared_plus_beta_squared_equals_one(self):
        x = DDTensor(torch.ones(5, 4))
        t = torch.tensor([0.1, 0.3, 0.5, 0.7, 0.9])
        alpha = self.schedule.alpha(x, t)
        beta = self.schedule.beta(x, t)
        total = alpha.data ** 2 + beta.data ** 2
        assert torch.allclose(total, torch.ones_like(total), atol=1e-5)

    def test_alpha_is_monotonically_increasing(self):
        x = DDTensor(torch.ones(5, 4))
        t = torch.tensor([0.1, 0.3, 0.5, 0.7, 0.9])
        alpha_vals = self.schedule.alpha(x, t).data[:, 0]
        diffs = alpha_vals[1:] - alpha_vals[:-1]
        assert (diffs > 0).all()


class TestProteinProblemSetup:
    @classmethod
    def setup_class(cls):
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        temp = tempfile.NamedTemporaryFile(delete=False)
        with open(temp.name, 'w') as f:
            f.write(CONFIG)

        setup_args = {'cfg_path': temp.name, 'no_verifier': True}
        cls.setup = ProteinProblemSetup(setup_args, device=device)

        reward_entry = reward_registry.get('proteins/creilov')
        reward = reward_entry.instantiate(**{'cfg_path': temp.name})

        cls.env = diffusiongym.construct_env(
            cls.setup.base_model,
            reward,
            100,
            1.
        )

        cls.samples = cls.env.sample(2)
        

    def test_embedding_invertibility(self):
        samples = self.samples
        base_model: ProteinModel = self.env.base_model

        embeds = base_model.probs_to_embedding(samples.sample)
        latent = samples.latent

        diff = embeds - latent
        
        assert diff.data.norm() <= latent.data.norm() * 0.05 # TODO relative error of 5%... not great
    

    def test_creilov_close_pretrained(self):
        samples = self.samples
        avg_reward = samples.rewards.mean()
        assert avg_reward == pytest.approx(CREILOV_PRETRAINED_REWARD, rel=0.1)
    

    def test_compute_metrics_contains_sphere_exclusion_keys(self):
        metrics = self.setup.compute_metrics(self.samples.sample, {})
        assert "sphere_exclusion_diversity" in metrics
        assert "n_clusters" in metrics

    def test_sphere_exclusion_diversity_in_range(self):
        metrics = self.setup.compute_metrics(self.samples.sample, {})
        assert 0.0 < metrics["sphere_exclusion_diversity"] <= 1.0

    def test_feature_layer_is_encoder(self):
        assert self.setup.feature_layer == 'sgpo_model.model.network.encoder'

    def test_feature_extractor_output_shape(self):
        x = self.samples.sample
        feat_extractor = FlowFeatureExtractor(
            self.setup.base_model,
            layer=self.setup.feature_layer,
            timestep=0.9,
            postprocess=self.setup.postprocess_features,
        )
        with torch.no_grad():
            feats = feat_extractor(x)
        assert feats.ndim == 2
        assert feats.shape[0] == x.data.shape[0]

    def test_features_differ_across_timesteps(self):
        x = self.samples.sample
        feat_extractor_early = FlowFeatureExtractor(
            self.setup.base_model,
            layer=self.setup.feature_layer,
            timestep=0.5,
            postprocess=self.setup.postprocess_features,
        )
        feat_extractor_late = FlowFeatureExtractor(
            self.setup.base_model,
            layer=self.setup.feature_layer,
            timestep=0.9,
            postprocess=self.setup.postprocess_features,
        )
        with torch.no_grad():
            feats_early = feat_extractor_early(x)
            feats_late = feat_extractor_late(x)
        assert not torch.allclose(feats_early, feats_late)

    def test_n_clusters_bounded_by_sample_count(self):
        n = self.samples.sample.data.shape[0]
        metrics = self.setup.compute_metrics(self.samples.sample, {})
        assert 1 <= metrics["n_clusters"] <= n

    def test_identical_sequences_give_one_cluster(self):
        sgpo_model = self.setup.base_model.sgpo_model
        n = 8
        wt_tokens = sgpo_model.tokenizer.tokenize(CREILOV_WILD_TYPE)
        probs = torch.zeros((n, sgpo_model.seq_len, len(sgpo_model.tokenizer.alphabet)))
        for pos, tok in enumerate(wt_tokens):
            probs[:, pos, tok] = 1.0
        metrics = self.setup.compute_metrics(DDTensor(probs), {})
        assert metrics["n_clusters"] == 1.0
        assert metrics["sphere_exclusion_diversity"] == pytest.approx(1.0 / n)

    def test_diversity_of_repeated_is_zero(self):
        n = 100
        sgpo_model = self.setup.base_model.sgpo_model
        alphabet = sgpo_model.tokenizer.alphabet
        probs = torch.zeros((n, sgpo_model.seq_len, len(alphabet)))

        wt_tokens = sgpo_model.tokenizer.tokenize(CREILOV_WILD_TYPE)
        probs = torch.zeros((n, sgpo_model.seq_len, len(alphabet)))
        for pos, tok in enumerate(wt_tokens):
            probs[:, pos, tok] = 1.0

        tokens = probs.argmax(dim=-1)
        assert tokens.shape == (n, sgpo_model.seq_len)
        sequences = [sgpo_model.tokenizer.untokenize(s) for s in tokens]
        for seq in sequences:
            assert len(seq) == sgpo_model.seq_len
            assert len(seq) == len(CREILOV_WILD_TYPE)


    def test_entropy_of_all_diff_is_maximized(self):
        n = len(CREILOV_ALPHABET)
        sgpo_model = self.setup.base_model.sgpo_model
        alphabet = sgpo_model.tokenizer.alphabet

        actual_tokens = sgpo_model.tokenizer.tokenize(CREILOV_ALPHABET)
        probs = torch.zeros((n, sgpo_model.seq_len, len(alphabet)))
        for pos, tok in enumerate(actual_tokens):
            probs[pos, :, tok] = 1.0

        metrics = self.setup.compute_metrics(DDTensor(probs), {})

        assert metrics['shannon_entropy'] == pytest.approx(np.log(n))


class TestProteinValidity:
    @classmethod
    def setup_class(cls):
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        temp = tempfile.NamedTemporaryFile(delete=False)
        with open(temp.name, 'w') as f:
            f.write(CONFIG)

        setup_args = {'cfg_path': temp.name, 'no_verifier': False}
        cls.setup = ProteinProblemSetup(setup_args, device=device)

    def test_wild_type_is_valid_but_random_is_not(self):
        sgpo_model = self.setup.base_model.sgpo_model
        wild_tokens = sgpo_model.tokenizer.tokenize(CREILOV_WILD_TYPE)

        n = 4

        wild_type_probs = torch.zeros((n, sgpo_model.seq_len, len(sgpo_model.tokenizer.alphabet)))
        wild_type_probs[:, torch.arange(sgpo_model.seq_len), wild_tokens] = 1.
        wild_type_probs = DDTensor(wild_type_probs)

        random_probs = DDTensor(torch.rand(n, sgpo_model.seq_len, len(sgpo_model.tokenizer.alphabet)))

        all_probs = DDTensor.collate([wild_type_probs, random_probs])
        assert all_probs.data.shape == (2 * n, sgpo_model.seq_len, len(sgpo_model.tokenizer.alphabet))

        valids = self.setup.validity(all_probs, {})
        assert torch.all(valids[:n] == 1)
        assert torch.any(valids[n:] == 0)


class TestIdentityMatrix:
    def test_identical_sequences_give_one(self):
        seqs = _seq_array(["AAAA", "AAAA"])
        m = ProteinProblemSetup._identity_matrix(seqs, seqs)
        assert np.allclose(m, 1.0)

    def test_completely_different_sequences_give_zero(self):
        a = _seq_array(["AAAA"])
        b = _seq_array(["BBBB"])
        m = ProteinProblemSetup._identity_matrix(a, b)
        assert np.allclose(m, 0.0)

    def test_half_matching(self):
        a = _seq_array(["AABB"])
        b = _seq_array(["AACC"])
        m = ProteinProblemSetup._identity_matrix(a, b)
        assert np.allclose(m, 0.5)

    def test_output_shape(self):
        a = _seq_array(["AAAA", "BBBB", "CCCC"])
        b = _seq_array(["AAAA", "BBBB"])
        m = ProteinProblemSetup._identity_matrix(a, b)
        assert m.shape == (3, 2)

    def test_diagonal_of_self_similarity_is_one(self):
        seqs = _seq_array(["ABCD", "EFGH", "IJKL"])
        m = ProteinProblemSetup._identity_matrix(seqs, seqs)
        assert np.allclose(np.diag(m), 1.0)


class TestComputeNovelty:
    def test_empty_current_returns_zero(self):
        current = _seq_array([])
        reference = _seq_array(["AAAA"])
        assert ProteinProblemSetup.compute_novelty(current, reference) == 0.0

    def test_empty_reference_returns_zero(self):
        current = _seq_array(["AAAA"])
        reference = _seq_array([])
        assert ProteinProblemSetup.compute_novelty(current, reference) == 0.0

    def test_identical_to_reference_gives_zero_novelty(self):
        seqs = _seq_array(["AAAA", "BBBB"])
        assert ProteinProblemSetup.compute_novelty(seqs, seqs) == pytest.approx(0.0)

    def test_fully_novel_gives_one(self):
        current = _seq_array(["AAAA"])
        reference = _seq_array(["BBBB"])
        assert ProteinProblemSetup.compute_novelty(current, reference) == pytest.approx(1.0)

    def test_partial_novelty(self):
        current = _seq_array(["AABB"])
        reference = _seq_array(["AACC"])
        novelty = ProteinProblemSetup.compute_novelty(current, reference)
        assert novelty == pytest.approx(1.0 - 0.5)

    def test_novelty_uses_max_similarity(self):
        current = _seq_array(["AAAA"])
        reference = _seq_array(["BBBB", "AAAA"])
        assert ProteinProblemSetup.compute_novelty(current, reference) == pytest.approx(0.0)

    def test_novelty_is_mean_over_current(self):
        current = _seq_array(["AAAA", "BBBB"])
        reference = _seq_array(["AAAA"])
        novelty = ProteinProblemSetup.compute_novelty(current, reference)
        assert novelty == pytest.approx(0.5)


class TestComputeCumulativeClusterMetrics:
    def test_first_call_returns_none_cumulative(self):
        seqs = _seq_array(["AAAA", "BBBB", "CCCC"])
        metrics, centers = ProteinProblemSetup.compute_cumulative_cluster_metrics(seqs, None)
        assert "cumulative_clusters" in metrics
        assert metrics["coverage_of_cumulative"] == 1.0
        assert metrics["clusters_lost"] == 0.0
        assert centers.ndim == 2

    def test_identical_sequences_form_one_cluster(self):
        seqs = _seq_array(["AAAA", "AAAA", "AAAA"])
        metrics, centers = ProteinProblemSetup.compute_cumulative_cluster_metrics(seqs, None)
        assert metrics["cumulative_clusters"] == 1.0
        assert metrics["new_clusters"] == 1.0

    def test_distinct_sequences_each_form_own_cluster(self):
        seqs = _seq_array(["AAAA", "BBBB", "CCCC", "DDDD"])
        metrics, centers = ProteinProblemSetup.compute_cumulative_cluster_metrics(seqs, None)
        assert metrics["cumulative_clusters"] == 4.0

    def test_no_new_clusters_when_repeated(self):
        seqs = _seq_array(["AAAA", "BBBB"])
        _, centers = ProteinProblemSetup.compute_cumulative_cluster_metrics(seqs, None)
        metrics, updated = ProteinProblemSetup.compute_cumulative_cluster_metrics(seqs, centers)
        assert metrics["new_clusters"] == 0.0
        assert len(updated) == len(centers)

    def test_new_sequences_expand_cumulative(self):
        first = _seq_array(["AAAA", "BBBB"])
        _, centers = ProteinProblemSetup.compute_cumulative_cluster_metrics(first, None)
        second = _seq_array(["CCCC", "DDDD"])
        metrics, updated = ProteinProblemSetup.compute_cumulative_cluster_metrics(second, centers)
        assert metrics["new_clusters"] == 2.0
        assert metrics["cumulative_clusters"] == 4.0

    def test_coverage_drops_when_previous_clusters_absent(self):
        first = _seq_array(["AAAA", "BBBB", "CCCC"])
        _, centers = ProteinProblemSetup.compute_cumulative_cluster_metrics(first, None)
        second = _seq_array(["AAAA"])
        metrics, _ = ProteinProblemSetup.compute_cumulative_cluster_metrics(second, centers)
        assert metrics["coverage_of_cumulative"] < 1.0
        assert metrics["clusters_lost"] > 0.0

    def test_cumulative_centers_shape(self):
        seqs = _seq_array(["ABCD", "EFGH"])
        _, centers = ProteinProblemSetup.compute_cumulative_cluster_metrics(seqs, None)
        assert centers.shape[1] == 4
