import unittest

from mlx_engine.utils.specprefill import (
    DEFAULT_SPECPREFILL_KEEP_PCT,
    DEFAULT_SPECPREFILL_THRESHOLD,
    SpecPrefillOptions,
    resolve_specprefill_options,
)


class TestSpecPrefillOptions(unittest.TestCase):
    """Tests for public SpecPrefill option resolution and validation."""

    def test_disabled_by_default(self):
        options = resolve_specprefill_options(
            specprefill_toggle=None,
            specprefill_keep_pct=None,
            specprefill_threshold=None,
            specprefill_system_tokens=None,
            draft_model=object(),
        )

        self.assertIsNone(options)

    def test_tuning_without_toggle_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "require specprefill_toggle=True"):
            resolve_specprefill_options(
                specprefill_toggle=None,
                specprefill_keep_pct=0.2,
                specprefill_threshold=None,
                specprefill_system_tokens=None,
                draft_model=object(),
            )

    def test_enabled_without_draft_model_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "requires a loaded compatible draft"):
            resolve_specprefill_options(
                specprefill_toggle=True,
                specprefill_keep_pct=None,
                specprefill_threshold=None,
                specprefill_system_tokens=None,
                draft_model=None,
            )

    def test_enabled_with_draft_model_uses_defaults(self):
        options = resolve_specprefill_options(
            specprefill_toggle=True,
            specprefill_keep_pct=None,
            specprefill_threshold=None,
            specprefill_system_tokens=None,
            draft_model=object(),
        )

        self.assertIsNotNone(options)
        self.assertTrue(options.enabled)
        self.assertEqual(options.keep_pct, DEFAULT_SPECPREFILL_KEEP_PCT)
        self.assertEqual(options.threshold, DEFAULT_SPECPREFILL_THRESHOLD)
        self.assertEqual(options.system_tokens, 0)

    def test_enabled_with_draft_model_uses_explicit_values(self):
        options = resolve_specprefill_options(
            specprefill_toggle=True,
            specprefill_keep_pct=0.4,
            specprefill_threshold=2048,
            specprefill_system_tokens=16,
            draft_model=object(),
        )

        self.assertEqual(options, SpecPrefillOptions(True, 0.4, 2048, 16))

    def test_invalid_keep_pct_is_rejected(self):
        for keep_pct in (0, -0.1, 1.1, True, "0.2"):
            with self.subTest(keep_pct=keep_pct):
                with self.assertRaisesRegex(ValueError, "specprefill_keep_pct"):
                    SpecPrefillOptions(enabled=True, keep_pct=keep_pct)

    def test_invalid_threshold_is_rejected(self):
        for threshold in (0, -1, True, 1.2, "1024"):
            with self.subTest(threshold=threshold):
                with self.assertRaisesRegex(ValueError, "specprefill_threshold"):
                    SpecPrefillOptions(enabled=True, threshold=threshold)

    def test_invalid_system_tokens_is_rejected(self):
        for system_tokens in (-1, True, 1.2, "0"):
            with self.subTest(system_tokens=system_tokens):
                with self.assertRaisesRegex(ValueError, "specprefill_system_tokens"):
                    SpecPrefillOptions(enabled=True, system_tokens=system_tokens)


if __name__ == "__main__":
    unittest.main(verbosity=2)
