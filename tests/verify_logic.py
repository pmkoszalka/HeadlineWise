from src.core.llm_provider import MockProvider
from src.core.schemas import PackagingOutput


class TestMockProvider:
    """Unified pytest-based verification for MockProvider."""

    def test_generate_packaging(self):
        provider = MockProvider()
        result = provider.generate_packaging("Some dummy article about the Fed.")

        assert isinstance(result, PackagingOutput)
        assert len(result.headlines) == 5
        assert "social_posts" in result.model_dump()

    def test_assess_packaging(self):
        provider = MockProvider()
        packaging = provider.generate_packaging("text")
        # Ensure assess_packaging doesn't crash and returns assessment
        result = provider.assess_packaging(packaging, "text")
        assert result.headline_assessment is not None
        assert len(result.headline_assessment) == 5
