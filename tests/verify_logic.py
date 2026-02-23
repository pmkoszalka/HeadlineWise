import sys
from src.core.llm_provider import MockProvider
from src.core.schemas import PackagingOutput


def test_mock_provider():
    print("Testing MockProvider...")
    provider = MockProvider()
    result = provider.generate_packaging("Some dummy article about the Fed.")

    assert isinstance(result, PackagingOutput)
    assert len(result.headlines) == 5
    assert "social_posts" in result.model_dump()
    print("✅ MockProvider test passed!")


if __name__ == "__main__":
    try:
        test_mock_provider()
    except Exception as e:
        print(f"❌ Test failed: {e}")
        sys.exit(1)
