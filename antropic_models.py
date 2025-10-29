import anthropic
import dotenv
import os

# Load environment variables
dotenv.load_dotenv()

# Initialize Anthropic client
client = anthropic.Anthropic()

print("Fetching available Anthropic models...")
print("=" * 60)

try:
    # List available models
    models_response = client.models.list(limit=20)
    
    if hasattr(models_response, 'data'):
        models = models_response.data
        print(f"\nFound {len(models)} models:\n")
        
        for model in models:
            print(f"Model ID: {model.id}")
            if hasattr(model, 'display_name'):
                print(f"  Display Name: {model.display_name}")
            if hasattr(model, 'created_at'):
                print(f"  Created: {model.created_at}")
            print()
    else:
        print("Response:", models_response)
        
except Exception as e:
    print(f"Error fetching models: {e}")
    print("\nNote: The Anthropic API may not support model listing.")
    print("Using known Claude model identifiers instead:\n")
    
    # Known Claude models as of October 2024
    known_models = [
        "claude-3-5-sonnet-20241022",  # Claude 3.5 Sonnet v2 (Oct 2024)
        "claude-3-5-haiku-20241022",   # Claude 3.5 Haiku (Oct 2024)
        "claude-3-5-sonnet-20240620",  # Claude 3.5 Sonnet v1 (June 2024)
        "claude-3-opus-20240229",      # Claude 3 Opus
        "claude-3-sonnet-20240229",    # Claude 3 Sonnet
        "claude-3-haiku-20240307",     # Claude 3 Haiku
    ]
    
    print("Known Claude Models:")
    print("-" * 40)
    for model_id in known_models:
        print(f"  • {model_id}")
    
    # Test if API key is working
    print("\n" + "=" * 60)
    print("Testing API key with a simple request...")
    try:
        test_response = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=10,
            messages=[{"role": "user", "content": "Hi"}]
        )
        print("✅ API key is valid and working!")
        print(f"Test response: {test_response.content[0].text}")
    except anthropic.NotFoundError:
        print("❌ Model not found. Your API key may not have access to this model.")
    except Exception as test_error:
        print(f"❌ API key test failed: {test_error}")
