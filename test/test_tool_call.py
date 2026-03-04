#!/usr/bin/env python3
"""Test tool calling functionality of OCI-Anthropic Gateway."""

import httpx
import json

GATEWAY_URL = "http://localhost:8000"

def test_tool_call():
    """Test that tool calls are properly detected and converted."""

    # Simple request with a tool
    payload = {
        "model": "claude-3-5-sonnet-20241022",
        "max_tokens": 1024,
        "tools": [
            {
                "name": "get_weather",
                "description": "Get the current weather in a given location",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g. San Francisco, CA"
                        }
                    },
                    "required": ["location"]
                }
            }
        ],
        "messages": [
            {
                "role": "user",
                "content": "What's the weather in Tokyo?"
            }
        ]
    }

    print("=" * 60)
    print("Testing Tool Call Detection")
    print("=" * 60)
    print(f"\nSending request to: {GATEWAY_URL}/v1/messages")
    print(f"Tool: get_weather")
    print(f"User message: What's the weather in Tokyo?")
    print()

    with httpx.Client(timeout=60.0) as client:
        response = client.post(
            f"{GATEWAY_URL}/v1/messages",
            headers={"Content-Type": "application/json"},
            json=payload
        )

        print(f"Status: {response.status_code}")

        if response.status_code != 200:
            print(f"Error: {response.text}")
            return False

        data = response.json()

        print(f"\nResponse:")
        print(f"  ID: {data.get('id')}")
        print(f"  Stop Reason: {data.get('stop_reason')}")
        print(f"  Content Blocks: {len(data.get('content', []))}")

        # Check if tool_use is present
        tool_uses = [b for b in data.get('content', []) if b.get('type') == 'tool_use']

        if tool_uses:
            print(f"\n✅ SUCCESS! Tool call detected and converted correctly!")
            for i, tool in enumerate(tool_uses):
                print(f"\n  Tool Call #{i+1}:")
                print(f"    Type: {tool.get('type')}")
                print(f"    ID: {tool.get('id')}")
                print(f"    Name: {tool.get('name')}")
                print(f"    Input: {json.dumps(tool.get('input', {}), ensure_ascii=False)}")
            return True
        else:
            print(f"\n❌ FAILED! No tool_use block found in response.")
            print(f"\n  Actual content:")
            for block in data.get('content', []):
                if block.get('type') == 'text':
                    print(f"    Text: {block.get('text', '')[:200]}...")
            return False


def test_streaming_tool_call():
    """Test streaming mode with tool calls."""

    payload = {
        "model": "claude-3-5-sonnet-20241022",
        "max_tokens": 1024,
        "stream": True,
        "tools": [
            {
                "name": "calculate",
                "description": "Perform a calculation",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "expression": {
                            "type": "string",
                            "description": "The math expression to evaluate"
                        }
                    },
                    "required": ["expression"]
                }
            }
        ],
        "messages": [
            {
                "role": "user",
                "content": "Calculate 123 * 456"
            }
        ]
    }

    print("\n" + "=" * 60)
    print("Testing STREAMING Tool Call Detection")
    print("=" * 60)
    print(f"\nSending streaming request...")
    print(f"Tool: calculate")
    print(f"User message: Calculate 123 * 456")
    print()

    tool_uses_found = []
    text_content = []

    with httpx.Client(timeout=60.0) as client:
        with client.stream(
            "POST",
            f"{GATEWAY_URL}/v1/messages",
            headers={"Content-Type": "application/json"},
            json=payload
        ) as response:
            print(f"Status: {response.status_code}")
            print("\nStreaming events:")

            for line in response.iter_lines():
                if not line:
                    continue

                if line.startswith("event: "):
                    event_type = line[7:]
                elif line.startswith("data: "):
                    data_str = line[6:]
                    if data_str == "[DONE]":
                        continue

                    try:
                        data = json.loads(data_str)

                        if event_type == "content_block_start":
                            block = data.get('content_block', {})
                            if block.get('type') == 'tool_use':
                                print(f"  - Tool use started: {block.get('name')}")
                                tool_uses_found.append({
                                    'id': block.get('id'),
                                    'name': block.get('name'),
                                    'input': {}
                                })

                        elif event_type == "content_block_delta":
                            delta = data.get('delta', {})
                            if delta.get('type') == 'text_delta':
                                text_content.append(delta.get('text', ''))
                            elif delta.get('type') == 'input_json_delta':
                                partial = delta.get('partial_json', '{}')
                                print(f"  - Tool input delta: {partial[:50]}...")
                                try:
                                    tool_uses_found[-1]['input'] = json.loads(partial)
                                except:
                                    pass

                        elif event_type == "message_delta":
                            stop_reason = data.get('delta', {}).get('stop_reason')
                            print(f"\n  Stop reason: {stop_reason}")

                    except json.JSONDecodeError:
                        pass

    print()
    if tool_uses_found:
        print(f"✅ SUCCESS! Streaming tool call detected!")
        for tool in tool_uses_found:
            print(f"  - {tool.get('name')}: {json.dumps(tool.get('input', {}), ensure_ascii=False)}")
        return True
    else:
        print(f"❌ FAILED! No tool_use in streaming response.")
        if text_content:
            print(f"  Text received: {''.join(text_content)[:200]}...")
        return False


if __name__ == "__main__":
    print("\n" + "#" * 60)
    print("# OCI-Anthropic Gateway Tool Call Test")
    print("#" * 60 + "\n")

    # Test non-streaming
    result1 = test_tool_call()

    # Test streaming
    result2 = test_streaming_tool_call()

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Non-streaming: {'✅ PASS' if result1 else '❌ FAIL'}")
    print(f"  Streaming:     {'✅ PASS' if result2 else '❌ FAIL'}")
    print("=" * 60)

    if result1 and result2:
        print("\n🎉 All tests passed! Tool calling is working correctly.")
    else:
        print("\n⚠️  Some tests failed. Check the output above for details.")
