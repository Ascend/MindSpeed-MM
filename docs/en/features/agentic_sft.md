# FSDP2 Supports Agentic SFT

## Feature Overview

Agentic SFT is a training method for multimodal models with tool call capabilities. Under the FSDP2 framework, this feature enables models to learn how to understand user intent, invoke external tools, and process results returned by tools, thereby achieving more intelligent multi-turn conversational interaction.

## Features

- Supports training on multi-turn conversation data containing tool calls (`tool_call`) and tool responses (`tool_response`).
- Supports injecting tool definitions (`tools schema`) into the system prompt, enabling a model to understand the available tool capabilities.
- Supports joint training of multimodal data (images, videos, audio) with tool calls.
- Compatible with the existing FSDP2 training pipeline and can be enabled without additional configuration.

## Data Format

### Data Structure

Agentic SFT data uses the JSON format, and each data record contains the following key fields:

```json
{
    "messages": [
        {"role": "system", "content": "System prompt content"},
        {"role": "user", "content": "User input"},
        {"role": "assistant", "content": "Assistant reply"},
        {"role": "user", "content": "User input"},
        {"role": "tool_call", "content": "Tool call request"},
        {"role": "tool_response", "content": "Tool-returned result"},
        {"role": "assistant", "content": "Assistant reply based on the tool result"}
    ],
    "audios": "Audio file path (optional)",
    "images": "Image file path (optional)",
    "videos": "Video file path (optional)",
    "tools": ["Tool definition list (optional)"]
}
```

### `messages` Details

`messages` is the core field and contains the complete conversation history. Each message object contains two properties: `role` and `content`:

| Role | Meaning | Description |
|---------|------|------|
| `system` | System prompt | Defines the assistant's role and behavioral norms, usually placed at the beginning of messages |
| `user` | User input | The user's question or request |
| `assistant` | Assistant reply | The model's reply  |
| `tool_call` | Tool call | A tool call request initiated by a model, using a special format |
| `tool_response` | Tool response | The result returned by an external tool |

### `tool_call` and `tool_response` Constraints

**Important rules**:

1. **Alternation principle**: `tool_call` and `tool_response` must appear in pairs, and `tool_call` must precede `tool_response`.
2. **Position constraint**: The tool call flow typically appears after user input, forming an interaction chain of "user → tool_call → tool_response → assistant".

## Example

```json
[
    {
        "messages": [
            {"role": "system", "content": "You are a professional and efficient AI customer service agent. The current time of this conversation is: 2024-06-15 14:30:00 (today is Saturday)."},
            {"role": "user", "content": "Hello, are you there?"},
            {"role": "assistant", "content": "Hello, may I ask if this is Ms. Du?"},
            {"role": "user", "content": "Hello, yes, I am Ms. Du."},
            {"role": "assistant", "content": "Ms. Du, thank you for confirming. I am a staff member of the Center for Disease Control and Prevention. We noticed that you have not yet received the new influenza vaccine. For your health and public health safety, we recommend that you get vaccinated as soon as possible."},
            {"role": "user", "content": "Then can I choose Wednesday afternoon at 3:00 PM?"},
            {"role": "tool_call", "content": "<tool_call>\n{name: register_vaccine_appointment, arguments: {appointment_time: Wednesday afternoon at 3:00 PM}}\n</tool_call>"},
            {"role": "tool_response", "content": "{status: success, message: Appointment successfully registered}"},
            {"role": "assistant", "content": "All right, your vaccination appointment has been registered for Wednesday afternoon at 3:00 PM. Please bring a valid ID and arrive at the designated vaccination site on time. Thank you for your support of the epidemic prevention and control efforts."},
            {"role": "user", "content": "Thank you for your hard work."},
            {"role": "assistant", "content": "You are welcome. This is our duty. Stay healthy, goodbye."}
        ],
        "audios": "/speeches/7_Katerina.wav",
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "register_vaccine_appointment",
                    "description": "Register the user's vaccination appointment",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "appointment_time": {
                                "type": "string",
                                "description": "The vaccination time selected by the user"
                            }
                        },
                        "required": ["appointment_time"]
                    }
                }
            }
        ]
    }
]
```

## Configuration

### Data Configuration

In the configuration file, set `formatting` to `multimodal_tool`.

```yaml
data:
  dataset_param:
    attr:
      formatting: multimodal_tool  # Use the multimodal_tool format converter.

    preprocess_parameters:
      template: qwen3_vl_nothink  # It is recommended to use the qwen3_vl_nothink or qwen3_omni_nothink template.
      # To use other templates, refer to the template registration code of qwen3_vl_nothink and pass tool_prompt = StringFormatter(slots=[tools_slot]).
```

### Supported Models

The models that support Agentic SFT:

- Qwen3.5 (the `qwen3_vl_nothink` template is recommended)
- Qwen3Omni (the `qwen3_omni_nothink` template is recommended)

## Precautions

1. **Data validation**: Before training, ensure that `tool_call` and `tool_response` are correctly paired. Unpaired tool calls will cause the data to be skipped.
2. **Template compatibility**: Ensure that the selected template supports the tool call format. Currently, the `qwen3_vl_nothink` template fully supports it.
3. **Multimodal support**: Agentic SFT supports joint training with image, video, and audio data, but ensure that the file paths are correct.
4. **Tool definition**: The `tools` field is optional. If you need to inject tool definitions into the system prompt, fill them in according to the specified format.
